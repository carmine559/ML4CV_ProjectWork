"""
Train SeqNet on PRW (official recipe by default).

  python Scripts/train.py --cfg configs/prw.yaml --data-root /path/to/PRW --output-dir exp_prw \
      [--amp bf16] [--batch-size 5 --lr 0.003] [--resume auto] [--eval-period 0] [--seed 1] \
      [--opts KEY VALUE ...]

--resume auto  resumes from the newest epoch_*.pth in --output-dir (no-op if none): use it in
               every SLURM submission. Checkpoints are written after every epoch.
"""
import argparse
import datetime
import json
import os
import os.path as osp
import sys
import time

import torch

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))

from config import dump_config, load_config  # noqa: E402
from engine import evaluate_performance, train_one_epoch  # noqa: E402
from prw import build_test_loaders, build_train_loader  # noqa: E402
from seqnet import SeqNet  # noqa: E402
from utils import (CSVLogger, amp_dtype, list_checkpoints, load_checkpoint,  # noqa: E402
                   prune_checkpoints, save_checkpoint, set_random_seed)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Train SeqNet on PRW.")
    p.add_argument("--cfg", help="YAML config (defaults = official SeqNet PRW recipe)")
    p.add_argument("--data-root", help="PRW root (overrides $PRW_ROOT and INPUT.DATA_ROOT)")
    p.add_argument("--output-dir")
    p.add_argument("--amp", choices=["off", "fp16", "bf16"])
    p.add_argument("--batch-size", type=int, help="INPUT.BATCH_SIZE_TRAIN (scale --lr linearly!)")
    p.add_argument("--lr", type=float, help="SOLVER.BASE_LR")
    p.add_argument("--epochs", type=int, help="SOLVER.MAX_EPOCHS")
    p.add_argument("--workers", type=int, help="INPUT.NUM_WORKERS_TRAIN/TEST")
    p.add_argument("--eval-period", type=int, help="evaluate every N epochs (0 = never)")
    p.add_argument("--seed", type=int)
    p.add_argument("--device")
    p.add_argument("--resume", help="'auto' or path to a checkpoint")
    p.add_argument("--max-iters", type=int, default=0, help="debug: stop each epoch after N iters")
    p.add_argument("--opts", nargs="*", default=[], help="KEY VALUE pairs, e.g. SOLVER.BASE_LR 0.0012")
    return p.parse_args(argv)


def apply_cli(cfg, args):
    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir
    if args.amp:
        cfg.AMP = args.amp
    if args.batch_size:
        cfg.INPUT.BATCH_SIZE_TRAIN = args.batch_size
    if args.lr:
        cfg.SOLVER.BASE_LR = args.lr
    if args.epochs:
        cfg.SOLVER.MAX_EPOCHS = args.epochs
    if args.workers is not None:
        cfg.INPUT.NUM_WORKERS_TRAIN = cfg.INPUT.NUM_WORKERS_TEST = args.workers
    if args.eval_period is not None:
        cfg.EVAL_PERIOD = args.eval_period
    if args.seed is not None:
        cfg.SEED = args.seed
    if args.device:
        cfg.DEVICE = args.device
    return cfg


class _Truncated:
    """Debug helper: limit an iterable DataLoader to n batches."""

    def __init__(self, loader, n):
        self.loader, self.n, self.dataset = loader, n, loader.dataset

    def __len__(self):
        return min(self.n, len(self.loader))

    def __iter__(self):
        for i, b in enumerate(self.loader):
            if i >= self.n:
                break
            yield b


def main(argv=None):
    args = parse_args(argv)
    cfg = apply_cli(load_config(args.cfg, args.opts, args.data_root), args)
    device = torch.device(cfg.DEVICE)
    if cfg.SEED >= 0:
        set_random_seed(cfg.SEED, cfg.CUDNN_BENCHMARK)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    dtype = amp_dtype(cfg.AMP)

    print("Creating model")
    model = SeqNet(cfg).to(device)

    print("Loading data")
    train_loader = build_train_loader(cfg)
    if args.max_iters > 0:
        train_loader = _Truncated(train_loader, args.max_iters)
    test_loaders = build_test_loaders(cfg) if cfg.EVAL_PERIOD > 0 else None

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=cfg.SOLVER.BASE_LR, momentum=cfg.SOLVER.SGD_MOMENTUM,
        weight_decay=cfg.SOLVER.WEIGHT_DECAY,
    )
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=cfg.SOLVER.LR_DECAY_MILESTONES, gamma=cfg.SOLVER.GAMMA
    )
    scaler = torch.amp.GradScaler(device.type) if cfg.AMP == "fp16" else None

    start_epoch = 0
    ckpt_path = args.resume
    if ckpt_path == "auto":
        ckpts = list_checkpoints(cfg.OUTPUT_DIR)
        ckpt_path = ckpts[-1] if ckpts else None
        if ckpt_path is None:
            print("--resume auto: no checkpoint found, starting from scratch")
    if ckpt_path:
        ckpt = load_checkpoint(ckpt_path, model, optimizer, lr_scheduler, scaler, restore_rng=True)
        start_epoch = ckpt["epoch"] + 1

    dump_config(cfg, osp.join(cfg.OUTPUT_DIR, "config.yaml"))
    print(json.dumps(cfg.to_dict(), indent=1))
    tfboard = None
    if cfg.TF_BOARD:
        try:
            from torch.utils.tensorboard import SummaryWriter

            tfboard = SummaryWriter(log_dir=osp.join(cfg.OUTPUT_DIR, "tf_log"))
        except ImportError:
            print("tensorboard not installed: logging to CSV only")
    csv_iter = CSVLogger(osp.join(cfg.OUTPUT_DIR, "train_log.csv"))
    csv_epoch = CSVLogger(osp.join(cfg.OUTPUT_DIR, "epoch_log.csv"))

    print(f"Start training from epoch {start_epoch}")
    start_time = time.time()
    for epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCHS):
        if hasattr(train_loader, "sampler") and hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)
        stats = train_one_epoch(cfg, model, optimizer, train_loader, device, epoch, dtype, scaler,
                                tfboard, csv_iter)
        lr_scheduler.step()
        row = {"epoch": epoch, "lr_end": optimizer.param_groups[0]["lr"]}
        row.update({k: round(v, 6) for k, v in stats.items()})

        last = epoch == cfg.SOLVER.MAX_EPOCHS - 1
        if (epoch + 1) % cfg.CKPT_PERIOD == 0 or last:
            save_checkpoint(osp.join(cfg.OUTPUT_DIR, f"epoch_{epoch}.pth"), model, optimizer,
                            lr_scheduler, epoch, scaler, cfg)
            prune_checkpoints(cfg.OUTPUT_DIR, cfg.CKPT_KEEP)

        if cfg.EVAL_PERIOD > 0 and ((epoch + 1) % cfg.EVAL_PERIOD == 0 or last):
            res, _ = evaluate_performance(model, *test_loaders, device,
                                          det_thresholds=cfg.EVAL_DET_THRESHOLDS[:1],
                                          flip_tta=cfg.EVAL_FLIP_TTA, dtype=dtype)
            s = res["search"][0]
            row.update(mAP=s["mAP"], top1=s["top1"], det_recall=res["detection"]["recall"],
                       det_ap=res["detection"]["ap"])
            if tfboard is not None:
                tfboard.add_scalar("eval/mAP", s["mAP"], epoch)
                tfboard.add_scalar("eval/top1", s["top1"], epoch)
        csv_epoch.log(row)

    if tfboard:
        tfboard.close()
    total = str(datetime.timedelta(seconds=int(time.time() - start_time)))
    print(f"Total training time {total}")


if __name__ == "__main__":
    main()
