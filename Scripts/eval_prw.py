"""
Evaluate a SeqNet checkpoint on PRW with the unchanged course function eval_search_prw.

  python Scripts/eval_prw.py --ckpt exp_prw/epoch_17.pth --data-root /path/to/PRW \
      [--det-thresh 0.5 0.6 0.7 0.8] [--flip-tta] [--amp bf16] [--out exp_prw/eval.json] \
      [--cache exp_prw/features.pt]

Gallery features are CWS-weighted (embedding * detection score), as in official SeqNet.
The config is read from the checkpoint (or --cfg); --cache stores/reuses extracted features
so the det_thresh sweep can be re-run on CPU without the GPU.
"""
import argparse
import json
import os.path as osp
import sys

import torch

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))

from config import load_config  # noqa: E402
from engine import evaluate_performance  # noqa: E402
from prw import build_test_loaders  # noqa: E402
from seqnet import SeqNet  # noqa: E402
from utils import amp_dtype, load_checkpoint  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description="Evaluate SeqNet on PRW.")
    p.add_argument("--ckpt", required=True)
    p.add_argument("--cfg", help="YAML merged on top of the config stored in the checkpoint")
    p.add_argument("--data-root")
    p.add_argument("--det-thresh", type=float, nargs="+", help="det_thresh values to sweep")
    p.add_argument("--flip-tta", action="store_true", help="horizontal-flip TTA of embeddings")
    p.add_argument("--amp", choices=["off", "fp16", "bf16"], default="off")
    p.add_argument("--batch-size", type=int, help="INPUT.BATCH_SIZE_TEST")
    p.add_argument("--workers", type=int)
    p.add_argument("--device")
    p.add_argument("--out", help="JSON output (default: <ckpt dir>/eval_<ckpt name>.json)")
    p.add_argument("--cache", help="feature cache file (.pt): loaded if valid, else written")
    p.add_argument("--opts", nargs="*", default=[])
    args = p.parse_args(argv)

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    stored = ckpt.get("cfg") if isinstance(ckpt, dict) else None  # None for official checkpoints
    cfg = load_config(args.cfg, args.opts, args.data_root, base=stored)
    del ckpt
    if args.batch_size:
        cfg.INPUT.BATCH_SIZE_TEST = args.batch_size
    if args.workers is not None:
        cfg.INPUT.NUM_WORKERS_TEST = args.workers
    if args.device:
        cfg.DEVICE = args.device
    thresholds = args.det_thresh or cfg.EVAL_DET_THRESHOLDS
    flip = args.flip_tta or cfg.EVAL_FLIP_TTA
    device = torch.device(cfg.DEVICE)

    cfg.MODEL.BACKBONE.PRETRAINED = False  # weights come from the checkpoint
    model = SeqNet(cfg).to(device)
    load_checkpoint(args.ckpt, model)
    gallery_loader, query_loader = build_test_loaders(cfg)

    key = {"ckpt": osp.abspath(args.ckpt), "flip_tta": flip, "amp": args.amp,
           "n_gallery": len(gallery_loader.dataset), "n_query": len(query_loader.dataset)}
    features = None
    if args.cache and osp.isfile(args.cache):
        cached = torch.load(args.cache, map_location="cpu", weights_only=False)
        if cached.get("key") == key:
            print(f"using cached features {args.cache}")
            features = cached["features"]
    results, features = evaluate_performance(
        model, gallery_loader, query_loader, device, det_thresholds=thresholds,
        flip_tta=flip, dtype=amp_dtype(args.amp), features=features,
    )
    if args.cache:
        torch.save({"key": key, "features": features}, args.cache)

    out = args.out or osp.join(osp.dirname(osp.abspath(args.ckpt)),
                               "eval_" + osp.splitext(osp.basename(args.ckpt))[0] + ".json")
    summary = {**key, "cws": True, "data_root": cfg.INPUT.DATA_ROOT, **results}
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print("det_thresh  mAP     top-1")
    for r in results["search"]:
        print(f"{r['det_thresh']:<10}  {r['mAP']:.2%}  {r['top1']:.2%}")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
