"""Seeding, LR warm-up, checkpointing and logging helpers.
set_random_seed / warmup_lr_scheduler adapted from SeqNet utils/utils.py
(Copyright (c) 2022 Zj Li, MIT License, see LICENSE-SeqNet)."""
import csv
import glob
import os
import os.path as osp
import random
import re
import time
from collections import defaultdict, deque

import numpy as np
import torch


def set_random_seed(seed, cudnn_benchmark=False):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.backends.cudnn.deterministic = not cudnn_benchmark
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):
    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


def amp_dtype(name):
    return {"off": None, "fp16": torch.float16, "bf16": torch.bfloat16}[name]


def autocast(device, dtype):
    return torch.autocast(device_type=torch.device(device).type, dtype=dtype, enabled=dtype is not None)


# ---------------------------------------------------------------- checkpoints
def _rng_state():
    return {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }


def _set_rng_state(s):
    torch.set_rng_state(s["torch"])
    if s["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(s["cuda"])
    np.random.set_state(s["numpy"])
    random.setstate(s["python"])


def save_checkpoint(path, model, optimizer, lr_scheduler, epoch, scaler=None, cfg=None):
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "epoch": epoch,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "rng": _rng_state(),
        "cfg": cfg.to_dict() if cfg is not None else None,
    }
    tmp = path + ".tmp"
    torch.save(state, tmp)
    os.replace(tmp, path)  # atomic: a job killed mid-save never leaves a corrupt checkpoint


def list_checkpoints(output_dir):
    ckpts = glob.glob(osp.join(output_dir, "epoch_*.pth"))
    return sorted(ckpts, key=lambda p: int(re.findall(r"epoch_(\d+)\.pth$", p)[0]))


def prune_checkpoints(output_dir, keep):
    if keep > 0:
        for p in list_checkpoints(output_dir)[:-keep]:
            os.remove(p)


def load_checkpoint(path, model, optimizer=None, lr_scheduler=None, scaler=None, restore_rng=False):
    # weights_only=False: our checkpoints contain RNG states / config (only load trusted files)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    state = ckpt["model"] if "model" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[ckpt] missing keys: {missing}\n[ckpt] unexpected keys: {unexpected}")
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if lr_scheduler is not None:
        lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
    if scaler is not None and ckpt.get("scaler"):
        scaler.load_state_dict(ckpt["scaler"])
    if restore_rng and ckpt.get("rng"):
        _set_rng_state(ckpt["rng"])
    print(f"loaded checkpoint {path} (epoch {ckpt.get('epoch', '?')})")
    return ckpt


# ---------------------------------------------------------------- logging
class SmoothedValue:
    def __init__(self, window_size=20):
        self.deque = deque(maxlen=window_size)
        self.total, self.count = 0.0, 0

    def update(self, v):
        self.deque.append(v)
        self.total += v
        self.count += 1

    @property
    def median(self):
        return float(np.median(self.deque))

    @property
    def global_avg(self):
        return self.total / max(self.count, 1)


class MetricLogger:
    def __init__(self):
        self.meters = defaultdict(SmoothedValue)

    def update(self, **kw):
        for k, v in kw.items():
            self.meters[k].update(float(v))

    def __str__(self):
        return "  ".join(f"{k}: {m.median:.4f} ({m.global_avg:.4f})" for k, m in self.meters.items())


class CSVLogger:
    """Appends rows to a CSV file; the header is (re)written when the column set changes."""

    def __init__(self, path):
        self.path = path
        self.fields = None

    def log(self, row):
        fields = list(row.keys())
        new = not osp.isfile(self.path) or self.fields != fields
        if new and osp.isfile(self.path) and self.fields is None:
            with open(self.path, newline="") as f:
                header = next(csv.reader(f), None)
            new = header != fields
        with open(self.path, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            if new:
                w.writeheader()
            w.writerow(row)
        self.fields = fields


class Timer:
    def __init__(self):
        self.t = time.time()

    def lap(self):
        now = time.time()
        dt, self.t = now - self.t, now
        return dt
