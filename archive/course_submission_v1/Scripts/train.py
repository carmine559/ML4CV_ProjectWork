"""
Training Engine
---------------
train_one_epoch() handles one full epoch including:
  - AMP (automatic mixed precision)
  - gradient clipping
  - linear LR warmup (optional, passed in as warmup_scheduler)
  - metric logging
"""

import time
from collections import defaultdict

import torch
from torch.amp import GradScaler, autocast


# --------------------------------------------------------------------------- #
#  Metric utilities
# --------------------------------------------------------------------------- #

class AverageMeter:
    """Tracks running average of a scalar."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = self.count = 0.0

    @property
    def avg(self):
        return self.sum / self.count if self.count > 0 else 0.0

    def update(self, val: float, n: int = 1):
        self.sum   += val * n
        self.count += n


class MetricLogger:
    def __init__(self):
        self.meters = defaultdict(AverageMeter)

    def update(self, **kwargs):
        for k, v in kwargs.items():
            self.meters[k].update(v.item() if isinstance(v, torch.Tensor) else float(v))

    def __str__(self):
        return "  ".join(f"{k}: {v.avg:.4f}" for k, v in self.meters.items())

    def averages(self) -> dict:
        return {k: v.avg for k, v in self.meters.items()}


# --------------------------------------------------------------------------- #
#  Training epoch
# --------------------------------------------------------------------------- #

def train_one_epoch(
    model,
    optimizer,
    data_loader,
    device,
    epoch: int,
    scaler: GradScaler,
    warmup_scheduler=None,
    print_freq: int = 100,
    grad_clip: float = 10.0,
) -> dict:
    """
    Run one epoch of training.

    Args:
        model             : PersonSearchModel
        optimizer         : SGD or Adam optimizer
        data_loader       : training DataLoader
        device            : torch.device
        epoch             : current epoch index (for logging only)
        scaler            : GradScaler instance — created ONCE outside and reused
        warmup_scheduler  : optional LambdaLR warmup scheduler stepped every iter
        print_freq        : log every N iterations
        grad_clip         : max gradient norm (0 = disabled)

    Returns:
        dict of average losses over the epoch
    """
    model.train()
    logger = MetricLogger()
    start  = time.time()

    for i, (images, targets) in enumerate(data_loader):
        images  = [img.to(device, non_blocking=True) for img in images]
        targets = [
            {k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
             for k, v in t.items()}
            for t in targets
        ]

        optimizer.zero_grad(set_to_none=True)   # faster than zero_grad()

        with autocast("cuda"):
            loss_dict = model(images, targets)
            losses    = sum(loss_dict.values())

        scaler.scale(losses).backward()

        if grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        scaler.step(optimizer)
        scaler.update()

        # Step warmup scheduler every iteration (no-op after warmup_iters)
        if warmup_scheduler is not None:
            warmup_scheduler.step()

        # Logging — detach to avoid holding computation graph
        log_dict = {k: v.detach() for k, v in loss_dict.items()}
        log_dict["loss_total"] = losses.detach()
        logger.update(**log_dict)

        if (i + 1) % print_freq == 0:
            elapsed = time.time() - start
            print(f"Epoch [{epoch}] [{i+1}/{len(data_loader)}]  "
                  f"Time: {elapsed:.1f}s  {logger}")
            start = time.time()

    return logger.averages()


# --------------------------------------------------------------------------- #
#  LR warm-up scheduler
# --------------------------------------------------------------------------- #

def warmup_lr_scheduler(
    optimizer,
    warmup_iters: int,
    warmup_factor: float = 1.0 / 1000,
) -> torch.optim.lr_scheduler.LambdaLR:
    """
    Linear LR warmup over `warmup_iters` iterations.
    LR scales from warmup_factor * base_lr → base_lr.
    After warmup_iters steps, returns 1.0 (no further change).
    Must be stepped every iteration, not every epoch.
    """
    def _lr_lambda(current_iter: int) -> float:
        if current_iter >= warmup_iters:
            return 1.0
        alpha = float(current_iter) / warmup_iters
        return warmup_factor * (1.0 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)