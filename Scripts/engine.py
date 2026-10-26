"""
Training epoch and inference/evaluation.
train_one_epoch / gallery+query inference follow SeqNet engine.py
(Copyright (c) 2022 Zj Li, MIT License, see LICENSE-SeqNet); eval_detection is adapted from
SeqNet eval_func.py. Person-search metrics come from the UNCHANGED course function
evaluate.eval_search_prw.
"""
import datetime
import math
import os.path as osp
import sys

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from evaluate import _compute_iou, eval_search_prw
from utils import MetricLogger, Timer, autocast, warmup_lr_scheduler


def to_device(images, targets, device):
    images = [image.to(device, non_blocking=True) for image in images]
    for t in targets:
        t["boxes"] = t["boxes"].to(device, non_blocking=True)
        t["labels"] = t["labels"].to(device, non_blocking=True)
    return images, targets


def train_one_epoch(cfg, model, optimizer, data_loader, device, epoch, dtype=None, scaler=None,
                    tfboard=None, csv_logger=None):
    model.train()
    metric_logger = MetricLogger()
    timer = Timer()
    n_iters = len(data_loader)

    # warmup learning rate in the first epoch (official: over the whole first epoch)
    warmup_scheduler = None
    if epoch == 0:
        warmup_iters = n_iters - 1
        warmup_scheduler = warmup_lr_scheduler(optimizer, warmup_iters, cfg.SOLVER.WARMUP_FACTOR)

    for i, (images, targets) in enumerate(data_loader):
        images, targets = to_device(images, targets, device)

        with autocast(device, dtype):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        loss_value = losses.item()
        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print({k: v.item() for k, v in loss_dict.items()})
            sys.exit(1)

        optimizer.zero_grad()
        if scaler is not None:  # fp16
            scaler.scale(losses).backward()
            if cfg.SOLVER.CLIP_GRADIENTS > 0:
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), cfg.SOLVER.CLIP_GRADIENTS)
            scaler.step(optimizer)
            scaler.update()
        else:
            losses.backward()
            if cfg.SOLVER.CLIP_GRADIENTS > 0:
                clip_grad_norm_(model.parameters(), cfg.SOLVER.CLIP_GRADIENTS)
            optimizer.step()

        if warmup_scheduler is not None:
            warmup_scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        scalars = {k: v.item() for k, v in loss_dict.items()}
        metric_logger.update(loss=loss_value, **scalars, time=timer.lap())
        it = epoch * n_iters + i
        if tfboard is not None:
            for k, v in scalars.items():
                tfboard.add_scalar(f"train/{k}", v, it)
            tfboard.add_scalar("train/loss", loss_value, it)
            tfboard.add_scalar("train/lr", lr, it)
        if (i + 1) % cfg.DISP_PERIOD == 0 or i == n_iters - 1:
            eta = metric_logger.meters["time"].global_avg * (n_iters - i - 1)
            mem = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0
            print(f"Epoch [{epoch}] [{i + 1}/{n_iters}]  eta {datetime.timedelta(seconds=int(eta))}"
                  f"  lr {lr:.6f}  {metric_logger}  mem {mem:.1f}G", flush=True)
            if csv_logger is not None:
                row = {"epoch": epoch, "iter": i + 1, "global_iter": it + 1, "lr": lr}
                row.update({k: round(m.median, 6) for k, m in metric_logger.meters.items()})
                csv_logger.log(row)
    return {k: m.global_avg for k, m in metric_logger.meters.items()}


# ------------------------------------------------------------------------- inference
def _mirror(boxes, width):
    """Horizontal flip of [x1, y1, x2, y2] boxes (same convention as RandomHorizontalFlip)."""
    out = boxes.clone()
    out[:, [0, 2]] = width - boxes[:, [2, 0]]
    return out


@torch.no_grad()
def extract_features(model, gallery_loader, query_loader, device, flip_tta=False, dtype=None):
    """
    Returns gallery_dets (list of [N,5] x1y1x2y2+score), gallery_feats (list of [N,256], CWS
    applied: embedding * detection score) and query_box_feats (list of [256], GT query box).
    flip_tta: average the L2-normalised embeddings of the image and its horizontal flip
    (same boxes, mirrored), renormalise, then apply CWS (gallery only).
    """
    model.eval()
    roi_heads = model.roi_heads
    roi_heads.cws = not flip_tta  # with TTA, CWS is applied after averaging
    try:
        gallery_dets, gallery_feats = [], []
        for images, _ in tqdm(gallery_loader, ncols=0, desc="gallery"):
            images = [img.to(device, non_blocking=True) for img in images]
            with autocast(device, dtype):
                outputs = model(images)
            if flip_tta:
                idx = [k for k, o in enumerate(outputs) if len(o["boxes"]) > 0]
                if idx:
                    flipped = [images[k].flip(-1) for k in idx]
                    tgts = [{"boxes": _mirror(outputs[k]["emb_boxes"], images[k].shape[-1])} for k in idx]
                    with autocast(device, dtype):
                        flip_embs = model(flipped, tgts)
                    for k, fe in zip(idx, flip_embs):
                        o = outputs[k]
                        e = F.normalize(o["embeddings"].float() + fe.float(), dim=1)
                        o["embeddings"] = e * o["scores"].view(-1, 1)
            for o in outputs:
                det = torch.cat([o["boxes"], o["scores"].unsqueeze(1)], dim=1)
                gallery_dets.append(det.float().cpu().numpy())
                gallery_feats.append(o["embeddings"].float().cpu().numpy())

        query_box_feats = []
        for images, targets in tqdm(query_loader, ncols=0, desc="query"):
            images, targets = to_device(images, targets, device)
            with autocast(device, dtype):
                embeddings = model(images, targets)
                if flip_tta:
                    tgts = [{"boxes": _mirror(t["boxes"], im.shape[-1])} for t, im in zip(targets, images)]
                    flip_embs = model([im.flip(-1) for im in images], tgts)
                    embeddings = [F.normalize(e.float() + fe.float(), dim=1)
                                  for e, fe in zip(embeddings, flip_embs)]
            for e in embeddings:
                query_box_feats.append(e.float().squeeze(0).cpu().numpy())
    finally:
        roi_heads.cws = True
    return gallery_dets, gallery_feats, query_box_feats


def eval_detection(gallery_dataset, gallery_dets, det_thresh=0.5, iou_thresh=0.5, labeled_only=False):
    """Detection recall / AP on the gallery (adapted from SeqNet eval_func.py)."""
    assert len(gallery_dataset) == len(gallery_dets)
    y_true, y_score = [], []
    count_gt, count_tp = 0, 0
    for anno, det in zip(gallery_dataset.annotations, gallery_dets):
        gt_boxes = anno["boxes"]
        if labeled_only:
            inds = np.where(anno["pids"].ravel() != 5555)[0]
            if len(inds) == 0:
                continue
            gt_boxes = gt_boxes[inds]
        num_gt = gt_boxes.shape[0]
        det = np.asarray(det).reshape(-1, 5)
        det = det[det[:, 4] >= det_thresh]
        num_det = det.shape[0]
        if num_det == 0:
            count_gt += num_gt
            continue
        ious = np.zeros((num_gt, num_det), dtype=np.float32)
        for i in range(num_gt):
            for j in range(num_det):
                ious[i, j] = _compute_iou(gt_boxes[i], det[j, :4])
        tfmat = ious >= iou_thresh
        # for each det, keep only the largest iou of all the gt
        for j in range(num_det):
            largest_ind = np.argmax(ious[:, j])
            for i in range(num_gt):
                if i != largest_ind:
                    tfmat[i, j] = False
        # for each gt, keep only the largest iou of all the det
        for i in range(num_gt):
            largest_ind = np.argmax(ious[i, :])
            for j in range(num_det):
                if j != largest_ind:
                    tfmat[i, j] = False
        for j in range(num_det):
            y_score.append(det[j, -1])
            y_true.append(tfmat[:, j].any())
        count_tp += tfmat.sum()
        count_gt += num_gt

    det_rate = count_tp * 1.0 / max(count_gt, 1)
    ap = average_precision_score(y_true, y_score) * det_rate if count_tp > 0 else 0.0
    print("{} detection: recall = {:.2%}  ap = {:.2%}".format(
        "labeled only" if labeled_only else "all", det_rate, ap))
    return float(det_rate), float(ap)


def evaluate_performance(model, gallery_loader, query_loader, device, det_thresholds=(0.5,),
                         flip_tta=False, dtype=None, features=None):
    """Runs inference (unless `features` is given) and the course eval for every det_thresh."""
    if features is None:
        features = extract_features(model, gallery_loader, query_loader, device, flip_tta, dtype)
    gallery_dets, gallery_feats, query_box_feats = features
    gallery_set, query_set = gallery_loader.dataset, query_loader.dataset

    recall, ap = eval_detection(gallery_set, gallery_dets, det_thresh=0.01)
    results = {"detection": {"recall": recall, "ap": ap}, "search": []}
    for t in det_thresholds:
        print(f"--- det_thresh = {t}")
        ret = eval_search_prw(gallery_set, query_set, gallery_dets, gallery_feats,
                              query_box_feats, det_thresh=t)
        results["search"].append(
            {"det_thresh": float(t), "mAP": float(ret["mAP"]), "top1": float(np.ravel(ret["accs"])[0])}
        )
    return results, features
