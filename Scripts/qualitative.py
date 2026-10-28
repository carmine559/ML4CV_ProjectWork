"""
Qualitative results on PRW from cached features (no model, no GPU).

  python Scripts/qualitative.py --data-root $PRW_ROOT --out results/qualitative \
      --run "SOLIDER Swin-S" exp_prw_solider_swin_s/features_epoch_17.pt \
      [--run "ResNet-50" exp_prw/features_epoch_17.pt] [--det-thresh 0.5] [--seed 0]

The features are the caches written by `eval_prw.py --cache` (slurm/train_prw.sbatch does it
after training). The first --run is the main model; a second one adds the comparison figures.
The ranking comes from the unchanged course eval_search_prw, so the figures show exactly the
retrievals behind the reported mAP / top-1. Each retrieved box is then compared with the
annotated people of its frame:
  correct    the course eval's flag (the query person, IoU >= min(0.5, wh/((w+10)(h+10))))
  pid N      another labelled person (best IoU >= 0.5)
  unlabeled  an annotated person without identity (PRW id -2)
  loose box  the query person, but the box overlaps it with 0.3 <= IoU < the eval threshold
  duplicate  the query person, already retrieved with a higher score in the same frame
  no person  no annotated person under the box (false detection)
Top-1 errors are grouped by these labels (plus "never detected": the detector found the
query person in none of their gallery frames). Examples are drawn at random with a fixed seed
inside each group, and summary.json gives the size of every group.

Outputs in --out: summary.json, top1_errors.png, recall_by_height.png, compare_*.jpg (two
runs), and per run <run>/queries.csv, search_random.jpg, failures_<group>.jpg,
detections_random.jpg, detections_misses.jpg.
"""
import argparse
import csv
import json
import os
import os.path as osp
import re
import sys
import zlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, osp.dirname(osp.abspath(__file__)))
from evaluate import eval_search_prw  # noqa: E402
from prw import PRW, UNLABELED_PID  # noqa: E402

# boxes on photos: status colours, always paired with a text label in the title
C_QUERY, C_HIT, C_WRONG, C_TRUE, C_PEOPLE = "#2a78d6", "#0ca30c", "#d03b3b", "#fab219", "#d9d9d9"
# charts: categorical slots 1-2 and chart chrome of the reference palette
SERIES = ("#2a78d6", "#eb6834")
INK, INK2, MUTED, GRID, AXIS, SURFACE = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"

GROUPS = {  # top-1 outcome -> description (order = order in the charts)
    "correct": "correct",
    "never_detected": "query person never detected",
    "same_frame": "right frame, wrong person",
    "lookalike": "another labelled person",
    "unlabeled": "an unlabelled person",
    "loose_box": "right person, inaccurate box",
    "no_person": "not a person (false detection)",
}
HEIGHT_BINS = (0, 75, 100, 150, 200, 300, 450, np.inf)


# ----------------------------------------------------------------------------- matching
def iou_matrix(a, b):
    """IoU between [N,4] and [M,4] boxes, same formula as evaluate._compute_iou."""
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    w = np.clip(np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None)
    h = np.clip(np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None)
    inter = w * h
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(area_a[:, None] + area_b[None, :] - inter, 1e-9)


def search_iou_thresh(gt):
    """The course eval's IoU threshold for a ground-truth box."""
    w, h = gt[2] - gt[0], gt[3] - gt[1]
    return min(0.5, (w * h * 1.0) / ((w + 10) * (h + 10)))


def match_detections(gt, det, iou_thresh=0.5):
    """engine.eval_detection's rule: a GT box and a detection match if each is the other's best
    IoU and IoU >= iou_thresh. Returns (gt matched, det matched) boolean arrays."""
    if len(gt) == 0 or len(det) == 0:
        return np.zeros(len(gt), bool), np.zeros(len(det), bool)
    ious = iou_matrix(gt, det[:, :4])
    tf = ious >= iou_thresh
    tf &= np.arange(len(gt))[:, None] == ious.argmax(0)[None, :]
    tf &= np.arange(len(det))[None, :] == ious.argmax(1)[:, None]
    return tf.any(1), tf.any(0)


def describe_box(roi, anno, query_pid, correct):
    """What a retrieved box really is: (label, pid of the matched person or None)."""
    if correct:
        return "correct", query_pid
    boxes, pids = anno["boxes"], np.atleast_1d(anno["pids"])
    if len(boxes) == 0:
        return "no_person", None
    ious = iou_matrix(roi[:4], boxes)[0]
    j = int(ious.argmax())
    if ious[j] >= 0.5 and pids[j] != query_pid:
        return ("unlabeled" if pids[j] == UNLABELED_PID else "other"), int(pids[j])
    q = np.where(pids == query_pid)[0]
    if len(q):
        k = q[int(ious[q].argmax())]
        if ious[k] >= search_iou_thresh(boxes[k]):
            return "duplicate", query_pid
        if ious[k] >= 0.3:
            return "loose_box", query_pid
    return "no_person", None


BOX_TEXT = {"correct": "✓ correct", "unlabeled": "✗ unlabelled person",
            "duplicate": "✗ duplicate", "loose_box": "✗ right person, loose box",
            "no_person": "✗ not a person"}


def box_text(label, pid):
    return f"✗ other person (pid {pid})" if label == "other" else BOX_TEXT[label]


# ----------------------------------------------------------------------------- analysis
def analyse_run(label, path, gallery, query, det_thresh):
    cached = torch.load(path, map_location="cpu", weights_only=False)
    gallery_dets, gallery_feats, query_feats = cached["features"]
    assert len(gallery_dets) == len(gallery) and len(query_feats) == len(query), \
        f"{path}: features do not match the PRW test split"
    print(f"=== {label}: {path}")
    ret = eval_search_prw(gallery, query, gallery_dets, gallery_feats, query_feats, det_thresh)

    annos = {a["img_name"]: a for a in gallery.annotations}
    dets = {}
    for a, d in zip(gallery.annotations, gallery_dets):
        d = np.asarray(d).reshape(-1, 5)
        dets[a["img_name"]] = d[d[:, 4] >= det_thresh]

    records = []
    for i, (q, res) in enumerate(zip(query.annotations, ret["results"])):
        pid = q["pids"]
        n_det = 0
        for name, gt in res["query_gt"].items():
            gt = np.asarray(gt).ravel()[:4]
            d = dets[name]
            n_det += bool(len(d)) and bool((iou_matrix(gt, d[:, :4]) >= search_iou_thresh(gt)).any())
        ranked = []
        for g in res["gallery"]:
            lab, lab_pid = describe_box(np.asarray(g["roi"]), annos[g["img"]], pid, g["correct"])
            ranked.append({**g, "label": lab, "label_pid": lab_pid,
                           "same_frame": g["img"] in res["query_gt"]})
        top = ranked[0] if ranked else None
        if top is not None and top["correct"]:
            group = "correct"
        elif n_det == 0:
            group = "never_detected"
        elif top is None or top["label"] == "no_person":
            group = "no_person"
        elif top["label"] in ("other", "unlabeled") and top["same_frame"]:
            group = "same_frame"
        else:
            group = {"other": "lookalike", "unlabeled": "unlabeled"}.get(top["label"], top["label"])
        hits = [k for k, g in enumerate(ranked) if g["correct"]]
        records.append({"index": i, "img_name": q["img_name"], "pid": int(pid), "cam": int(q["cam_id"]),
                        "group": group, "first_hit_rank": hits[0] + 1 if hits else "",
                        "gt_frames": len(res["query_gt"]), "gt_frames_detected": int(n_det),
                        "top1_sim": round(top["score"], 4) if top else "",
                        "query_roi": res["query_roi"], "query_gt": res["query_gt"], "ranked": ranked})

    # detection against all annotated people
    heights, found, frames = [], [], []
    for a in gallery.annotations:
        gt_ok, det_ok = match_detections(a["boxes"], dets[a["img_name"]])
        heights.extend((a["boxes"][:, 3] - a["boxes"][:, 1]).tolist())
        found.extend(gt_ok.tolist())
        frames.append({"name": a["img_name"], "missed": int((~gt_ok).sum()), "fp": int((~det_ok).sum())})
    heights, found = np.asarray(heights), np.asarray(found)
    bins = []
    for lo, hi in zip(HEIGHT_BINS[:-1], HEIGHT_BINS[1:]):
        m = (heights >= lo) & (heights < hi)
        bins.append({"min_px": lo, "max_px": None if np.isinf(hi) else hi, "n": int(m.sum()),
                     "recall": float(found[m].mean()) if m.any() else None})

    n = len(records)
    first = [r["first_hit_rank"] for r in records]
    n_gt = sum(r["gt_frames"] for r in records)
    summary = {
        "label": label, "features": path, "det_thresh": det_thresh, "n_queries": n,
        "mAP": float(ret["mAP"]), "top1": float(np.ravel(ret["accs"])[0]),
        # from the eval's top-10 lists (eval_search_prw itself reports top-1 only)
        "top5": sum(1 for f in first if f != "" and f <= 5) / n,
        "top10": sum(1 for f in first if f != "") / n,
        "top1_groups": {g: sum(r["group"] == g for r in records) for g in GROUPS},
        "query_person_frames": n_gt,
        "query_person_frames_detected": sum(r["gt_frames_detected"] for r in records),
        "detection_recall": float(found.mean()), "recall_by_height": bins,
    }
    print(f"mAP {summary['mAP']:.2%} | top-1 {summary['top1']:.2%} | top-5 {summary['top5']:.2%}"
          f" | top-10 {summary['top10']:.2%} | detection recall {summary['detection_recall']:.2%}")
    print("top-1 groups:", summary["top1_groups"])
    return {"label": label, "summary": summary, "records": records, "dets": dets, "frames": frames,
            "annos": annos}


# ----------------------------------------------------------------------------- drawing
def _rng(seed, name):
    return np.random.default_rng([seed, zlib.crc32(name.encode())])


def _sample(rng, items, k):
    items = list(items)
    if len(items) <= k:
        return items
    return [items[i] for i in sorted(rng.choice(len(items), size=k, replace=False))]


def _window(boxes, W, H, aspect=2 / 3, pad=0.4, min_h=200):
    """A crop of the frame around `boxes`: width / height = `aspect`, or wider when the boxes
    are far apart (the crop always contains all of them)."""
    b = np.asarray(boxes, dtype=float).reshape(-1, 4)
    x1, y1, x2, y2 = b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max()
    need_w, need_h = (x2 - x1) * (1 + pad), (y2 - y1) * (1 + pad)
    h = min(max(need_h, need_w / aspect, min_h), H)
    w = min(max(h * aspect, need_w), W)
    x0 = min(max((x1 + x2) / 2 - w / 2, 0), W - w)
    y0 = min(max((y1 + y2) / 2 - h / 2, 0), H - h)
    return tuple(int(round(v)) for v in (x0, y0, x0 + w, y0 + h))


def _box(ax, b, off, color, lw=2.2, ls="-", scale=1.0):
    x, y = (b[0] - off[0]) * scale, (b[1] - off[1]) * scale
    ax.add_patch(Rectangle((x, y), (b[2] - b[0]) * scale, (b[3] - b[1]) * scale, fill=False,
                           edgecolor=color, linewidth=lw, linestyle=ls))


def _cell(ax, img_path, focus, draw, others, optional=()):
    """Draw a crop around `focus` boxes, widened to the `optional` boxes unless that makes it
    more than 2.5x wider; `draw` = [(box, colour, linestyle)], `others` = grey boxes.
    Returns the crop window."""
    img = Image.open(img_path).convert("RGB")
    win = _window(focus, *img.size)
    if len(optional):
        wide = _window(list(focus) + list(optional), *img.size)
        if wide[2] - wide[0] <= 2.5 * (win[2] - win[0]):
            win = wide
    ax.imshow(img.crop(win))
    for b in others:
        _box(ax, b, win, C_PEOPLE, lw=0.8)
    for b, c, ls in draw:
        _box(ax, b, win, c, ls=ls)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    return win


def _others(anno, exclude):
    """Annotated people of the frame except the boxes in `exclude` (drawn on their own)."""
    out = []
    for b in anno["boxes"]:
        if all(iou_matrix(b, e)[0, 0] < 0.9 for e in exclude):
            out.append(b)
    return out


def _query_cell(ax, run, rec, query):
    q = query.annotations[rec["index"]]
    qb = np.asarray(rec["query_roi"])
    anno = run["annos"].get(q["img_name"], {"boxes": np.zeros((0, 4))})
    _cell(ax, q["img_path"], [qb], [(qb, C_QUERY, "-")], _others(anno, [qb]))
    ax.set_title(f"query\npid {rec['pid']} · cam {rec['cam']}", fontsize=8, color=INK)


def _result_cell(ax, run, rec, k, prefix=""):
    if k >= len(rec["ranked"]):
        ax.axis("off")
        return
    g = rec["ranked"][k]
    roi = np.asarray(g["roi"][:4])
    anno = run["annos"][g["img"]]
    draw = [(roi, C_HIT if g["correct"] else C_WRONG, "-")]
    true = []
    if not g["correct"] and g["same_frame"]:  # where the query person really is
        true = list(np.asarray(rec["query_gt"][g["img"]]).reshape(-1, 4))
        draw += [(t, C_TRUE, "--") for t in true]
    win = _cell(ax, anno["img_path"], [roi], draw, _others(anno, [d[0] for d in draw]), optional=true)
    head = f"{prefix}#{k + 1}  sim {g['score']:.2f}"
    if not prefix:
        head += f" · cam {int(re.search(r'c(\d)', g['img']).group(1))}"
    text = f"{head}\n{box_text(g['label'], g['label_pid'])}"
    if any(t[0] < win[0] or t[1] < win[1] or t[2] > win[2] or t[3] > win[3] for t in true):
        text += "\n(query person elsewhere in frame)"
    ax.set_title(text, fontsize=8, color=INK)


def _frame_cell(ax, run, rec, width=960):
    """The whole frame of the top-1 result: retrieved box, the query person if present, others."""
    g = rec["ranked"][0]
    anno = run["annos"][g["img"]]
    img = Image.open(anno["img_path"]).convert("RGB")
    s = width / img.size[0]
    ax.imshow(img.resize((width, int(round(img.size[1] * s)))))
    roi = np.asarray(g["roi"][:4])
    true = np.asarray(rec["query_gt"].get(g["img"], np.zeros((0, 4)))).reshape(-1, 4)
    for b in _others(anno, [roi, *true]):
        _box(ax, b, (0, 0), C_PEOPLE, lw=0.8, scale=s)
    for t in true:
        _box(ax, t, (0, 0), C_TRUE, ls="--", scale=s)
    _box(ax, roi, (0, 0), C_HIT if g["correct"] else C_WRONG, scale=s)
    where = "the query person is in this frame" if len(true) else "the query person is not in this frame"
    ax.set_title(f"#1 whole frame ({g['img']})\n{where}", fontsize=8, color=INK)
    ax.set_xticks([])
    ax.set_yticks([])


def _legend(fig, with_true=True):
    h = [Line2D([], [], color=C_QUERY, lw=2.2, label="query"),
         Line2D([], [], color=C_HIT, lw=2.2, label="retrieved: the query person"),
         Line2D([], [], color=C_WRONG, lw=2.2, label="retrieved: someone / something else")]
    if with_true:
        h.append(Line2D([], [], color=C_TRUE, lw=2.2, ls="--", label="where the query person really is"))
    h.append(Line2D([], [], color=C_PEOPLE, lw=1.5, label="other annotated people"))
    fig.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=len(h), frameon=False,
               fontsize=8.5, labelcolor=INK2)


def search_figure(run, recs, query, title, path, top_k=5, whole_frame=False):
    """Rows: query crop | top-k crops [| whole frame of the top-1 result]."""
    rows = max(len(recs), 1)
    ratios = [1] * (top_k + 1) + ([2.9] if whole_frame else [])
    fig, axes = plt.subplots(rows, len(ratios), figsize=(2.05 * sum(ratios), 3.45 * rows + 0.9), squeeze=False,
                             gridspec_kw={"width_ratios": ratios})
    for r, rec in enumerate(recs):
        _query_cell(axes[r, 0], run, rec, query)
        for k in range(top_k):
            _result_cell(axes[r, k + 1], run, rec, k)
        if whole_frame:
            if rec["ranked"]:
                _frame_cell(axes[r, -1], run, rec)
            else:
                axes[r, -1].axis("off")
    for ax in axes[len(recs):].ravel():
        ax.axis("off")
    fig.suptitle(title, fontsize=11, color=INK, y=1.0 - 0.32 / fig.get_figheight())
    _legend(fig)
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=1 - 0.95 / fig.get_figheight(),
                        wspace=0.06, hspace=0.42)
    fig.savefig(path, dpi=100, facecolor="white", pil_kwargs={"quality": 88})
    plt.close(fig)


def compare_figure(run_a, run_b, idx, query, title, path, top_k=3):
    rows = max(len(idx), 1)
    fig, axes = plt.subplots(rows, 1 + 2 * top_k, figsize=(2.05 * (1 + 2 * top_k), 3.35 * rows + 0.9),
                             squeeze=False)
    for r, i in enumerate(idx):
        _query_cell(axes[r, 0], run_a, run_a["records"][i], query)
        for k in range(top_k):
            _result_cell(axes[r, 1 + k], run_b, run_b["records"][i], k, prefix=run_b["label"] + " ")
            _result_cell(axes[r, 1 + top_k + k], run_a, run_a["records"][i], k, prefix=run_a["label"] + " ")
    for ax in axes[len(idx):].ravel():
        ax.axis("off")
    fig.suptitle(title, fontsize=11, color=INK, y=1.0 - 0.32 / fig.get_figheight())
    _legend(fig)
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=1 - 0.95 / fig.get_figheight(),
                        wspace=0.06, hspace=0.42)
    fig.savefig(path, dpi=100, facecolor="white", pil_kwargs={"quality": 88})
    plt.close(fig)


def detection_figure(run, names, title, path, width=960):
    cols = 2
    rows = max((len(names) + 1) // 2, 1)
    fig, axes = plt.subplots(rows, cols, figsize=(8 * cols, 4.7 * rows + 0.8), squeeze=False)
    for ax, name in zip(axes.ravel(), names):
        anno, det = run["annos"][name], run["dets"][name]
        img = Image.open(anno["img_path"]).convert("RGB")
        s = width / img.size[0]
        ax.imshow(img.resize((width, int(round(img.size[1] * s)))))
        gt_ok, det_ok = match_detections(anno["boxes"], det)
        for b, ok in zip(det[:, :4], det_ok):
            _box(ax, b, (0, 0), C_HIT if ok else C_WRONG, lw=1.6, scale=s)
        for b, ok in zip(anno["boxes"], gt_ok):
            if not ok:
                _box(ax, b, (0, 0), C_TRUE, lw=1.6, ls="--", scale=s)
        ax.set_title(f"{name}: {len(gt_ok)} people, {int(gt_ok.sum())} detected, {int((~gt_ok).sum())} missed,"
                     f" {int((~det_ok).sum())} false detections", fontsize=9, color=INK)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes.ravel()[len(names):]:
        ax.axis("off")
    h = [Line2D([], [], color=C_HIT, lw=2, label="detection matching a person (IoU ≥ 0.5)"),
         Line2D([], [], color=C_WRONG, lw=2, label="false detection"),
         Line2D([], [], color=C_TRUE, lw=2, ls="--", label="missed person (ground truth)")]
    fig.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False,
               fontsize=9, labelcolor=INK2)
    fig.suptitle(title, fontsize=11, color=INK, y=1.0 - 0.3 / fig.get_figheight())
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.01, top=1 - 0.85 / fig.get_figheight(),
                        wspace=0.03, hspace=0.12)
    fig.savefig(path, dpi=100, facecolor="white", pil_kwargs={"quality": 88})
    plt.close(fig)


def _chart_axes(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelcolor=INK2, labelsize=9)
    ax.set_axisbelow(True)


def top1_chart(runs, path):
    groups = [g for g in GROUPS if g != "correct"]
    n = runs[0]["summary"]["n_queries"]
    fig, ax = plt.subplots(figsize=(8.5, 0.4 * len(groups) * len(runs) + 1.6), facecolor=SURFACE)
    _chart_axes(ax)
    bar_h = 0.8 / len(runs)
    for r, run in enumerate(runs):
        counts = [run["summary"]["top1_groups"][g] for g in groups]
        vals = [100 * c / n for c in counts]
        y = np.arange(len(groups)) + (r - (len(runs) - 1) / 2) * bar_h
        top1 = run["summary"]["top1"]
        ax.barh(y, vals, height=bar_h * 0.9, color=SERIES[r], label=f"{run['label']} (top-1 {top1:.1%})")
        for yy, v, c in zip(y, vals, counts):
            if c:
                ax.annotate(f"{v:.1f}% ({c})", (v, yy), xytext=(4, 0), textcoords="offset points",
                            va="center", fontsize=8, color=INK2)
    ax.set_yticks(np.arange(len(groups)), [GROUPS[g] for g in groups])
    ax.invert_yaxis()
    ax.xaxis.grid(True, color=GRID, lw=0.8)
    ax.set_xlabel(f"share of the {n} queries whose top-1 result is wrong (%)", color=INK2, fontsize=9)
    ax.set_xlim(0, max(100 * max(run["summary"]["top1_groups"][g] for g in groups for run in runs) / n * 1.3, 1))
    ax.set_title("Why the top-1 result is wrong", fontsize=11, color=INK, loc="left", pad=26)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="lower left", bbox_to_anchor=(0, 1.0),
              ncol=len(runs), borderaxespad=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def recall_chart(runs, path):
    bins = runs[0]["summary"]["recall_by_height"]
    labels = [(f"<{b['max_px']}" if b["min_px"] == 0 else f"≥{b['min_px']}" if b["max_px"] is None
               else f"{b['min_px']}–{b['max_px']}") + f"\nn={b['n']}" for b in bins]
    fig, ax = plt.subplots(figsize=(8.5, 4.2), facecolor=SURFACE)
    _chart_axes(ax)
    x = np.arange(len(bins))
    ends = []
    for r, run in enumerate(runs):
        y = [np.nan if b["recall"] is None else 100 * b["recall"] for b in run["summary"]["recall_by_height"]]
        ax.plot(x, y, color=SERIES[r], lw=2, marker="o", ms=7, mec=SURFACE, mew=1.5,
                label=f"{run['label']} (all people: {run['summary']['detection_recall']:.1%})")
        last = max(i for i, v in enumerate(y) if not np.isnan(v))
        ends.append((x[last], y[last], run["label"]))
    lo, hi = ax.get_ylim()
    for r, (xe, ye, lab) in enumerate(ends):  # direct labels, pushed apart when the lines end close
        other = [e[1] for k, e in enumerate(ends) if k != r]
        dy = 0
        if other and abs(ye - other[0]) < 0.06 * (hi - lo):
            dy = 7 if ye > other[0] or (ye == other[0] and r == 0) else -7
        ax.annotate(lab, (xe, ye), xytext=(8, dy), textcoords="offset points", va="center",
                    fontsize=8.5, color=INK2)
    ax.set_xticks(x, labels)
    ax.set_xlim(-0.3, len(bins) + 0.5)
    ax.yaxis.grid(True, color=GRID, lw=0.8)
    ax.set_xlabel("height of the annotated person (pixels, 1920x1080 frame)", color=INK2, fontsize=9)
    ax.set_ylabel("people detected (%)", color=INK2, fontsize=9)
    ax.set_title(f"Detection recall by person height (score ≥ {runs[0]['summary']['det_thresh']},"
                 " IoU ≥ 0.5)", fontsize=11, color=INK, loc="left", pad=26)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="lower left", bbox_to_anchor=(0, 1.0),
              ncol=len(runs), borderaxespad=0.2)
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


# ----------------------------------------------------------------------------- main
def _slug(s):
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def main(argv=None):
    p = argparse.ArgumentParser(description="Qualitative PRW results from cached features.")
    p.add_argument("--run", nargs=2, action="append", required=True, metavar=("LABEL", "FEATURES"),
                   help="model label and feature cache (.pt from eval_prw.py --cache); at most 2")
    p.add_argument("--data-root", default=os.environ.get("PRW_ROOT", "data/PRW"))
    p.add_argument("--out", default="results/qualitative")
    p.add_argument("--det-thresh", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-random", type=int, default=6, help="queries in search_random.jpg")
    p.add_argument("--n-fail", type=int, default=3, help="queries per failure group")
    p.add_argument("--n-compare", type=int, default=4, help="queries per comparison figure")
    args = p.parse_args(argv)
    if len(args.run) > 2:
        p.error("at most two --run")

    gallery, query = PRW(args.data_root, None, "gallery"), PRW(args.data_root, None, "query")
    runs = [analyse_run(label, path, gallery, query, args.det_thresh) for label, path in args.run]
    os.makedirs(args.out, exist_ok=True)

    for run in runs:
        d = osp.join(args.out, _slug(run["label"]))
        os.makedirs(d, exist_ok=True)
        recs, s = run["records"], run["summary"]
        with open(osp.join(d, "queries.csv"), "w", newline="") as f:
            cols = ["index", "img_name", "pid", "cam", "group", "first_hit_rank", "gt_frames",
                    "gt_frames_detected", "top1_sim"]
            w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
            w.writeheader()
            w.writerows(recs)
        name = run["label"]
        pick = _sample(_rng(args.seed, name + "random"), recs, args.n_random)
        search_figure(run, pick, query, f"{name}: {len(pick)} random queries (seed {args.seed}), top-5 of"
                      f" the course eval ranking. mAP {s['mAP']:.2%}, top-1 {s['top1']:.2%}",
                      osp.join(d, "search_random.jpg"))
        for g in GROUPS:
            members = [r for r in recs if r["group"] == g]
            if g == "correct" or not members:
                continue
            pick = _sample(_rng(args.seed, name + g), members, args.n_fail)
            search_figure(run, pick, query, f"{name}: top-1 wrong, {GROUPS[g]}: {len(members)} of {len(recs)}"
                          f" queries ({len(members) / len(recs):.1%}); {len(pick)} drawn at random",
                          osp.join(d, f"failures_{g}.jpg"), whole_frame=True)
        frames = run["frames"]
        pick = _sample(_rng(args.seed, name + "det"), [f["name"] for f in frames], 4)
        detection_figure(run, pick, f"{name}: detections on 4 random gallery frames (score ≥ {args.det_thresh})",
                         osp.join(d, "detections_random.jpg"))
        missed = [f["name"] for f in frames if f["missed"] > 0]
        pick = _sample(_rng(args.seed, name + "miss"), missed, 4)
        detection_figure(run, pick, f"{name}: 4 random frames among the {len(missed)} of {len(frames)} with a"
                         " missed person", osp.join(d, "detections_misses.jpg"))

    if len(runs) == 2:
        a, b = runs
        hits_a = [r["group"] == "correct" for r in a["records"]]
        hits_b = [r["group"] == "correct" for r in b["records"]]
        fixed = [i for i, (x, y) in enumerate(zip(hits_a, hits_b)) if x and not y]
        broken = [i for i, (x, y) in enumerate(zip(hits_a, hits_b)) if y and not x]
        for tag, idx, desc in (("fixed", fixed, f"{a['label']} right, {b['label']} wrong"),
                               ("broken", broken, f"{b['label']} right, {a['label']} wrong")):
            if not idx:
                continue
            pick = _sample(_rng(args.seed, "compare" + tag), idx, args.n_compare)
            compare_figure(a, b, pick, query, f"Top-1 {desc}: {len(idx)} of {len(hits_a)} queries;"
                           f" {len(pick)} drawn at random. Top-3 of each model",
                           osp.join(args.out, f"compare_{tag}.jpg"))
        print(f"top-1: {a['label']} fixes {len(fixed)} queries of {b['label']}, breaks {len(broken)}")

    top1_chart(runs, osp.join(args.out, "top1_errors.png"))
    recall_chart(runs, osp.join(args.out, "recall_by_height.png"))
    summary = {"seed": args.seed, "runs": [r["summary"] for r in runs]}
    if len(runs) == 2:
        summary["compare"] = {"fixed": len(fixed), "broken": len(broken)}
    with open(osp.join(args.out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
