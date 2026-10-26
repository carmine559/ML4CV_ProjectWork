# Adapted from SeqNet (https://github.com/serend1p1ty/SeqNet), datasets/{base,prw,build}.py
# Copyright (c) 2022 Zj Li. MIT License, see LICENSE-SeqNet.
#
# Box convention (official SeqNet, used everywhere: training targets, gallery GT, query boxes,
# and therefore detections): pixel coordinates [x1, y1, x2, y2] in the original frame, with
# x2 = x + w, y2 = y + h computed from the raw [x, y, w, h], negatives clipped to 0, cast to int32.
# PID handling (official): raw PRW ids, unlabeled (-2) -> 5555. Training pids are
# 1..478, 480..483 and 932; with LUT_SIZE=482 the OIM maps pid p to label p-1.
#
# Changes: robust root discovery (dataset may sit in a sub-folder, e.g. on Kaggle), robust .mat /
# query_info.txt parsing (line endings, nested cell arrays), query `pids` stored as int (course
# eval contract).
import os
import os.path as osp
import re

import numpy as np
import torch
from PIL import Image
from scipy.io import loadmat

from transforms import build_transforms

UNLABELED_PID = 5555
_MARKERS = ("frame_test.mat", "query_info.txt")
_SKIP_DIRS = {"frames", "annotations", "query_box"}


def resolve_prw_root(root, max_depth=3):
    """Return the folder that contains frames/, annotations/, frame_*.mat and query_info.txt."""
    root = osp.abspath(osp.expanduser(root))

    def ok(d):
        return all(osp.isfile(osp.join(d, m)) for m in _MARKERS) and osp.isdir(
            osp.join(d, "annotations")
        )

    if ok(root):
        return root
    for dirpath, dirnames, _ in os.walk(root):
        depth = osp.relpath(dirpath, root).count(os.sep) + (dirpath != root)
        dirnames[:] = [] if depth >= max_depth else [d for d in dirnames if d not in _SKIP_DIRS]
        if ok(dirpath):
            return dirpath
    raise FileNotFoundError(
        f"PRW not found under {root}: expected a folder with frames/, annotations/, "
        f"frame_train.mat, frame_test.mat and query_info.txt"
    )


def _mat_strings(arr):
    """Flatten a MATLAB cell array of strings loaded by scipy into a list of str."""
    out = []
    for x in np.asarray(arr, dtype=object).ravel():
        while isinstance(x, np.ndarray):
            x = x.ravel()[0]
        out.append(str(x).strip())
    return out


class PRW:
    name = "PRW"

    def __init__(self, root, transforms, split):
        assert split in ("train", "gallery", "query")
        self.root = resolve_prw_root(root)
        self.img_prefix = osp.join(self.root, "frames")
        if not osp.isdir(self.img_prefix):
            raise FileNotFoundError(f"{self.img_prefix} does not exist")
        self.transforms = transforms
        self.split = split
        self.annotations = self._load_annotations()

    # ------------------------------------------------------------------ loading
    def _get_cam_id(self, img_name):
        match = re.search(r"c\d", img_name).group().replace("c", "")
        return int(match)

    def _load_queries(self):
        query_info = osp.join(self.root, "query_info.txt")
        queries = []
        with open(query_info, "r") as f:
            for line in f:
                parts = line.split()
                if not parts:
                    continue
                pid = int(float(parts[0]))
                x, y, w, h = (float(v) for v in parts[1:5])
                roi = np.array([x, y, x + w, y + h]).astype(np.int32)
                roi = np.clip(roi, 0, None)  # several coordinates are negative
                img_name = parts[5] + ".jpg"
                queries.append(
                    {
                        "img_name": img_name,
                        "img_path": osp.join(self.img_prefix, img_name),
                        "boxes": roi[np.newaxis, :],
                        "pids": pid,
                        "cam_id": self._get_cam_id(img_name),
                    }
                )
        return queries

    def _load_split_img_names(self):
        """
        Load the image names for the specific split.
        """
        fname, key = (
            ("frame_train.mat", "img_index_train")
            if self.split == "train"
            else ("frame_test.mat", "img_index_test")
        )
        mat = loadmat(osp.join(self.root, fname))
        if key not in mat:
            key = [k for k in mat if not k.startswith("__")][0]
        return [n if n.endswith(".jpg") else n + ".jpg" for n in _mat_strings(mat[key])]

    def _load_annotations(self):
        if self.split == "query":
            return self._load_queries()

        annotations = []
        for img_name in self._load_split_img_names():
            anno_path = osp.join(self.root, "annotations", img_name + ".mat")
            if not osp.isfile(anno_path):  # tolerate "<name>.mat" without ".jpg"
                anno_path = osp.join(self.root, "annotations", img_name[:-4] + ".mat")
            anno = loadmat(anno_path)
            box_key = "box_new"
            if box_key not in anno.keys():
                box_key = "anno_file"
            if box_key not in anno.keys():
                box_key = "anno_previous"

            raw = np.asarray(anno[box_key], dtype=np.float64).reshape(-1, 5)
            rois = raw[:, 1:]
            ids = raw[:, 0].copy()
            rois = np.clip(rois, 0, None)  # several coordinates are negative

            assert len(rois) == len(ids)

            rois[:, 2:] += rois[:, :2]
            ids[ids == -2] = UNLABELED_PID  # assign pid = 5555 for unlabeled people
            annotations.append(
                {
                    "img_name": img_name,
                    "img_path": osp.join(self.img_prefix, img_name),
                    "boxes": rois.astype(np.int32),
                    "pids": ids.astype(np.int32),
                    "cam_id": self._get_cam_id(img_name),
                }
            )
        return annotations

    # ------------------------------------------------------------------ access
    def __getitem__(self, index):
        anno = self.annotations[index]
        img = Image.open(anno["img_path"]).convert("RGB")
        boxes = torch.as_tensor(anno["boxes"], dtype=torch.float32)
        labels = torch.as_tensor(np.atleast_1d(anno["pids"]), dtype=torch.int64)
        target = {"img_name": anno["img_name"], "boxes": boxes, "labels": labels}
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target

    def __len__(self):
        return len(self.annotations)

    def statistics(self):
        pids = set()
        n_boxes = 0
        for a in self.annotations:
            n_boxes += a["boxes"].shape[0]
            pids.update(int(p) for p in np.atleast_1d(a["pids"]))
        labeled = sorted(p for p in pids if p != UNLABELED_PID)
        s = f"PRW-{self.split}: {len(self)} images, {n_boxes} boxes, {len(labeled)} labeled pids"
        if labeled:
            s += f" (min {labeled[0]}, max {labeled[-1]})"
        if UNLABELED_PID in pids:
            s += f", unlabeled pid {UNLABELED_PID}"
        return s + f"  [root: {self.root}]"


def collate_fn(batch):
    return tuple(zip(*batch))


def build_dataset(cfg, split, verbose=True):
    assert cfg.INPUT.DATASET == "PRW", "only PRW is supported"
    ds = PRW(cfg.INPUT.DATA_ROOT, build_transforms(is_train=(split == "train")), split)
    if verbose:
        print("=> " + ds.statistics())
    return ds


def build_train_loader(cfg):
    dataset = build_dataset(cfg, "train")
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=cfg.INPUT.BATCH_SIZE_TRAIN,
        shuffle=True,
        num_workers=cfg.INPUT.NUM_WORKERS_TRAIN,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_fn,
    )


def build_test_loaders(cfg):
    loaders = []
    for split in ("gallery", "query"):
        loaders.append(
            torch.utils.data.DataLoader(
                build_dataset(cfg, split),
                batch_size=cfg.INPUT.BATCH_SIZE_TEST,
                shuffle=False,
                num_workers=cfg.INPUT.NUM_WORKERS_TEST,
                pin_memory=True,
                collate_fn=collate_fn,
            )
        )
    return tuple(loaders)
