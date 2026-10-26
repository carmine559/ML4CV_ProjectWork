"""
PRW Dataset Loader
------------------
Handles the PRW dataset structure as distributed on Kaggle:

    PRW/
    ├── annotations/       # per-frame .mat files: [x1, y1, w, h, pid] rows
    ├── frames/            # full scene images (.jpg)
    ├── query_box/         # pre-cropped query person images (.jpg)
    ├── ID_train.mat       # labeled person IDs used in training
    ├── ID_test.mat        # labeled person IDs in test set
    ├── frame_train.mat    # list of training frame names
    ├── frame_test.mat     # list of test frame names
    ├── query_info.txt     # query set: one row per query
    └── readme.txt

query_info.txt format (space-separated):
    x1  y1  w  h  pid  cam_id  frame_name
    (coordinates are 1-indexed MATLAB style)

Person IDs:
    pid > 0   → labeled identity
    pid == -2 → unlabeled / distractor pedestrian

Frame name format: c{cam}s{seq}_{frame:06d}_{idx:02d}
    e.g. c1s3_000001_01  →  camera 1
"""

import os
import os.path as osp
import zipfile

import numpy as np
import scipy.io as sio
import torch
from PIL import Image
from torch.utils.data import Dataset


# --------------------------------------------------------------------------- #
#  Setup helper: unzip if needed
# --------------------------------------------------------------------------- #

def prepare_dataset(zip_path: str, extract_to: str) -> str:
    """
    If the PRW dataset is provided as a .zip, extract it once and return
    the path to the extracted root directory.

    Args:
        zip_path   : path to the .zip file (e.g. '/kaggle/input/prw/PRW.zip')
        extract_to : directory where the zip will be extracted

    Returns:
        root : path to the PRW root folder inside the extracted directory
    """
    if not osp.exists(extract_to):
        os.makedirs(extract_to)

    print(f"Extracting {zip_path} → {extract_to} ...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)
    print("Extraction complete.")

    # The zip may contain a single top-level folder (e.g. 'PRW-v16.04.20/')
    # or dump files directly. Find the actual root.
    entries = os.listdir(extract_to)
    if len(entries) == 1 and osp.isdir(osp.join(extract_to, entries[0])):
        root = osp.join(extract_to, entries[0])
    else:
        root = extract_to

    return root


# --------------------------------------------------------------------------- #
#  Low-level helpers
# --------------------------------------------------------------------------- #

def _load_mat_frames(mat_path: str) -> list:
    """Return a flat list of frame name strings from a .mat cell array."""
    data = sio.loadmat(mat_path)
    key = [k for k in data.keys() if not k.startswith("__")][0]
    raw = data[key].squeeze()
    return [str(f[0]) if isinstance(f, np.ndarray) else str(f) for f in raw]


def _load_mat_ids(mat_path: str) -> np.ndarray:
    """Return a 1-D array of integer person IDs from ID_train.mat / ID_test.mat."""
    data = sio.loadmat(mat_path)
    key = [k for k in data.keys() if not k.startswith("__")][0]
    return data[key].ravel().astype(np.int64)


def _load_frame_annotations(ann_path: str) -> np.ndarray:
    """
    Load a per-frame annotation .mat file.
    Raw format in box_new: [pid, x1, y1, w, h]  (MATLAB 1-indexed coords).
    We reorder to [x1, y1, w, h, pid] for consistency with the rest of the code.
    """
    data = sio.loadmat(ann_path)
    key = [k for k in data.keys() if not k.startswith("__")][0]
    boxes = data[key]
    if boxes.ndim == 1:
        boxes = boxes[np.newaxis, :]
    boxes = boxes.astype(np.float32)
    # Reorder: [pid, x1, y1, w, h] -> [x1, y1, w, h, pid]
    return np.column_stack([boxes[:, 1:5], boxes[:, 0]])


def _parse_cam_id(frame_name: str) -> int:
    """
    Extract camera ID from PRW frame name.
    Format: c{cam_id}s{seq}_{frame}_{idx}  e.g. 'c1s3_000001_01' → 1
    """
    try:
        return int(frame_name[1])
    except (IndexError, ValueError):
        return -1


def _mat_box_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    """
    Convert [x1, y1, w, h] (1-indexed MATLAB) → [x1, y1, x2, y2] (0-indexed).
    Input shape: (N, 4) or (N, 5) — only the first 4 columns are touched.
    """
    out = boxes[:, :4].copy()
    out[:, 0] -= 1                        # x1: 1-indexed → 0-indexed
    out[:, 1] -= 1                        # y1
    out[:, 2] = out[:, 0] + boxes[:, 2]  # x2 = x1 + w
    out[:, 3] = out[:, 1] + boxes[:, 3]  # y2 = y1 + h
    return out


def _parse_query_info_txt(txt_path: str) -> list:
    """
    Parse query_info.txt into a list of dicts.

    Actual column format (space-separated, 1-indexed MATLAB coords):
        pid  x1  y1  w  h  frame_name

    Note: pid is the FIRST column, there is NO cam_id column.
    cam_id is derived from the frame name (e.g. 'c1s3_016471' -> cam 1).
    """
    queries = []
    with open(txt_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            try:
                frame_name     = parts[-1]
                nums           = [float(p) for p in parts[:-1]]
                pid            = int(nums[0])
                x1, y1, w, h  = nums[1], nums[2], nums[3], nums[4]
                cam_id         = _parse_cam_id(frame_name)
                queries.append({
                    "pid": pid, "x1": x1, "y1": y1, "w": w, "h": h,
                    "cam_id": cam_id, "frame_name": frame_name,
                })
            except (ValueError, IndexError) as e:
                print(f"[query_info.txt] Skipping malformed line: {line!r}  ({e})")
    return queries


# --------------------------------------------------------------------------- #
#  Training Dataset
# --------------------------------------------------------------------------- #

class PRWTrainDataset(Dataset):
    """
    Each sample is one training scene image with all its annotations.

    Returns:
        image  : Tensor (C, H, W)  after transforms
        target : dict
            'boxes'  : FloatTensor (N, 4)  x1y1x2y2, 0-indexed
            'labels' : LongTensor  (N,)    always 1 (pedestrian)
            'pids'   : LongTensor  (N,)    contiguous index ≥0 for labeled,
                                           -1 for unlabeled/distractor
    """

    def __init__(self, root: str, transforms=None):
        self.root = root
        self.transforms = transforms

        # Use ID_train.mat to get the canonical set of labeled training pids
        id_train_path = osp.join(root, "ID_train.mat")
        if osp.exists(id_train_path):
            labeled_pids = sorted(_load_mat_ids(id_train_path).tolist())
        else:
            # Fallback: collect from annotations
            frame_names = _load_mat_frames(osp.join(root, "frame_train.mat"))
            labeled_pids = set()
            for name in frame_names:
                ann_path = osp.join(root, "annotations", name + ".jpg.mat")
                if osp.exists(ann_path):
                    anns = _load_frame_annotations(ann_path)
                    for pid in anns[:, 4]:
                        if pid > 0:
                            labeled_pids.add(int(pid))
            labeled_pids = sorted(labeled_pids)

        self.pid2label = {pid: idx for idx, pid in enumerate(labeled_pids)}
        self.num_labeled_pids = len(labeled_pids)

        # Build sample list
        frame_names = _load_mat_frames(osp.join(root, "frame_train.mat"))
        self.samples = self._build_samples(frame_names)

    def _build_samples(self, frame_names):
        samples = []
        for name in frame_names:
            img_path = osp.join(self.root, "frames", name + ".jpg")
            ann_path = osp.join(self.root, "annotations", name + ".jpg.mat")
            if not osp.exists(img_path) or not osp.exists(ann_path):
                continue
            anns = _load_frame_annotations(ann_path)
            if anns.shape[0] == 0:
                continue
            samples.append((img_path, anns))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, anns = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        boxes = _mat_box_to_xyxy(anns)  # (N, 4) float32, 0-indexed

        pids_raw = anns[:, 4].astype(np.int64)
        pids = np.array(
            [self.pid2label[p] if p in self.pid2label else -1 for p in pids_raw],
            dtype=np.int64,
        )

        target = {
            "boxes":  torch.as_tensor(boxes, dtype=torch.float32),
            "labels": torch.ones(len(boxes), dtype=torch.int64),
            "pids":   torch.as_tensor(pids,  dtype=torch.int64),
        }

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target


# --------------------------------------------------------------------------- #
#  Gallery Dataset
# --------------------------------------------------------------------------- #

class PRWGalleryDataset(Dataset):
    """
    Test-set scene images.

    Attributes:
        img_prefix  : path to frames/ directory         (used by eval_search_prw)
        annotations : list of per-frame annotation dicts (used by eval_search_prw)
            {
                'img_name' : str            frame name (no extension)
                'boxes'    : ndarray (N,4)  GT boxes x1y1x2y2, 0-indexed, float32
                'pids'     : ndarray (N,)   person IDs (raw, not remapped), int64
                'cam_id'   : int
            }
    """

    def __init__(self, root: str, transforms=None):
        self.root = root
        self.transforms = transforms
        self.img_prefix = osp.join(root, "frames")

        frame_names = _load_mat_frames(osp.join(root, "frame_test.mat"))
        self.samples = []
        self.annotations = []

        for name in frame_names:
            img_path = osp.join(self.root, "frames", name + ".jpg")
            ann_path = osp.join(self.root, "annotations", name + ".jpg.mat")
            if not osp.exists(img_path):
                continue

            if osp.exists(ann_path):
                anns = _load_frame_annotations(ann_path)
                boxes = _mat_box_to_xyxy(anns).astype(np.float32)
                pids  = anns[:, 4].astype(np.int64)
            else:
                boxes = np.zeros((0, 4), dtype=np.float32)
                pids  = np.zeros(0,      dtype=np.int64)

            self.samples.append((img_path, name, boxes, pids))
            self.annotations.append(
                {
                    "img_name": name,
                    "boxes":    boxes,
                    "pids":     pids,
                    "cam_id":   _parse_cam_id(name),
                }
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, frame_name, boxes, pids = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transforms is not None:
            image, _ = self.transforms(image, None)

        meta = {
            "frame_name": frame_name,
            "gt_boxes":   torch.as_tensor(boxes, dtype=torch.float32),
            "gt_pids":    torch.as_tensor(pids,  dtype=torch.int64),
        }
        return image, meta


# --------------------------------------------------------------------------- #
#  Query Dataset
# --------------------------------------------------------------------------- #

class PRWQueryDataset(Dataset):
    """
    Query set. Reads pre-cropped images from query_box/ and metadata from
    query_info.txt.

    query_info.txt columns (space-separated, 1-indexed):
        x1  y1  w  h  pid  cam_id  frame_name

    Attributes:
        annotations : list of per-query dicts        (used by eval_search_prw)
            {
                'img_name' : str            source frame name (no extension)
                'boxes'    : ndarray (1,4)  query box x1y1x2y2, 0-indexed
                'pids'     : int            person ID
                'cam_id'   : int
            }
    """

    def __init__(self, root: str, transforms=None):
        self.root = root
        self.transforms = transforms
        self.annotations = []
        self.queries = self._load_queries()

    def _load_queries(self):
        txt_path = osp.join(self.root, "query_info.txt")
        raw_queries = _parse_query_info_txt(txt_path)

        queries = []
        for q in raw_queries:
            frame_name = q["frame_name"]

            # Pre-cropped image: query_box/{frame_name}.jpg  (common convention)
            # Fall back to cropping from the full frame if the file doesn't exist.
            crop_path = osp.join(self.root, "query_box", f"{q['pid']}_{frame_name}.jpg")
            full_frame_path = osp.join(self.root, "frames", frame_name + ".jpg")

            if not osp.exists(crop_path) and not osp.exists(full_frame_path):
                continue  # skip if neither source exists

            # Box in 0-indexed x1y1x2y2
            box = np.array(
                [[q["x1"] - 1, q["y1"] - 1,
                  q["x1"] - 1 + q["w"], q["y1"] - 1 + q["h"]]],
                dtype=np.float32,
            )  # shape (1, 4)

            queries.append(
                {
                    "crop_path":       crop_path if osp.exists(crop_path) else None,
                    "full_frame_path": full_frame_path,
                    "box":             box,
                    "pid":             q["pid"],
                    "cam_id":          q["cam_id"],
                    "frame_name":      frame_name,
                }
            )
            self.annotations.append(
                {
                    "img_name": frame_name,
                    "boxes":    box,         # (1, 4) ndarray — squeezed in eval fn
                    "pids":     q["pid"],    # int
                    "cam_id":   q["cam_id"],
                }
            )

        print(f"Loaded {len(queries)} queries from query_info.txt")
        return queries

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        q = self.queries[idx]

        if q["crop_path"] is not None:
            # Use the pre-cropped image directly from query_box/
            image = Image.open(q["crop_path"]).convert("RGB")
        else:
            # Fall back: crop the person from the full frame
            image = Image.open(q["full_frame_path"]).convert("RGB")
            x1, y1, x2, y2 = [int(round(v)) for v in q["box"].ravel()]
            image = image.crop((x1, y1, x2, y2))

        if self.transforms is not None:
            image, _ = self.transforms(image, None)

        return image, {"pid": q["pid"], "frame": q["frame_name"], "box": q["box"]}


# --------------------------------------------------------------------------- #
#  DataLoader collate
# --------------------------------------------------------------------------- #

def collate_fn(batch):
    """Custom collate: return images and targets as plain lists."""
    images, targets = zip(*batch)
    return list(images), list(targets)