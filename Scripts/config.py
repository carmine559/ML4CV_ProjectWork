"""
Configuration. Defaults = official SeqNet PRW recipe (SeqNet defaults.py merged with
configs/prw.yaml); key names follow the official yacs config so values can be checked 1:1.
Precedence: DEFAULTS < --cfg YAML < --opts KEY VALUE < explicit CLI flags.
Dataset root: --data-root > $PRW_ROOT > INPUT.DATA_ROOT.
"""
import copy
import os

import yaml


class CfgNode(dict):
    """dict with attribute access (cfg.MODEL.RPN.NMS_THRESH), like yacs."""

    def __getattr__(self, k):
        try:
            return self[k]
        except KeyError as e:
            raise AttributeError(k) from e

    def __setattr__(self, k, v):
        self[k] = v

    def to_dict(self):
        return {k: v.to_dict() if isinstance(v, CfgNode) else v for k, v in self.items()}


DEFAULTS = {
    "INPUT": {
        "DATASET": "PRW",
        "DATA_ROOT": "data/PRW",
        "MIN_SIZE": 900,  # shorter side
        "MAX_SIZE": 1500,  # longer side
        "BATCH_SIZE_TRAIN": 5,
        "BATCH_SIZE_TEST": 1,
        "NUM_WORKERS_TRAIN": 5,
        "NUM_WORKERS_TEST": 1,
    },
    "SOLVER": {
        "MAX_EPOCHS": 18,
        "BASE_LR": 0.003,
        "HEAD_LR_MULT": 1.0,  # lr x this for the non-pre-trained layers (RPN, box predictors, NAE); 1 = official
        "WARMUP_FACTOR": 0.001,  # linear warm-up over the whole first epoch
        "LR_DECAY_MILESTONES": [16],
        "GAMMA": 0.1,
        "WEIGHT_DECAY": 0.0005,
        "SGD_MOMENTUM": 0.9,
        "LW_RPN_REG": 1.0,
        "LW_RPN_CLS": 1.0,
        "LW_PROPOSAL_REG": 10.0,
        "LW_PROPOSAL_CLS": 1.0,
        "LW_BOX_REG": 1.0,
        "LW_BOX_CLS": 1.0,
        "LW_BOX_REID": 1.0,
        "CLIP_GRADIENTS": 10.0,  # <= 0 disables
    },
    "MODEL": {
        "BACKBONE": {
            "NAME": "resnet50",  # resnet50 | swin_tiny | swin_small | swin_base (SOLIDER)
            "PRETRAINED": True,  # resnet50: ImageNet IMAGENET1K_V1; swin_*: the WEIGHTS file
            "FROZEN_BN": False,  # official: plain trainable BatchNorm2d (resnet50 only)
            "SEMANTIC_WEIGHT": 0.6,  # swin_* only: SOLIDER lambda (0.6 in SOLIDER-PersonSearch)
            "WEIGHTS": "",  # swin_* only: SOLIDER checkpoint path; $VARS are expanded
        },
        "RPN": {
            "NMS_THRESH": 0.7,
            "BATCH_SIZE_TRAIN": 256,
            "POS_FRAC_TRAIN": 0.5,
            "POS_THRESH_TRAIN": 0.7,
            "NEG_THRESH_TRAIN": 0.3,
            "PRE_NMS_TOPN_TRAIN": 12000,
            "PRE_NMS_TOPN_TEST": 6000,
            "POST_NMS_TOPN_TRAIN": 2000,
            "POST_NMS_TOPN_TEST": 300,
        },
        "ROI_HEAD": {
            "BN_NECK": True,
            "BATCH_SIZE_TRAIN": 128,
            "POS_FRAC_TRAIN": 0.5,
            "POS_THRESH_TRAIN": 0.5,
            "NEG_THRESH_TRAIN": 0.5,
            "SCORE_THRESH_TEST": 0.5,
            "NMS_THRESH_TEST": 0.4,  # after the first (Faster R-CNN) head
            "NMS_THRESH_TEST_2": 0.5,  # after the second head (hard-coded in official code)
            "DETECTIONS_PER_IMAGE_TEST": 300,
        },
        "LOSS": {
            "LUT_SIZE": 482,
            "CQ_SIZE": 500,
            "OIM_MOMENTUM": 0.5,
            "OIM_SCALAR": 30.0,
        },
    },
    # ---- engineering options (not part of the official recipe) ----
    "AMP": "off",  # off (official fp32) | fp16 | bf16
    "EVAL_PERIOD": 0,  # official: 1 (eval every epoch); 0 = no eval during training
    "EVAL_DET_THRESHOLDS": [0.5, 0.6, 0.7, 0.8],  # 0.5 = SeqNet default
    "EVAL_FLIP_TTA": False,
    "CKPT_PERIOD": 1,
    "CKPT_KEEP": 2,  # keep the N most recent full checkpoints (0 = keep all)
    "DISP_PERIOD": 10,
    "TF_BOARD": True,
    "CUDNN_BENCHMARK": False,  # official: deterministic cuDNN when SEED >= 0
    "DEVICE": "cuda",
    "SEED": 1,
    "OUTPUT_DIR": "./exp_prw",
}


def _to_node(d):
    return CfgNode({k: _to_node(v) if isinstance(v, dict) else v for k, v in d.items()})


def _merge(base, new, prefix=""):
    for k, v in new.items():
        if k not in base:
            raise KeyError(f"Unknown config key: {prefix}{k}")
        if isinstance(base[k], dict):
            if not isinstance(v, dict):
                raise TypeError(f"{prefix}{k} must be a mapping")
            _merge(base[k], v, prefix + k + ".")
        else:
            base[k] = v


def set_by_path(cfg, dotted, value):
    keys = dotted.split(".")
    node = cfg
    for k in keys[:-1]:
        if k not in node:
            raise KeyError(f"Unknown config key: {dotted}")
        node = node[k]
    if keys[-1] not in node or isinstance(node[keys[-1]], dict):
        raise KeyError(f"Unknown config key: {dotted}")
    node[keys[-1]] = value


def load_config(cfg_file=None, opts=None, data_root=None, base=None):
    """base: optional dict (e.g. the config stored in a checkpoint) merged before cfg_file."""
    cfg = copy.deepcopy(DEFAULTS)
    if base:
        _merge(cfg, base)
    if cfg_file:
        with open(cfg_file, "r") as f:
            _merge(cfg, yaml.safe_load(f) or {})
    opts = opts or []
    if len(opts) % 2:
        raise ValueError("--opts expects KEY VALUE pairs")
    for k, v in zip(opts[0::2], opts[1::2]):
        set_by_path(cfg, k, yaml.safe_load(v))
    if data_root:
        cfg["INPUT"]["DATA_ROOT"] = data_root
    elif os.environ.get("PRW_ROOT"):
        cfg["INPUT"]["DATA_ROOT"] = os.environ["PRW_ROOT"]
    if cfg["AMP"] not in ("off", "fp16", "bf16"):
        raise ValueError(f"AMP must be off|fp16|bf16, got {cfg['AMP']}")
    return _to_node(cfg)


def dump_config(cfg, path):
    with open(path, "w") as f:
        yaml.safe_dump(cfg.to_dict(), f, sort_keys=False)
