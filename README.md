# SeqNet person search on PRW (reimplementation)

This is a clean reimplementation of **SeqNet** (Z. Li and D. Miao, *Sequential End-to-end Network for Efficient Person Search*, AAAI 2021) for the PRW dataset. It was written for the *Machine Learning for Computer Vision* course (University of Bologna).

The model code is adapted closely from the official repository [serend1p1ty/SeqNet](https://github.com/serend1p1ty/SeqNet) (MIT License, Copyright (c) 2022 Zj Li; see `LICENSE-SeqNet`). Every adapted file starts with an attribution header that lists its changes. All credit for the method goes to the original authors. Search metrics are computed with the course-provided `eval_search_prw` (`Scripts/evaluate.py`), which is kept unchanged.

> `main.ipynb`, `report.md` and `weights/` belong to the earlier course submission: a single-stage Faster R-CNN FPN with OIM and NAE. They do not use this code. The original code is preserved in `archive/course_submission_v1/`.

## Layout

```text
Scripts/
  seqnet.py      SeqNet: RPN, Faster R-CNN head, baseline (re-ID) head, NAE, FCS, CWS
  backbone.py    build_backbone(cfg): ResNet-50 conv1-layer3 stem + res5 head (single builder)
  oim.py         OIM loss (LUT 482 / CQ 500, updated in backward like the official code)
  prw.py         PRW dataset and loaders (official box and pid conventions)
  transforms.py  ToTensor + horizontal flip (the official augmentation)
  config.py      defaults = official SeqNet PRW recipe; YAML and CLI overrides
  engine.py      training epoch, gallery/query feature extraction (CWS, optional flip TTA)
  evaluate.py    course eval_search_prw (unchanged)
  train.py       training entry point
  eval_prw.py    evaluation entry point (det_thresh sweep, JSON output)
configs/prw.yaml         the full recipe, identical to config.py defaults
tools/download_prw.sh    Kaggle download
slurm/train_prw.sbatch   1x L40 template: train, auto-resume/requeue, then evaluate
```

## Setup and data

```bash
pip install -r requirements.txt            # torch 2.7.1 / torchvision 0.22.1 (CUDA 12.x, Ada OK)
bash tools/download_prw.sh data/PRW        # Kaggle CLI + ~/.kaggle/kaggle.json; prints PRW_ROOT
export PRW_ROOT=/path/printed/above        # or pass --data-root
```

The loader expects `frames/`, `annotations/*.jpg.mat`, `frame_train.mat`, `frame_test.mat` and `query_info.txt`. It also finds them up to three levels below the folder you pass, as in Kaggle's `/kaggle/input/<dataset>/...` layout.

## Train

**UniBo DISI GPU cluster** (partition `l40`: 1x L40 48 GB, 8 CPU cores, 64 GB RAM; driver 590 / CUDA 13.1; SLURM 24.11). Jobs are submitted from `giano.cs.unibo.it`. The home quota is 400 MB, so the code, venv, dataset and caches all live in `/scratch.hpc/$USER`. Files there are deleted after 40 days without access, so copy your results elsewhere.

```bash
# from your PC: copy the code (not weights/ or archive/)
ssh <name.surname>@giano.cs.unibo.it "mkdir -p /scratch.hpc/<name.surname>/prw-seqnet"
scp -r Scripts configs slurm tools requirements.txt README.md <name.surname>@giano.cs.unibo.it:/scratch.hpc/<name.surname>/prw-seqnet/

# on giano, once: venv (torch cu128), cached ImageNet weights, PRW from Kaggle (needs ~/.kaggle/kaggle.json)
cd /scratch.hpc/$USER/prw-seqnet && bash slurm/setup_giano.sh

mkdir -p logs
SMOKE=1 sbatch --time=01:00:00 slurm/train_prw.sbatch   # 20-iteration check + full eval (~pipeline test)
sbatch slurm/train_prw.sbatch                           # official recipe, fp32
AMP=bf16 sbatch slurm/train_prw.sbatch                  # faster autocast variant
squeue -u $USER ; tail -f logs/seqnet_prw_<jobid>.out
```

The job saves `exp_prw/epoch_N.pth` after every epoch and always restarts with `--resume auto`. Five minutes before the time limit it requeues itself; if requeueing is not permitted, run `sbatch` again and it resumes. After training it evaluates the last epoch into `exp_prw/eval_epoch_N.json`. Check the partition's time limit with `sinfo -p l40 -o "%P %l"` and keep `--time` within it.

Running directly:

```bash
python Scripts/train.py --cfg configs/prw.yaml --data-root $PRW_ROOT --output-dir exp_prw \
    [--amp off|fp16|bf16] [--resume auto] [--eval-period 0] [--seed 1] [--opts KEY VALUE ...]
```

The defaults are the official recipe: batch 5, SGD lr 0.003 (momentum 0.9, wd 5e-4), linear warm-up over epoch 0, 18 epochs, lr x0.1 at epoch 16, 900x1500 input, gradient clipping at 10. Logs go to `train_log.csv`, `epoch_log.csv` and `tf_log/` (TensorBoard).

**Batch size and LR:** follow the linear scaling rule, `lr = 0.003 * batch / 5`, for example `--batch-size 2 --lr 0.0012`. The official README says batch 5 needs about 28 GB in fp32, which fits on a 48 GB L40.

**Kaggle notebook (T4/P100, 16 GB, 12 h sessions):** run with `--data-root /kaggle/input/<PRW dataset folder> --output-dir /kaggle/working/exp_prw --amp fp16 --batch-size 2 --lr 0.0012`. To continue in a later session, attach the previous output and run with `--resume <path to epoch_N.pth>`.

## Evaluate

```bash
python Scripts/eval_prw.py --ckpt exp_prw/epoch_17.pth --data-root $PRW_ROOT \
    [--det-thresh 0.5 0.6 0.7 0.8] [--flip-tta] [--cache exp_prw/features.pt]
```

The script extracts gallery detections and features and query features (GT query box). It applies **CWS** (gallery embedding multiplied by the detection score, as in official SeqNet), calls the unchanged `eval_search_prw` for every `det_thresh`, and writes a JSON file. That file also contains the detection recall and AP. `--cache` stores the features so a sweep can be re-run on CPU.

Note on thresholds: `0.5` is SeqNet's default, and with `SCORE_THRESH_TEST=0.5` lower values give the same result. Picking a threshold on the test set amounts to tuning on the test set, so report 0.5. `--flip-tta` (off by default) averages the normalised embeddings of the image and its mirror, renormalises, then applies CWS.

The official released PRW checkpoint loads into this model; the only missing key is the OIM queue pointer, which inference does not use. This gives a cheap way to check the pipeline before spending GPU hours: `python Scripts/eval_prw.py --ckpt <official epoch_*.pth>`.

## Results (PRW test, `eval_search_prw`, det_thresh 0.5, no CBGM)

| Model | mAP | top-1 |
| --- | --- | --- |
| SeqNet, paper (NAE+SeqNet, w/o CBGM) | 46.7 | 83.4 |
| This reimplementation, fp32 | TBD | TBD |
| This reimplementation, bf16 | TBD | TBD |
| + flip TTA | TBD | TBD |

CBGM (paper: 47.6 / 87.6) is not used because it lives inside the evaluation function.

## Differences from the official code

- **Mixed precision:** fp16/bf16 autocast is optional; the config default is fp32 as in the official code. Under autocast, OIM runs in fp32, and the RPN and box-head outputs are cast to fp32 before box decoding; these casts do nothing in fp32.
- **OIM:** same LUT/CQ update rule, done in `backward`. The CQ pointer is now saved in checkpoints. A batch with no labelled person returns 0 instead of NaN. The official PRW pid quirk is kept: pids 483 and 932 exceed LUT size 482 and end up as CQ targets.
- **Inference with test batch size > 1:** FCS and the empty-detection check handle it correctly (identical for batch 1). The CBGM-only query-as-gallery path is removed.
- **Engineering:** no evaluation during training by default (official: every epoch), atomic checkpoints (last 2 kept), RNG state restored on resume. torchvision `weights=` API: IMAGENET1K_V1 is the same weights as the old `pretrained=True`.
- **Batch norm:** the official code fine-tunes plain BatchNorm (only `conv1` and `bn1` are frozen). `MODEL.BACKBONE.FROZEN_BN` exists but is off.
- **Library versions:** the official code targeted much older PyTorch. Its authors warn that results depend on the PyTorch version.

A parity test built the official `SeqNet` from the original sources with the same torchvision. It found identical parameter names, shapes and counts (48,418,735), identical frozen parameters, buffers and random initialisation, and identical training losses, gradients, LUT/CQ updates, detections and embeddings on the same input.
