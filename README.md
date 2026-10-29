# SeqNet person search on PRW (reimplementation)

This is a clean reimplementation of **SeqNet** (Z. Li and D. Miao, *Sequential End-to-end Network for Efficient Person Search*, AAAI 2021) for the PRW dataset. It was written for the *Machine Learning for Computer Vision* course (University of Bologna).

The model code is adapted closely from the official repository [serend1p1ty/SeqNet](https://github.com/serend1p1ty/SeqNet) (MIT License, Copyright (c) 2022 Zj Li; see `LICENSE-SeqNet`). Every adapted file starts with an attribution header that lists its changes. All credit for the method goes to the original authors. Search metrics are computed with the course-provided `eval_search_prw` (`Scripts/evaluate.py`), which is kept unchanged.

The optional Swin backbone follows [SOLIDER-PersonSearch](https://github.com/tinyvision/SOLIDER-PersonSearch) (W. Chen et al., *Beyond Appearance: a Semantic Controllable Self-Supervised Learning Framework for Human-Centric Visual Tasks*, CVPR 2023). Its Swin Transformer code is copied from [SOLIDER](https://github.com/tinyvision/SOLIDER) (Apache License 2.0, see `LICENSE-SOLIDER`), and it uses SOLIDER's self-supervised weights, pre-trained on LUPerson (4.18M unlabelled person images).

> `main.ipynb` presents this reimplementation: method, training, quantitative and qualitative results. `report.md` and `weights/` belong to the earlier course submission, a single-stage Faster R-CNN FPN with OIM and NAE, and do not use this code. That submission, including its original notebook, is preserved in `archive/course_submission_v1/`.

## Layout

```text
Scripts/
  seqnet.py      SeqNet: RPN, Faster R-CNN head, baseline (re-ID) head, NAE, FCS, CWS
  backbone.py    build_backbone(cfg): ResNet-50 conv1-layer3 stem + res5 head, or SOLIDER Swin
  swin.py        SOLIDER Swin split for SeqNet: stages 1-3 stem, stage 4 RoI head; weight loader
  swin_transformer.py  SOLIDER's Swin Transformer with the semantic controller (Apache-2.0)
  oim.py         OIM loss (LUT 482 / CQ 500, updated in backward like the official code)
  prw.py         PRW dataset and loaders (official box and pid conventions)
  transforms.py  ToTensor + horizontal flip (the official augmentation)
  config.py      defaults = official SeqNet PRW recipe; YAML and CLI overrides
  engine.py      training epoch, gallery/query feature extraction (CWS, optional flip TTA)
  evaluate.py    course eval_search_prw (unchanged)
  train.py       training entry point
  eval_prw.py    evaluation entry point (det_thresh sweep, JSON output)
  qualitative.py figures from cached features: retrievals vs annotated people, error causes
configs/prw.yaml         the full recipe, identical to config.py defaults
configs/prw_solider_swin_s.yaml   SOLIDER Swin-Small backbone (SOLIDER-PersonSearch recipe)
configs/prw_solider_swin_s_head10x.yaml   the same, with a 10x lr for the non-pre-trained layers
tools/download_prw.sh    Kaggle download
tools/download_solider.sh         SOLIDER pre-trained Swin weights (Google Drive, via gdown)
slurm/train_prw.sbatch   1x L40 template: train, auto-resume/requeue, then evaluate
```

## Setup and data

```bash
pip install -r requirements.txt            # torch 2.7.1 / torchvision 0.22.1 (CUDA 12.x, Ada OK)
bash tools/download_prw.sh data/PRW        # Kaggle CLI + ~/.kaggle/access_token; prints PRW_ROOT
export PRW_ROOT=/path/printed/above        # or pass --data-root
```

The loader expects `frames/`, `annotations/*.jpg.mat`, `frame_train.mat`, `frame_test.mat` and `query_info.txt`. It also finds them up to three levels below the folder you pass, as in Kaggle's `/kaggle/input/<dataset>/...` layout.

## Train

**UniBo DISI GPU cluster** (partition `l40`: 1x L40 48 GB, 8 CPU cores, 64 GB RAM; driver 590 / CUDA 13.1; SLURM 24.11). Jobs are submitted from `giano.cs.unibo.it`. The home quota is 400 MB, so the code, venv, dataset and caches all live in `/scratch.hpc/$USER`. Files there are deleted after 40 days without access, so copy your results elsewhere.

```bash
# from your PC: copy the code (not weights/ or archive/)
ssh <name.surname>@giano.cs.unibo.it "mkdir -p /scratch.hpc/<name.surname>/prw-seqnet"
scp -r Scripts configs slurm tools requirements.txt README.md <name.surname>@giano.cs.unibo.it:/scratch.hpc/<name.surname>/prw-seqnet/

# on giano, once: venv (torch cu128), cached ImageNet weights, PRW from Kaggle (needs ~/.kaggle/access_token)
cd /scratch.hpc/$USER/prw-seqnet && bash slurm/setup_giano.sh

mkdir -p logs
SMOKE=1 sbatch --time=01:00:00 slurm/train_prw.sbatch   # 20-iteration check + full eval (~pipeline test)
sbatch slurm/train_prw.sbatch                           # official recipe, fp32
AMP=bf16 sbatch slurm/train_prw.sbatch                  # faster autocast variant
squeue -u $USER ; tail -f logs/prw_<jobid>.out

# SOLIDER Swin-Small backbone (setup_giano.sh also downloads its weights to /scratch.hpc/$USER/weights)
CFG=configs/prw_solider_swin_s.yaml SMOKE=1 sbatch --time=01:00:00 slurm/train_prw.sbatch
CFG=configs/prw_solider_swin_s.yaml sbatch slurm/train_prw.sbatch
CFG=configs/prw_solider_swin_s_head10x.yaml sbatch slurm/train_prw.sbatch
```

Each config writes to its own folder, `exp_<config name>` (for example `exp_prw` or `exp_prw_solider_swin_s`). The job saves `epoch_N.pth` there after every epoch and always restarts with `--resume auto`. Five minutes before the time limit it requeues itself; if requeueing is not permitted, run `sbatch` again and it resumes. After training it evaluates the last epoch into `eval_epoch_N.json` in the same folder. Check the partition's time limit with `sinfo -p l40 -o "%P %l"` and keep `--time` within it.

Running directly:

```bash
python Scripts/train.py --cfg configs/prw.yaml --data-root $PRW_ROOT --output-dir exp_prw \
    [--amp off|fp16|bf16] [--resume auto] [--eval-period 0] [--seed 1] [--opts KEY VALUE ...]
```

The defaults are the official recipe: batch 5, SGD lr 0.003 (momentum 0.9, wd 5e-4), linear warm-up over epoch 0, 18 epochs, lr x0.1 at epoch 16, 900x1500 input, gradient clipping at 10. Logs go to `train_log.csv`, `epoch_log.csv` and `tf_log/` (TensorBoard).

**Batch size and LR:** follow the linear scaling rule, `lr = 0.003 * batch / 5`, for example `--batch-size 2 --lr 0.0012`. The official README says batch 5 needs about 28 GB in fp32, which fits on a 48 GB L40.

**Head learning rate:** `SOLVER.HEAD_LR_MULT` (default 1, the official single lr) multiplies the lr of the layers that start from random initialisation: the RPN head, both box predictors and the NAE embedding. The pre-trained backbone and res5 / stage-4 heads keep `BASE_LR`, which is the lr in the logs. `prw_solider_swin_s_head10x.yaml` uses 10, so the heads train at SeqNet's 0.003 and the Swin at SOLIDER's 3e-4.

**Kaggle notebook (T4/P100, 16 GB, 12 h sessions):** run with `--data-root /kaggle/input/<PRW dataset folder> --output-dir /kaggle/working/exp_prw --amp fp16 --batch-size 2 --lr 0.0012`. To continue in a later session, attach the previous output and run with `--resume <path to epoch_N.pth>`.

## Evaluate

```bash
python Scripts/eval_prw.py --ckpt exp_prw/epoch_17.pth --data-root $PRW_ROOT \
    [--det-thresh 0.5 0.6 0.7 0.8] [--flip-tta] [--cache exp_prw/features.pt]
```

The script extracts gallery detections and features and query features (GT query box). It applies **CWS** (gallery embedding multiplied by the detection score, as in official SeqNet), calls the unchanged `eval_search_prw` for every `det_thresh`, and writes a JSON file. That file also contains the detection recall and AP. `--cache` stores the features so a sweep can be re-run on CPU.

Note on thresholds: `0.5` is SeqNet's default, and with `SCORE_THRESH_TEST=0.5` lower values give the same result. Picking a threshold on the test set amounts to tuning on the test set, so report 0.5. `--flip-tta` (off by default) averages the normalised embeddings of the image and its mirror, renormalises, then applies CWS.

The official released PRW checkpoint loads into this model; the only missing key is the OIM queue pointer, which inference does not use. This gives a cheap way to check the pipeline before spending GPU hours: `python Scripts/eval_prw.py --ckpt <official epoch_*.pth>`.

## Qualitative results

```bash
python Scripts/qualitative.py --data-root $PRW_ROOT --out results/qualitative \
    --run "SOLIDER Swin-S" exp_prw_solider_swin_s/features_epoch_17.pt \
    --run "ResNet-50" exp_prw/features_epoch_17.pt      # optional second model

# on giano: CPU only, so the (usually free) rtx2080 partition is enough
sbatch -p rtx2080 --gres=gpu:1 --cpus-per-task=4 --mem=16G --time=00:45:00 -J prw_quali -o logs/%x_%j.out --wrap \
  "/scratch.hpc/$USER/venv-prw/bin/python3 Scripts/qualitative.py --data-root /scratch.hpc/$USER/data/PRW --out results/qualitative --run 'SOLIDER Swin-S' exp_prw_solider_swin_s/features_epoch_17.pt --run 'ResNet-50' exp_prw/features_epoch_17.pt"
```

The script needs no model and no GPU. It reads the feature caches that `eval_prw.py --cache` writes (`train_prw.sbatch` saves `features_epoch_N.pt` next to the checkpoint). It ranks the gallery with the unchanged `eval_search_prw`, so the figures show the retrievals behind the reported numbers, and it prints the same mAP. Each retrieved box is then compared with the people annotated in its frame: the query person (the eval's `correct` flag), another labelled person, an unlabelled person, the query person with a box too loose to count, or no annotated person. The last case is either a false detection or a person PRW left unannotated in that frame, sometimes the query person themselves.

- `<model>/search_random.jpg`: random queries with their top-5 results. Green and red boxes mark right and wrong results; a dashed yellow box shows where the query person really is when they were missed in that frame.
- `<model>/failures_<cause>.jpg`: queries whose top-1 result is wrong, one figure per cause, with the whole top-1 frame.
- `<model>/detections_*.jpg`: detections compared with every annotated person (matched, no annotated person, missed).
- `compare_fixed.jpg`, `compare_broken.jpg` (two models): queries that the first model gets right and the second wrong, and the reverse.
- `top1_errors.png` (causes of top-1 errors), `recall_by_height.png` (detection recall by person height), `summary.json` (every count, plus top-5/top-10 from the eval's top-10 lists), `<model>/queries.csv` (one row per query).

Examples are drawn at random with a fixed `--seed` within each group, and every figure title gives the size of its group, so they are representative rather than hand-picked.

## Notebook

`main.ipynb` reads the files in the repository (`exp_*/`, `results/qualitative/`) and runs without a GPU; only Section 1 needs the PRW frames. Its flags `DO_TRAIN`, `RUN_EVAL` and `RUN_QUALI` (off by default) repeat training, evaluation and the qualitative figures. To execute it with outputs on giano, on the CPU:

```bash
/scratch.hpc/$USER/venv-prw/bin/pip install --no-cache-dir nbconvert ipykernel   # once
sbatch -p rtx2080 --gres=gpu:1 --cpus-per-task=4 --mem=16G --time=00:30:00 -J prw_nb -o logs/%x_%j.out --wrap \
  "export PATH=/scratch.hpc/$USER/venv-prw/bin:\$PATH; jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 main.ipynb"
```

## Results (PRW test, `eval_search_prw`, det_thresh 0.5, no CBGM)

| Model | mAP | top-1 |
| --- | --- | --- |
| SeqNet, paper (NAE+SeqNet, w/o CBGM) | 46.7 | 83.4 |
| This reimplementation, fp32 (1x L40, 3h20m) | 46.40 | 83.08 |
| SOLIDER Swin-S + SeqNet, paper (SOLIDER-PersonSearch README) | 59.84 | 86.73 |
| This reimplementation, SOLIDER Swin-S, fp32 (1x L40, 6h00m) | 60.19 | 87.36 |
| + flip TTA (`eval_prw.py --flip-tta`, same checkpoint) | **63.06** | **88.48** |
| SOLIDER Swin-S, head lr x10 (`prw_solider_swin_s_head10x.yaml`, 6h00m) | 58.79 | 85.85 |

Flip TTA changes only how the features are extracted (the course `eval_search_prw` is untouched), but the published numbers above are without test-time augmentation, so compare them with 60.19. The 10x head lr improves detection (recall 95.99 vs 95.66, AP 92.26 vs 91.49) but lowers search accuracy; each configuration is a single run.

CBGM (paper: 47.6 / 87.6) is not used because it lives inside the evaluation function. SOLIDER-PersonSearch uses SeqNet's evaluation code with CBGM off by default.

## Differences from the official code

- **Mixed precision:** fp16/bf16 autocast is optional; the config default is fp32 as in the official code. Under autocast, OIM runs in fp32, and the RPN and box-head outputs are cast to fp32 before box decoding; these casts do nothing in fp32.
- **OIM:** same LUT/CQ update rule, done in `backward`. The CQ pointer is now saved in checkpoints. A batch with no labelled person returns 0 instead of NaN. The official PRW pid quirk is kept: pids 483 and 932 exceed LUT size 482 and end up as CQ targets.
- **Inference with test batch size > 1:** FCS and the empty-detection check handle it correctly (identical for batch 1). The CBGM-only query-as-gallery path is removed.
- **Engineering:** no evaluation during training by default (official: every epoch), atomic checkpoints (last 2 kept), RNG state restored on resume. torchvision `weights=` API: IMAGENET1K_V1 is the same weights as the old `pretrained=True`.
- **Batch norm:** the official code fine-tunes plain BatchNorm (only `conv1` and `bn1` are frozen). `MODEL.BACKBONE.FROZEN_BN` exists but is off.
- **Library versions:** the official code targeted much older PyTorch. Its authors warn that results depend on the PyTorch version.

**SOLIDER Swin backbone** (`MODEL.BACKBONE.NAME swin_tiny|swin_small|swin_base`): as in SOLIDER-PersonSearch, Swin stages 1-3 replace ResNet conv1-layer3, and stage 3's patch merging plus stage 4 replace res5 in both heads. RoI features are average-pooled, and the semantic weight is 0.6. The recipe is SeqNet's with batch 3 and lr 3e-4 for Swin-Small (SOLIDER-PersonSearch `run.sh`). Differences: the pre-trained checkpoint is loaded into the Swin before the re-ID head is copied, which gives the same weights as loading it afterwards, and the semantic-weight vector follows the input's device instead of `.cuda()`. A CPU self-test checks four things:

- the stem's output equals the stage-3 output of SOLIDER's `SwinTransformer.forward`;
- SOLIDER checkpoints, raw or converted, load into the stem and the re-ID head, and a checkpoint of the wrong size is rejected;
- a training step gives finite losses, with gradients in the stem and in both stage-4 heads;
- the gallery embeddings keep CWS (norm equal to the detection score).

A parity test built the official `SeqNet` from the original sources with the same torchvision. It found identical parameter names, shapes and counts (48,418,735), identical frozen parameters, buffers and random initialisation, and identical training losses, gradients, LUT/CQ updates, detections and embeddings on the same input.
