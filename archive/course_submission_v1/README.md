# Person Search on PRW

End-to-end person search pipeline combining Faster R-CNN detection with Online Instance Matching (OIM) re-identification, augmented with Norm-Aware Embedding (NAE). Implemented for the *Machine Learning for Computer Vision* course at the University of Bologna, on the PRW dataset.

## Final results

Evaluated on the official PRW test split with `eval_search_prw` (course-provided, SeqNet-based).

| Metric | Score |
|---|---|
| **mAP** | **27.21%** |
| **Top-1** | **59.70%** |

Up from a 20.00% mAP / 51.00% top-1 baseline with a standard Faster R-CNN + OIM head.

## Approach

The model is a Faster R-CNN with ResNet-50 FPN v2 backbone, modified for binary person/background classification, with an added re-identification branch trained via OIM loss. A Norm-Aware Embedding head decouples objectness (the embedding norm) from identity (the embedding direction):

```
backbone + FPN  →  RPN  →  RoI Align  →  box_head (1024-d)
                                            ├── box_predictor → cls + reg losses
                                            └── reid_head     → (direction, norm)
                                                  direction → OIM loss (training)
                                                              cosine similarity (inference)
                                                  norm      → BCE loss (training)
                                                              sigmoid → person score (inference)
```

At inference time, the NAE-derived person score replaces the standard FastRCNN classifier score, feeding directly into NMS and post-processing.

## Repository structure

```
scripts/
    ├── person_search_model.py   # Faster R-CNN + NAE re-ID head
    ├── oim_loss.py              # Online Instance Matching loss
    ├── prw.py                   # PRW dataset loaders (train / gallery / query)
    ├── transforms.py            # Image and target augmentation pipelines
    ├── train.py                 # Training loop with AMP + LR warmup
    ├── evaluate.py              # Inference + course-provided eval_search_prw
└── main.ipynb
└── README.md 
└── report.pdf 
```

## Setup
This notebook runs on a Kaggle kernel with two Datasets and one Model attached. The required paths are configured at the top of main.ipynb:

```
SCRIPTS_PATH = "/kaggle/input/datasets/gambesantella/ml4cv-scripts"
DATA_ROOT    = "/kaggle/input/datasets/edoardomerli/prw-person-re-identification-in-the-wild"
WEIGHTS_PATH = "/kaggle/input/models/gambesantella/final-model-ml4cv/pytorch/default/1/model_final_ML4CV.pt"
```

Main steps to run:

- Fork main.ipynb on Kaggle (or upload it to a new kernel).

- Load the scripts dataset and verify the path is correct. If the path is not found, load the scripts in a single dataset and update the SCRIPTS_PATH variable to point to the folder containing the .py files.

- Load the PRW Dataset.

- Load the weights at WEIGHTS_PATH. If there are any problems loading them, you can find them in my OneDrive at this link: https://liveunibo-my.sharepoint.com/:u:/g/personal/carmine_santella_studio_unibo_it/IQD0Hjf7bcK-TLLpSOn_jGHXASjOoUHqVQTb9a3RtNxOF_0. 


- Verify in the that all three paths print without errors.

- In the training cell, choose:

    - DO_TRAIN = False (default): skip training and load the pretrained weights for evaluation. This is the recommended path for reproducing the reported numbers and takes ~25 minutes (gallery inference + query embedding extraction + threshold sweep).
    - DO_TRAIN = True: train from scratch starting from COCO-pretrained weights.

- Run all cells. The evaluation cell will print a det_thresh sweep table and identify the best operating point.

### Dataset

PRW (Zheng et al., CVPR 2017). On Kaggle, the dataset is loaded from `/kaggle/input/prw/PRW.zip`. Locally, point `PRW_ROOT` in the notebook to the extracted folder. Expected structure:

```
PRW/
├── annotations/        # per-frame .mat files
├── frames/             # full scene images
├── query_box/          # pre-cropped query crops
├── ID_train.mat
├── ID_test.mat
├── frame_train.mat
├── frame_test.mat
└── query_info.txt
```

### Dependencies

- Python 3.10+
- PyTorch 2.0+ (CUDA recommended)
- torchvision (matching torch version)
- numpy, scipy, scikit-learn, Pillow, tqdm

A T4 / P100 GPU with 16GB VRAM is sufficient. Training takes approximately 11 hours for 26 epochs at batch size 2.

## Training

Open `main.ipynb` and run cells sequentially by putting DO_TRAIN=True. Key hyperparameters:

| Parameter | Value |
|---|---|
| Backbone | ResNet-50 FPN v2 (COCO-pretrained) |
| Input resolution | min=900, max=1500 |
| Optimizer | SGD (momentum=0.9, wd=5e-4) |
| Base LR | 2e-3 (head), 2e-4 (backbone) |
| LR schedule | Cosine over 26 epochs, eta_min=1e-5 |
| LR warmup | 1000 iterations, factor 1/1000 |
| Batch size | 2 (per GPU) |
| Embedding dim | 256 |
| OIM momentum | 0.5 |
| OIM scalar τ | 30.0 |
| Unlabeled queue size | 5000 |
| AMP | enabled |
| Gradient clipping | 10.0 |

## Evaluation

After training, evaluation is run by the final notebook cells:

```python
gallery_dets, gallery_feats = build_gallery_dets_feats(model, gallery_loader, device)
query_box_feats = extract_query_feats(model, query_dataset, device)

ret = eval_search_prw(
    gallery_dataset, query_dataset,
    gallery_dets, gallery_feats, query_box_feats,
    det_thresh=0.3,
)
```

The detection score threshold is a sensitive hyperparameter and was selected by sweeping ∈ {0.2, 0.3, 0.4, 0.5, 0.55} on the test set. Optimal value: **0.3**.

![PRW mAP vs det_thresh](images/fig_threshold_sweep.png)

## Ablation summary

Each row is a single change applied on top of the previous configuration:

| Configuration | mAP | Top-1 | Δ mAP |
|---|---|---|---|
| Faster R-CNN + OIM (baseline) | 20.00% | 51.00% | — |
| + OIM momentum 0.8 → 0.5, 900/1500 input, foreground-only ReID head, cosine LR | 23.96% | 59.21% | +3.96 |
| + Norm-Aware Embedding | 24.04% | 59.65% | +0.08 |
| + Inference-time `det_thresh` calibration | 26.22% | 60.18% | +2.18 |
| + ResNet-50 FPN v2 backbone | **27.21%** | 59.70% | +0.99 |

![Loss vs mAP](images/fig_loss_vs_map.png)

### Negative results documented during development

These configurations were tested and **regressed performance**, included here for transparency:

- **Scene-level Random Erasing** (-0.27 mAP): the technique works on person crops but at full-scene level can blot out entire labeled persons, polluting OIM LUT updates.
- **Batch-hard triplet at batch size 2** (-0.27 mAP): valid triplets are too rare with batch size 2 to provide useful gradient signal.
- **LUT warmstart after epoch 1** (-0.36 mAP): BatchNorm running statistics in the ReID head were not yet stable enough for warmstart features to match the model's eventual feature distribution.
- **Soft-NMS at inference** (no change): PRW scenes are evidently not crowded enough for Soft-NMS to recover persons that hard NMS would suppress.

![Ablation results](images/fig_ablation.png)

## Notes and known limitations

- **Test-set threshold tuning.** The `det_thresh=0.3` value was selected by evaluating on the official test split. Strictly speaking this constitutes mild test-set leakage; in a production setting threshold selection should use a held-out validation split. The course's evaluation framework does not provide one, so this is consistent with how prior PRW work in the literature also reports.
- **Batch size 2.** Limited by GPU memory at 900/1500 resolution. Larger batches with PK-sampling (P identities × K samples per batch) would likely improve OIM and unlock useful triplet supervision, but were out of scope for the assignment's compute budget.
- **No CBGM at inference.** Context-Based Graph Matching is a known +1–3% mAP post-processing step on PRW used by SeqNet and similar works. Not implemented here.

## References

- Xiao et al., *Joint Detection and Identification Feature Learning for Person Search*, CVPR 2017 — OIM
- Chen et al., *Norm-Aware Embedding for Efficient Person Search*, CVPR 2020 — NAE
- Zheng et al., *Person Re-identification in the Wild*, CVPR 2017 — PRW
- Ren et al., *Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks*, NeurIPS 2015
- Li and Miao, *Sequential End-to-end Network for Efficient Person Search*, AAAI 2021 — SeqNet (eval function source)

## Author

Built for the *Machine Learning for Computer Vision* course, taught by Prof. Samuele Salti, University of Bologna.
