"""
SeqNet Person Search Model
---------------------------
"Sequential End-to-end Network for Efficient Person Search" (Li & Miao, AAAI 2021).

Architecture changes vs the previous version:
  - NAEHead (Norm-Aware Embedding) replaces the plain L2-norm ReIDHead.
    direction (unit vector) → OIM re-ID loss (Lreid).
    norm_score (sigmoid of BN'd L2-norm) → BCE detection loss (Lcls2).
  - FCS (First Classification Score): stage-1 softmax scores are forwarded to
    the final detection output so high-IoU stage-2 boxes are not mis-classified.
  - Loss weight λ1 = 10 applied to Lreg1, all others = 1 (paper Eq. 4).
  - Stage-2 IoU threshold changed to 0.5 / 0.5 (paper: "trained with 0.5 IoU").
  - ResNet-50 FPN v2 backbone supported alongside v1.
  - OIM queue size default = 500 (PRW convention; use 5000 for CUHK-SYSU).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    fasterrcnn_resnet50_fpn_v2,
    FasterRCNN_ResNet50_FPN_Weights,
    FasterRCNN_ResNet50_FPN_V2_Weights,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, TwoMLPHead
from torchvision.models.detection.roi_heads import RoIHeads, fastrcnn_loss
from torchvision.ops import boxes as box_ops

from oim_loss import OIMLoss


# --------------------------------------------------------------------------- #
#  NAE head — Norm-Aware Embedding
# --------------------------------------------------------------------------- #

class NAEHead(nn.Module):
    """
    Norm-Aware Embedding (NAE) head.

    FC(1024→512) → BN → ReLU → FC(512→D) → split into:
      direction  : feat / ||feat||  — L2-normalised re-ID embedding
      norm_score : sigmoid(BN(||feat||)) — person objectness in [0, 1]

    At training, norm_score drives the BCE classification loss Lcls2.
    At inference, FCS (stage-1 score) replaces norm_score for final detection
    so that high-IoU boxes are not mis-classified by the 0.5-IoU-trained head.
    """

    def __init__(self, in_channels: int = 1024, embed_dim: int = 256):
        super().__init__()
        self.fc1     = nn.Linear(in_channels, 512)
        self.bn1     = nn.BatchNorm1d(512)
        self.fc2     = nn.Linear(512, embed_dim)
        self.bn_norm = nn.BatchNorm1d(1)

    def forward(self, x: torch.Tensor):
        x    = F.relu(self.bn1(self.fc1(x)), inplace=True)
        feat = self.fc2(x)
        norm = feat.norm(p=2, dim=1, keepdim=True).clamp(min=1e-8)  # (N, 1)
        direction  = feat / norm                      # unit vector (N, D)
        norm_logit = self.bn_norm(norm)              # raw logit (N, 1) — no sigmoid here
        return direction, norm_logit                 # use BCEWithLogits for AMP safety


# --------------------------------------------------------------------------- #
#  SeqNet RoI Heads — two-stage cascade
# --------------------------------------------------------------------------- #

class SeqNetRoIHeads(RoIHeads):
    """
    Stage 1 (inherited box_head / box_predictor):
        Standard Faster R-CNN head on RPN proposals.
        Losses: Lreg1 * 10 (λ1=10) + Lcls1.

    Stage 2 (box_head_2 / box_predictor_2 / nae_head):
        Operates on stage-1 refined boxes.
        box_predictor_2 → Lreg2 (box part only; its cls logits are discarded).
        nae_head        → Lcls2 (BCE on norm_score) + Lreid (OIM on direction).

    Inference — First Classification Score (FCS):
        Stage-1 softmax person scores propagate to the final output because
        stage 2 is trained at 0.5 IoU but at test time receives boxes with
        IoU > 0.8, making its classifier unreliable.
    """

    def __init__(
        self,
        box_head_2: nn.Module,
        box_predictor_2: nn.Module,
        nae_head: NAEHead,
        oim_loss: OIMLoss,
        nae_weight: float = 1.0,
        *args, **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.box_head_2      = box_head_2
        self.box_predictor_2 = box_predictor_2
        self.nae_head        = nae_head
        self.oim_loss        = oim_loss
        self.nae_weight      = nae_weight

    # ------------------------------------------------------------------ #
    #  Stage-2 sampling — 0.5 / 0.5 IoU (paper §Method, Training)
    # ------------------------------------------------------------------ #
    def _sample_stage2(
        self, refined_boxes, targets,
        fg_iou: float = 0.5, bg_iou: float = 0.5,
        samples_per_image: int = 128, positive_fraction: float = 0.5,
    ):
        sampled_proposals  = []
        matched_idxs_list  = []
        labels_list        = []
        regression_targets = []

        for rb, t in zip(refined_boxes, targets):
            gt_boxes = t["boxes"]
            n_gt     = gt_boxes.shape[0]
            device   = rb.device

            if n_gt == 0:
                sampled_proposals.append(torch.zeros((0, 4), device=device))
                matched_idxs_list.append(torch.zeros((0,), dtype=torch.int64, device=device))
                labels_list.append(torch.zeros((0,), dtype=torch.int64, device=device))
                regression_targets.append(torch.zeros((0, 4), device=device))
                continue

            all_props  = torch.cat([gt_boxes, rb], dim=0)
            iou_matrix = box_ops.box_iou(gt_boxes, all_props)
            max_iou, max_gt_idx = iou_matrix.max(dim=0)

            is_pos = max_iou >= fg_iou
            is_neg = max_iou < bg_iou
            is_pos[:n_gt] = True    # GT boxes are always positive
            is_neg[:n_gt] = False

            pos_indices = torch.where(is_pos)[0]
            neg_indices = torch.where(is_neg)[0]

            n_pos_target = int(samples_per_image * positive_fraction)
            n_pos = min(pos_indices.numel(), n_pos_target)
            n_neg = min(neg_indices.numel(), samples_per_image - n_pos)

            if pos_indices.numel() > n_pos:
                perm = torch.randperm(pos_indices.numel(), device=device)[:n_pos]
                pos_indices = pos_indices[perm]
            if neg_indices.numel() > n_neg:
                perm = torch.randperm(neg_indices.numel(), device=device)[:n_neg]
                neg_indices = neg_indices[perm]

            sampled        = torch.cat([pos_indices, neg_indices])
            sampled_boxes  = all_props[sampled]
            sampled_gt_idx = max_gt_idx[sampled]
            sampled_labels = torch.cat([
                torch.ones(n_pos,  dtype=torch.int64, device=device),
                torch.zeros(n_neg, dtype=torch.int64, device=device),
            ])

            sampled_proposals.append(sampled_boxes)
            matched_idxs_list.append(sampled_gt_idx)
            labels_list.append(sampled_labels)
            # clamp prevents inf from degenerate boxes reaching smooth_l1_loss
            regression_targets.append(
                self.box_coder.encode_single(
                    gt_boxes[sampled_gt_idx], sampled_boxes
                ).clamp(-10.0, 10.0)
            )

        return sampled_proposals, matched_idxs_list, labels_list, regression_targets

    # ------------------------------------------------------------------ #
    #  Re-ID PID labels for OIM
    # ------------------------------------------------------------------ #
    @staticmethod
    def _gather_pids(targets, matched_idxs, labels):
        """
        pid >= 0  : labeled identity  → LUT update + OIM loss
        pid == -1 : unlabeled         → CQ push only
        pid == -2 : background        → ignored
        """
        pid_list = []
        for t, midx, lbl in zip(targets, matched_idxs, labels):
            gt_pids      = t["pids"]
            matched_pids = gt_pids[midx.clamp(min=0)]
            matched_pids = torch.where(
                lbl > 0, matched_pids, torch.full_like(matched_pids, -2),
            )
            pid_list.append(matched_pids)
        return torch.cat(pid_list)

    # ------------------------------------------------------------------ #
    #  Forward
    # ------------------------------------------------------------------ #
    def forward(self, features, proposals, image_shapes, targets=None):

        # ====== STAGE 1 ======
        if self.training:
            proposals_s1, matched_idxs_s1, labels_s1, reg_targets_s1 = \
                self.select_training_samples(proposals, targets)
        else:
            proposals_s1                                    = proposals
            matched_idxs_s1 = labels_s1 = reg_targets_s1  = None

        box_feats_s1 = self.box_head(
            self.box_roi_pool(features, proposals_s1, image_shapes)
        )
        class_logits_s1, box_reg_s1 = self.box_predictor(box_feats_s1)

        # Decode stage-1 regressions → refined boxes (person class = index 1)
        props_flat   = torch.cat(proposals_s1, dim=0)
        decoded_s1   = self.box_coder.decode(box_reg_s1, [props_flat]).squeeze(1)
        refined_flat = decoded_s1[:, 1, :]
        bpi_s1       = [p.shape[0] for p in proposals_s1]
        refined_boxes = []
        for rb, sh in zip(refined_flat.split(bpi_s1, 0), image_shapes):
            clipped = box_ops.clip_boxes_to_image(rb, sh)
            keep    = box_ops.remove_small_boxes(clipped, min_size=1.0)
            refined_boxes.append(clipped[keep])    # remove_small_boxes returns indices

        # ====== STAGE 2 ======
        if self.training:
            proposals_s2, matched_idxs_s2, labels_s2, reg_targets_s2 = \
                self._sample_stage2(refined_boxes, targets)
            s1_scores_s2 = None
        else:
            proposals_s2, s1_scores_s2 = self._postprocess_stage1_for_inference(
                class_logits_s1, refined_boxes, proposals_s1, image_shapes,
            )

        # Safety: all images have zero proposals after stage-1 filtering
        embed_dim = self.nae_head.fc2.out_features
        if all(p.shape[0] == 0 for p in proposals_s2):
            if self.training:
                zero = box_feats_s1.sum() * 0.0
                loss_cls_s1, loss_box_s1 = fastrcnn_loss(
                    class_logits_s1, box_reg_s1, labels_s1, reg_targets_s1,
                )
                return [], {
                    "loss_classifier_s1": loss_cls_s1,
                    "loss_box_reg_s1":    loss_box_s1 * 10.0,
                    "loss_classifier_s2": zero,
                    "loss_box_reg_s2":    zero,
                    "loss_reid":          zero,
                }
            return [
                {"boxes":      torch.zeros(0, 4, device=features["0"].device),
                 "scores":     torch.zeros(0,    device=features["0"].device),
                 "labels":     torch.zeros(0, dtype=torch.int64, device=features["0"].device),
                 "embeddings": torch.zeros(0, embed_dim, device=features["0"].device)}
                for _ in image_shapes
            ], {}

        # Stage-2 forward pass
        box_feats_s2 = self.box_head_2(
            self.box_roi_pool(features, proposals_s2, image_shapes)
        )
        cls_logits_s2, box_reg_s2  = self.box_predictor_2(box_feats_s2)
        direction_s2, norm_score_s2 = self.nae_head(box_feats_s2)

        # ====== TRAINING LOSSES ======
        if self.training:
            # Stage-1 losses
            loss_cls_s1, loss_box_s1 = fastrcnn_loss(
                class_logits_s1, box_reg_s1, labels_s1, reg_targets_s1,
            )
            # Stage-2 box regression (cls loss from box_predictor_2 is discarded
            # and replaced by the NAE BCE loss below)
            _, loss_box_s2 = fastrcnn_loss(
                cls_logits_s2, box_reg_s2, labels_s2, reg_targets_s2,
            )
            # NAE classification loss: BCEWithLogits on raw norm_logit vs fg/bg labels
            # (AMP-safe; equivalent to sigmoid + BCE but numerically stable in float16)
            all_lbl     = torch.cat(labels_s2).float()
            loss_cls_s2 = (
                F.binary_cross_entropy_with_logits(norm_score_s2.squeeze(1), all_lbl)
                * self.nae_weight
            )
            # OIM re-ID loss on direction embeddings of foreground proposals
            reid_labels = self._gather_pids(targets, matched_idxs_s2, labels_s2)
            fg_mask     = reid_labels != -2
            if fg_mask.any():
                loss_reid = self.oim_loss(direction_s2[fg_mask], reid_labels[fg_mask])
            else:
                loss_reid = direction_s2.sum() * 0.0

            return [], {
                "loss_classifier_s1": loss_cls_s1,
                "loss_box_reg_s1":    loss_box_s1 * 10.0,   # λ1 = 10 (paper Eq. 4)
                "loss_classifier_s2": loss_cls_s2,
                "loss_box_reg_s2":    loss_box_s2,
                "loss_reid":          loss_reid,
            }

        # ====== INFERENCE — First Classification Score (FCS) ======
        # Decode stage-2 box regression for final box positions
        props_s2_flat = torch.cat(proposals_s2, dim=0)
        s1_flat       = torch.cat(s1_scores_s2, dim=0)
        decoded_s2    = self.box_coder.decode(box_reg_s2, [props_s2_flat]).squeeze(1)
        final_flat    = decoded_s2[:, 1, :]   # person-class boxes

        bpi_s2          = [p.shape[0] for p in proposals_s2]
        boxes_split     = list(final_flat.split(bpi_s2, 0))
        s1_scores_split = list(s1_flat.split(bpi_s2, 0))
        dir_split       = list(direction_s2.split(bpi_s2, 0))

        result = []
        for fb, s1s, dirv, img_shape in zip(
            boxes_split, s1_scores_split, dir_split, image_shapes
        ):
            fb  = box_ops.clip_boxes_to_image(fb, img_shape)

            # Score filter using FCS (stage-1 scores)
            keep = s1s > self.score_thresh
            fb, s1s, dirv = fb[keep], s1s[keep], dirv[keep]

            if fb.shape[0] == 0:
                result.append({
                    "boxes":      fb,
                    "scores":     s1s,
                    "labels":     torch.zeros(0, dtype=torch.int64, device=fb.device),
                    "embeddings": torch.zeros(0, embed_dim, device=fb.device),
                })
                continue

            # NMS with FCS scores (stage-2 NMS threshold = 0.5, paper §Impl. Details)
            keep_nms = box_ops.nms(fb, s1s, self.nms_thresh)[:self.detections_per_img]
            fb, s1s, dirv = fb[keep_nms], s1s[keep_nms], dirv[keep_nms]

            # Re-pool features at final stage-2 boxes for clean re-ID embeddings
            roi_feats = self.box_head_2(
                self.box_roi_pool(features, [fb], [img_shape])
            )
            direction_final, _ = self.nae_head(roi_feats)

            result.append({
                "boxes":      fb,
                "scores":     s1s,
                "labels":     torch.ones(fb.shape[0], dtype=torch.int64, device=fb.device),
                "embeddings": direction_final,
            })

        return result, {}

    # ------------------------------------------------------------------ #
    #  Stage-1 NMS (inference) — returns (boxes, stage-1 scores)
    # ------------------------------------------------------------------ #
    def _postprocess_stage1_for_inference(
        self, class_logits_s1, refined_boxes, proposals_s1, image_shapes,
        nms_thresh: float = 0.4, score_thresh: float = 0.05, topk: int = 300,
    ):
        """NMS on stage-1 outputs; propagates scores for FCS."""
        pred_scores  = F.softmax(class_logits_s1, -1)
        bpi          = [p.shape[0] for p in proposals_s1]
        scores_split = pred_scores.split(bpi, 0)

        kept_boxes  = []
        kept_scores = []
        for rb, sc in zip(refined_boxes, scores_split):
            person_sc = sc[:, 1]
            keep      = person_sc > score_thresh
            rb, person_sc = rb[keep], person_sc[keep]

            if rb.shape[0] == 0:
                kept_boxes.append(rb)
                kept_scores.append(person_sc)
                continue

            nms_keep = box_ops.nms(rb, person_sc, nms_thresh)[:topk]
            kept_boxes.append(rb[nms_keep])
            kept_scores.append(person_sc[nms_keep])

        return kept_boxes, kept_scores


# --------------------------------------------------------------------------- #
#  Full PersonSearchModel
# --------------------------------------------------------------------------- #

class PersonSearchModel(nn.Module):
    """
    SeqNet person search model (Li & Miao, AAAI 2021).

    backbone_name : "resnet50"    → FasterRCNN_ResNet50_FPN (v1 COCO weights)
                    "resnet50_v2" → FasterRCNN_ResNet50_FPN_V2 (improved v2 weights)
    num_unlabeled : OIM circular-queue size.  Use 500 for PRW, 5000 for CUHK-SYSU.
    nae_weight    : scalar multiplier on the NAE BCE classification loss (Lcls2).
    triplet_weight: auxiliary batch-hard triplet loss weight in OIM (0 = disabled).
    """

    def __init__(
        self,
        num_pids: int,
        backbone_name: str = "resnet50",
        embed_dim: int = 256,
        num_unlabeled: int = 500,
        oim_momentum: float = 0.5,
        oim_scalar: float = 30.0,
        nae_weight: float = 1.0,
        triplet_weight: float = 0.0,
        pretrained: bool = True,
    ):
        super().__init__()

        if backbone_name == "resnet50_v2":
            base = fasterrcnn_resnet50_fpn_v2(
                weights=FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT if pretrained else None,
            )
        else:
            base = fasterrcnn_resnet50_fpn(
                weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT if pretrained else None,
            )

        # Stage-1 box predictor: 2 classes (background + person)
        in_features = base.roi_heads.box_predictor.cls_score.in_features
        base.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes=2)

        # Stage-2 modules: duplicate TwoMLPHead + predictor + NAE head
        out_channels = base.backbone.out_channels               # 256 for ResNet-50 FPN
        resolution   = base.roi_heads.box_roi_pool.output_size[0]  # 7
        box_head_2   = TwoMLPHead(
            in_channels=out_channels * resolution * resolution,  # 256*7*7 = 12544
            representation_size=1024,
        )
        box_predictor_2 = FastRCNNPredictor(1024, num_classes=2)
        nae_head        = NAEHead(in_channels=1024, embed_dim=embed_dim)

        oim = OIMLoss(
            num_features=embed_dim,
            num_pids=num_pids,
            num_unlabeled=num_unlabeled,
            momentum=oim_momentum,
            scalar=oim_scalar,
            triplet_weight=triplet_weight,
        )

        old = base.roi_heads
        base.roi_heads = SeqNetRoIHeads(
            # Stage-2 modules
            box_head_2=box_head_2,
            box_predictor_2=box_predictor_2,
            nae_head=nae_head,
            oim_loss=oim,
            nae_weight=nae_weight,
            # Stage-1 modules (inherited from torchvision)
            box_roi_pool=old.box_roi_pool,
            box_head=old.box_head,
            box_predictor=old.box_predictor,
            # Stage-1 sampling config (0.5 IoU, paper §Method)
            fg_iou_thresh=0.5,
            bg_iou_thresh=0.5,
            batch_size_per_image=256,
            positive_fraction=0.25,
            bbox_reg_weights=None,
            # Inference config
            score_thresh=0.05,
            nms_thresh=0.5,           # stage-2 NMS threshold (paper §Impl. Details)
            detections_per_img=300,
        )

        self.model     = base
        self.embed_dim = embed_dim

    def forward(self, images, targets=None):
        return self.model(images, targets)

    @torch.no_grad()
    def extract_query_embedding(self, scene_tensor, query_box):
        """
        Extract the L2-normalised direction embedding for a query crop.
        scene_tensor : (1, 3, H, W) float tensor in [0, 1]
        query_box    : (4,) tensor [x1, y1, x2, y2] in original pixel coords
        """
        device = scene_tensor.device

        images_list = [scene_tensor.squeeze(0)]
        transformed, _ = self.model.transform(images_list, None)
        t = transformed.tensors
        actual_h, actual_w = transformed.image_sizes[0]

        orig_h = scene_tensor.shape[2]
        orig_w = scene_tensor.shape[3]
        scale_x = actual_w / orig_w
        scale_y = actual_h / orig_h

        box = query_box.float().to(device)
        scaled_box = torch.stack([
            box[0] * scale_x, box[1] * scale_y,
            box[2] * scale_x, box[3] * scale_y,
        ]).unsqueeze(0)

        features  = self.model.backbone(t)
        roi_feat  = self.model.roi_heads.box_roi_pool(
            features, [scaled_box], [(actual_h, actual_w)]
        )
        flat      = self.model.roi_heads.box_head_2(roi_feat)
        direction, _ = self.model.roi_heads.nae_head(flat)   # return direction only
        return direction
