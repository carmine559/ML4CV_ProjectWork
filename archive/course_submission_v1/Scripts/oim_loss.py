import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------------------------------- #
#  OIM Loss implementation
# --------------------------------------------------------------------------- #

# OIM Loss implementation based on the original paper "OIM: Online Instance Matching Loss for Person Search" (https://arxiv.org/abs/2002.10560).
# This implementation includes the OIM loss itself, as well as an optional batch-hard triplet loss as an auxiliary loss. 
# The LUT (Look-Up Table) and CQ (Circular Queue) are implemented as buffers that are updated during training. 
# The forward method computes the OIM loss and optionally the triplet loss, and updates the LUT with the features of the labeled samples.
class OIMLoss(nn.Module):
    """
    OIM Loss + optional batch-hard triplet auxiliary.
    If triplet_weight == 0 the triplet branch is a no-op.
    """

    def __init__(self, num_features, num_pids, num_unlabeled=5000,
                 scalar=30.0, momentum=0.5,
                 triplet_weight=0.0, triplet_margin=0.3):
        super().__init__()
        self.num_features = num_features
        self.num_pids = num_pids
        self.num_unlabeled = num_unlabeled
        self.momentum = momentum
        self.scalar = scalar
        self.triplet_weight = triplet_weight
        self.triplet_margin = triplet_margin

        self.register_buffer("lut", torch.zeros(num_pids, num_features))
        self.register_buffer("cq", torch.zeros(num_unlabeled, num_features))
        self.register_buffer("cq_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _enqueue(self, feats: torch.Tensor):
        n = feats.shape[0]
        if n == 0:
            return
        ptr = int(self.cq_ptr)
        idx = (torch.arange(n, device=feats.device) + ptr) % self.num_unlabeled
        self.cq[idx] = feats.detach().float()
        self.cq_ptr[0] = (ptr + n) % self.num_unlabeled

    @staticmethod
    def _batch_hard_triplet(feats, pids, margin):
        if feats.shape[0] < 2:
            return feats.sum() * 0.0
        sim = feats @ feats.t()
        dist = 1.0 - sim
        same = pids.unsqueeze(0) == pids.unsqueeze(1)
        same.fill_diagonal_(False)
        diff = ~(pids.unsqueeze(0) == pids.unsqueeze(1))
        valid = same.any(dim=1) & diff.any(dim=1)
        if not valid.any():
            return feats.sum() * 0.0
        hardest_pos = dist.masked_fill(~same, -1e9).max(dim=1).values
        hardest_neg = dist.masked_fill(~diff,  1e9).min(dim=1).values
        loss = F.relu(hardest_pos - hardest_neg + margin)
        return loss[valid].mean()

    def forward(self, features: torch.Tensor, pids: torch.Tensor) -> torch.Tensor:
        # Upcast to float32: scalar=30 means logits can reach 30, and
        # exp(30) ≈ 1e13 which overflows float16 (max ≈ 65504) under AMP,
        # producing inf → NaN in cross_entropy.  float32 max is 3.4e38, safe.
        features = features.float()

        labeled_mask = pids >= 0
        unlabeled_mask = pids == -1

        if unlabeled_mask.any():
            self._enqueue(features[unlabeled_mask])

        if not labeled_mask.any():
            return features.sum() * 0.0

        l_feats = features[labeled_mask]
        l_pids = pids[labeled_mask]

        l_feats = F.normalize(l_feats, p=2, dim=1)

        # Snapshot LUT/CQ before F.linear so in-place updates later don't
        # invalidate the autograd graph (the backward of F.linear w.r.t.
        # `l_feats` needs the weight tensor unchanged until backward runs).
        lut_snapshot = self.lut.clone()
        cq_snapshot = self.cq.clone()

        sim_labeled = F.linear(l_feats, lut_snapshot)
        sim_unlabeled = F.linear(l_feats, cq_snapshot)
        projected = torch.cat([sim_labeled, sim_unlabeled], dim=1) * self.scalar
        oim_loss = F.cross_entropy(projected, l_pids)

        if self.training:
            with torch.no_grad():
                for feat, pid in zip(l_feats, l_pids):
                    self.lut[pid] = self.momentum * self.lut[pid] + (1. - self.momentum) * feat
                    self.lut[pid] = F.normalize(self.lut[pid], p=2, dim=0)

        if self.triplet_weight > 0:
            triplet = self._batch_hard_triplet(l_feats, l_pids, self.triplet_margin)
            return oim_loss + self.triplet_weight * triplet

        return oim_loss