# SOLIDER Swin backbone for SeqNet, following SOLIDER-PersonSearch
# (https://github.com/tinyvision/SOLIDER-PersonSearch: models/swin.py and the pre-trained weight
# loading in utils/utils.py; that repository has no license file, so this is a re-implementation).
# The Swin model is SOLIDER's swin_transformer.py (Apache-2.0, see LICENSE-SOLIDER).
#
# Split, as in SOLIDER-PersonSearch: Swin stages 1-3 are the stem (feat_res4, stride 16) and
# stage 4, with the patch merging of stage 3, is the RoI head on 14x14 RoIs. The stem and the box
# head share one Swin instance; SeqNet deep-copies the box head for the re-ID head.
#
# Changes: the semantic-weight vector is built on the input's device instead of with .cuda(), and
# the SOLIDER checkpoint is loaded into the Swin before SeqNet copies the head (the official code
# copies it into both heads afterwards: same result), with a check that the tensors really loaded.
from collections import OrderedDict

import torch
from torch import nn

from swin_transformer import (
    swin_base_patch4_window7_224,
    swin_small_patch4_window7_224,
    swin_tiny_patch4_window7_224,
)

SWIN_BUILDERS = {
    "swin_tiny": swin_tiny_patch4_window7_224,
    "swin_small": swin_small_patch4_window7_224,
    "swin_base": swin_base_patch4_window7_224,
}
STEM_STAGES = 3  # stages 1-3 -> feat_res4 (stride 16); stage 4 -> RoI head


def _semantic_vector(swin, x):
    """SOLIDER's controller input [lambda, 1 - lambda] for every image (or RoI) in the batch."""
    w = torch.ones(x.shape[0], 1, device=x.device) * swin.semantic_weight
    return torch.cat([w, 1 - w], dim=-1)


def _modulate(swin, i, x, sem):
    sw = swin.semantic_embed_w[i](sem).unsqueeze(1)
    sb = swin.semantic_embed_b[i](sem).unsqueeze(1)
    return x * swin.softplus(sw) + sb


class SwinStem(nn.Module):
    """Swin stages 1-3: OrderedDict(feat_res4=[B, C4, H/16, W/16])."""

    def __init__(self, swin):
        super(SwinStem, self).__init__()
        self.swin = swin
        self.out_channels = swin.num_features[STEM_STAGES - 1]

    def forward(self, x):
        swin = self.swin
        sem = _semantic_vector(swin, x) if swin.semantic_weight >= 0 else None

        x, hw_shape = swin.patch_embed(x)
        if swin.use_abs_pos_embed:
            x = x + swin.absolute_pos_embed
        x = swin.drop_after_pos(x)

        for i, stage in enumerate(swin.stages[:STEM_STAGES]):
            x, hw_shape, out, out_hw_shape = stage(x, hw_shape)
            if sem is not None:
                x = _modulate(swin, i, x, sem)
        out = getattr(swin, f"norm{STEM_STAGES - 1}")(out)
        out = out.view(-1, *out_hw_shape, swin.num_features[STEM_STAGES - 1])
        return OrderedDict([["feat_res4", out.permute(0, 3, 1, 2).contiguous()]])


class SwinRes5Head(nn.Module):
    """Stage 3 patch merging + stage 4 on RoI features; avg-pooled stage-3 input and stage-4 output."""

    def __init__(self, swin):
        super(SwinRes5Head, self).__init__()
        self.swin = swin
        self.out_channels = [swin.num_features[STEM_STAGES - 1], swin.num_features[-1]]

    def forward(self, x):
        swin = self.swin
        sem = _semantic_vector(swin, x) if swin.semantic_weight >= 0 else None

        feat = x
        hw_shape = x.shape[-2:]
        x = torch.flatten(x, 2).permute(0, 2, 1)
        x, hw_shape = swin.stages[STEM_STAGES - 1].downsample(x, hw_shape)
        if sem is not None:
            x = _modulate(swin, STEM_STAGES - 1, x, sem)
        for i, stage in enumerate(swin.stages[STEM_STAGES:], start=STEM_STAGES):
            x, hw_shape, out, out_hw_shape = stage(x, hw_shape)
            if sem is not None:
                x = _modulate(swin, i, x, sem)
        last = len(swin.stages) - 1
        out = getattr(swin, f"norm{last}")(out)
        out = out.view(-1, *out_hw_shape, swin.num_features[last]).permute(0, 3, 1, 2).contiguous()
        return OrderedDict([["feat_res4", swin.avgpool(feat)], ["feat_res5", swin.avgpool(out)]])


def load_solider_weights(swin, path):
    """Load a SOLIDER checkpoint (raw training checkpoint with 'teacher', or converted) into swin."""
    # weights_only=False: SOLIDER training checkpoints also pickle non-tensor objects (args).
    # Only load files downloaded from the official SOLIDER links.
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if "teacher" in ckpt:
        ckpt = ckpt["teacher"]
    if "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    elif "model" in ckpt:
        ckpt = ckpt["model"]
    ckpt = {k[7:] if k.startswith("module.") else k: v for k, v in ckpt.items()}
    state = {k[9:]: v for k, v in ckpt.items() if k.startswith("backbone.")}
    if not state:  # a bare Swin state_dict
        state = ckpt
    missing, unexpected = swin.load_state_dict(state, strict=False)
    loaded = len(state) - len(unexpected)
    print(f"SOLIDER weights {path}: {loaded} tensors loaded | not in checkpoint: {missing}")
    if loaded == 0 or unexpected:
        raise RuntimeError(f"{path} does not match this Swin: unexpected keys {unexpected[:10]}")


def build_swin(name="swin_small", semantic_weight=0.6, weights=None):
    """Returns (stem, box_head) like backbone.build_resnet; weights = SOLIDER checkpoint or None."""
    swin = SWIN_BUILDERS[name](drop_path_rate=0.1, semantic_weight=semantic_weight)
    if weights:
        load_solider_weights(swin, weights)
    return SwinStem(swin), SwinRes5Head(swin)
