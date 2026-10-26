# Adapted from SeqNet (https://github.com/serend1p1ty/SeqNet), models/resnet.py
# Copyright (c) 2022 Zj Li. MIT License, see LICENSE-SeqNet.
#
# Changes: torchvision `weights=` API (IMAGENET1K_V1 = the old `pretrained=True` weights),
# optional FrozenBatchNorm2d (off by default: the official code uses plain, trainable
# nn.BatchNorm2d), and a single `build_backbone(cfg)` entry point so other backbones
# (e.g. ConvNeXt/Swin) can be added later without touching the detector.
from collections import OrderedDict

import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.ops import misc as misc_nn_ops


class Backbone(nn.Sequential):
    """Stem network: ResNet conv1..layer3 (res1-res4), stride 16, 1024 channels."""

    def __init__(self, resnet):
        super(Backbone, self).__init__(
            OrderedDict(
                [
                    ["conv1", resnet.conv1],
                    ["bn1", resnet.bn1],
                    ["relu", resnet.relu],
                    ["maxpool", resnet.maxpool],
                    ["layer1", resnet.layer1],  # res2
                    ["layer2", resnet.layer2],  # res3
                    ["layer3", resnet.layer3],  # res4
                ]
            )
        )
        self.out_channels = 1024

    def forward(self, x):
        # using the forward method from nn.Sequential
        feat = super(Backbone, self).forward(x)
        return OrderedDict([["feat_res4", feat]])


class Res5Head(nn.Sequential):
    """RoI head: ResNet layer4 (res5); returns max-pooled res4 input and res5 output."""

    def __init__(self, resnet):
        super(Res5Head, self).__init__(OrderedDict([["layer4", resnet.layer4]]))  # res5
        self.out_channels = [1024, 2048]

    def forward(self, x):
        feat = super(Res5Head, self).forward(x)
        x = F.adaptive_max_pool2d(x, 1)
        feat = F.adaptive_max_pool2d(feat, 1)
        return OrderedDict([["feat_res4", x], ["feat_res5", feat]])


_RESNET_WEIGHTS = {"resnet50": torchvision.models.ResNet50_Weights.IMAGENET1K_V1}


def build_resnet(name="resnet50", pretrained=True, frozen_bn=False):
    weights = _RESNET_WEIGHTS[name] if pretrained else None
    kwargs = {"norm_layer": misc_nn_ops.FrozenBatchNorm2d} if frozen_bn else {}
    resnet = torchvision.models.resnet.__dict__[name](weights=weights, **kwargs)

    # freeze layers (official policy: only conv1 and bn1 affine parameters)
    resnet.conv1.weight.requires_grad_(False)
    resnet.bn1.weight.requires_grad_(False)
    resnet.bn1.bias.requires_grad_(False)

    return Backbone(resnet), Res5Head(resnet)


def build_backbone(cfg):
    """
    Returns (stem, box_head):
      stem(x)      -> OrderedDict(feat_res4=[B, stem.out_channels, H/16, W/16])
      box_head(r)  -> OrderedDict(feat_res4=[N, C4, 1, 1], feat_res5=[N, C5, 1, 1]) on 14x14 RoIs,
                      with box_head.out_channels = [C4, C5].
    The stage-2 re-ID head is a deepcopy of box_head (done in SeqNet).
    Add new backbones here.
    """
    bcfg = cfg.MODEL.BACKBONE
    if bcfg.NAME in _RESNET_WEIGHTS:
        return build_resnet(bcfg.NAME, pretrained=bcfg.PRETRAINED, frozen_bn=bcfg.FROZEN_BN)
    raise NotImplementedError(f"Unknown backbone: {bcfg.NAME}")
