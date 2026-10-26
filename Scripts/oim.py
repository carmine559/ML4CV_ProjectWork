# Adapted from SeqNet (https://github.com/serend1p1ty/SeqNet), models/oim.py
# Copyright (c) 2022 Zj Li. MIT License, see LICENSE-SeqNet.
#
# Changes w.r.t. the official file (semantics unchanged):
#   * forward/backward wrapped with torch.amp.custom_fwd/custom_bwd so OIM always runs in fp32
#     under autocast (logits up to +-30, and the LUT/CQ buffers stay fp32);
#   * header/momentum passed as Python numbers instead of tensors; targets read with one
#     .tolist() instead of one GPU sync per sample (same sequential update order);
#   * the CQ write pointer is stored in the state_dict (extra state) so --resume is exact;
#   * if a batch contains no labelled identity at all, return a zero loss instead of the
#     NaN that F.cross_entropy produces when every target equals ignore_index.
# As in the official code, the LUT/CQ are updated in-place inside backward(), *after* the
# input gradient has been computed with the forward-time memory. Saved tensors are checked
# by autograd's version counter, so this assumes one backward per forward (no grad accumulation).
import torch
import torch.nn.functional as F
from torch import autograd, nn

UNLABELED_LABEL = 5554  # pid 5555 (unlabeled) minus 1


class OIM(autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda", cast_inputs=torch.float32)
    def forward(ctx, inputs, targets, lut, cq, header, momentum):
        ctx.save_for_backward(inputs, targets, lut, cq)
        ctx.header = header
        ctx.momentum = momentum
        outputs_labeled = inputs.mm(lut.t())
        outputs_unlabeled = inputs.mm(cq.t())
        return torch.cat([outputs_labeled, outputs_unlabeled], dim=1)

    @staticmethod
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_outputs):
        inputs, targets, lut, cq = ctx.saved_tensors
        header, momentum = ctx.header, ctx.momentum

        grad_inputs = None
        if ctx.needs_input_grad[0]:
            grad_inputs = grad_outputs.mm(torch.cat([lut, cq], dim=0))

        for x, y in zip(inputs, targets.tolist()):
            if y < len(lut):
                lut[y] = momentum * lut[y] + (1.0 - momentum) * x
                lut[y] /= lut[y].norm()
            else:
                cq[header] = x
                header = (header + 1) % cq.size(0)
        return grad_inputs, None, None, None, None, None


def oim(inputs, targets, lut, cq, header, momentum=0.5):
    return OIM.apply(inputs, targets, lut, cq, int(header), float(momentum))


class OIMLoss(nn.Module):
    def __init__(self, num_features, num_pids, num_cq_size, oim_momentum, oim_scalar):
        super(OIMLoss, self).__init__()
        self.num_features = num_features
        self.num_pids = num_pids
        self.num_unlabeled = num_cq_size
        self.momentum = oim_momentum
        self.oim_scalar = oim_scalar

        self.register_buffer("lut", torch.zeros(self.num_pids, self.num_features))
        self.register_buffer("cq", torch.zeros(self.num_unlabeled, self.num_features))

        self.header_cq = 0

    def get_extra_state(self):
        return {"header_cq": int(self.header_cq)}

    def set_extra_state(self, state):
        self.header_cq = int(state["header_cq"])

    def forward(self, inputs, roi_label):
        # merge into one batch, background label = 0
        targets = torch.cat(roi_label)
        label = targets - 1  # background label = -1

        inds = label >= 0
        label = label[inds]
        inputs = inputs[inds.unsqueeze(1).expand_as(inputs)].view(-1, self.num_features)

        projected = oim(inputs, label, self.lut, self.cq, self.header_cq, momentum=self.momentum)
        projected = projected * self.oim_scalar

        self.header_cq = (
            self.header_cq + (label >= self.num_pids).long().sum().item()
        ) % self.num_unlabeled
        if not bool((label != UNLABELED_LABEL).any()):
            # no labelled identity in the batch: keep the graph (CQ still gets updated)
            return projected.sum() * 0.0
        # NOTE (official behaviour, kept on purpose): labels >= LUT_SIZE other than 5554
        # (PRW train pids 483 and 932 with LUT_SIZE=482) are pushed to the CQ in backward but
        # are still used as CE targets that point into the CQ columns.
        loss_oim = F.cross_entropy(projected, label, ignore_index=UNLABELED_LABEL)
        return loss_oim
