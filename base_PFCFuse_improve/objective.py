import kornia
import torch
from torch import nn
from torch.nn import functional as F


import math

import kornia
import torch
from torch import nn
from torch.nn import functional as F


def _pearson_correlation(x, y, eps=1e-8):
    x = x - x.mean(dim=(1, 2, 3), keepdim=True)
    y = y - y.mean(dim=(1, 2, 3), keepdim=True)
    numerator = (x * y).sum(dim=(1, 2, 3))
    # eps must be INSIDE sqrt: sqrt's backward is 1/(2*sqrt(t)), which is +inf at t=0
    # (e.g. constant images), so adding eps only after sqrt still produces a NaN gradient.
    denominator = torch.sqrt(x.square().sum(dim=(1, 2, 3)) * y.square().sum(dim=(1, 2, 3)) + eps)
    return numerator / denominator


def _scd_score(fused, visible, infrared):
    """Differentiable analogue of Evaluator.SCD, per sample."""
    return (_pearson_correlation(visible, fused - infrared)
            + _pearson_correlation(infrared, fused - visible))


class QabfProxy(nn.Module):
    """Differentiable analogue of Evaluator.Qabf (Xydeas & Petrovic edge-preservation index).

    Uses the same Sobel kernels and Tg/kg/Dg/Ta/ka/Da constants as utils/Evaluator.py, but with
    torch.atan/eps-stabilized ratios instead of the numpy special-cased branches, so it stays
    differentiable. It is a training-time proxy, not a replacement for the authoritative
    8-bit numpy metric that evaluation.py uses for gating and reporting.
    """

    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[1., 2., 1.], [0., 0., 0.], [-1., -2., -1.]]).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def _gradient(self, image, eps=1e-6):
        gx = F.conv2d(image, self.sobel_x, padding=1)
        gy = F.conv2d(image, self.sobel_y, padding=1)
        magnitude = torch.sqrt(gx.square() + gy.square() + eps)
        angle = torch.atan(gy / (gx + eps))
        return magnitude, angle

    @staticmethod
    def _edge_quality(magnitude_ref, angle_ref, magnitude_fused, angle_fused, eps=1e-6):
        ratio = torch.minimum(magnitude_ref, magnitude_fused) / (torch.maximum(magnitude_ref, magnitude_fused) + eps)
        orientation = 1 - (angle_ref - angle_fused).abs() / (math.pi / 2)
        magnitude_term = 0.9994 / (1 + torch.exp(-15 * (ratio - 0.5)))
        orientation_term = 0.9879 / (1 + torch.exp(-22 * (orientation - 0.8)))
        return magnitude_term * orientation_term

    def forward(self, fused, visible, infrared, eps=1e-6):
        magnitude_vi, angle_vi = self._gradient(visible)
        magnitude_ir, angle_ir = self._gradient(infrared)
        magnitude_fused, angle_fused = self._gradient(fused)
        quality_vi = self._edge_quality(magnitude_vi, angle_vi, magnitude_fused, angle_fused)
        quality_ir = self._edge_quality(magnitude_ir, angle_ir, magnitude_fused, angle_fused)
        numerator = (quality_vi * magnitude_vi + quality_ir * magnitude_ir).sum(dim=(1, 2, 3))
        denominator = (magnitude_vi + magnitude_ir).sum(dim=(1, 2, 3)) + eps
        return numerator / denominator


class FusionObjective(nn.Module):
    """Self-referential ranking loss against the frozen additive baseline (v3).

    v1/v2 optimized heuristic surrogates (SSIM-to-source, blended intensity, Sobel gradient
    matching) that only correlated loosely with the acceptance gate's actual metrics, which let
    the network find shortcuts that reduced the proxy loss while making Qabf/EN/SF/MI worse
    (see ANALYSIS.md Sec. 5.1). v3 instead computes differentiable analogues of the SAME metrics
    the gate checks (Qabf, SSIM, SCD, clipping) for both the candidate and the teacher (frozen
    additive baseline) on the SAME batch, and trains with one-sided hinges: strictly beat the
    teacher on Qabf by `qabf_margin`, and do not fall behind the teacher by more than each
    metric's own gate tolerance on SSIM/SCD/clipping. This directly targets "beat original",
    instead of hoping a generic fusion heuristic happens to correlate with it.
    """

    def __init__(self, qabf_margin=0.01, ssim_margin=0.001, scd_margin=0.01, clip_margin=0.001,
                 weight_qabf=4.0, weight_ssim=2.0, weight_scd=1.0, weight_clip=1.0, weight_trust=1.0):
        super().__init__()
        self.qabf = QabfProxy()
        self.qabf_margin = qabf_margin
        self.ssim_margin = ssim_margin
        self.scd_margin = scd_margin
        self.clip_margin = clip_margin
        self.weight_qabf = weight_qabf
        self.weight_ssim = weight_ssim
        self.weight_scd = weight_scd
        self.weight_clip = weight_clip
        self.weight_trust = weight_trust

    def _ssim_score(self, image, visible, infrared):
        return 1 - 0.5 * (kornia.losses.ssim_loss(image, visible, 7, reduction="none").mean(dim=(1, 2, 3))
                          + kornia.losses.ssim_loss(image, infrared, 7, reduction="none").mean(dim=(1, 2, 3)))

    def forward(self, fused_base, batch, teacher_base):
        visible, infrared, detail = batch["visible"], batch["infrared"], batch["detail"]
        teacher_base = teacher_base.detach()
        candidate_unclipped = fused_base + detail
        teacher_unclipped = teacher_base + detail
        candidate = candidate_unclipped.clamp(0, 1)
        teacher = teacher_unclipped.clamp(0, 1)

        qabf_candidate = self.qabf(candidate, visible, infrared)
        qabf_teacher = self.qabf(teacher, visible, infrared)
        loss_qabf = F.relu(self.qabf_margin - (qabf_candidate - qabf_teacher)).mean()

        ssim_candidate = self._ssim_score(candidate, visible, infrared)
        ssim_teacher = self._ssim_score(teacher, visible, infrared)
        loss_ssim = F.relu((ssim_teacher - ssim_candidate) - self.ssim_margin).mean()

        scd_candidate = _scd_score(candidate, visible, infrared)
        scd_teacher = _scd_score(teacher, visible, infrared)
        loss_scd = F.relu((scd_teacher - scd_candidate) - self.scd_margin).mean()

        clip_candidate = (candidate_unclipped - candidate).abs().mean(dim=(1, 2, 3))
        clip_teacher = (teacher_unclipped - teacher).abs().mean(dim=(1, 2, 3))
        loss_clip = F.relu((clip_candidate - clip_teacher) - self.clip_margin).mean()

        trust = F.relu((fused_base - teacher_base).abs() - 0.02).mean()

        total = (self.weight_qabf * loss_qabf + self.weight_ssim * loss_ssim
                + self.weight_scd * loss_scd + self.weight_clip * loss_clip
                + self.weight_trust * trust)
        return total