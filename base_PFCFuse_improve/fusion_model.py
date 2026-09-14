from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .net import (
    BaseFeatureExtraction,
    DetailFeatureExtraction,
    Restormer_Decoder,
    Restormer_Encoder,
)


DEFAULT_CHECKPOINT = Path(__file__).with_name("PFCFuse_IVF.pth")
FORMAT = "pfcfuse_residual_v1"


class ResidualFusion(nn.Module):
    """Bounded redistribution around pretrained additive fusion.

    v2: wider multi-dilation stack (rates 1/2/4, hidden=24) for a larger receptive field, so
    the gate has enough spatial context (e.g. a hot object vs its background) to make a useful
    decision. v1 used only rates 1/2 with hidden=16.
    """

    def __init__(self, dim=64, hidden=24, limit=0.25):
        super().__init__()
        if not 0 < limit <= 1:
            raise ValueError("limit must be in (0, 1]")
        self.limit = limit
        self.project = nn.Conv2d(dim * 4, hidden, 1)
        self.branch1 = nn.Conv2d(hidden, hidden, 3, padding=1, dilation=1, groups=hidden)
        self.branch2 = nn.Conv2d(hidden, hidden, 3, padding=2, dilation=2, groups=hidden)
        self.branch4 = nn.Conv2d(hidden, hidden, 3, padding=4, dilation=4, groups=hidden)
        self.output = nn.Conv2d(hidden, dim, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, visible, infrared):
        scale = (visible.square().mean(1, keepdim=True)
                 + infrared.square().mean(1, keepdim=True) + 1e-6).sqrt()
        inputs = torch.cat((visible / scale, infrared / scale,
                            (visible - infrared).abs() / scale,
                            F.avg_pool2d((visible + infrared) / scale, 5, 1, 2)), dim=1)
        hidden = F.gelu(self.project(inputs))
        context = F.gelu(self.branch1(hidden) + self.branch2(hidden) + self.branch4(hidden))
        gate = self.limit * torch.tanh(self.output(context))
        return visible + infrared + gate * (visible - infrared)


class FusionModel(nn.Module):
    def __init__(self, limit=0.25):
        super().__init__()
        self.limit = limit
        self.backbone = nn.ModuleDict({
            "DIDF_Encoder": Restormer_Encoder(),
            "DIDF_Decoder": Restormer_Decoder(),
            "BaseFuseLayer": BaseFeatureExtraction(dim=64),
            "DetailFuseLayer": DetailFeatureExtraction(num_layers=1),
        })
        self.adapters = nn.ModuleDict({
            "base": ResidualFusion(limit=limit),
            "detail": ResidualFusion(limit=limit),
        })
        self.backbone.requires_grad_(False)

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def encode(self, visible, infrared):
        with torch.no_grad():
            visible_base, visible_detail, _ = self.backbone["DIDF_Encoder"](visible)
            infrared_base, infrared_detail, _ = self.backbone["DIDF_Encoder"](infrared)
        return visible_base, visible_detail, infrared_base, infrared_detail

    def decode(self, visible, features, baseline=False):
        visible_base, visible_detail, infrared_base, infrared_detail = features
        if baseline:
            base = visible_base + infrared_base
            detail = visible_detail + infrared_detail
        else:
            base = self.adapters["base"](visible_base, infrared_base)
            detail = self.adapters["detail"](visible_detail, infrared_detail)
        base = self.backbone["BaseFuseLayer"](base)
        detail = self.backbone["DetailFuseLayer"](detail)
        fused, _ = self.backbone["DIDF_Decoder"](visible, base, detail)
        return fused

    def forward(self, visible, infrared, baseline=False):
        return self.decode(visible, self.encode(visible, infrared), baseline=baseline)


def _strip_module_prefix(key):
    return key[len("module."):] if key.startswith("module.") else key


def load_model(checkpoint=DEFAULT_CHECKPOINT, device="cpu", limit=0.25):
    payload = torch.load(Path(checkpoint), map_location="cpu", weights_only=True)
    if payload.get("format") == FORMAT:
        model = FusionModel(limit=payload["limit"])
        model.load_state_dict(payload["model"], strict=True)
    else:
        model = FusionModel(limit=limit)
        for name, module in model.backbone.items():
            state = {_strip_module_prefix(key): value for key, value in payload[name].items()}
            module.load_state_dict(state, strict=True)
    return model.to(device).eval()