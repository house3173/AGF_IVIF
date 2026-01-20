import torch
import torch.nn as nn
import torch.nn.functional as F

from base_fusion.model.cbamf_block import CBAMFusion
from base_fusion.model.convnext_block import ConvNeXtBlock

class Encoder(nn.Module):
    def __init__(self, channels, depth=3):
        super().__init__()
        self.blocks = nn.ModuleList([
            ConvNeXtBlock(channels) for _ in range(depth)
        ])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x

class Decoder(nn.Module):
    def __init__(self, channels, depth=3):
        super().__init__()
        self.blocks = nn.ModuleList([
            ConvNeXtBlock(channels) for _ in range(depth)
        ])

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return x

class ConvNeXtFusionNetVer1(nn.Module):
    def __init__(self, in_channels=1, feat_channels=64, depth=3):
        super().__init__()

        self.ir_embed = nn.Conv2d(in_channels, feat_channels, 3, padding=1)
        self.vi_embed = nn.Conv2d(in_channels, feat_channels, 3, padding=1)

        self.encoder_ir = Encoder(feat_channels, depth)
        self.encoder_vi = Encoder(feat_channels, depth)

        self.fusion_blocks = CBAMFusion(feat_channels) 

        self.decoder = Decoder(feat_channels, depth)

        self.out_conv = nn.Conv2d(feat_channels, 1, kernel_size=1)

    def forward(self, ir, vi):
        ir = self.ir_embed(ir)
        vi = self.vi_embed(vi)

        feat_ir = self.encoder_ir(ir)
        feat_vi = self.encoder_vi(vi)

        feat_fused = self.fusion_blocks(feat_ir, feat_vi)

        x = self.decoder(feat_fused)
        out = self.out_conv(x)
        return out

if __name__ == "__main__":
    model = ConvNeXtFusionNetVer1()
    ir = torch.randn(1, 1, 256, 256)
    vi = torch.randn(1, 1, 256, 256)

    out = model(ir, vi)
    print(out.shape)  # (1, 1, 256, 256)