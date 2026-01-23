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

class ConvDecoder(nn.Module):
    def __init__(self, channels, depth=3):
        super().__init__()
        blocks = []
        for _ in range(depth):
            blocks.append(
                nn.Sequential(
                    nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(channels),
                    nn.GELU()
                )
            )
        self.blocks = nn.Sequential(*blocks)

    def forward(self, x):
        return self.blocks(x)

class ConvNeXtFusionNetVer0(nn.Module):
    def __init__(self, in_channels=1, feat_channels=64, depth=3):
        super().__init__()

        # Input embedding
        self.ir_embed = nn.Conv2d(in_channels, feat_channels, kernel_size=3, padding=1)
        self.vi_embed = nn.Conv2d(in_channels, feat_channels, kernel_size=3, padding=1)

        # Encoders
        self.encoder_ir = Encoder(feat_channels, depth)
        self.encoder_vi = Encoder(feat_channels, depth)

        # Fusion
        self.fusion_blocks = CBAMFusion(feat_channels)

        # Conv-based Decoder 
        self.decoder = ConvDecoder(feat_channels, depth)

        # Output projection
        self.out_conv = nn.Conv2d(feat_channels, 1, kernel_size=1)

    def forward(self, ir, vi):
        # Embed
        ir_feat = self.ir_embed(ir)
        vi_feat = self.vi_embed(vi)

        # Encode
        feat_ir = self.encoder_ir(ir_feat)
        feat_vi = self.encoder_vi(vi_feat)

        # Fuse
        feat_fused = self.fusion_blocks(feat_ir, feat_vi)

        # Decode
        x = self.decoder(feat_fused)

        # Output
        out = self.out_conv(x)
        return out

if __name__ == "__main__":
    model = ConvNeXtFusionNetVer0(
        in_channels=1,
        feat_channels=64,
        depth=3
    )

    ir = torch.randn(1, 1, 256, 256)
    vi = torch.randn(1, 1, 256, 256)

    out = model(ir, vi)
    print(out.shape)  # Expected: (1, 1, 256, 256)
