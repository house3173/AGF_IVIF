import torch
import torch.nn as nn
import torch.nn.functional as F

from base_fusion.model.cbamf_block import CBAMFusion
from base_fusion.model.convnext_block import ConvNeXtBlock

class EncoderStage(nn.Module):
    def __init__(self, in_ch, feat_ch, depth):
        super().__init__()

        self.embed = nn.Conv2d(in_ch, feat_ch, kernel_size=3, padding=1)

        self.blocks = nn.Sequential(*[
            ConvNeXtBlock(feat_ch) for _ in range(depth)
        ])

    def forward(self, x):
        x = self.embed(x)
        x = self.blocks(x)
        return x

class DecoderStage(nn.Module):
    def __init__(self, feat_ch):
        super().__init__()
        self.block = ConvNeXtBlock(feat_ch)

    def forward(self, x, skip):
        x = x + skip
        x = self.block(x)
        return x

class ConvNeXtFusionNetVer3(nn.Module):
    """
    AutoEncoder ConvNeXt-based IR-VI Image Fusion Network
    (Improved version with CBAMF + skip-add decoder)
    """

    def __init__(
        self,
        in_channels=1,
        feat_channels=64,
        encoder_depth=3,
        num_stages=3
    ):
        super().__init__()

        # ===============================
        # Encoder (IR & VI)
        # ===============================
        self.encoder_ir = nn.ModuleList([
            EncoderStage(
                in_ch=in_channels if i == 0 else feat_channels,
                feat_ch=feat_channels,
                depth=encoder_depth
            ) for i in range(num_stages)
        ])

        self.encoder_vi = nn.ModuleList([
            EncoderStage(
                in_ch=in_channels if i == 0 else feat_channels,
                feat_ch=feat_channels,
                depth=encoder_depth
            ) for i in range(num_stages)
        ])

        # ===============================
        # CBAM Fusion at each stage
        # ===============================
        self.fusion_blocks = nn.ModuleList([
            CBAMFusion(feat_channels) for _ in range(num_stages)
        ])

        # ===============================
        # Decoder
        # ===============================
        self.decoder_stages = nn.ModuleList([
            DecoderStage(feat_channels) for _ in range(num_stages)
        ])

        # ===============================
        # Output projection
        # ===============================
        self.out_conv = nn.Conv2d(
            feat_channels, 1, kernel_size=3, padding=1
        )

    def forward(self, ir, vi):
        """
        ir, vi: (B, 1, H, W)
        """

        # ===============================
        # Encoder forward
        # ===============================
        fused_features = []

        x_ir, x_vi = ir, vi
        for i in range(len(self.encoder_ir)):
            x_ir = self.encoder_ir[i](x_ir)
            x_vi = self.encoder_vi[i](x_vi)

            f = self.fusion_blocks[i](x_ir, x_vi)
            fused_features.append(f)

        # ===============================
        # Decoder forward (reverse order)
        # ===============================
        x = fused_features[-1]
        for i in reversed(range(len(self.decoder_stages) - 1)):
            x = self.decoder_stages[i](x, fused_features[i])

        # ===============================
        # Output
        # ===============================
        out = self.out_conv(x)
        return out

if __name__ == "__main__":
    model = ConvNeXtFusionNetVer3()
    ir = torch.randn(1, 1, 256, 256)
    vi = torch.randn(1, 1, 256, 256)

    out = model(ir, vi)
    print(out.shape)  # (1, 1, 256, 256)
