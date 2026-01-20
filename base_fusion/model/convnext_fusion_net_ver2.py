import torch
import torch.nn as nn
import torch.nn.functional as F

from base_fusion.model.cbamf_block import CBAMFusion
from base_fusion.model.convnext_block import ConvNeXtBlock

class EncoderStage(nn.Module):
    def __init__(self, in_ch, feat_ch, depth=3):
        super().__init__()

        self.embed = nn.Conv2d(in_ch, feat_ch, kernel_size=3, padding=1)

        self.blocks = nn.ModuleList([
            ConvNeXtBlock(feat_ch) for _ in range(depth)
        ])

    def forward(self, x):
        x = self.embed(x)

        features = []
        for blk in self.blocks:
            x = blk(x)
            features.append(x)
        return features

class DecoderStage(nn.Module):
    def __init__(self, feat_ch):
        super().__init__()
        self.block = ConvNeXtBlock(feat_ch)

    def forward(self, x, skip):
        x = x + skip
        x = self.block(x)
        return x

class ConvNeXtFusionNetVer2(nn.Module):
    """
    AutoEncoder ConvNeXt-based IR-VI Image Fusion Network
    (Improved version with CBAMF + skip-add decoder)
    """

    def __init__(
        self,
        in_channels=1,
        feat_channels=64,
        encoder_depth=3,
    ):
        super().__init__()

        # ===============================
        # Encoder (IR & VI)
        # ===============================
        self.encoder_ir = EncoderStage(
            in_ch=in_channels,
            feat_ch=feat_channels,
            depth=encoder_depth
        )

        self.encoder_vi = EncoderStage(
            in_ch=in_channels,
            feat_ch=feat_channels,
            depth=encoder_depth
        )

        # ===============================
        # CBAM Fusion at each stage
        # ===============================
        self.fusion_blocks = CBAMFusion(feat_channels) 

        # ===============================
        # Decoder
        # ===============================
        self.decoder_first_stage = ConvNeXtBlock(feat_channels)
        self.decoder_stages = nn.ModuleList([
            DecoderStage(feat_channels) for _ in range(encoder_depth - 1)
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

        feats_ir = self.encoder_ir(ir)
        feats_vi = self.encoder_vi(vi)

        for i, (f_ir, f_vi) in enumerate(zip(feats_ir, feats_vi)):
            feats_fused = self.fusion_blocks(f_ir, f_vi)
            fused_features.append(feats_fused)

        # ===============================
        # Decoder forward (reverse order)
        # ===============================
        x = fused_features[-1]
        x = self.decoder_first_stage(x)
        for i in range(len(self.decoder_stages) - 1, -1, -1):
            x = self.decoder_stages[i](x, fused_features[i])

        # ===============================
        # Output
        # ===============================
        out = self.out_conv(x)
        return out
    
if __name__ == "__main__":
    model = ConvNeXtFusionNetVer2()
    ir = torch.randn(1, 1, 256, 256)
    vi = torch.randn(1, 1, 256, 256)

    out = model(ir, vi)
    print(out.shape)  # (1, 1, 256, 256)

