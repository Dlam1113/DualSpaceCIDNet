# -*- coding: utf-8 -*-
"""
DualSpaceCIDNet: Cascaded Dual-Space Color-Illumination Decoupling Network.
Combines continuous HVI color-space restoration with output-domain RGB residual refinement.
"""
import torch
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin

from net.HVI_transform import RGB_HVI
from net.transformer_utils import NormDownsample, NormUpsample
from net.LCA import HV_LCA, I_LCA
from net.NeuralCurve import NeuralCurveLayer


class RGBRefiner(nn.Module):
    """
    Lightweight Output-Domain Residual Refiner.
    Refines coarse reconstruction results directly in sRGB space to restore fine-grained spatial textures.
    Zero-initialized at the final projection layer to guarantee identity behavior at initialization.
    """
    def __init__(self, in_channels=3, mid_channels=32):
        super(RGBRefiner, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.act1 = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=1, padding=1, bias=True)
        self.act2 = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.conv3 = nn.Conv2d(mid_channels, in_channels, kernel_size=1, stride=1, padding=0, bias=True)

        # Zero initialization guarantees correction ≈ 0 initially
        nn.init.zeros_(self.conv3.weight)
        nn.init.zeros_(self.conv3.bias)

    def forward(self, x):
        """
        Forward pass of RGB Refiner.

        Args:
            x (torch.Tensor): Coarse sRGB reconstruction tensor [B, 3, H, W].

        Returns:
            torch.Tensor: Residual-corrected sRGB output [B, 3, H, W].
        """
        h = self.act1(self.conv1(x))
        h = self.act2(self.conv2(h))
        correction = self.conv3(h)
        return x + correction


class DualSpaceCIDNet(nn.Module, PyTorchModelHubMixin):
    """
    DualSpaceCIDNet Architecture.
    Processes features across decoupled continuous HVI space and output sRGB space.
    """
    def __init__(self,
                 channels=[36, 36, 72, 144],
                 heads=[1, 2, 4, 8],
                 norm=False,
                 use_rgb_refiner=True,
                 refiner_mid_ch=32,
                 use_curve=False,
                 curve_M=11):
        super(DualSpaceCIDNet, self).__init__()

        self.use_rgb_refiner = use_rgb_refiner
        self.use_curve = use_curve
        [ch1, ch2, ch3, ch4] = channels
        [head1, head2, head3, head4] = heads

        # Continuous Polarized HVI transformation
        self.trans = RGB_HVI()

        # Chromaticity (HV) Stream Encoder
        self.HVE_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(3, ch1, 3, stride=1, padding=0, bias=False)
        )
        self.HVE_block1 = NormDownsample(ch1, ch2, use_norm=norm)
        self.HVE_block2 = NormDownsample(ch2, ch3, use_norm=norm)
        self.HVE_block3 = NormDownsample(ch3, ch4, use_norm=norm)

        # Chromaticity (HV) Stream Decoder
        self.HVD_block3 = NormUpsample(ch4, ch3, use_norm=norm)
        self.HVD_block2 = NormUpsample(ch3, ch2, use_norm=norm)
        self.HVD_block1 = NormUpsample(ch2, ch1, use_norm=norm)
        self.HVD_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(ch1, 2, 3, stride=1, padding=0, bias=False)
        )

        # Intensity (I) Stream Encoder
        self.IE_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(1, ch1, 3, stride=1, padding=0, bias=False),
        )
        self.IE_block1 = NormDownsample(ch1, ch2, use_norm=norm)
        self.IE_block2 = NormDownsample(ch2, ch3, use_norm=norm)
        self.IE_block3 = NormDownsample(ch3, ch4, use_norm=norm)

        # Intensity (I) Stream Decoder
        self.ID_block3 = NormUpsample(ch4, ch3, use_norm=norm)
        self.ID_block2 = NormUpsample(ch3, ch2, use_norm=norm)
        self.ID_block1 = NormUpsample(ch2, ch1, use_norm=norm)
        self.ID_block0 = nn.Sequential(
            nn.ReplicationPad2d(1),
            nn.Conv2d(ch1, 1, 3, stride=1, padding=0, bias=False),
        )

        # Bidirectional Cross-Attention Modules (HV-I interaction)
        self.HV_LCA1 = HV_LCA(ch2, head2)
        self.HV_LCA2 = HV_LCA(ch3, head3)
        self.HV_LCA3 = HV_LCA(ch4, head4)
        self.HV_LCA4 = HV_LCA(ch4, head4)
        self.HV_LCA5 = HV_LCA(ch3, head3)
        self.HV_LCA6 = HV_LCA(ch2, head2)

        self.I_LCA1 = I_LCA(ch2, head2)
        self.I_LCA2 = I_LCA(ch3, head3)
        self.I_LCA3 = I_LCA(ch4, head4)
        self.I_LCA4 = I_LCA(ch4, head4)
        self.I_LCA5 = I_LCA(ch3, head3)
        self.I_LCA6 = I_LCA(ch2, head2)

        # Optional Adaptive Neural Curve Adjustment
        if self.use_curve:
            self.i_curve = NeuralCurveLayer(in_channels=ch1, M=curve_M, num_curves=1)

        # Output-Domain Residual Refiner
        if self.use_rgb_refiner:
            self.rgb_refiner = RGBRefiner(in_channels=3, mid_channels=refiner_mid_ch)

    def forward(self, x):
        """
        Forward execution of DualSpaceCIDNet.

        Args:
            x (torch.Tensor): Degraded sRGB image tensor [B, 3, H, W] normalized in [0, 1].

        Returns:
            torch.Tensor: Restored sRGB image tensor [B, 3, H, W] in [0, 1].
        """
        dtypes = x.dtype

        # Phase 1: Decoupling into Continuous HVI Space
        hvi = self.trans.HVIT(x)
        i = hvi[:, 2, :, :].unsqueeze(1).to(dtypes)

        # Intensity stream encoding
        i_enc0 = self.IE_block0(i)
        i_enc1 = self.IE_block1(i_enc0)

        # Chromaticity stream encoding
        hv_0 = self.HVE_block0(hvi)
        hv_1 = self.HVE_block1(hv_0)

        i_jump0 = i_enc0
        hv_jump0 = hv_0

        # Stage 1 Cross-attention interaction
        i_enc2 = self.I_LCA1(i_enc1, hv_1)
        hv_2 = self.HV_LCA1(hv_1, i_enc1)
        v_jump1 = i_enc2
        hv_jump1 = hv_2

        # Downsampling stage 2
        i_enc2 = self.IE_block2(i_enc2)
        hv_2 = self.HVE_block2(hv_2)

        # Stage 2 Cross-attention interaction
        i_enc3 = self.I_LCA2(i_enc2, hv_2)
        hv_3 = self.HV_LCA2(hv_2, i_enc2)
        v_jump2 = i_enc3
        hv_jump2 = hv_3

        # Bottleneck stage
        i_enc3 = self.IE_block3(i_enc2)
        hv_3 = self.HVE_block3(hv_2)

        i_enc4 = self.I_LCA3(i_enc3, hv_3)
        hv_4 = self.HV_LCA3(hv_3, i_enc3)

        i_dec4 = self.I_LCA4(i_enc4, hv_4)
        hv_4 = self.HV_LCA4(hv_4, i_enc4)

        # Decoding stage
        hv_3 = self.HVD_block3(hv_4, hv_jump2)
        i_dec3 = self.ID_block3(i_dec4, v_jump2)

        i_dec2 = self.I_LCA5(i_dec3, hv_3)
        hv_2 = self.HV_LCA5(hv_3, i_dec3)

        hv_2 = self.HVD_block2(hv_2, hv_jump1)
        i_dec2 = self.ID_block2(i_dec3, v_jump1)

        i_dec1 = self.I_LCA6(i_dec2, hv_2)
        hv_1 = self.HV_LCA6(hv_2, i_dec2)

        i_dec1 = self.ID_block1(i_dec1, i_jump0)
        i_dec0 = self.ID_block0(i_dec1)

        # Optional Neural Curve mapping on Intensity channel
        if self.use_curve:
            i_normalized = torch.clamp(i_dec0, 0.0, 1.0)
            i_dec0 = self.i_curve(i_dec1, i_normalized)

        hv_1 = self.HVD_block1(hv_1, hv_jump0)
        hv_0 = self.HVD_block0(hv_1)

        # Reconstruct coarse sRGB image via inverse HVI transformation
        output_hvi = torch.cat([hv_0, i_dec0], dim=1) + hvi
        coarse_rgb = self.trans.PHVIT(output_hvi)

        # Phase 2: Output-Domain Residual Refinement
        if self.use_rgb_refiner:
            output_rgb = self.rgb_refiner(coarse_rgb)
        else:
            output_rgb = coarse_rgb

        output_rgb = torch.clamp(output_rgb, 0.0, 1.0)
        return output_rgb

    def HVIT(self, x):
        """
        Utility method to extract HVI representation.

        Args:
            x (torch.Tensor): sRGB image tensor [B, 3, H, W].

        Returns:
            torch.Tensor: HVI representation [B, 3, H, W].
        """
        return self.trans.HVIT(x)


if __name__ == '__main__':
    batch_size = 2
    height, width = 256, 256
    x = torch.clamp(torch.randn(batch_size, 3, height, width), 0.0, 1.0)

    # Sanity check: DualSpaceCIDNet with RGB Refiner
    model = DualSpaceCIDNet(use_rgb_refiner=True, refiner_mid_ch=32)
    out = model(x)
    total_params = sum(p.numel() for p in model.parameters())

    assert out.shape == x.shape and isinstance(out, torch.Tensor)
    print(f"[OK] DualSpaceCIDNet forward pass successful. Output shape: {out.shape}, Total params: {total_params:,}")
