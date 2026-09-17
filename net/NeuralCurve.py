# -*- coding: utf-8 -*-
"""
Neural Curve Layer Module.
Implements dynamic piecewise-linear illumination mapping via image-adaptive curve estimation.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def sgn_m(x):
    """
    Piecewise linear basis function delta(x).

    Args:
        x (torch.Tensor): Input tensor.

    Returns:
        torch.Tensor: Bounded tensor clamped to [0, 1].
    """
    return torch.clamp(x, 0.0, 1.0)


def piece_function(x, curve_params, M):
    """
    Evaluates piecewise-linear mapping S(x) with M control points.

    Args:
        x (torch.Tensor): Input feature/image channel [B, 1, H, W] in range [0, 1].
        curve_params (torch.Tensor): Estimated curve control points [B, M].
        M (int): Number of control points.

    Returns:
        torch.Tensor: Mapped tensor of shape [B, 1, H, W].
    """
    b, c, h, w = x.shape
    r = curve_params[:, 0].view(b, 1, 1, 1).expand(b, c, h, w)

    # Accumulate piecewise linear segments: r = k_0 + sum((k_{i+1} - k_i) * delta(M * x - i))
    for i in range(M - 1):
        slope = (curve_params[:, i + 1] - curve_params[:, i]).view(b, 1, 1, 1).expand(b, c, h, w)
        delta = sgn_m(M * x - i)
        r = r + slope * delta

    return r


class NeuralCurveLayer(nn.Module):
    """
    Neural Curve Adjustment Layer.
    Predicts adaptive piecewise-linear illumination mapping parameters dynamically from deep features.
    """
    def __init__(self, in_channels, M=11, num_curves=1):
        super().__init__()
        self.M = M
        self.num_curves = num_curves

        # Global feature pooling and curve parameter regression
        self.curve_predictor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, num_curves * M),
            nn.Sigmoid()
        )

        self._init_identity()

    def _init_identity(self):
        """
        Initializes curve parameters to approximate an identity mapping at training onset.
        """
        last_linear = self.curve_predictor[-2]
        nn.init.zeros_(last_linear.weight)

        identity_curve = torch.linspace(0, 1, self.M)
        identity_bias = identity_curve.repeat(self.num_curves)
        identity_bias = torch.log(identity_bias / (1.0 - identity_bias + 1e-6))
        last_linear.bias.data = identity_bias

    def forward(self, feat, img_channel, return_curve=False):
        """
        Applies predicted curves to target image channels.

        Args:
            feat (torch.Tensor): Conditioning feature map [B, in_channels, H, W].
            img_channel (torch.Tensor): Target image channel [B, num_curves, H, W] in [0, 1].
            return_curve (bool): Whether to return estimated control points.

        Returns:
            torch.Tensor or tuple: Adjusted image channel [B, num_curves, H, W] (and curve points if requested).
        """
        b = feat.shape[0]
        curve_params = self.curve_predictor(feat).view(b, self.num_curves, self.M)

        outputs = []
        for i in range(self.num_curves):
            ch_in = img_channel[:, i:i + 1, :, :]
            curve = curve_params[:, i, :]
            ch_out = piece_function(ch_in, curve, self.M)
            outputs.append(ch_out)

        output = torch.cat(outputs, dim=1)
        output = torch.clamp(output, 0.0, 1.0)

        if return_curve:
            return output, curve_params
        return output


if __name__ == '__main__':
    batch_size = 2
    height, width = 128, 128
    in_channels = 36

    feat = torch.randn(batch_size, in_channels, height // 4, width // 4)
    i_channel = torch.rand(batch_size, 1, height, width)

    curve_layer = NeuralCurveLayer(in_channels=in_channels, M=11, num_curves=1)
    i_out, curve_params = curve_layer(feat, i_channel, return_curve=True)
    assert i_out.shape == i_channel.shape
    print("[OK] NeuralCurveLayer sanity check passed successfully.")
