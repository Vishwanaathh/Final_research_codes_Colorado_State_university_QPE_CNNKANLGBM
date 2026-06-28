"""
cnn_feature_extractor.py
=========================
Shallow 4-layer CNN feature extractor for QPE.

Takes a T_{4x9x9} dual-polarisation radar tensor and produces a compact
32-dimensional feature vector for input to the KAN rainfall predictor.

Input
-----
    (N, 4, 9, 9)  float32  -- normalised, per-channel z-score
    Channels: [Z_low, ZDR_low, Z_high, ZDR_high]

Output
------
    (N, 32)  float32  -- feature vector for KAN

Architecture
------------
    Conv1 : Conv2d(4,  16, 3x3, pad=1) + BatchNorm2d + GELU  -> (N, 16, 9, 9)
    Conv2 : Conv2d(16, 32, 3x3, pad=1) + BatchNorm2d + GELU  -> (N, 32, 9, 9)
    Conv3 : Conv2d(32, 64, 3x3, pad=1) + BatchNorm2d + GELU  -> (N, 64, 9, 9)
    Dropout2d(p=0.25)
    Conv4 : Conv2d(64, 64, 3x3, pad=1) + BatchNorm2d + GELU  -> (N, 64, 9, 9)
    Dropout2d(p=0.25)
    SE    : SqueezeExcitation(64, reduction=16)               -> (N, 64, 9, 9)
    GlobalAvgPool2d                                           -> (N, 64)
    Concat(GAP output, center_pixel)                          -> (N, 68)
    Linear(68, 32) + LayerNorm(32) + GELU                     -> (N, 32)

Design decisions
----------------
  4 conv layers, padding=1 throughout:
    The spatial dimension stays 9x9 across all conv layers. Downsampling
    early on a 9x9 input would discard too much spatial context. Global
    average pooling at the end summarises spatial structure without forcing
    the network to commit to a specific spatial location.

  GELU activations (not ReLU):
    GELU is smooth and differentiable everywhere. This produces smoother
    feature representations which help the downstream KAN learn cleaner
    spline activations. ReLU's hard zero-cutoff can create discontinuities
    that propagate into the KAN's input space.

  Center-pixel skip connection:
    After global average pooling, the 4 normalised values at the exact
    center pixel (ray=4, gate=4 in the 9x9 window -- the gauge co-located
    gate) are concatenated before the final projection. This guarantees
    that the precise Z and ZDR values at the gauge location are never
    diluted by spatial averaging, which is critical for QPE accuracy.
    The network can learn to combine local spatial context (from GAP)
    with the exact point measurement (from center pixel).

  Progressive channel expansion 4->16->32->64->64:
    Each layer learns increasingly abstract spatial patterns. The final
    64-channel representation captures multi-scale Z/ZDR structure across
    the 9x9 neighbourhood before being compressed to 32 features.

  Squeeze-and-Excitation (SE) block after Conv4:
    Before global average pooling, an SE block learns to reweight the 64
    feature channels based on their global importance for the current sample.
    The block computes a per-channel scaling vector:
        GAP(h) -> FC(64->4) -> ReLU -> FC(4->64) -> Sigmoid -> scale h
    The reduction ratio of 16 (64/4=16) keeps the block lightweight (~600
    params). Effect: for heavy convective rain the network upweights channels
    sensitive to high-Z patterns; for stratiform rain it upweights ZDR-
    sensitive channels. This is a well-established technique in remote
    sensing CNNs with negligible parameter cost.

  LayerNorm on projection output:
    The KAN B-spline grid is initialised over [-1, 1]. Without normalisation
    the CNN projection can output values well outside this range, forcing the
    splines to extrapolate rather than interpolate. LayerNorm(32) applied
    after the linear projection keeps the KAN input distribution centred and
    bounded, ensuring the splines operate in their well-defined regime.

  Output dimension 32:
    KAN spline parameters scale quadratically with input dimension.
    32 inputs keeps the KAN compact (~20K training samples available)
    while providing enough features to represent the R(Z,ZDR) mapping.

  BatchNorm momentum=0.01 (not default 0.1):
    With ~20 batches per epoch, default momentum=0.1 shifts the running
    mean/var by 10% per batch, making BN statistics unstable on small
    datasets. momentum=0.01 accumulates statistics over ~100 batches,
    producing stable, representative running statistics.

  ~63K trainable parameters:
    Appropriate for a dataset of ~20K samples. Large enough to capture
    spatial patterns, small enough to avoid overfitting without heavy
    regularisation.

  Dropout (p=0.25) after Conv3 and Conv4:
    Spatial dropout applied after the deeper conv layers where overfitting
    is most likely. Not applied to Conv1/Conv2 — early layers learn basic
    edge/gradient features that should be stable. p=0.25 is conservative
    enough to avoid underfitting while providing meaningful regularisation
    for a ~13K sample dataset.

Usage
-----
    from cnn_feature_extractor import RadarCNN

    model = RadarCNN()
    features = model(x)   # x: (N, 4, 9, 9) -> features: (N, 32)
"""

import torch
import torch.nn as nn
from typing import Optional


# Center pixel index in the 9x9 window (HALF_WIN=4, so center = index 4)
CENTER_IDX = 4


class ConvBlock(nn.Module):
    """Conv2d + BatchNorm2d + GELU. Reused across all 4 layers."""

    def __init__(self, in_ch: int, out_ch: int,
                 kernel_size: int = 3, padding: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size,
                      padding=padding, bias=False),
            nn.BatchNorm2d(out_ch, momentum=0.01),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SqueezeExcitation(nn.Module):
    """
    Channel attention block (Hu et al., 2018 — Squeeze-and-Excitation Networks).

    Learns a per-channel scaling vector from global context:
        GAP -> FC(C -> C//reduction) -> ReLU -> FC(C//reduction -> C) -> Sigmoid

    The output scales each channel of the input feature map, effectively
    letting the network upweight informative channels and suppress noise.

    Parameters
    ----------
    channels  : int  number of input channels (64 in our CNN)
    reduction : int  bottleneck reduction ratio (default 16 -> 64//16 = 4)
    """

    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        bottleneck = max(channels // reduction, 4)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),           # (N, C, H, W) -> (N, C, 1, 1)
            nn.Flatten(),                      # (N, C)
            nn.Linear(channels, bottleneck),   # (N, C//reduction)
            nn.ReLU(inplace=True),
            nn.Linear(bottleneck, channels),   # (N, C)
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # scale: (N, C) -> (N, C, 1, 1) for broadcasting
        scale = self.se(x).unsqueeze(-1).unsqueeze(-1)
        return x * scale


class RadarCNN(nn.Module):
    """
    4-layer shallow CNN feature extractor for T_{4x9x9} radar tensors.

    Parameters
    ----------
    in_channels : int
        Number of input channels (default 4: Z_low, ZDR_low, Z_high, ZDR_high).
    feature_dim : int
        Output feature vector dimension (default 32, KAN input size).
    dropout_p : float
        Dropout probability applied after conv3 and conv4 (default 0.25).
        Set to 0.0 to disable dropout entirely.
    se_reduction : int
        Channel reduction ratio for the SE block (default 16).
        Lower values = more SE parameters; higher = fewer.

    Forward input
    -------------
    x : torch.Tensor  shape (N, 4, 9, 9)  float32  normalised

    Forward output
    --------------
    torch.Tensor  shape (N, 32)  float32
    """

    def __init__(self,
                 in_channels:  int   = 4,
                 feature_dim:  int   = 32,
                 dropout_p:    float = 0.25,
                 se_reduction: int   = 16) -> None:
        super().__init__()

        # 4 convolutional layers -- spatial dims stay 9x9 throughout
        self.conv1 = ConvBlock(in_channels, 16)   # (N,  4, 9, 9) -> (N, 16, 9, 9)
        self.conv2 = ConvBlock(16, 32)             # (N, 16, 9, 9) -> (N, 32, 9, 9)
        self.conv3 = ConvBlock(32, 64)             # (N, 32, 9, 9) -> (N, 64, 9, 9)
        self.conv4 = ConvBlock(64, 64)             # (N, 64, 9, 9) -> (N, 64, 9, 9)

        # Spatial dropout after deep conv layers (regularisation)
        # Applied after conv3 and conv4 where overfitting risk is highest
        self.drop3 = nn.Dropout2d(p=dropout_p)
        self.drop4 = nn.Dropout2d(p=dropout_p)

        # Squeeze-and-Excitation: channel attention after conv4
        self.se = SqueezeExcitation(channels=64, reduction=se_reduction)

        # Global average pool: summarise spatial structure
        self.gap = nn.AdaptiveAvgPool2d(1)         # (N, 64, 9, 9) -> (N, 64, 1, 1)

        # Projection: GAP(64) + center_pixel(in_channels) -> feature_dim
        # LayerNorm normalises KAN input to stay within spline grid [-1, 1]
        self.project = nn.Sequential(
            nn.Linear(64 + in_channels, feature_dim, bias=True),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
        )

        self._in_channels = in_channels
        self._feature_dim = feature_dim
        self._dropout_p    = dropout_p
        self._se_reduction = se_reduction

        # Weight initialisation
        self._init_weights()

    def _init_weights(self) -> None:
        """
        Kaiming initialisation for conv layers (appropriate for GELU).
        Zero-initialise BN biases, one-initialise BN weights.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (N, 4, 9, 9)  normalised radar tensor

        Returns
        -------
        (N, 32)  feature vector
        """
        # Extract center pixel values before conv processing
        # center_pixel shape: (N, 4)
        center_pixel = x[:, :, CENTER_IDX, CENTER_IDX]

        # 4 conv layers
        h = self.conv1(x)
        h = self.conv2(h)
        h = self.conv3(h)
        h = self.drop3(h)
        h = self.conv4(h)
        h = self.drop4(h)

        # Channel attention: reweight features by importance
        h = self.se(h)

        # Global average pool -> (N, 64)
        h = self.gap(h).flatten(1)

        # Concatenate with center pixel -> (N, 68)
        h = torch.cat([h, center_pixel], dim=1)

        # Project to feature_dim -> (N, 32)
        features = self.project(h)

        return features

    @property
    def output_dim(self) -> int:
        """Feature vector dimension (KAN input size)."""
        return self._feature_dim


# ---------------------------------------------------------------------------
# Utility: parameter count
# ---------------------------------------------------------------------------

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import torch

    model = RadarCNN(in_channels=4, feature_dim=32)
    print(model)
    print(f"\nTrainable parameters: {count_parameters(model):,}")

    # Forward pass test
    batch = torch.randn(8, 4, 9, 9)
    features = model(batch)
    print(f"\nInput shape:   {batch.shape}")
    print(f"Output shape:  {features.shape}")
    assert features.shape == (8, 32), f"Unexpected shape: {features.shape}"
    print("\nForward pass: OK")

    # Gradient flow test
    loss = features.sum()
    loss.backward()
    grad_norms = [
        p.grad.norm().item()
        for p in model.parameters()
        if p.grad is not None
    ]
    assert all(g > 0 for g in grad_norms), "Zero gradients detected"
    print("Gradient flow: OK")
    print(f"\nAll checks passed.")
    from torchview import draw_graph
    graph = draw_graph(model, input_size=(1, 4, 9, 9), expand_nested=True)
    graph.visual_graph.render('CNN', '../images/', format='png')