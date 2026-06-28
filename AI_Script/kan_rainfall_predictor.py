"""
kan_rainfall_predictor.py
==========================
KAN (Kolmogorov-Arnold Network) rainfall predictor for QPE.

Fully self-contained — no external KAN libraries required.
Implements B-spline KAN layers from scratch using only PyTorch.

Based on:
  "KAN: Kolmogorov-Arnold Networks" (Liu et al., 2024)
  "Improving Precipitation Estimation Accuracy with Knowledge-Aware Networks"
  Remote Sensing 16(24), 4713 (2024)

Architecture
------------
    Input    : (N, 32)   CNN feature vector
    Hidden 1 : (N, 64)   KAN layer  [32 -> 64]
    Hidden 2 : (N, 32)   KAN layer  [64 -> 32]
    Output   : (N,  1)   KAN layer  [32 ->  1]  log1p(R) space

KAN Layer
---------
    Each connection i -> j computes:
        phi_{ij}(x) = w_base * SiLU(x) + w_spline * spline(x)

    spline(x) is a B-spline of order 3 (cubic) on a uniform grid
    of grid_size intervals, evaluated using the Cox-de Boor recursion.

    All spline coefficients, base weights, and scale factors are
    learnable parameters updated via backprop.

Target transformation
---------------------
    y_train  = log1p(R)  = log(1 + R)   during training
    y_pred   = expm1(out) = exp(out) - 1  at inference

Regularisation
--------------
    L1 on spline weights per layer, weighted by lambda_l1.
    Add to training loss: total = mse + model.regularisation_loss()

Parameter counts
----------------
    KAN layer 32->64 :  20,544
    KAN layer 64->32 :  20,512
    KAN layer 32-> 1 :     321
    Total KAN        :  41,377
    CNN (RadarCNN)   :  63,040
    Combined         : 104,417

Dependencies
------------
    pip install torch
"""

import torch
import torch.nn as nn
import math
from typing import Tuple


# ---------------------------------------------------------------------------
# B-spline basis evaluation
# ---------------------------------------------------------------------------

def b_spline_basis(x: torch.Tensor,
                   grid: torch.Tensor,
                   spline_order: int) -> torch.Tensor:
    """
    Evaluate B-spline basis functions using the Cox-de Boor recursion.

    Parameters
    ----------
    x            : (N, in_features)        input values
    grid         : (in_features, G+2k+1)   extended knot grid per input feature
    spline_order : int                     polynomial order (3 = cubic)

    Returns
    -------
    basis : (N, in_features, G + spline_order)
            B-spline basis values for each input and each basis function
    """
    # x: (N, in) -> (N, in, 1) for broadcasting against grid
    x = x.unsqueeze(-1)   # (N, in, 1)

    # Order-0 basis: indicator function for each grid interval
    # grid has shape (in, num_knots), num_knots = G + 2k + 1
    # basis_0[n, i, j] = 1 if grid[i,j] <= x[n,i] < grid[i,j+1]
    basis = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).float()
    # basis: (N, in, num_knots - 1)

    # Cox-de Boor recursion up to desired order
    for k in range(1, spline_order + 1):
        # Left term:  (x - t_j)   / (t_{j+k}   - t_j)   * B_{j,k-1}
        # Right term: (t_{j+k+1} - x) / (t_{j+k+1} - t_{j+1}) * B_{j+1,k-1}
        t_left  = grid[:, :-(k + 1)]   # (in, num_knots - k - 1)
        t_right = grid[:, k:-1]        # (in, num_knots - k - 1)
        t_left2 = grid[:, 1:-k]        # (in, num_knots - k - 1)
        t_right2= grid[:, k + 1:]      # (in, num_knots - k - 1)

        denom_l = (t_right  - t_left).clamp(min=1e-8)
        denom_r = (t_right2 - t_left2).clamp(min=1e-8)

        coeff_l = (x - t_left)  / denom_l   # (N, in, *)
        coeff_r = (t_right2 - x) / denom_r  # (N, in, *)

        basis = coeff_l * basis[:, :, :-1] + coeff_r * basis[:, :, 1:]

    return basis   # (N, in, G + spline_order)


# ---------------------------------------------------------------------------
# KANLinear — single KAN layer
# ---------------------------------------------------------------------------

class KANLinear(nn.Module):
    """
    Single KAN layer implementing the Kolmogorov-Arnold representation.

    Each connection i->j learns a 1D spline function:
        phi_{ij}(x_i) = w_base_{ij} * SiLU(x_i)
                      + w_spline_{ij} * sum_k( c_{ijk} * B_k(x_i) )

    Parameters
    ----------
    in_features  : int   number of input features
    out_features : int   number of output features
    grid_size    : int   number of spline grid intervals (default 5)
    spline_order : int   B-spline polynomial order (default 3 = cubic)
    grid_range   : tuple input range for grid initialisation (default [-1, 1])
    """

    def __init__(self,
                 in_features:  int,
                 out_features: int,
                 grid_size:    int   = 5,
                 spline_order: int   = 3,
                 grid_range:   Tuple = (-1.0, 1.0)) -> None:
        super().__init__()

        self.in_features  = in_features
        self.out_features = out_features
        self.grid_size    = grid_size
        self.spline_order = spline_order

        # Number of B-spline basis functions per input feature
        # = grid_size + spline_order
        n_basis = grid_size + spline_order

        # Uniform knot grid extended on both ends for boundary handling
        # Shape: (in_features, grid_size + 2*spline_order + 1)
        grid_knots = torch.linspace(
            grid_range[0] - spline_order * (grid_range[1] - grid_range[0]) / grid_size,
            grid_range[1] + spline_order * (grid_range[1] - grid_range[0]) / grid_size,
            grid_size + 2 * spline_order + 1,
        )
        grid = grid_knots.unsqueeze(0).expand(in_features, -1)
        self.register_buffer("grid", grid.clone())

        # Learnable spline coefficients: (out, in, n_basis)
        self.spline_weight = nn.Parameter(
            torch.empty(out_features, in_features, n_basis)
        )

        # Learnable base weight (SiLU branch): (out, in)
        self.base_weight = nn.Parameter(
            torch.empty(out_features, in_features)
        )

        # Per-output scale for spline contribution
        self.spline_scaler = nn.Parameter(
            torch.empty(out_features, in_features)
        )

        # Bias
        self.bias = nn.Parameter(torch.zeros(out_features))

        # Base activation: SiLU = x * sigmoid(x)  (matches paper exactly)
        self.base_activation = nn.SiLU()

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.kaiming_uniform_(self.base_weight,   a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.spline_weight, a=math.sqrt(5))
        nn.init.ones_(self.spline_scaler)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (N, in_features)

        Returns
        -------
        (N, out_features)
        """
        # ── Base (SiLU) branch ──────────────────────────────────────────────
        # base_output[n, j] = sum_i w_base[j,i] * SiLU(x[n,i])
        base_out = self.base_activation(x)                       # (N, in)
        base_out = base_out @ self.base_weight.t()               # (N, out)

        # ── Spline branch ───────────────────────────────────────────────────
        # Evaluate B-spline basis for all input features
        basis = b_spline_basis(x, self.grid, self.spline_order)
        # basis: (N, in, n_basis)

        # Scaled spline weights: w_spline * scaler
        scaled_w = self.spline_weight * self.spline_scaler.unsqueeze(-1)
        # scaled_w: (out, in, n_basis)

        # spline_out[n, j] = sum_i sum_k scaled_w[j,i,k] * basis[n,i,k]
        # Use einsum for clarity: n=batch, i=in, k=basis, j=out
        spline_out = torch.einsum("nik,jik->nj", basis, scaled_w)  # (N, out)

        return base_out + spline_out + self.bias

    def l1_loss(self) -> torch.Tensor:
        """L1 regularisation on spline weights."""
        return self.spline_weight.abs().mean()


# ---------------------------------------------------------------------------
# RainfallKAN
# ---------------------------------------------------------------------------

class RainfallKAN(nn.Module):
    """
    KAN rainfall predictor: (N, 32) -> (N, 1) in log1p space.

    Architecture
    ------------
        KANLinear(32 -> 64)
        KANLinear(64 -> 32)
        KANLinear(32 ->  1)

    Parameters
    ----------
    input_dim    : int   CNN output dimension (default 32)
    hidden1      : int   first hidden width   (default 64)
    hidden2      : int   second hidden width  (default 32)
    grid_size    : int   B-spline intervals   (default 5)
    spline_order : int   B-spline order       (default 3)
    """

    def __init__(self,
                 input_dim:    int = 32,
                 hidden1:      int = 64,
                 hidden2:      int = 32,
                 grid_size:    int = 5,
                 spline_order: int = 3) -> None:
        super().__init__()

        self.layer1 = KANLinear(input_dim, hidden1, grid_size, spline_order)
        self.layer2 = KANLinear(hidden1,   hidden2, grid_size, spline_order)
        self.layer3 = KANLinear(hidden2,   1,       grid_size, spline_order)

        self._input_dim = input_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (N, 32)  normalised CNN feature vector

        Returns
        -------
        (N, 1)  log1p(R) prediction
        """
        h = self.layer1(x)
        h = self.layer2(h)
        h = self.layer3(h)
        return h

    def predict_mmh(self, x: torch.Tensor) -> torch.Tensor:
        """
        End-to-end inference in mm/h.
        Applies inverse log1p: R = exp(output) - 1

        Parameters
        ----------
        x : (N, 32)

        Returns
        -------
        (N,)  rain rate in mm/h  (>= 0)
        """
        with torch.no_grad():
            log1p_pred = self.forward(x)
            rain_pred  = torch.expm1(log1p_pred).squeeze(1)
            return torch.clamp(rain_pred, min=0.0)

    def regularisation_loss(self, lambda_l1: float = 1e-5) -> torch.Tensor:
        """
        L1 regularisation on spline weights across all layers.
        Add to training loss: total_loss = mse_loss + model.regularisation_loss()
        """
        return lambda_l1 * (
            self.layer1.l1_loss() +
            self.layer2.l1_loss() +
            self.layer3.l1_loss()
        )

    @property
    def input_dim(self) -> int:
        return self._input_dim


# ---------------------------------------------------------------------------
# Target transformation helpers
# ---------------------------------------------------------------------------

def transform_target(y: torch.Tensor) -> torch.Tensor:
    """log1p transform: log(1 + R)  ->  compresses right-skewed distribution."""
    return torch.log1p(y)


def inverse_transform_target(y_pred: torch.Tensor) -> torch.Tensor:
    """Inverse log1p: exp(y_pred) - 1  ->  back to mm/h."""
    return torch.clamp(torch.expm1(y_pred), min=0.0)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("RainfallKAN — sanity check")
    print("=" * 50)

    model = RainfallKAN(
        input_dim    = 32,
        hidden1      = 64,
        hidden2      = 32,
        grid_size    = 5,
        spline_order = 3,
    )

    total_params = count_parameters(model)
    print(f"Trainable parameters: {total_params:,}")

    # Forward pass
    torch.manual_seed(0)
    batch = torch.randn(16, 32)
    log1p_out = model(batch)
    assert log1p_out.shape == (16, 1), f"Bad shape: {log1p_out.shape}"
    print(f"Forward pass:  (16, 32) -> {log1p_out.shape}  OK")

    # predict_mmh
    rain = model.predict_mmh(batch)
    assert rain.shape == (16,)
    assert (rain >= 0).all()
    print(f"predict_mmh:   {rain.shape}  "
          f"min={rain.min():.3f}  max={rain.max():.3f} mm/h  OK")

    # Regularisation
    reg = model.regularisation_loss(lambda_l1=1e-5)
    assert reg.shape == torch.Size([])
    print(f"Reg loss:      {reg.item():.8f}  OK")

    # Target transform roundtrip
    y      = torch.tensor([1.0, 5.0, 10.0, 50.0, 100.0])
    y_t    = transform_target(y)
    y_back = inverse_transform_target(y_t)
    assert torch.allclose(y, y_back, atol=1e-5)
    print(f"log1p roundtrip: {y.tolist()} -> OK")

    # Gradient flow
    loss = log1p_out.sum() + model.regularisation_loss()
    loss.backward()
    n_grads = sum(1 for p in model.parameters() if p.grad is not None)
    print(f"Gradient flow: {n_grads} parameter tensors  OK")

    print()
    print("=" * 50)
    print("ALL CHECKS PASSED")
    print()
    print(f"CNN parameters : 63,040")
    print(f"KAN parameters : {total_params:,}")
    print(f"Combined       : {63040 + total_params:,}")