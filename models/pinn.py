"""
Physics-Informed Neural Network for aluminium extrusion.

Architecture
------------
  Input  : [billet_temperature_C, container_temperature_C, die_temperature_C,
             ram_speed_mm_per_sec, billet_diameter_mm, extrusion_ratio,
             density, specific_heat, thermal_conductivity,
             activation_energy, A_constant, alpha_constant, stress_exponent]
  Hidden : N fully-connected layers with Tanh activation
  Output : [exit_temperature_C, ram_pressure_MPa, exit_speed_mm_per_sec, surface_quality_score]
           (all normalised to [0,1])

Loss
----
  L_total = w_data * L_data  +  w_phys * L_physics  +  w_bc * L_bc

  L_data    — MSE between predictions and (noisy) mock sensor values
  L_physics — residuals of the governing equations evaluated on collocation points
  L_bc      — volume-conservation hard constraint (v_exit = v_ram * R)
"""

import torch
import torch.nn as nn
import numpy as np

R_GAS = 8.314   # J/(mol·K)


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

class ExtrusionPINN(nn.Module):
    """Fully-connected PINN with Tanh activations."""

    def __init__(
        self,
        input_dim:  int = 13,   # 6 process vars + 7 material properties
        hidden_dim: int = 128,
        n_layers:   int = 6,
        output_dim: int = 4,    # T_exit, P, v_exit, surface_quality
    ):
        super().__init__()

        layers = [nn.Linear(input_dim, hidden_dim), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.Tanh()]
        layers.append(nn.Linear(hidden_dim, output_dim))

        self.net = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ---------------------------------------------------------------------------
# Physics residuals  (operate on raw / un-normalised values)
# ---------------------------------------------------------------------------

def compute_physics_residuals(
    inputs_raw: torch.Tensor,
    outputs_raw: torch.Tensor,
) -> dict:
    """
    Compute residuals of the governing equations.

    inputs_raw columns (un-normalised):
      0  billet_temperature_C
      1  container_temperature_C
      2  die_temperature_C
      3  ram_speed_mm_per_sec
      4  billet_diameter_mm
      5  extrusion_ratio
      6  density           (kg/m³)
      7  specific_heat     (J/kg·K)
      8  thermal_conductivity (W/m·K)
      9  activation_energy (J/mol)
      10 A_constant        (s^-1)
      11 alpha_constant    (MPa^-1)
      12 stress_exponent

    outputs_raw columns (un-normalised):
      0  exit_temperature_C
      1  ram_pressure_MPa
      2  exit_speed_mm_per_sec
      3  surface_quality_score

    Returns dict of scalar residual tensors (each should be ≈ 0).
    """
    # Unpack inputs
    T_billet    = inputs_raw[:, 0]
    T_container = inputs_raw[:, 1]
    v_ram_mm_s  = inputs_raw[:, 3]
    D_billet_mm = inputs_raw[:, 4]
    R_ratio     = inputs_raw[:, 5]
    rho         = inputs_raw[:, 6]
    cp          = inputs_raw[:, 7]
    Q_act       = inputs_raw[:, 9]
    A           = inputs_raw[:, 10]
    alpha       = inputs_raw[:, 11]
    n           = inputs_raw[:, 12]

    # Unpack outputs (network predictions)
    T_exit_pred = outputs_raw[:, 0]
    P_pred      = outputs_raw[:, 1]
    v_exit_pred = outputs_raw[:, 2]

    # Convert units
    v_ram       = v_ram_mm_s  / 1000.0          # mm/s → m/s
    D_billet    = D_billet_mm / 1000.0          # mm   → m
    T_billet_K  = T_billet    + 273.15
    T_cont_K    = T_container  + 273.15
    T_mean_K    = 0.85 * T_billet_K + 0.15 * T_cont_K

    # --- Intermediate physics (analytic, no learnable params) ---

    # Mean effective strain rate (s^-1)
    eps_dot = (6.0 * v_ram * torch.log(R_ratio)) / (torch.pi * D_billet)

    # Zener-Hollomon parameter
    Z = eps_dot * torch.exp(Q_act / (R_GAS * T_mean_K))

    # Flow stress via Sellars-Tegart (MPa)
    sigma = (1.0 / alpha) * torch.arcsinh((Z / A) ** (1.0 / n))

    # --- Physics-predicted target values ---

    eta = 0.9  # Taylor-Quinney coefficient
    eps_bar   = torch.log(R_ratio)  # total effective strain

    # Energy balance: expected exit temperature
    dT_def  = (eta * sigma * 1e6 * eps_bar) / (rho * cp)
    dT_loss = 5.0 + 0.02 * torch.clamp(T_billet - 400.0, min=0.0)
    T_exit_phys = T_mean_K - 273.15 + dT_def - dT_loss

    # Upper-bound pressure
    P_phys = sigma * (2.0 / (3.0 ** 0.5)) * eps_bar + 0.1 * sigma * 0.5

    # Volume conservation: v_exit = v_ram * R  (exact constraint)
    v_exit_phys = v_ram * 1000.0 * R_ratio    # m/s → mm/s * R

    # --- Residuals (prediction minus physics), normalised to similar scale ---
    r_temperature = (T_exit_pred - T_exit_phys) / 20.0    # scale: ~20°C
    r_pressure    = (P_pred      - P_phys)      / 20.0    # scale: ~20 MPa
    r_volume      = (v_exit_pred - v_exit_phys) / 100.0   # scale: ~100 mm/s

    return {
        'temperature': r_temperature,
        'pressure':    r_pressure,
        'volume':      r_volume,
    }


# ---------------------------------------------------------------------------
# Loss function
# ---------------------------------------------------------------------------

class PINNLoss(nn.Module):
    """
    Combined data + physics loss.

    Weights are tunable; physics weight is typically annealed
    upward during training once the data loss has converged.
    """

    def __init__(
        self,
        w_data: float = 1.0,
        w_phys: float = 0.1,
        w_bc:   float = 1.0,
    ):
        super().__init__()
        self.w_data = w_data
        self.w_phys = w_phys
        self.w_bc   = w_bc

    def forward(
        self,
        y_pred_norm:    torch.Tensor,    # network output (normalised)
        y_true_norm:    torch.Tensor,    # ground-truth (normalised)
        inputs_raw:     torch.Tensor,    # un-normalised inputs
        outputs_raw:    torch.Tensor,    # un-normalised network predictions
    ) -> tuple[torch.Tensor, dict]:
        """
        Returns
        -------
        total_loss : scalar tensor
        components : dict with individual loss values for logging
        """
        # Data loss
        L_data = nn.functional.mse_loss(y_pred_norm, y_true_norm)

        # Physics residuals
        residuals = compute_physics_residuals(inputs_raw, outputs_raw)

        L_temp   = residuals['temperature'].pow(2).mean()
        L_press  = residuals['pressure'].pow(2).mean()
        L_vol    = residuals['volume'].pow(2).mean()

        L_phys = L_temp + L_press
        L_bc   = L_vol

        total = (
            self.w_data * L_data
            + self.w_phys * L_phys
            + self.w_bc   * L_bc
        )

        components = {
            'total':       total.item(),
            'data':        L_data.item(),
            'phys_temp':   L_temp.item(),
            'phys_press':  L_press.item(),
            'bc_volume':   L_vol.item(),
        }

        return total, components
