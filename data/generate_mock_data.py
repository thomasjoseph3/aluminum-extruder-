"""
Mock data generator for aluminium extrusion process.

Uses real physics equations:
  - Sellars-Tegart (Garofalo) flow stress model
  - Zener-Hollomon parameter
  - Energy balance for exit temperature
  - Upper bound analysis for extrusion pressure

Outputs a CSV ready for PINN training.
"""

import numpy as np
import pandas as pd

R_GAS = 8.314  # J/(mol·K)

# Material constants for common extrusion alloys
# Source: literature values for hot deformation behaviour
ALLOY_PARAMS = {
    'AA6063': {
        'rho':   2700,       # kg/m³
        'cp':    900,        # J/(kg·K)
        'k':     167,        # W/(m·K)
        'Q_act': 149_776,    # J/mol  activation energy
        'A':     2.41e9,     # s^-1
        'alpha': 0.045,      # MPa^-1
        'n':     5.385,
        'eta':   0.9,        # Taylor-Quinney: fraction of plastic work → heat
    },
    'AA6061': {
        'rho':   2700,
        'cp':    896,
        'k':     152,
        'Q_act': 152_000,
        'A':     3.12e9,
        'alpha': 0.048,
        'n':     5.2,
        'eta':   0.9,
    },
    'AA7075': {
        'rho':   2810,
        'cp':    960,
        'k':     130,
        'Q_act': 156_000,
        'A':     4.57e9,
        'alpha': 0.050,
        'n':     4.8,
        'eta':   0.9,
    },
}


# ---------------------------------------------------------------------------
# Physics equations
# ---------------------------------------------------------------------------

def zener_hollomon(strain_rate: float, T_K: float, alloy: str) -> float:
    """Z = strain_rate * exp(Q / R*T)"""
    p = ALLOY_PARAMS[alloy]
    return strain_rate * np.exp(p['Q_act'] / (R_GAS * T_K))


def flow_stress_sellars_tegart(Z: float, alloy: str) -> float:
    """
    Sellars-Tegart (Garofalo) equation:
        sigma = (1/alpha) * arcsinh( (Z/A)^(1/n) )
    Returns flow stress in MPa.
    """
    p = ALLOY_PARAMS[alloy]
    return (1.0 / p['alpha']) * np.arcsinh((Z / p['A']) ** (1.0 / p['n']))


def mean_strain_rate(v_ram: float, D_billet: float, R_ratio: float) -> float:
    """
    Mean effective strain rate through the die (s^-1).
    Simplified formula: eps_dot = 6 * v_ram * ln(R) / (pi * D_billet)
    """
    return (6.0 * v_ram * np.log(R_ratio)) / (np.pi * D_billet)


def exit_temperature(
    T_billet: float,
    T_container: float,
    sigma_MPa: float,
    R_ratio: float,
    alloy: str,
) -> float:
    """
    Exit temperature via energy balance (°C).
    T_exit = T_mean + dT_deformation - dT_die_loss

    dT_deformation = eta * sigma * ln(R) / (rho * cp)
    """
    p = ALLOY_PARAMS[alloy]
    T_mean = 0.85 * T_billet + 0.15 * T_container

    eps_bar = np.log(R_ratio)  # total effective strain
    dT_def = (p['eta'] * sigma_MPa * 1e6 * eps_bar) / (p['rho'] * p['cp'])

    # Small empirical die-cooling loss
    dT_loss = 5.0 + 0.02 * max(0.0, T_billet - 400.0)

    return T_mean + dT_def - dT_loss


def extrusion_pressure(
    sigma_MPa: float,
    R_ratio: float,
    mu: float = 0.1,
) -> float:
    """
    Extrusion pressure via upper-bound analysis (MPa).
    P = sigma * (2/sqrt(3)) * ln(R)  +  friction term
    """
    P_deform  = sigma_MPa * (2.0 / np.sqrt(3.0)) * np.log(R_ratio)
    P_friction = mu * sigma_MPa * 0.5
    return P_deform + P_friction


def surface_quality_index(
    T_exit: float,
    v_exit_m_s: float,
    sigma_MPa: float,
    alloy: str,
) -> float:
    """
    Heuristic surface quality score 0–100 (higher = better).
    Penalises:
      - exit temperature outside optimal window
      - exit speed too high
      - flow stress too high (cracking risk)
    Optimal window varies by alloy.
    """
    # Optimal exit temps calibrated to match the energy-balance output range (470-520°C)
    optimal = {'AA6063': 495.0, 'AA6061': 505.0, 'AA7075': 420.0}
    T_opt = optimal.get(alloy, 495.0)

    temp_score   = np.exp(-((T_exit - T_opt) ** 2) / (2 * 30.0 ** 2))
    speed_score  = np.exp(-max(0.0, v_exit_m_s - 0.020) / 0.005)
    stress_score = np.exp(-max(0.0, sigma_MPa - 80.0)   / 20.0)

    return 100.0 * temp_score * speed_score * stress_score


# ---------------------------------------------------------------------------
# Dataset generation
# ---------------------------------------------------------------------------

def generate_mock_data(
    n_samples: int = 2000,
    alloys: list = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a synthetic extrusion dataset.

    Process parameter ranges are based on industrial practice for
    direct hot extrusion of aluminium alloys on a horizontal press.
    """
    if alloys is None:
        alloys = ['AA6063', 'AA6061', 'AA7075']

    np.random.seed(seed)
    records = []

    for _ in range(n_samples):
        alloy = np.random.choice(alloys)

        # --- Setpoint inputs (what the operator can control) ---
        T_billet    = np.random.uniform(420, 500)   # °C
        T_container = np.random.uniform(380, 460)   # °C
        T_die       = np.random.uniform(380, 460)   # °C
        v_ram       = np.random.uniform(1e-3, 8e-3) # m/s  (1–8 mm/s)

        # --- Die / billet geometry ---
        D_billet        = np.random.uniform(0.08, 0.15)  # m  (80–150 mm)
        extrusion_ratio = np.random.uniform(10, 50)       # dimensionless

        # --- Derived quantities ---
        T_billet_K  = T_billet + 273.15
        T_cont_K    = T_container + 273.15
        T_mean_K    = 0.85 * T_billet_K + 0.15 * T_cont_K

        eps_dot  = mean_strain_rate(v_ram, D_billet, extrusion_ratio)
        Z        = zener_hollomon(eps_dot, T_mean_K, alloy)
        sigma    = flow_stress_sellars_tegart(Z, alloy)

        T_exit   = exit_temperature(T_billet, T_container, sigma, extrusion_ratio, alloy)
        P        = extrusion_pressure(sigma, extrusion_ratio)
        v_exit   = v_ram * extrusion_ratio   # volume conservation (m/s)
        quality  = surface_quality_index(T_exit, v_exit, sigma, alloy)

        # --- Add realistic sensor noise ---
        noise = {
            'billet_temperature_C':    np.random.normal(0, 1.0),
            'container_temperature_C': np.random.normal(0, 1.0),
            'die_temperature_C':       np.random.normal(0, 1.0),
            'ram_speed_mm_per_sec':    np.random.normal(0, 0.05),
            'exit_temperature_C':      np.random.normal(0, 2.0),
            'ram_pressure_MPa':        np.random.normal(0, 1.5),
        }

        records.append({
            # --- Inputs (setpoints + geometry) ---
            'alloy_grade':              alloy,
            'billet_temperature_C':     round(T_billet    + noise['billet_temperature_C'],    2),
            'container_temperature_C':  round(T_container + noise['container_temperature_C'], 2),
            'die_temperature_C':        round(T_die       + noise['die_temperature_C'],       2),
            'ram_speed_mm_per_sec':     round(v_ram * 1000 + noise['ram_speed_mm_per_sec'],   3),
            'billet_diameter_mm':       round(D_billet * 1000,                                2),
            'extrusion_ratio':          round(extrusion_ratio,                                2),

            # --- Physics intermediates (useful for PINN loss terms) ---
            'mean_strain_rate_per_sec': round(eps_dot,  6),
            'zener_hollomon_parameter': round(Z,        3),
            'flow_stress_MPa':          round(sigma,    3),

            # --- Outputs (what we want to predict) ---
            'exit_temperature_C':       round(T_exit + noise['exit_temperature_C'],           2),
            'ram_pressure_MPa':         round(P      + noise['ram_pressure_MPa'],             2),
            'exit_speed_mm_per_sec':    round(v_exit * 1000,                                  3),
            'surface_quality_score':    round(np.clip(quality, 0, 100),                       2),
        })

    df = pd.DataFrame(records)

    # Flag physically implausible rows (press overload, melting point)
    df['run_is_valid'] = (
        (df['ram_pressure_MPa'] < 700) &        # typical press limit
        (df['exit_temperature_C'] < 620) &      # below melting onset
        (df['exit_temperature_C'] > 350)
    )

    return df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import os

    df = generate_mock_data(n_samples=3000)

    out_path = os.path.join(os.path.dirname(__file__), 'extrusion_mock_data.csv')
    df.to_csv(out_path, index=False)

    print("=== Dataset Summary ===")
    print(f"Total samples : {len(df)}")
    print(f"Valid samples : {df['run_is_valid'].sum()}  ({df['run_is_valid'].mean()*100:.1f}%)")
    print(f"\nAlloy distribution:\n{df['alloy_grade'].value_counts()}")
    print(f"\nOutput ranges:")
    print(df[['exit_temperature_C', 'ram_pressure_MPa', 'exit_speed_mm_per_sec', 'surface_quality_score']].describe().round(2))
    print(f"\nSaved to: {out_path}")
