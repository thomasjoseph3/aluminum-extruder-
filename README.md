# Aluminium Extrusion — Physics-Informed Neural Network (PINN)

## Overview

This project builds a **Physics-Informed Neural Network (PINN)** to predict optimal process setpoints for aluminium hot extrusion. Instead of relying purely on historical data, the model is constrained by the actual governing equations of heat transfer and material flow, making it generalise better — especially for new alloys or die geometries where little data exists.

The goal: **given an alloy, a die geometry, and a target output quality, predict the billet temperature, container temperature, die temperature, and ram speed that will produce the best result.**

---

## The Extrusion Process (Quick Summary)

In direct hot extrusion:
1. An aluminium billet is heated to 420–500°C
2. A hydraulic ram pushes it through a steel die
3. The metal flows out as a continuous profile (rod, tube, structural section)
4. The profile is quenched, stretched, and aged to achieve final mechanical properties

The key challenge: **temperature, speed, and pressure interact in nonlinear ways**. Getting them wrong causes surface defects, cracking, press overload, or poor mechanical properties. Currently operators find good setpoints by trial and error — this project automates that.

---

## Inputs (What We Feed the Model)

### Process Setpoints — what the operator controls

| Variable | Symbol | Typical Range | Unit | How Measured |
|----------|--------|--------------|------|-------------|
| Billet preheat temperature | T_billet | 420 – 500 | °C | Thermocouple in furnace |
| Container temperature | T_container | 380 – 460 | °C | Thermocouple in container wall |
| Die temperature | T_die | 380 – 460 | °C | Thermocouple in die stack |
| Ram speed | v_ram | 1 – 8 | mm/s | Encoder on hydraulic ram |

### Geometry & Material — known before each run

| Variable | Symbol | Typical Range | Unit |
|----------|--------|--------------|------|
| Billet diameter | D_billet | 80 – 150 | mm |
| Extrusion ratio (billet area / profile area) | R | 10 – 50 | — |
| Alloy grade | — | AA6061, AA6063, AA7075 | — |

---

## Outputs (What the Model Predicts)

| Output | Symbol | Why It Matters |
|--------|--------|---------------|
| Profile exit temperature | T_exit | Must hit quench window for correct aging response |
| Extrusion pressure | P | Must stay below press capacity (~700 MPa) |
| Profile exit speed | v_exit | Too fast → surface defects |
| Surface quality score | Q | Composite score 0–100 based on process window |

The **inverse problem** — what the PINN ultimately solves — is:

> *Given target T_exit and Q, find the setpoints (T_billet, T_container, T_die, v_ram) that achieve them.*

---

## The Physics (Governing Equations)

### 1. Zener-Hollomon Parameter

Captures the combined effect of temperature and strain rate on material behaviour:

```
Z = ε̇ · exp( Q / (R·T) )
```

- `ε̇`  — mean effective strain rate (s⁻¹)
- `Q`  — activation energy for hot deformation (J/mol), alloy-specific
- `R`  — universal gas constant (8.314 J/mol·K)
- `T`  — absolute temperature (K)

### 2. Sellars-Tegart Flow Stress Model (Garofalo Equation)

Relates the Zener-Hollomon parameter to the material's resistance to deformation:

```
σ = (1/α) · arcsinh( (Z/A)^(1/n) )
```

- `σ`  — flow stress (MPa)
- `α`, `A`, `n` — material constants (fitted from hot compression tests)

| Alloy  | A (s⁻¹)  | α (MPa⁻¹) | n    | Q (J/mol) |
|--------|----------|-----------|------|-----------|
| AA6063 | 2.41×10⁹ | 0.045     | 5.39 | 149 776   |
| AA6061 | 3.12×10⁹ | 0.048     | 5.20 | 152 000   |
| AA7075 | 4.57×10⁹ | 0.050     | 4.80 | 156 000   |

### 3. Mean Effective Strain Rate

Simplified formula for strain rate inside the die bearing:

```
ε̇ = ( 6 · v_ram · ln(R) ) / ( π · D_billet )
```

### 4. Exit Temperature — Energy Balance

```
T_exit = T_mean + ΔT_deformation − ΔT_die_loss

ΔT_deformation = (η · σ · ln(R)) / (ρ · c_p)
```

- `η`  — Taylor-Quinney coefficient (~0.9): fraction of plastic work converted to heat
- `ρ`  — density (kg/m³)
- `c_p` — specific heat (J/kg·K)

### 5. Extrusion Pressure — Upper Bound Analysis

```
P = σ · (2/√3) · ln(R)  +  friction term
```

The friction term accounts for billet-container wall friction (Coulomb model, µ ≈ 0.1).

---

## The PINN Methodology

### Why Physics-Informed?

A standard neural network trained only on data can predict outputs within the training distribution but will produce physically impossible results outside it (e.g., negative temperatures, pressure violating energy conservation). A PINN embeds the governing equations directly into the loss function, so the model is **physically consistent even where data is sparse**.

### Network Architecture

```
Input layer:
  [T_billet, T_container, T_die, v_ram, D_billet, R, alloy_encoding]

Hidden layers:
  4–6 fully-connected layers, 64–128 neurons each, tanh activation
  (tanh is preferred over ReLU for PINNs — it is smooth and differentiable)

Output layer:
  [T_exit, P, v_exit, surface_quality]
```

### Loss Function

```
L_total = w1·L_data  +  w2·L_physics  +  w3·L_boundary

L_data     = MSE between predictions and measured/simulated values
L_physics  = residual of the Sellars-Tegart and energy-balance equations
L_boundary = inlet/outlet temperature and velocity boundary conditions
```

The physics residual penalises predictions that violate the governing equations, even at points with no sensor measurements (collocation points). This is the core idea of PINNs.

### Training Strategy

1. **Pre-train on synthetic data** (this repo) — fast, no real data needed
2. **Fine-tune on real sensor data** when available (transfer learning)
3. Optimiser: Adam for initial training, L-BFGS to refine near convergence
4. Non-dimensionalise all variables (temperatures → 0–1, pressures → 0–1) for stable gradients

---

## Mock Data Generation

Since real sensor logs are not yet available, `data/generate_mock_data.py` generates synthetic data by:

1. Sampling process parameters uniformly from realistic industrial ranges
2. Computing outputs using the physics equations above
3. Adding Gaussian noise to simulate sensor measurement error

| Sensor | Noise std |
|--------|-----------|
| Thermocouples (billet, container, die) | ±1°C |
| Pyrometer (exit temperature) | ±2°C |
| Pressure transducer | ±1.5 MPa |
| Ram speed encoder | ±0.05 mm/s |

Generated dataset contains 3 000 samples across AA6061, AA6063, and AA7075.

**To regenerate:**
```bash
python data/generate_mock_data.py
```

Output: `data/extrusion_mock_data.csv`

---

## Setup

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### Mac / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> After setup, always activate the environment before running any script.
> You will see `(.venv)` at the start of your terminal prompt when it is active.

---

## Running the Project

```bash
# 1. Generate the mock dataset
python data/generate_mock_data.py

# 2. Train the PINN
python train.py
```

---

## Project Structure

```
aluminium extruder/
│
├── data/
│   ├── generate_mock_data.py   # synthetic dataset generator (physics-based)
│   └── extrusion_mock_data.csv # generated dataset
│
├── models/
│   └── pinn.py                 # PINN architecture and loss functions
│
├── utils/
│   └── preprocessing.py        # data loading, normalisation, train/val/test split
│
├── train.py                    # training loop
├── requirements.txt            # Python dependencies
└── README.md
```

---

## Roadmap

- [x] Mock data generation with real physics equations
- [x] PINN model architecture (PyTorch)
- [x] Training loop with physics loss
- [ ] Setpoint optimiser (inverse problem solver)
- [ ] Validation against FEM simulation (Deform-3D or HyperXtrude)
- [ ] Real sensor data integration
