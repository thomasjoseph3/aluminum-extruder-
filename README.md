# Aluminium Extrusion — Physics-Informed Neural Network (PINN)

## What This Project Does

This project builds a **Physics-Informed Neural Network (PINN)** to predict and optimise process setpoints for aluminium hot extrusion.

Most extrusion plants find the right setpoints through **operator experience and trial and error** — wasting material, press time, and energy. This system replaces that with a model that:

1. **Predicts** what will happen given a set of process inputs (forward prediction)
2. **Recommends** the optimal setpoints to hit a desired output (inverse optimisation)

The model is not a black-box neural network — it is constrained by the **actual governing physics equations** of hot deformation and heat transfer, which means it stays physically consistent even where data is sparse.

---

## The Problem It Solves

An aluminium extruder has four knobs the operator controls:
- Billet temperature
- Container temperature
- Die temperature
- Ram speed

Get them wrong and the profile comes out with surface defects, wrong dimensions, or the press overloads. The model answers two questions:

> *"If I use these settings, what will I get?"* — Forward Prediction

> *"I want exit temperature 495°C and quality score above 80 — what settings should I use?"* — Inverse Optimiser

---

## Model Accuracy

Evaluated on a held-out test set (data the model never saw during training):

| Output | Mean Absolute Error |
|--------|-------------------|
| Exit Temperature | **±1.74°C** |
| Ram Pressure | **±1.55 MPa** |
| Exit Speed | **±2.40 mm/s** |
| Surface Quality Score | **±0.98 / 100** |

For context: industrial thermocouples have ±1°C inherent noise, so the model is operating near sensor-level accuracy.

---

## Inputs and Outputs

### Inputs — what you provide

| Input | Range | Unit | Source |
|-------|-------|------|--------|
| Alloy grade | AA6061, AA6063, AA7075 | — | Known before run |
| Billet temperature | 420 – 500 | °C | Furnace thermocouple |
| Container temperature | 380 – 460 | °C | Container wall thermocouple |
| Die temperature | 380 – 460 | °C | Die stack thermocouple |
| Ram speed | 1 – 8 | mm/s | Ram encoder |
| Billet diameter | 80 – 150 | mm | Known before run |
| Extrusion ratio | 10 – 50 | — | Die geometry |

The alloy grade is automatically converted to 7 material property numbers (density, specific heat, activation energy, etc.) so the model understands the physics of the metal, not just a label.

**Total: 13 numbers into the network.**

### Outputs — what the model predicts

| Output | Range | Unit | Why It Matters |
|--------|-------|------|----------------|
| Exit temperature | 430 – 580 | °C | Must hit quench window for correct aging |
| Ram pressure | 50 – 700 | MPa | Must stay below press limit |
| Exit speed | 10 – 400 | mm/s | Too fast causes surface defects |
| Surface quality score | 0 – 100 | — | Overall process health indicator (computed analytically from exit temperature and exit speed — not predicted by the neural network) |

---

## The Physics (Governing Equations)

These are the real equations used in commercial extrusion FEM software (Deform-3D, HyperXtrude). They are embedded into the model's loss function so predictions are always physically consistent.

### 1. Zener-Hollomon Parameter
Captures the combined effect of temperature and strain rate:
```
Z = ε̇ · exp( Q / (R·T) )
```
- `ε̇` — mean effective strain rate (s⁻¹)
- `Q` — activation energy for hot deformation (J/mol)
- `R` — universal gas constant (8.314 J/mol·K)
- `T` — absolute temperature (K)

### 2. Sellars-Tegart Flow Stress (Garofalo Equation)
Relates the Zener-Hollomon parameter to the metal's resistance to flow:
```
σ = (1/α) · arcsinh( (Z/A)^(1/n) )
```

Material constants used (from published literature):

| Alloy | A (s⁻¹) | α (MPa⁻¹) | n | Q (J/mol) |
|-------|---------|-----------|---|-----------|
| AA6063 | 2.41×10⁹ | 0.045 | 5.39 | 149,776 |
| AA6061 | 3.12×10⁹ | 0.048 | 5.20 | 152,000 |
| AA7075 | 4.57×10⁹ | 0.050 | 4.80 | 156,000 |

### 3. Mean Effective Strain Rate
```
ε̇ = ( 6 · v_ram · ln(R) ) / ( π · D_billet )
```

### 4. Exit Temperature — Energy Balance
```
T_exit = T_mean + ΔT_deformation − ΔT_die_loss

ΔT_deformation = (η · σ · ln(R)) / (ρ · c_p)
```
- `η = 0.9` — Taylor-Quinney coefficient (fraction of plastic work converted to heat)

### 5. Extrusion Pressure — Upper Bound Analysis
```
P = σ · (2/√3) · ln(R)  +  friction term   (µ = 0.1)
```

---

## How the PINN Works

### Why Not Just a Normal Neural Network?

A standard neural network trained only on data will:
- Require thousands of examples to generalise
- Predict physically impossible results outside the training range
- Give no guarantee that energy is conserved or flow equations are satisfied

A PINN solves this by adding the physics residuals directly to the loss function:

```
Total Loss = (Data Loss) + (Physics Residual Loss) + (Boundary Condition Loss)

Data Loss         — how wrong are predictions vs measured/simulated values
Physics Loss      — how much do predictions violate the governing equations
Boundary Loss     — volume conservation: v_exit = v_ram × R  (exact law)
```

The physics weight starts small and is gradually increased during training, so the network first learns from data, then tightens to match the physics.

### Network Architecture
```
Inputs (13 numbers)
       ↓
  6 hidden layers
  128 neurons each
  Tanh activation        ← smooth & differentiable, required for physics gradients
       ↓
Outputs (4 numbers)      ← exit temperature, ram pressure, exit speed, quality
```
Total trainable parameters: **84,868**

### Inverse Optimiser

After training, the model is run in reverse using L-BFGS-B optimisation (scipy):
- Given target exit temperature and quality score
- Searches the setpoint space to minimise the gap between target and prediction
- Tries 3 different starting points and returns the best result

---

## Training Procedure

### Step 1 — Data Normalisation
All inputs and outputs are scaled to the range [0, 1] using min-max normalisation fitted on the training set only. This prevents any single variable (e.g. activation energy at 150,000 J/mol vs ram speed at 3 mm/s) from dominating the gradients.

### Step 2 — Optimiser
The network is trained using the **Adam optimiser** (learning rate 0.001) with a `ReduceLROnPlateau` scheduler — if validation loss stops improving for 15 epochs, the learning rate is halved automatically.

### Step 3 — Physics Weight Annealing
The physics residual weight `w_phys` is not fixed — it follows a deliberate schedule:

| Epochs | Physics Weight | What is happening |
|--------|---------------|-------------------|
| 1 – 50 | 0.01 (small) | Network learns from data first — gets the basic input/output mapping right |
| 50 – 150 | 0.01 → 0.50 (ramping up) | Physics constraint gradually tightens — predictions must increasingly satisfy the governing equations |
| 150 – 200 | 0.50 (fixed) | Full physics enforcement — model is constrained by both data and physics |

This two-phase strategy is critical. If the physics weight is too high from the start, the network cannot learn from the data. If it is always too low, the physics is never enforced.

### Step 4 — Gradient Clipping
Gradients are clipped to a maximum norm of 1.0 per step to prevent instability during the physics-weight ramp-up phase.

### Training Results
```
Epoch   1:  Total loss 2.073   Data loss 0.063   Physics residual 1.693
Epoch  50:  Total loss 0.007   Data loss 0.001   Physics residual 0.012
Epoch 100:  Total loss 0.004   Data loss 0.001   Physics residual 0.007
Epoch 200:  Total loss 0.003   Data loss 0.000   Physics residual 0.003
```
Training time: ~30 seconds on CPU. No GPU required.

The best checkpoint (lowest validation loss) is saved automatically during training.

---

## Synthetic Data Generation

### Why Synthetic Data?

Real extrusion plants do not typically log structured sensor data at the detail level needed to train a model. Getting access to such data requires plant partnerships, NDAs, and significant integration work. The standard approach in ML for physical systems is:

1. Build and validate the model on **physics-based simulated data** first
2. **Fine-tune on real machine data** once plant access is available (transfer learning)

This is exactly how tools like Deform-3D and HyperXtrude are validated — against equations first, then against physical experiments.

### How the Data is Generated

Each sample is produced by:
1. Randomly sampling process parameters from realistic industrial ranges
2. Running the governing equations (Sellars-Tegart, energy balance, upper bound pressure) to compute what the outputs would be
3. Adding Gaussian noise to each value to simulate real sensor measurement error

| Sensor | Noise Applied |
|--------|--------------|
| Thermocouples (billet, container, die) | ±1°C std dev |
| Pyrometer (exit temperature) | ±2°C std dev |
| Pressure transducer | ±1.5 MPa std dev |
| Ram speed encoder | ±0.05 mm/s std dev |

### What to Tell a Client About the Data

> *"The model is currently trained on physics-based simulation data using the same governing equations as commercial extrusion FEM software. The equations and material constants are taken from published literature for AA6061, AA6063, and AA7075. The next step is connecting to real machine sensor logs to fine-tune the model on actual production data — the architecture is designed for this."*

Dataset: **3,000 samples** across AA6061, AA6063, and AA7075.
Split: **70% train / 15% validation / 15% test**

---

## Setup

### Step 1 — Clone the repo

```bash
git clone https://github.com/thomasjoseph3/aluminum-extruder-.git
cd aluminum-extruder-
```

### Step 2 — Create virtual environment

**Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**Mac / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

You will see `(.venv)` at the start of your terminal when the environment is active.

### Step 3 — Install dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Project

### Run the App (recommended)

The trained model and data are already included in the repo. Just launch the app:

```bash
# Windows
streamlit run app.py

# Mac / Linux
streamlit run app.py
```

Then open **http://localhost:8501** in your browser.

### Retrain the Model (optional)

Only needed if you want to regenerate data or retrain from scratch:

```bash
# Step 1 — regenerate the dataset
python data/generate_mock_data.py

# Step 2 — train the PINN (200 epochs, ~30 seconds on CPU)
python train.py
```

Checkpoints are saved to `checkpoints/best_model.pt` and `checkpoints/scalers.pkl`.

---

## The App

### Sidebar
Always visible. Shows model architecture summary, accuracy numbers, supported alloys, and a quick how-to guide for each tab.

### Tab 1 — Forward Prediction
Move the sliders to set process parameters. Predictions update **live** in real-time — no button needed.

- **Colour-coded metric cards** — green (good), orange (warning), red (out of range) for each output
- **Surface quality gauge** — 0–100, colour-banded, updates live
- **Process flow diagram** — arrow-connected stages from billet → container → die → ram → profile out

### Tab 2 — Inverse Optimiser
Enter the output you want to achieve:
- Alloy grade, billet diameter, extrusion ratio
- Target exit temperature
- Target quality score

Click **Find Optimal Setpoints** — the optimiser runs L-BFGS-B from 3 starting points and returns:
- Recommended billet temperature, container temperature, die temperature, ram speed (styled as a recommendation panel)
- Predicted result with those setpoints shown immediately below

### Tab 3 — Model Accuracy
Shows how well the model performs on test data it never saw during training:
- Mean Absolute Error for each output in real engineering units (colour-coded cards)
- 4 parity plots (predicted vs actual — points should lie on the diagonal)
- Error distribution histogram for exit temperature with ±2°C band marked

---

## Project Structure

```
aluminium-extruder/
│
├── app.py                      # Streamlit web app (forward + inverse + accuracy)
├── train.py                    # Training loop with physics weight annealing
├── requirements.txt            # Python dependencies
│
├── data/
│   ├── generate_mock_data.py   # Physics-based synthetic dataset generator
│   └── extrusion_mock_data.csv # Generated dataset (3,000 samples)
│
├── models/
│   └── pinn.py                 # PINN architecture, physics residuals, loss function
│
├── utils/
│   └── preprocessing.py        # Data loading, normalisation, train/val/test split
│
├── checkpoints/
│   ├── best_model.pt           # Trained model weights
│   └── scalers.pkl             # Min-max scalers (fitted on training data)
│
└── loss_curve.png              # Training and validation loss curves
```

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `torch` | Neural network and automatic differentiation |
| `numpy` / `pandas` | Data handling |
| `scipy` | L-BFGS-B optimisation for inverse solver |
| `streamlit` | Web app frontend |
| `plotly` | Interactive charts and gauges |
| `matplotlib` | Loss curve plots |

---

## Current Status and Limitations

### What Works Now
- Forward prediction of exit temperature, pressure, exit speed, and quality for any combination of setpoints within the training range
- Inverse optimiser that recommends setpoints for a given target
- Accuracy near sensor-level noise (±1.74°C exit temperature)
- Works on any computer with Python — no GPU, no cloud, no special hardware

### Honest Limitations
| Limitation | Impact | Path to Fix |
|------------|--------|-------------|
| Trained on simulated data, not real machine data | Predictions are physically consistent but not validated against a real press | Collect real sensor logs and fine-tune the model |
| Surface quality is a heuristic, not derived from metallurgy | Quality score is indicative only — based on exit temperature window and exit speed, not grain structure or surface roughness | Replace with a published Hot Tearing Criterion or real quality measurements once plant data is available |
| Sensor drift, spike noise, and missing data not modelled | Real sensors are noisier than the simulation assumes | Add realistic fault injection to the data generator |
| Only 3 alloys supported | Cannot predict for other grades | Add material constants from literature for additional alloys |
| No confidence intervals | Model gives a point prediction with no uncertainty range | Add Bayesian or ensemble uncertainty quantification |

### What a Client Should Know
This is a working proof-of-concept that demonstrates the full system — data pipeline, physics-informed model, inverse optimiser, and interactive interface. It is **ready for validation against real machine data**. The architecture does not need to change — only the training data source changes when real sensor logs become available.

---

## Roadmap

- [x] Physics-based mock data generation (Sellars-Tegart, Zener-Hollomon, energy balance)
- [x] PINN model architecture (PyTorch, 6 layers, tanh)
- [x] Training loop with physics weight annealing
- [x] Physics residual normalisation
- [x] Inverse setpoint optimiser (L-BFGS-B)
- [x] Streamlit frontend — forward prediction, inverse optimiser, accuracy tab
- [ ] Real sensor data integration and fine-tuning
- [ ] Validation against FEM simulation (Deform-3D or HyperXtrude)
- [ ] Uncertainty quantification (confidence intervals on predictions)
- [ ] Die wear as a model input
