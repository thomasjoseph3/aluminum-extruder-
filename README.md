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

> *"I want exit temperature 510°C and quality score above 80 — what settings should I use?"* — Inverse Optimiser

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
| Surface quality score | 0 – 100 | — | Overall process health indicator |

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

## Synthetic Data Generation

Since real sensor logs are not yet available, the training data is generated using the physics equations above. This is standard practice in ML for physical systems — simulate first, fine-tune on real data later.

Each sample is generated by:
1. Sampling process parameters from realistic industrial ranges
2. Running the physics equations to compute outputs
3. Adding sensor noise to simulate real measurement error

| Sensor | Noise (std dev) |
|--------|----------------|
| Thermocouples (billet, container, die) | ±1°C |
| Pyrometer (exit temperature) | ±2°C |
| Pressure transducer | ±1.5 MPa |
| Ram speed encoder | ±0.05 mm/s |

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

### Tab 1 — Forward Prediction
Move the sliders to set process parameters. Predictions update **live** in real-time.

- 4 output metrics with ✅ / ⚠️ / 🔴 status indicators
- Surface quality gauge chart (0–100, colour coded)
- Process flow diagram showing the full extrusion path

### Tab 2 — Inverse Optimiser
Enter the output you want to achieve:
- Target exit temperature
- Minimum quality score

Click **Find Optimal Setpoints** — the optimiser searches the input space and returns:
- Recommended billet temperature
- Recommended container temperature
- Recommended die temperature
- Recommended ram speed
- Predicted outputs with those setpoints

### Tab 3 — Model Accuracy
Shows how well the model performs on test data it never saw during training:
- MAE for each output in real engineering units
- Parity plots (predicted vs actual — points should sit on the diagonal)
- Error distribution histogram for exit temperature

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
