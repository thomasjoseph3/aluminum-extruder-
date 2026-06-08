"""
Streamlit frontend for the Aluminium Extrusion PINN.

Two tabs:
  1. Forward Prediction  — adjust setpoints, see predicted outputs in real-time
  2. Inverse Optimiser   — enter target outputs, get recommended setpoints
"""

import os
import sys
import pickle

import numpy as np
import torch
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from scipy.optimize import minimize

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from models.pinn import ExtrusionPINN
from utils.preprocessing import ALLOY_FEATURES, load_and_split

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Aluminium Extrusion Optimiser",
    page_icon="⚙️",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Load model and scalers (cached — loads once per session)
# ---------------------------------------------------------------------------

@st.cache_data
def load_test_predictions():
    """Run model on test split and return actual vs predicted arrays."""
    _, scalers, splits = load_and_split(
        os.path.join("data", "extrusion_mock_data.csv")
    )
    with open(os.path.join("checkpoints", "scalers.pkl"), "rb") as f:
        sc = pickle.load(f)

    model = ExtrusionPINN(input_dim=13, hidden_dim=128, n_layers=6, output_dim=4)
    model.load_state_dict(
        torch.load(os.path.join("checkpoints", "best_model.pt"),
                   map_location="cpu", weights_only=True)
    )
    model.eval()

    X_test, y_test = splits["test"]
    X_norm = sc["X"].transform(X_test)
    with torch.no_grad():
        y_pred_norm = model(torch.tensor(X_norm, dtype=torch.float32)).numpy()
    y_pred = sc["y"].inverse_transform(y_pred_norm)

    return y_test, y_pred


@st.cache_resource
def load_model_and_scalers():
    model_path   = os.path.join("checkpoints", "best_model.pt")
    scalers_path = os.path.join("checkpoints", "scalers.pkl")

    if not os.path.exists(model_path) or not os.path.exists(scalers_path):
        return None, None, None

    with open(scalers_path, "rb") as f:
        scalers = pickle.load(f)

    model = ExtrusionPINN(input_dim=13, hidden_dim=128, n_layers=6, output_dim=4)
    model.load_state_dict(
        torch.load(model_path, map_location="cpu", weights_only=True)
    )
    model.eval()

    return model, scalers["X"], scalers["y"]


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------

def build_input_vector(alloy, T_billet, T_container, T_die, v_ram, D_billet, R):
    mat = ALLOY_FEATURES[alloy]
    return np.array([[T_billet, T_container, T_die, v_ram, D_billet, R, *mat]])


def predict(model, x_scaler, y_scaler, input_raw: np.ndarray) -> dict:
    x_norm = x_scaler.transform(input_raw)
    tensor = torch.tensor(x_norm, dtype=torch.float32)
    with torch.no_grad():
        y_norm = model(tensor).numpy()
    y_raw = y_scaler.inverse_transform(y_norm)[0]
    return {
        "exit_temperature_C":    float(y_raw[0]),
        "ram_pressure_MPa":      float(y_raw[1]),
        "exit_speed_mm_per_sec": float(y_raw[2]),
        "surface_quality_score": float(np.clip(y_raw[3], 0, 100)),
    }


def optimise_setpoints(
    model, x_scaler, y_scaler,
    alloy, D_billet, R,
    target_temp, target_quality,
):
    mat = ALLOY_FEATURES[alloy]

    def objective(x):
        inp = np.array([[x[0], x[1], x[2], x[3], D_billet, R, *mat]])
        x_norm = x_scaler.transform(inp)
        t = torch.tensor(x_norm, dtype=torch.float32)
        with torch.no_grad():
            y_norm = model(t).numpy()
        y = y_scaler.inverse_transform(y_norm)[0]
        temp_err    = ((y[0] - target_temp)    / 20.0) ** 2
        quality_err = ((y[3] - target_quality) / 20.0) ** 2
        return temp_err + quality_err

    bounds  = [(420, 500), (380, 460), (380, 460), (1.0, 8.0)]
    starts  = [[460, 420, 410, 3.0], [480, 440, 430, 5.0], [440, 400, 395, 2.0]]
    best    = None

    for x0 in starts:
        res = minimize(objective, x0, method="L-BFGS-B", bounds=bounds,
                       options={"maxiter": 500})
        if best is None or res.fun < best.fun:
            best = res

    T_b, T_c, T_d, v = best.x
    final = predict(model, x_scaler, y_scaler,
                    np.array([[T_b, T_c, T_d, v, D_billet, R, *mat]]))

    setpoints = {
        "billet_temperature_C":    round(T_b, 1),
        "container_temperature_C": round(T_c, 1),
        "die_temperature_C":       round(T_d, 1),
        "ram_speed_mm_per_sec":    round(v,   2),
    }
    return setpoints, final


# ---------------------------------------------------------------------------
# UI components
# ---------------------------------------------------------------------------

def quality_gauge(value: float):
    bar_color = "#28a745" if value >= 70 else ("#ffc107" if value >= 40 else "#dc3545")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(value, 1),
        number={"suffix": " / 100", "font": {"size": 26}},
        title={"text": "Surface Quality Score", "font": {"size": 15}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1},
            "bar":  {"color": bar_color, "thickness": 0.3},
            "steps": [
                {"range": [0,  40], "color": "#ffe0e0"},
                {"range": [40, 70], "color": "#fff8e0"},
                {"range": [70, 100],"color": "#e0ffe0"},
            ],
            "threshold": {
                "line": {"color": "black", "width": 3},
                "thickness": 0.8,
                "value": 70,
            },
        },
    ))
    fig.update_layout(height=260, margin=dict(t=40, b=0, l=20, r=20))
    return fig


def output_bar_chart(preds: dict):
    outputs = ["Exit Temp (°C)", "Pressure (MPa)", "Exit Speed (mm/s)", "Quality (0-100)"]
    values  = [
        preds["exit_temperature_C"],
        preds["ram_pressure_MPa"],
        preds["exit_speed_mm_per_sec"],
        preds["surface_quality_score"],
    ]
    colors = []
    for label, val in zip(outputs, values):
        if "Temp" in label:
            colors.append("#28a745" if 480 <= val <= 560 else "#ffc107" if 450 <= val <= 580 else "#dc3545")
        elif "Pressure" in label:
            colors.append("#28a745" if val <= 400 else "#ffc107" if val <= 600 else "#dc3545")
        elif "Speed" in label:
            colors.append("#28a745" if val <= 100 else "#ffc107" if val <= 200 else "#dc3545")
        else:
            colors.append("#28a745" if val >= 70 else "#ffc107" if val >= 40 else "#dc3545")

    fig = go.Figure(go.Bar(
        x=outputs, y=values,
        marker_color=colors,
        text=[f"{v:.1f}" for v in values],
        textposition="outside",
    ))
    fig.update_layout(
        title="Predicted Outputs",
        height=320,
        margin=dict(t=40, b=40, l=20, r=20),
        yaxis_title="Value",
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def status(value, good_lo, good_hi, warn_lo=None, warn_hi=None):
    if good_lo <= value <= good_hi:
        return "✅ Good"
    if warn_lo is not None and warn_hi is not None and warn_lo <= value <= warn_hi:
        return "⚠️ Warning"
    return "🔴 Out of range"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    st.title("⚙️ Aluminium Extrusion Process Optimiser")
    st.caption(
        "Physics-Informed Neural Network — predicts process outputs from setpoints "
        "and recommends optimal setpoints for a target result."
    )

    model, x_scaler, y_scaler = load_model_and_scalers()

    if model is None:
        st.error("Model not found. Run  `python train.py`  first, then restart the app.")
        st.stop()

    tab1, tab2, tab3 = st.tabs(["🔍  Forward Prediction", "🎯  Inverse Optimiser", "📊  Model Accuracy"])

    # -----------------------------------------------------------------------
    # TAB 1 — FORWARD PREDICTION
    # -----------------------------------------------------------------------
    with tab1:
        st.markdown("### What will happen if I use these setpoints?")
        st.caption("Predictions update live as you move the sliders.")

        col_in, col_out = st.columns([1, 1.5], gap="large")

        with col_in:
            st.markdown("**Material & Geometry**")
            alloy    = st.selectbox("Alloy Grade", ["AA6063", "AA6061", "AA7075"])
            D_billet = st.slider("Billet Diameter (mm)", 80, 150, 100)
            R        = st.slider("Extrusion Ratio",       10,  50,  25)

            st.markdown("**Process Setpoints**")
            T_billet    = st.slider("Billet Temperature (°C)",    420, 500, 460)
            T_container = st.slider("Container Temperature (°C)", 380, 460, 420)
            T_die       = st.slider("Die Temperature (°C)",       380, 460, 410)
            v_ram       = st.slider("Ram Speed (mm/s)", 1.0, 8.0, 3.0, step=0.1)

        with col_out:
            inp   = build_input_vector(alloy, T_billet, T_container, T_die,
                                       v_ram, D_billet, R)
            preds = predict(model, x_scaler, y_scaler, inp)

            st.markdown("**Predicted Outputs**")

            m1, m2 = st.columns(2)
            m3, m4 = st.columns(2)

            m1.metric(
                "Exit Temperature",
                f"{preds['exit_temperature_C']:.1f} °C",
                status(preds['exit_temperature_C'], 480, 560, 450, 580),
            )
            m2.metric(
                "Ram Pressure",
                f"{preds['ram_pressure_MPa']:.1f} MPa",
                status(preds['ram_pressure_MPa'], 0, 400, 400, 600),
            )
            m3.metric(
                "Exit Speed",
                f"{preds['exit_speed_mm_per_sec']:.1f} mm/s",
                status(preds['exit_speed_mm_per_sec'], 0, 100, 100, 200),
            )
            m4.metric(
                "Surface Quality",
                f"{preds['surface_quality_score']:.1f} / 100",
                status(preds['surface_quality_score'], 70, 100, 40, 70),
            )

            st.plotly_chart(quality_gauge(preds["surface_quality_score"]),
                            use_container_width=True, key="fwd_gauge")

        st.divider()
        st.markdown("**Process Flow**")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.info(f"🔥 **Billet**\n\n{T_billet} °C")
        c2.info(f"🛢️ **Container**\n\n{T_container} °C")
        c3.info(f"🔩 **Die**\n\nRatio {R}× | {T_die} °C")
        c4.info(f"➡️ **Ram**\n\n{v_ram} mm/s")
        c5.success(
            f"✅ **Profile Out**\n\n"
            f"{preds['exit_temperature_C']:.0f} °C  |  "
            f"{preds['ram_pressure_MPa']:.0f} MPa"
        )

    # -----------------------------------------------------------------------
    # TAB 2 — INVERSE OPTIMISER
    # -----------------------------------------------------------------------
    with tab2:
        st.markdown("### What setpoints should I use to hit my target?")
        st.caption(
            "Enter the output you want to achieve. "
            "The optimiser searches for the setpoints that produce it."
        )

        col_t, col_r = st.columns([1, 1.5], gap="large")

        with col_t:
            st.markdown("**Material & Geometry**")
            alloy_inv    = st.selectbox("Alloy Grade", ["AA6063", "AA6061", "AA7075"],
                                        key="inv_alloy")
            D_billet_inv = st.number_input("Billet Diameter (mm)", 80, 150, 100,
                                           key="inv_D")
            R_inv        = st.number_input("Extrusion Ratio", 10, 50, 25, key="inv_R")

            st.markdown("**Target Outputs**")
            target_temp    = st.slider("Target Exit Temperature (°C)", 450, 560, 510,
                                       key="inv_temp")
            target_quality = st.slider("Minimum Quality Score",          0, 100,  75,
                                       key="inv_q")

            run = st.button("🔍  Find Optimal Setpoints", use_container_width=True,
                            type="primary")

        with col_r:
            if run:
                with st.spinner("Optimising — searching setpoint space..."):
                    setpoints, final_preds = optimise_setpoints(
                        model, x_scaler, y_scaler,
                        alloy_inv, D_billet_inv, R_inv,
                        target_temp, target_quality,
                    )

                st.markdown("**Recommended Setpoints**")
                s1, s2 = st.columns(2)
                s3, s4 = st.columns(2)
                s1.metric("Billet Temperature",    f"{setpoints['billet_temperature_C']} °C")
                s2.metric("Container Temperature", f"{setpoints['container_temperature_C']} °C")
                s3.metric("Die Temperature",        f"{setpoints['die_temperature_C']} °C")
                s4.metric("Ram Speed",              f"{setpoints['ram_speed_mm_per_sec']} mm/s")

                st.divider()
                st.markdown("**Predicted result with these setpoints**")

                r1, r2, r3, r4 = st.columns(4)
                r1.metric(
                    "Exit Temperature",
                    f"{final_preds['exit_temperature_C']:.1f} °C",
                    f"Target {target_temp} °C",
                )
                r2.metric("Ram Pressure",
                          f"{final_preds['ram_pressure_MPa']:.1f} MPa")
                r3.metric("Exit Speed",
                          f"{final_preds['exit_speed_mm_per_sec']:.1f} mm/s")
                r4.metric(
                    "Surface Quality",
                    f"{final_preds['surface_quality_score']:.1f} / 100",
                    f"Target {target_quality}",
                )

                st.plotly_chart(
                    quality_gauge(final_preds["surface_quality_score"]),
                    use_container_width=True, key="inv_gauge",
                )
            else:
                st.info(
                    "Set your target values on the left and click  "
                    "**Find Optimal Setpoints**."
                )


    # -----------------------------------------------------------------------
    # TAB 3 — MODEL ACCURACY
    # -----------------------------------------------------------------------
    with tab3:
        st.markdown("### How accurate is the model?")
        st.caption("Evaluated on the held-out test set — data the model never saw during training.")

        y_test, y_pred = load_test_predictions()

        OUTPUT_META = [
            {"name": "Exit Temperature",  "col": 0, "unit": "°C",     "good_err": 5},
            {"name": "Ram Pressure",       "col": 1, "unit": "MPa",    "good_err": 5},
            {"name": "Exit Speed",         "col": 2, "unit": "mm/s",   "good_err": 5},
            {"name": "Surface Quality",    "col": 3, "unit": "/ 100",  "good_err": 3},
        ]

        # --- Summary metrics row ---
        st.markdown("**Prediction Error (Mean Absolute Error)**")
        cols_m = st.columns(4)
        for meta, col_m in zip(OUTPUT_META, cols_m):
            i   = meta["col"]
            mae = float(np.mean(np.abs(y_pred[:, i] - y_test[:, i])))
            col_m.metric(
                meta["name"],
                f"± {mae:.2f} {meta['unit']}",
                "✅ Good" if mae <= meta["good_err"] else "⚠️ Review",
            )

        st.divider()

        # --- Parity plots (predicted vs actual) ---
        st.markdown("**Parity Plots — Predicted vs Actual**")
        st.caption("Points should lie along the diagonal. The closer to the line, the better.")

        p1, p2 = st.columns(2)
        p3, p4 = st.columns(2)

        plot_cols = [p1, p2, p3, p4]
        for meta, pcol in zip(OUTPUT_META, plot_cols):
            i      = meta["col"]
            actual = y_test[:, i]
            pred   = y_pred[:, i]
            lo     = min(actual.min(), pred.min())
            hi     = max(actual.max(), pred.max())

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=actual, y=pred,
                mode="markers",
                marker=dict(size=4, opacity=0.5, color="#1f77b4"),
                name="Test samples",
            ))
            fig.add_trace(go.Scatter(
                x=[lo, hi], y=[lo, hi],
                mode="lines",
                line=dict(color="red", dash="dash", width=2),
                name="Perfect prediction",
            ))
            fig.update_layout(
                title=meta["name"],
                xaxis_title=f"Actual ({meta['unit']})",
                yaxis_title=f"Predicted ({meta['unit']})",
                height=320,
                margin=dict(t=40, b=40, l=40, r=20),
                showlegend=False,
            )
            pcol.plotly_chart(fig, use_container_width=True, key=f"parity_{meta['col']}")

        st.divider()

        # --- Error distribution ---
        st.markdown("**Error Distribution — Exit Temperature**")
        st.caption("Shows how often the model is within a given error band.")

        errors = y_pred[:, 0] - y_test[:, 0]
        fig_hist = px.histogram(
            x=errors, nbins=40,
            labels={"x": "Prediction Error (°C)", "y": "Count"},
            color_discrete_sequence=["#1f77b4"],
        )
        fig_hist.add_vline(x=0,  line_dash="dash", line_color="red",   annotation_text="Zero error")
        fig_hist.add_vline(x=2,  line_dash="dot",  line_color="orange", annotation_text="+2°C")
        fig_hist.add_vline(x=-2, line_dash="dot",  line_color="orange")
        fig_hist.update_layout(height=300, margin=dict(t=20, b=40, l=40, r=20))
        st.plotly_chart(fig_hist, use_container_width=True, key="err_hist")


if __name__ == "__main__":
    main()
