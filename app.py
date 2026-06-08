"""
Streamlit frontend for the Aluminium Extrusion PINN.

Tabs:
  1. Forward Prediction  — live predictions as sliders move
  2. Inverse Optimiser   — find setpoints for a target output
  3. Model Accuracy      — parity plots and error metrics
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
    page_title="Extrusion Optimiser",
    page_icon="⚙️",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Global CSS
# ---------------------------------------------------------------------------

st.markdown("""
<style>
    /* Metric cards */
    .card {
        border-radius: 10px;
        padding: 18px 16px 14px;
        text-align: center;
        margin-bottom: 8px;
    }
    .card-good   { background: #e8f5e9; border-left: 5px solid #2e7d32; }
    .card-warn   { background: #fff8e1; border-left: 5px solid #f9a825; }
    .card-bad    { background: #ffebee; border-left: 5px solid #c62828; }
    .card-label  { font-size: 0.82rem; color: #555; margin: 0 0 4px; font-weight: 500; letter-spacing: 0.03em; }
    .card-value  { font-size: 1.75rem; font-weight: 700; margin: 0 0 4px; }
    .card-status { font-size: 0.78rem; margin: 0; }
    .card-good  .card-value  { color: #1b5e20; }
    .card-warn  .card-value  { color: #e65100; }
    .card-bad   .card-value  { color: #b71c1c; }

    /* Section headers */
    .section-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #333;
        border-bottom: 2px solid #e0e0e0;
        padding-bottom: 6px;
        margin: 16px 0 12px;
    }

    /* Setpoint result box */
    .setpoint-box {
        background: #f0f4ff;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
        border: 1px solid #c5cae9;
    }
    .setpoint-label { font-size: 0.8rem; color: #555; margin: 0 0 4px; }
    .setpoint-value { font-size: 1.5rem; font-weight: 700; color: #1a237e; margin: 0; }
    .setpoint-unit  { font-size: 0.78rem; color: #777; margin: 0; }

    /* Process step boxes */
    .step-box {
        background: #fafafa;
        border-radius: 8px;
        padding: 12px 8px;
        text-align: center;
        border: 1px solid #e0e0e0;
        height: 90px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .step-box-out {
        background: #e8f5e9;
        border: 1px solid #81c784;
    }
    .step-icon  { font-size: 1.3rem; margin: 0; }
    .step-label { font-size: 0.75rem; font-weight: 600; color: #444; margin: 2px 0; }
    .step-val   { font-size: 0.85rem; color: #222; margin: 0; }

    /* Hide Streamlit branding */
    #MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Load model and scalers
# ---------------------------------------------------------------------------

@st.cache_resource
def load_model_and_scalers():
    mp = os.path.join("checkpoints", "best_model.pt")
    sp = os.path.join("checkpoints", "scalers.pkl")
    if not os.path.exists(mp) or not os.path.exists(sp):
        return None, None, None
    with open(sp, "rb") as f:
        sc = pickle.load(f)
    model = ExtrusionPINN(input_dim=13, hidden_dim=128, n_layers=6, output_dim=4)
    model.load_state_dict(torch.load(mp, map_location="cpu", weights_only=True))
    model.eval()
    return model, sc["X"], sc["y"]


@st.cache_data
def load_test_predictions():
    _, scalers, splits = load_and_split(os.path.join("data", "extrusion_mock_data.csv"))
    with open(os.path.join("checkpoints", "scalers.pkl"), "rb") as f:
        sc = pickle.load(f)
    model = ExtrusionPINN(input_dim=13, hidden_dim=128, n_layers=6, output_dim=4)
    model.load_state_dict(torch.load(os.path.join("checkpoints", "best_model.pt"),
                                     map_location="cpu", weights_only=True))
    model.eval()
    X_test, y_test = splits["test"]
    X_norm = sc["X"].transform(X_test)
    with torch.no_grad():
        y_pred_norm = model(torch.tensor(X_norm, dtype=torch.float32)).numpy()
    y_pred = sc["y"].inverse_transform(y_pred_norm)
    return y_test, y_pred


# ---------------------------------------------------------------------------
# Prediction helpers
# ---------------------------------------------------------------------------

def build_input(alloy, T_b, T_c, T_d, v, D, R):
    return np.array([[T_b, T_c, T_d, v, D, R, *ALLOY_FEATURES[alloy]]])


def compute_quality(alloy: str, T_exit: float, v_exit_mm_s: float) -> float:
    """
    Quality score computed analytically from exit conditions.
    Derived from physics — not predicted by the neural network — for reliability.
    v_exit is from volume conservation (exact law): v_exit = v_ram * R
    """
    optimal = {"AA6063": 495.0, "AA6061": 505.0, "AA7075": 420.0}
    T_opt      = optimal.get(alloy, 495.0)
    v_exit_m_s = v_exit_mm_s / 1000.0

    temp_score  = np.exp(-((T_exit - T_opt) ** 2) / (2 * 30.0 ** 2))
    speed_score = np.exp(-max(0.0, v_exit_m_s - 0.020) / 0.005)
    return float(np.clip(100.0 * temp_score * speed_score, 0, 100))


def predict(model, xs, ys, inp, alloy: str, v_ram: float, R: float):
    xn = xs.transform(inp)
    with torch.no_grad():
        yn = model(torch.tensor(xn, dtype=torch.float32)).numpy()
    raw = ys.inverse_transform(yn)[0]

    v_exit = v_ram * R                                # exact: volume conservation
    quality = compute_quality(alloy, float(raw[0]), v_exit)

    return {
        "exit_temperature_C":    float(raw[0]),
        "ram_pressure_MPa":      float(raw[1]),
        "exit_speed_mm_per_sec": v_exit,
        "surface_quality_score": quality,
    }


def optimise(model, xs, ys, alloy, D, R, t_temp, t_qual):
    mat = ALLOY_FEATURES[alloy]

    def loss(x):
        inp = np.array([[x[0], x[1], x[2], x[3], D, R, *mat]])
        xn  = xs.transform(inp)
        with torch.no_grad():
            yn = model(torch.tensor(xn, dtype=torch.float32)).numpy()
        y = ys.inverse_transform(yn)[0]
        v_exit  = x[3] * R
        quality = compute_quality(alloy, float(y[0]), v_exit)
        return ((y[0] - t_temp) / 20.0)**2 + ((quality - t_qual) / 20.0)**2

    best = None
    for x0 in [[460, 420, 410, 3.0], [490, 450, 440, 1.5], [440, 400, 395, 5.0]]:
        r = minimize(loss, x0, method="L-BFGS-B",
                     bounds=[(420,500),(380,460),(380,460),(1.0,8.0)],
                     options={"maxiter": 500})
        if best is None or r.fun < best.fun:
            best = r

    T_b, T_c, T_d, v = best.x
    final = predict(model, xs, ys,
                    np.array([[T_b, T_c, T_d, v, D, R, *mat]]),
                    alloy, v, R)
    return {"billet_temperature_C": round(T_b,1), "container_temperature_C": round(T_c,1),
            "die_temperature_C": round(T_d,1), "ram_speed_mm_per_sec": round(v,2)}, final


# ---------------------------------------------------------------------------
# UI components
# ---------------------------------------------------------------------------

def card(label, value, status):
    css = {"✅ Good": "card-good", "⚠️ Warning": "card-warn", "🔴 Out of range": "card-bad"}
    cls = css.get(status, "card-warn")
    return f"""
    <div class="card {cls}">
        <p class="card-label">{label}</p>
        <p class="card-value">{value}</p>
        <p class="card-status">{status}</p>
    </div>"""


def setpoint_box(label, value, unit):
    return f"""
    <div class="setpoint-box">
        <p class="setpoint-label">{label}</p>
        <p class="setpoint-value">{value}</p>
        <p class="setpoint-unit">{unit}</p>
    </div>"""


def status(val, good_lo, good_hi, warn_lo, warn_hi):
    if good_lo <= val <= good_hi:  return "✅ Good"
    if warn_lo <= val <= warn_hi:  return "⚠️ Warning"
    return "🔴 Out of range"


def quality_gauge(value, key):
    color = "#2e7d32" if value >= 70 else ("#f9a825" if value >= 40 else "#c62828")
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(value, 1),
        number={"suffix": " / 100", "font": {"size": 28, "color": color}},
        title={"text": "Surface Quality Score", "font": {"size": 14, "color": "#444"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#aaa"},
            "bar":  {"color": color, "thickness": 0.25},
            "bgcolor": "white",
            "steps": [
                {"range": [0,  40], "color": "#ffebee"},
                {"range": [40, 70], "color": "#fff8e1"},
                {"range": [70, 100],"color": "#e8f5e9"},
            ],
            "threshold": {"line": {"color": "#333", "width": 3},
                          "thickness": 0.8, "value": 70},
        },
    ))
    fig.update_layout(height=250, margin=dict(t=40, b=0, l=20, r=20),
                      paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True, key=key)


def process_flow(T_b, T_c, T_d, R, v, t_exit, p):
    st.markdown('<p class="section-title">Process Flow</p>', unsafe_allow_html=True)
    c = st.columns([3, 1, 3, 1, 3, 1, 3, 1, 3])
    boxes = [
        (c[0], "step-box",     "🔥", "Billet",    f"{T_b}°C"),
        (c[2], "step-box",     "🛢️", "Container",  f"{T_c}°C"),
        (c[4], "step-box",     "🔩", "Die",        f"Ratio {R}× | {T_d}°C"),
        (c[6], "step-box",     "➡️", "Ram",        f"{v} mm/s"),
        (c[8], "step-box-out", "✅", "Profile Out", f"{t_exit:.0f}°C | {p:.0f} MPa"),
    ]
    arrows = [c[1], c[3], c[5], c[7]]
    for col, cls, icon, label, val in boxes:
        col.markdown(f"""<div class="step-box {cls}">
            <p class="step-icon">{icon}</p>
            <p class="step-label">{label}</p>
            <p class="step-val">{val}</p>
        </div>""", unsafe_allow_html=True)
    for a in arrows:
        a.markdown("<div style='text-align:center;font-size:1.4rem;padding-top:28px;color:#999'>→</div>",
                   unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def sidebar():
    with st.sidebar:
        st.markdown("## ⚙️ Extrusion Optimiser")
        st.caption("Physics-Informed Neural Network")
        st.divider()

        st.markdown("**Model**")
        st.markdown("- 6 hidden layers · 128 neurons · Tanh")
        st.markdown("- 84,868 trainable parameters")
        st.markdown("- Trained 200 epochs · ~30s on CPU")
        st.divider()

        st.markdown("**Accuracy (test set)**")
        st.markdown("- Exit temp: **±1.74°C**")
        st.markdown("- Pressure:  **±1.55 MPa**")
        st.markdown("- Exit speed:**±2.40 mm/s**")
        st.markdown("- Quality:   **±0.98 / 100**")
        st.divider()

        st.markdown("**Supported Alloys**")
        st.markdown("AA6061 · AA6063 · AA7075")
        st.divider()

        st.markdown("**How to use**")
        st.markdown(
            "**Tab 1:** Drag sliders — predictions update live.\n\n"
            "**Tab 2:** Enter targets → click to get recommended setpoints.\n\n"
            "**Tab 3:** Inspect model accuracy and parity plots."
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    sidebar()

    st.markdown("# ⚙️ Aluminium Extrusion Process Optimiser")
    st.markdown(
        "A **Physics-Informed Neural Network** that predicts process outputs "
        "and recommends optimal setpoints — replacing operator trial and error."
    )
    st.divider()

    model, xs, ys = load_model_and_scalers()
    if model is None:
        st.error("Model not found. Run `python train.py` first.")
        st.stop()

    tab1, tab2, tab3 = st.tabs([
        "🔍  Forward Prediction",
        "🎯  Inverse Optimiser",
        "📊  Model Accuracy",
    ])

    # -------------------------------------------------------------------
    # TAB 1
    # -------------------------------------------------------------------
    with tab1:
        st.markdown("### What will happen if I use these setpoints?")
        st.caption("Predictions update live as you move the sliders — no button needed.")

        col_in, col_out = st.columns([1, 1.4], gap="large")

        with col_in:
            st.markdown('<p class="section-title">Material & Geometry</p>',
                        unsafe_allow_html=True)
            alloy = st.selectbox("Alloy Grade", ["AA6063", "AA6061", "AA7075"])
            D     = st.slider("Billet Diameter (mm)", 80, 150, 100)
            R     = st.slider("Extrusion Ratio", 10, 50, 12)

            st.markdown('<p class="section-title">Process Setpoints</p>',
                        unsafe_allow_html=True)
            T_b = st.slider("Billet Temperature (°C)",    420, 500, 490)
            T_c = st.slider("Container Temperature (°C)", 380, 460, 450)
            T_d = st.slider("Die Temperature (°C)",       380, 460, 440)
            v   = st.slider("Ram Speed (mm/s)", 1.0, 8.0, 1.5, step=0.1)

        with col_out:
            p = predict(model, xs, ys, build_input(alloy, T_b, T_c, T_d, v, D, R), alloy, v, R)

            st.markdown('<p class="section-title">Predicted Outputs</p>',
                        unsafe_allow_html=True)

            m1, m2 = st.columns(2)
            m3, m4 = st.columns(2)

            m1.markdown(card("Exit Temperature",
                             f"{p['exit_temperature_C']:.1f} °C",
                             status(p['exit_temperature_C'], 480, 520, 450, 540)),
                        unsafe_allow_html=True)
            m2.markdown(card("Ram Pressure",
                             f"{p['ram_pressure_MPa']:.1f} MPa",
                             status(p['ram_pressure_MPa'], 0, 400, 400, 600)),
                        unsafe_allow_html=True)
            m3.markdown(card("Exit Speed",
                             f"{p['exit_speed_mm_per_sec']:.1f} mm/s",
                             status(p['exit_speed_mm_per_sec'], 0, 100, 100, 200)),
                        unsafe_allow_html=True)
            m4.markdown(card("Surface Quality",
                             f"{p['surface_quality_score']:.1f} / 100",
                             status(p['surface_quality_score'], 70, 100, 40, 70)),
                        unsafe_allow_html=True)

            quality_gauge(p["surface_quality_score"], key="fwd_gauge")

        st.divider()
        process_flow(T_b, T_c, T_d, R, v,
                     p["exit_temperature_C"], p["ram_pressure_MPa"])

    # -------------------------------------------------------------------
    # TAB 2
    # -------------------------------------------------------------------
    with tab2:
        st.markdown("### What setpoints will give me the result I want?")
        st.caption("Enter your targets. The optimiser searches the full setpoint space to find the best match.")

        col_t, col_r = st.columns([1, 1.4], gap="large")

        with col_t:
            st.markdown('<p class="section-title">Material & Geometry</p>',
                        unsafe_allow_html=True)
            alloy_i = st.selectbox("Alloy Grade", ["AA6063", "AA6061", "AA7075"],
                                   key="inv_alloy")
            D_i     = st.number_input("Billet Diameter (mm)", 80, 150, 100, key="inv_D")
            R_i     = st.number_input("Extrusion Ratio", 10, 50, 12, key="inv_R")

            st.markdown('<p class="section-title">Target Outputs</p>',
                        unsafe_allow_html=True)
            t_temp = st.slider("Target Exit Temperature (°C)", 450, 540, 495, key="inv_t")
            t_qual = st.slider("Target Quality Score",           0, 100,  85, key="inv_q")

            run = st.button("🔍  Find Optimal Setpoints",
                            use_container_width=True, type="primary")

        with col_r:
            if run:
                with st.spinner("Searching setpoint space..."):
                    sp, fp = optimise(model, xs, ys, alloy_i, D_i, R_i, t_temp, t_qual)

                st.markdown('<p class="section-title">Recommended Setpoints</p>',
                            unsafe_allow_html=True)
                s1, s2, s3, s4 = st.columns(4)
                s1.markdown(setpoint_box("Billet Temp",
                                         sp["billet_temperature_C"], "°C"),
                            unsafe_allow_html=True)
                s2.markdown(setpoint_box("Container Temp",
                                         sp["container_temperature_C"], "°C"),
                            unsafe_allow_html=True)
                s3.markdown(setpoint_box("Die Temp",
                                         sp["die_temperature_C"], "°C"),
                            unsafe_allow_html=True)
                s4.markdown(setpoint_box("Ram Speed",
                                         sp["ram_speed_mm_per_sec"], "mm/s"),
                            unsafe_allow_html=True)

                st.markdown('<p class="section-title">Predicted Result with These Setpoints</p>',
                            unsafe_allow_html=True)

                r1, r2, r3, r4 = st.columns(4)
                r1.markdown(card("Exit Temperature",
                                 f"{fp['exit_temperature_C']:.1f} °C",
                                 status(fp['exit_temperature_C'], 480, 520, 450, 540)),
                            unsafe_allow_html=True)
                r2.markdown(card("Ram Pressure",
                                 f"{fp['ram_pressure_MPa']:.1f} MPa",
                                 status(fp['ram_pressure_MPa'], 0, 400, 400, 600)),
                            unsafe_allow_html=True)
                r3.markdown(card("Exit Speed",
                                 f"{fp['exit_speed_mm_per_sec']:.1f} mm/s",
                                 status(fp['exit_speed_mm_per_sec'], 0, 100, 100, 200)),
                            unsafe_allow_html=True)
                r4.markdown(card("Surface Quality",
                                 f"{fp['surface_quality_score']:.1f} / 100",
                                 status(fp['surface_quality_score'], 70, 100, 40, 70)),
                            unsafe_allow_html=True)

                quality_gauge(fp["surface_quality_score"], key="inv_gauge")

            else:
                st.info("Set your targets on the left and click **Find Optimal Setpoints**.")

    # -------------------------------------------------------------------
    # TAB 3
    # -------------------------------------------------------------------
    with tab3:
        st.markdown("### How accurate is the model?")
        st.caption("All metrics evaluated on the held-out test set — data never seen during training.")

        y_test, y_pred = load_test_predictions()

        OUTPUT_META = [
            {"name": "Exit Temperature",  "col": 0, "unit": "°C",    "good_err": 5},
            {"name": "Ram Pressure",       "col": 1, "unit": "MPa",   "good_err": 5},
            {"name": "Exit Speed",         "col": 2, "unit": "mm/s",  "good_err": 5},
            {"name": "Surface Quality",    "col": 3, "unit": "/ 100", "good_err": 3},
        ]

        st.markdown('<p class="section-title">Mean Absolute Error</p>',
                    unsafe_allow_html=True)
        cols_m = st.columns(4)
        for meta, cm in zip(OUTPUT_META, cols_m):
            i   = meta["col"]
            mae = float(np.mean(np.abs(y_pred[:, i] - y_test[:, i])))
            s   = "✅ Good" if mae <= meta["good_err"] else "⚠️ Review"
            cm.markdown(card(meta["name"], f"± {mae:.2f} {meta['unit']}", s),
                        unsafe_allow_html=True)

        st.divider()
        st.markdown('<p class="section-title">Parity Plots — Predicted vs Actual</p>',
                    unsafe_allow_html=True)
        st.caption("Points lying on the red diagonal line = perfect prediction.")

        p1, p2 = st.columns(2)
        p3, p4 = st.columns(2)

        for meta, pcol in zip(OUTPUT_META, [p1, p2, p3, p4]):
            i      = meta["col"]
            actual = y_test[:, i]
            pred   = y_pred[:, i]
            lo, hi = min(actual.min(), pred.min()), max(actual.max(), pred.max())

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=actual, y=pred, mode="markers",
                                     marker=dict(size=4, opacity=0.5, color="#1565c0"),
                                     name="Samples"))
            fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines",
                                     line=dict(color="#c62828", dash="dash", width=2),
                                     name="Perfect"))
            fig.update_layout(
                title=dict(text=meta["name"], font=dict(size=14)),
                xaxis_title=f"Actual ({meta['unit']})",
                yaxis_title=f"Predicted ({meta['unit']})",
                height=310, showlegend=False,
                margin=dict(t=40, b=40, l=50, r=20),
                plot_bgcolor="#fafafa",
            )
            pcol.plotly_chart(fig, use_container_width=True,
                              key=f"parity_{meta['col']}")

        st.divider()
        st.markdown('<p class="section-title">Exit Temperature Error Distribution</p>',
                    unsafe_allow_html=True)
        errors = y_pred[:, 0] - y_test[:, 0]
        fig_h = px.histogram(x=errors, nbins=40,
                             labels={"x": "Prediction Error (°C)", "y": "Count"},
                             color_discrete_sequence=["#1565c0"])
        fig_h.add_vline(x=0,   line_dash="dash", line_color="#c62828",
                        annotation_text="Zero error", annotation_position="top right")
        fig_h.add_vline(x=2,   line_dash="dot",  line_color="#f9a825")
        fig_h.add_vline(x=-2,  line_dash="dot",  line_color="#f9a825",
                        annotation_text="±2°C band", annotation_position="top left")
        fig_h.update_layout(height=280, margin=dict(t=20, b=40, l=50, r=20),
                            plot_bgcolor="#fafafa")
        st.plotly_chart(fig_h, use_container_width=True, key="err_hist")


if __name__ == "__main__":
    main()
