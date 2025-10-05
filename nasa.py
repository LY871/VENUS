import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.graph_objects as go
import shap

# -----------------------------
# Sidebar: select dataset
# -----------------------------
dataset_choice = st.sidebar.selectbox("Select Dataset", ["Kepler", "TESS"])

if dataset_choice == "Kepler":
    model_path = "C:/Users/Layan/OneDrive/Desktop/nasa/stacked_model.pkl"
    scaler_path = "C:/Users/Layan/OneDrive/Desktop/nasa/scaler.pkl"
    encoder_path = "C:/Users/Layan/OneDrive/Desktop/nasa/encoder.pkl"
else:  # TESS
    model_path = "C:/Users/Layan/OneDrive/Desktop/nasa/stacked_model_tess.pkl"
    scaler_path = "C:/Users/Layan/OneDrive/Desktop/nasa/scaler_tess.pkl"
    encoder_path = "C:/Users/Layan/OneDrive/Desktop/nasa/encoder_tess.pkl"

# -----------------------------
# Load model, scaler, encoder
# -----------------------------
stack_model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
encoder = joblib.load(encoder_path)

st.set_page_config(page_title="Exoplanet Prediction & Simulation", layout="wide")
st.title(f"Exoplanet Prediction & Animated Transit Simulation ({dataset_choice})")

# -----------------------------
# Sidebar sliders for input features
# -----------------------------
st.sidebar.header("Adjust Candidate Features")
transit_depth = st.sidebar.slider("Transit Depth (ppm)", 50, 1000, 616)
transit_duration = st.sidebar.slider("Transit Duration (hours)", 0.5, 10.0, 2.9575)
orbital_period = st.sidebar.slider("Orbital Period (days)", 1.0, 50.0, 9.488)
planet_radius = st.sidebar.slider("Planet Radius (Earth radii)", 0.5, 15.0, 2.26)
snr = st.sidebar.slider("Transit SNR", 1.0, 100.0, 35.8)

# FP flags
fp_nt = int(st.sidebar.checkbox("FP Noise", False))
fp_ss = int(st.sidebar.checkbox("FP Secondary Eclipse", False))
fp_co = int(st.sidebar.checkbox("FP Contamination", False))
fp_ec = int(st.sidebar.checkbox("FP Eclipsing Companion", False))

# -----------------------------
# Build input dataframe for ML model
# -----------------------------
X_new = pd.DataFrame([{
    "koi_depth": transit_depth,
    "koi_duration": transit_duration,
    "koi_period": orbital_period,
    "koi_impact": 0,
    "koi_ror": 0,
    "koi_dor": 0,
    "koi_model_snr": snr,
    "koi_prad": planet_radius,
    "koi_sma": 1,
    "koi_teq": 0,
    "koi_incl": 90,
    "koi_eccen": 0,
    "koi_steff": 5800,
    "koi_srad": 1,
    "koi_smass": 1,
    "koi_slogg": 4.4,
    "koi_smet": 0,
    "koi_insol": 1,
    "koi_fpflag_nt": fp_nt,
    "koi_fpflag_ss": fp_ss,
    "koi_fpflag_co": fp_co,
    "koi_fpflag_ec": fp_ec,
    "depth_duration": transit_depth * transit_duration,
    "relative_duration": transit_duration / (orbital_period + 1),
    "fp_sum": fp_nt + fp_ss + fp_co + fp_ec,
    "prad_sma_ratio": planet_radius / (1 + 1e-6)
}])

# Ensure columns order matches model training
all_features = [
    "koi_depth", "koi_duration", "koi_period", "koi_impact", "koi_ror", "koi_dor", 
    "koi_model_snr", "koi_prad", "koi_sma", "koi_teq", "koi_incl", "koi_eccen", 
    "koi_steff", "koi_srad", "koi_smass", "koi_slogg", "koi_smet", "koi_insol", 
    "koi_fpflag_nt", "koi_fpflag_ss", "koi_fpflag_co", "koi_fpflag_ec",
    "depth_duration", "relative_duration", "fp_sum", "prad_sma_ratio"
]
X_new = X_new[all_features]

# Scale input
X_scaled = scaler.transform(X_new)

# -----------------------------
# Predict class and probabilities
# -----------------------------
probs = stack_model.predict_proba(X_scaled)[0]  # probabilities for all classes
pred_index = np.argmax(probs)
pred_label = encoder.inverse_transform([pred_index])[0]

st.subheader("Prediction Results")
st.write(f"Predicted Class: **{pred_label}**")
st.write("Class Probabilities:")
for cls, prob in zip(encoder.classes_, probs):
    st.write(f"- {cls}: {prob*100:.2f}%")

# -----------------------------
# Planet transit simulation
# -----------------------------
time_steps = 300
time = np.linspace(0, 1, time_steps)
theta = 2*np.pi*time

# Planet orbit
orbit_radius = 2
x_planet = orbit_radius*np.cos(theta)
y_planet = orbit_radius*np.sin(theta)

# Kepler/TESS satellite orbit
sat_radius = 3
x_sat = sat_radius*np.cos(theta*0.5)
y_sat = sat_radius*np.sin(theta*0.5)

# Background stars
num_stars = 100
star_x = np.random.uniform(-5, 5, num_stars)
star_y = np.random.uniform(-5, 5, num_stars)
star_sizes = np.random.uniform(1, 3, num_stars)

# Transit fraction
transit_duration_fraction = transit_duration / (24*orbital_period)

# Animation frames
frames = []
for i in range(time_steps):
    in_transit = np.abs((time[i]-0.5) % 1) < transit_duration_fraction/2
    fp_shapes = []
    if in_transit:
        if fp_nt:
            fp_shapes.append(go.Scatter(x=[0], y=[0], mode="markers",
                                        marker=dict(size=110 + 10*np.sin(i/5),
                                                    color="red", opacity=0.3 + 0.3*np.sin(i/5)),
                                        name="FP Noise"))
        if fp_ss:
            fp_shapes.append(go.Scatter(x=[0], y=[0], mode="markers",
                                        marker=dict(size=120 + 10*np.sin(i/4),
                                                    color="orange", opacity=0.3 + 0.3*np.cos(i/5)),
                                        name="FP Secondary"))
        if fp_co:
            fp_shapes.append(go.Scatter(x=[0], y=[0], mode="markers",
                                        marker=dict(size=130 + 10*np.sin(i/6),
                                                    color="purple", opacity=0.2 + 0.3*np.sin(i/7)),
                                        name="FP Contamination"))
        if fp_ec:
            fp_shapes.append(go.Scatter(x=[0], y=[0], mode="markers",
                                        marker=dict(size=140 + 20*np.abs(np.sin(i/3)),
                                                    color="red", opacity=0.1 + 0.4*np.abs(np.sin(i/3))),
                                        name="FP Eclipsing Companion"))

    frame = go.Frame(
        data=[
            go.Scatter(x=star_x, y=star_y, mode="markers",
                       marker=dict(size=star_sizes, color="white"), name="Stars"),
            go.Scatter(x=[0], y=[0], mode="markers",
                       marker=dict(size=80, color="yellow", opacity=0.6,
                                   line=dict(color="gold", width=4)), name="Star"),
            go.Scatter(x=[0], y=[0], mode="markers",
                       marker=dict(size=100, color="cyan", opacity=(transit_depth/1000)*(snr/50)), name="Transit Dip") if in_transit else go.Scatter(x=[], y=[], mode="markers"),
            go.Scatter(x=[x_planet[i]], y=[y_planet[i]], mode="markers",
                       marker=dict(size=planet_radius*10, color="blue"), name="Planet"),
            go.Scatter(x=[x_sat[i]], y=[y_sat[i]], mode="markers",
                       marker=dict(size=20, color="gray"), name=dataset_choice),
            go.Scatter(x=[x_sat[i],0], y=[y_sat[i],0], mode="lines",
                       line=dict(color="yellow", width=2, dash="dot"), name="Light Beam")
        ] + fp_shapes
    )
    frames.append(frame)

# Initial figure
fig = go.Figure(
    data=frames[0].data,
    layout=go.Layout(
        xaxis=dict(range=[-6,6], zeroline=False, title="X Position / Space"),
        yaxis=dict(range=[-6,6], zeroline=False, title="Y Position / Space"),
        paper_bgcolor="black", plot_bgcolor="black",
        font=dict(color="white"),
        title=f"Animated Exoplanet Transit Simulation ({dataset_choice})",
        updatemenus=[dict(
            type="buttons", showactive=False,
            buttons=[dict(label="Play", method="animate",
                          args=[None, {"frame":{"duration":30,"redraw":True}, "fromcurrent":True}])])
        ]
    ),
    frames=frames
)

st.plotly_chart(fig, use_container_width=True)

st.markdown("""
### Animated False Positive Explanation
- **Cyan glow**: normal planet transit  
- **Red pulse**: FP Noise (`nt`)  
- **Orange pulse**: FP Secondary Eclipse (`ss`)  
- **Purple pulse**: FP Contamination (`co`)  
- **Flashing red**: FP Eclipsing Companion (`ec`)  
- FP halos **appear only when planet is in transit** to simulate observation  
""")
