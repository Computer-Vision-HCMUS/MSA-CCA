"""
Streamlit app for self-learning CCA geometry.

Restates the CCA problem geometrically, loads data via utils, fits CCA via core,
and explains the link between the formulation (max Corr(U,V), Var(U)=Var(V)=1)
and the fitted weights/scores. Uses the geometry_cca package for plots and text.

Run with: streamlit run visualize_explain_streamlit.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from utils import read_csv, standardize_data
from core import CCA
from geometry_cca import (
    get_geometry_description,
    get_canonical_angles_degrees,
    plot_correlation_angle,
    plot_geometry_schematic,
    plot_first_pair_scatter_with_angle,
)

# Page config
st.set_page_config(
    page_title="CCA Geometry – Self-learning",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("📐 CCA Geometry – Self-learning")
st.markdown("Use this app to understand the **geometric formulation** of Canonical Correlation Analysis and see it on real data.")

# ----- Sidebar: data and options -----
st.sidebar.header("Data & options")
use_default = st.sidebar.checkbox("Use default files (AQ_X1.csv, AQ_X2.csv)", value=True)
uploaded_x1 = st.sidebar.file_uploader("Upload X¹ (CSV)", type=["csv"], key="x1")
uploaded_x2 = st.sidebar.file_uploader("Upload X² (CSV)", type=["csv"], key="x2")
standardize = st.sidebar.checkbox("Standardize data (recommended; gives Var(U)=Var(V)=1 up to scale)", value=True)
st.sidebar.markdown("---")

# ----- Section 1: Restate the CCA problem geometrically -----
st.header("1. Restate the CCA problem geometrically")

st.markdown("""
**Input:** Two random groups:
- **Group 1 (X¹):** $X^{(1)} = (X_1^{(1)}, X_2^{(1)}, \\ldots, X_p^{(1)})^T$ — a vector of $p$ random variables.
- **Group 2 (X²):** $X^{(2)} = (X_1^{(2)}, X_2^{(2)}, \\ldots, X_q^{(2)})^T$ — a vector of $q$ random variables.

**Linear combinations:** We consider
- $U = a^T X^{(1)}$ (scalar),
- $V = b^T X^{(2)}$ (scalar),

where $a$, $b$ are coefficient (weight) vectors.

**Output:** Find $a$ and $b$ such that the **correlation between $U$ and $V$ is maximized**, under the constraints:
- $\\mathrm{Var}(U) = 1$
- $\\mathrm{Var}(V) = 1$

That is, we seek
"""
)
st.latex(r"\max_{a,b} \; \mathrm{Corr}(U, V) \quad \text{subject to} \quad \mathrm{Var}(U) = 1,\; \mathrm{Var}(V) = 1")
st.markdown("")

# ----- Section 2: Load data -----
st.header("2. Load data (X¹ and X²)")

def load_data():
    if use_default:
        try:
            X1 = read_csv("AQ_X1.csv")
            X2 = read_csv("AQ_X2.csv")
            return X1, X2, None
        except Exception as e:
            return None, None, str(e)
    if uploaded_x1 is not None and uploaded_x2 is not None:
        try:
            X1 = pd.read_csv(uploaded_x1)
            X2 = pd.read_csv(uploaded_x2)
            X1 = X1.loc[:, ~X1.columns.str.contains("^Unnamed")].dropna(axis=1, how="all").dropna()
            X2 = X2.loc[:, ~X2.columns.str.contains("^Unnamed")].dropna(axis=1, how="all").dropna()
            return X1, X2, None
        except Exception as e:
            return None, None, str(e)
    return None, None, "Upload both X¹ and X² or use default files."

X1, X2, load_err = load_data()

if load_err:
    st.warning(load_err)
    if use_default:
        st.info("Place AQ_X1.csv and AQ_X2.csv in the project root, or uncheck 'Use default files' and upload CSVs.")
    st.stop()

if len(X1) != len(X2):
    st.error(f"X¹ and X² must have the same number of samples. Got {len(X1)} vs {len(X2)}.")
    st.stop()

st.success(f"Loaded X¹: {X1.shape[0]} samples × {X1.shape[1]} variables (p).  X²: {X2.shape[0]} samples × {X2.shape[1]} variables (q).")
max_comp = min(X1.shape[1], X2.shape[1])
n_components = st.sidebar.slider("Number of components", 1, max_comp, max_comp)

# ----- Section 3: Fit CCA (core + utils) -----
st.header("3. Fit CCA (core + utils)")

if standardize:
    X1_proc, _, _ = standardize_data(X1)
    X2_proc, _, _ = standardize_data(X2)
    st.caption("Data standardized → variances of linear combinations are on a common scale; CCA then finds directions that maximize correlation.")
else:
    X1_proc = X1
    X2_proc = X2

if st.button("Run CCA", type="primary"):
    with st.spinner("Fitting CCA..."):
        cca = CCA(n_components=n_components)
        cca.fit(X1_proc, X2_proc)
        st.session_state["cca"] = cca
        st.session_state["cca_done"] = True
    st.success("CCA fitted. Coefficients $a$ and $b$ are the canonical weights; U and V are the canonical variates (scores).")

if not st.session_state.get("cca_done", False):
    st.info("Click **Run CCA** to fit the model and unlock the geometry interpretation below.")
    st.stop()

cca = st.session_state["cca"]

# ----- Section 4: Link to geometry (a, b = weights; U, V = scores; Var and Corr) -----
st.header("4. Geometry: weights $a$, $b$ and variates $U$, $V$")

st.markdown("""
- **$a$ (X¹ weights)** and **$b$ (X² weights)** are stored as the columns of the weight matrices; each column is one canonical direction.
- **$U$** = `x_scores`, **$V$** = `y_scores`: for each sample, these are the values of the linear combinations $a^T X^{(1)}$ and $b^T X^{(2)}$ for each component.
- With **standardized data**, the algorithm effectively enforces scale so that the correlation $\\mathrm{Corr}(U, V)$ is maximized; the resulting variates have unit variance (or proportional to it) per component.
""")

col1, col2 = st.columns(2)
with col1:
    st.subheader("Canonical correlations (maximized)")
    corr_df = pd.DataFrame({
        "Component": [f"CC{i+1}" for i in range(cca.n_components)],
        "Corr(U,V)": cca.canonical_correlations,
        "Angle θ (deg)": get_canonical_angles_degrees(cca),
    })
    st.dataframe(corr_df.style.format({"Corr(U,V)": "{:.4f}", "Angle θ (deg)": "{:.1f}"}), use_container_width=True)
with col2:
    st.caption("**Correlation and angle:** For standardized variates, $r = \\cos(\\theta)$, so $\\theta = \\arccos(r)$.")

# ----- Section 5: Full geometry text and plots (geometry_cca) -----
st.header("5. Geometric meaning and plots")

st.markdown(get_geometry_description(cca, language="en"))

st.subheader("Plots")
fig_schematic = plot_geometry_schematic()
st.pyplot(fig_schematic)
plt.close(fig_schematic)

fig_r_angle = plot_correlation_angle(cca)
st.pyplot(fig_r_angle)
plt.close(fig_r_angle)

st.markdown("**U₁ vs V₁:** points close to the diagonal U = V correspond to high correlation (small angle θ).")
fig_scatter = plot_first_pair_scatter_with_angle(cca)
st.pyplot(fig_scatter)
plt.close(fig_scatter)

st.markdown("---")
st.caption("CCA Geometry self-learning app — uses `core.CCA`, `utils.read_csv` / `utils.standardize_data`, and `geometry_cca` for explanations and plots.")
