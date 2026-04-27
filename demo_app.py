"""
Streamlit Demo App for Canonical Correlation Analysis
Run with: streamlit run demo_app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2
from sklearn.cross_decomposition import CCA as SklearnCCA
from utils import read_csv, analyze_data, standardize_data
from core import CCA
from geometry_cca import (
    get_geometry_description,
    plot_correlation_angle,
    plot_cca_variable_spaces_canonical,
    plot_first_pair_scatter_with_angle,
)
import io


# Page configuration
st.set_page_config(
    page_title="CCA Demo App",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem;
        background: linear-gradient(90deg, #f0f2f6 0%, #ffffff 100%);
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 5px;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# Main title
st.markdown('<div class="main-header">📊 Canonical Correlation Analysis (CCA) Demo</div>', 
            unsafe_allow_html=True)

# Sidebar
st.sidebar.title("⚙️ Configuration")
st.sidebar.markdown("---")

# Data source
st.sidebar.subheader("📁 Nguồn dữ liệu")
data_source = st.sidebar.radio(
    "Chọn cách nhập dữ liệu",
    ["📁 File mặc định (AQ_X1, AQ_X2)", "📤 Upload file CSV", "✏️ Nhập trực tiếp X, Y"],
    index=0,
    help="File mặc định / Upload CSV / Nhập/paste bảng X, Y trực tiếp"
)

uploaded_file_x1 = None
uploaded_file_x2 = None
if data_source == "📤 Upload file CSV":
    uploaded_file_x1 = st.sidebar.file_uploader("Upload X1 (CSV)", type=['csv'], key='x1')
    uploaded_file_x2 = st.sidebar.file_uploader("Upload X2 (CSV)", type=['csv'], key='x2')

# Analysis options
st.sidebar.markdown("---")
st.sidebar.subheader("🔧 Analysis Options")

standardize_data_flag = st.sidebar.checkbox("Standardize data", value=True, 
                                            help="Recommended for CCA")
show_analysis = st.sidebar.checkbox("Show data analysis", value=False)
show_raw_data = st.sidebar.checkbox("Show raw data", value=False)


# Initialize session state
if 'cca_fitted' not in st.session_state:
    st.session_state.cca_fitted = False

# Main content
try:
    # Load data based on source
    if data_source == "📁 File mặc định (AQ_X1, AQ_X2)":
        try:
            X1 = read_csv("AQ_X1.csv")
            X2 = read_csv("AQ_X2.csv")
            st.sidebar.success("✓ Đã dùng file mặc định")
        except Exception as e:
            st.sidebar.error(f"✗ Lỗi: {str(e)}")
            st.info("Vui lòng chọn 'Upload file CSV' hoặc 'Nhập trực tiếp' và cung cấp dữ liệu.")
            st.stop()
    elif data_source == "📤 Upload file CSV":
        if uploaded_file_x1 is not None and uploaded_file_x2 is not None:
            X1 = pd.read_csv(uploaded_file_x1)
            X2 = pd.read_csv(uploaded_file_x2)
            X1 = X1.loc[:, ~X1.columns.str.contains('^Unnamed')]
            X1 = X1.dropna(axis=1, how='all').dropna()
            X2 = X2.loc[:, ~X2.columns.str.contains('^Unnamed')]
            X2 = X2.dropna(axis=1, how='all').dropna()
            st.sidebar.success("✓ Đã tải file lên")
        else:
            st.info("👈 Vui lòng upload cả hai file X1 và X2 trong sidebar.")
            st.stop()
    else:
        # ✏️ Nhập trực tiếp X, Y
        st.subheader("✏️ Nhập trực tiếp hai tập X, Y")
        input_mode = st.radio(
            "Cách nhập",
            ["📋 Sửa bảng trực tiếp", "📄 Dán từ Excel/CSV (paste)"],
            horizontal=True,
            key="input_mode"
        )
        
        if input_mode == "📋 Sửa bảng trực tiếp":
            st.caption("Chỉnh kích thước và điền số vào bảng. X và Y phải có cùng số dòng.")
            c_rx, c_cx, c_cy = st.columns([1, 1, 1])
            with c_rx:
                n_samples = st.number_input("Số mẫu (số dòng)", min_value=2, max_value=500, value=5, key="n_samples")
            with c_cx:
                n_cols_x = st.number_input("Số cột tập X", min_value=1, max_value=50, value=3, key="n_cols_x")
            with c_cy:
                n_cols_y = st.number_input("Số cột tập Y", min_value=1, max_value=50, value=2, key="n_cols_y")
            
            col_x, col_y = st.columns(2)
            with col_x:
                st.write("**Tập X (X1)** — sửa trực tiếp trong bảng:")
                default_x = pd.DataFrame(
                    np.zeros((n_samples, n_cols_x)),
                    columns=[f"X{i}" for i in range(n_cols_x)]
                )
                edited_x = st.data_editor(default_x, use_container_width=True, key="direct_df_x", num_rows="fixed")
                X1 = edited_x.astype(float)
            with col_y:
                st.write("**Tập Y (X2)** — sửa trực tiếp trong bảng:")
                default_y = pd.DataFrame(
                    np.zeros((n_samples, n_cols_y)),
                    columns=[f"Y{i}" for i in range(n_cols_y)]
                )
                edited_y = st.data_editor(default_y, use_container_width=True, key="direct_df_y", num_rows="fixed")
                X2 = edited_y.astype(float)
        else:
            st.caption("Dán dữ liệu (mỗi dòng = 1 mẫu, các cột cách nhau bởi Tab hoặc dấu phẩy). Dán tập X xong rồi tập Y.")
            raw_x = st.text_area(
                "Tập X (X1) — mỗi dòng một mẫu, ví dụ: 1.2, 3, 4.5",
                height=120,
                placeholder="1, 2, 3\n4, 5, 6\n7, 8, 9",
                key="paste_x"
            )
            raw_y = st.text_area(
                "Tập Y (X2) — cùng số dòng với X",
                height=120,
                placeholder="10, 20\n30, 40\n50, 60",
                key="paste_y"
            )
            if not raw_x.strip() or not raw_y.strip():
                st.warning("Vui lòng dán cả hai tập X và Y.")
                st.stop()
            def parse_paste(text):
                """Execute parse paste."""
                rows = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
                data = []
                for ln in rows:
                    parts = ln.replace(",", " ").split()
                    data.append([float(x) for x in parts])
                if not data:
                    return None
                return pd.DataFrame(data, columns=[f"C{i}" for i in range(len(data[0]))])
            X1 = parse_paste(raw_x)
            X2 = parse_paste(raw_y)
            if X1 is None or X2 is None:
                st.error("Không đọc được số. Mỗi dòng phải là các số cách nhau bởi dấu phẩy hoặc space.")
                st.stop()
            if len(X1) != len(X2):
                st.error(f"Số dòng không khớp: X có {len(X1)} dòng, Y có {len(X2)} dòng.")
                st.stop()
            X1.columns = [f"X{i}" for i in range(X1.shape[1])]
            X2.columns = [f"Y{i}" for i in range(X2.shape[1])]
        
        if X1.isna().any().any() or X2.isna().any().any():
            st.warning("⚠️ Bảng chứa ô trống; ô trống sẽ được thay bằng 0.")
            X1 = X1.fillna(0)
            X2 = X2.fillna(0)
        st.success("✓ Dùng dữ liệu vừa nhập. Cuộn xuống để chọn số component và chạy CCA.")
    
    # Check data consistency
    if len(X1) != len(X2):
        st.error(f"⚠️ Error: Datasets have different number of samples! X1: {len(X1)}, X2: {len(X2)}")
        st.stop()
    
    # Display dataset information
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📊 Total Samples", f"{len(X1):,}")
    with col2:
        st.metric("📈 X1 Features", X1.shape[1])
    with col3:
        st.metric("📉 X2 Features", X2.shape[1])
    with col4:
        max_comp = min(X1.shape[1], X2.shape[1])
        st.metric("🔢 Max Components", max_comp)
    
    st.markdown("---")
    
    # Show raw data if requested
    if show_raw_data:
        st.subheader("📋 Raw Data Preview")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**X1 Dataset:**")
            st.dataframe(X1.head(20), use_container_width=True)
        
        with col2:
            st.write("**X2 Dataset:**")
            st.dataframe(X2.head(20), use_container_width=True)
        
        st.markdown("---")
    
    # Show data analysis if requested
    if show_analysis:
        st.subheader("📊 Data Analysis")
        
        tab1, tab2 = st.tabs(["X1 Analysis", "X2 Analysis"])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Descriptive Statistics:**")
                st.dataframe(X1.describe(), use_container_width=True)
            
            with col2:
                st.write("**Correlation Matrix:**")
                fig, ax = plt.subplots(figsize=(6, 5))
                sns.heatmap(X1.corr(), annot=True, cmap='coolwarm', center=0, ax=ax)
                st.pyplot(fig)
        
        with tab2:
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Descriptive Statistics:**")
                st.dataframe(X2.describe(), use_container_width=True)
            
            with col2:
                st.write("**Correlation Matrix:**")
                fig, ax = plt.subplots(figsize=(6, 5))
                sns.heatmap(X2.corr(), annot=True, cmap='coolwarm', center=0, ax=ax)
                st.pyplot(fig)
        
        st.markdown("---")
    
    # CCA parameters
    st.subheader("🎯 CCA Parameters")
    
    max_components = min(X1.shape[1], X2.shape[1])
    n_components = st.slider(
        "Number of components",
        min_value=1,
        max_value=max_components,
        value=max_components,
        help="Number of canonical components to compute"
    )

    st.markdown("##### 🧾 Mô tả dữ liệu đang dùng")
    info_col1, info_col2 = st.columns(2)
    with info_col1:
        st.markdown(
            f"""
            - **X1**: `{X1.shape[0]}` samples, `{X1.shape[1]}` cột
            - **Cột X1**: {", ".join(map(str, X1.columns))}
            """
        )
    with info_col2:
        st.markdown(
            f"""
            - **X2**: `{X2.shape[0]}` samples, `{X2.shape[1]}` cột
            - **Cột X2**: {", ".join(map(str, X2.columns))}
            """
        )

    st.markdown("##### 🔍 Preview 5 dòng dữ liệu")
    preview_col1, preview_col2 = st.columns(2)
    with preview_col1:
        st.caption("X1 (5 dòng đầu)")
        st.dataframe(X1.head(5), use_container_width=True)
    with preview_col2:
        st.caption("X2 (5 dòng đầu)")
        st.dataframe(X2.head(5), use_container_width=True)
    
    # Run CCA button
    if st.button("🚀 Run CCA Analysis", type="primary", use_container_width=True):
        with st.spinner("Running CCA analysis..."):
            # Standardize if requested
            if standardize_data_flag:
                X1_proc, _, _ = standardize_data(X1)
                X2_proc, _, _ = standardize_data(X2)
            else:
                X1_proc = X1
                X2_proc = X2
            
            # Fit CCA
            cca = CCA(n_components=n_components)
            cca.fit(X1_proc, X2_proc)
            
            # Store in session state
            st.session_state.cca = cca
            st.session_state.cca_fitted = True
            st.session_state.X1 = X1
            st.session_state.X2 = X2
            st.session_state.X1_proc = X1_proc
            st.session_state.X2_proc = X2_proc
            
            st.success("✓ CCA analysis completed successfully!")
    
    # Display results if CCA is fitted
    if st.session_state.cca_fitted:
        st.markdown("---")
        st.subheader("📈 CCA Results")
        
        cca = st.session_state.cca
        
        # Canonical correlations
        st.write("**Canonical Correlations:**")
        
        corr_df = pd.DataFrame({
            'Component': [f"CC{i+1}" for i in range(cca.n_components)],
            'Correlation': cca.canonical_correlations,
            'R²': cca.canonical_correlations ** 2,
            'Variance Explained %': (cca.canonical_correlations ** 2) / 
                                   np.sum(cca.canonical_correlations ** 2) * 100
        })

        corr_df_show = corr_df.copy()
        corr_df_show[['Correlation', 'R²']] = corr_df_show[['Correlation', 'R²']].round(6)
        corr_df_show['Variance Explained %'] = corr_df_show['Variance Explained %'].round(2)
        st.dataframe(corr_df_show, use_container_width=True)
        
        # ---- Step-by-step algorithm ----
        st.markdown("---")
        st.subheader("📐 Từng bước thuật toán CCA")
        
        def _df_mat(mat, index=None, columns=None):
            """Internal helper that df mat."""
            return pd.DataFrame(mat, index=index, columns=columns)
        
        with st.expander("1️⃣ Center — centerX, centerY (trung bình mẫu)", expanded=True):
            st.markdown("**centerX** = mean(X1), **centerY** = mean(X2). Dữ liệu sau khi trừ mean: X1_centered, X2_centered.")
            c1, c2 = st.columns(2)
            with c1:
                center_x = pd.DataFrame(cca.X1_mean.reshape(1, -1), columns=cca.feature_names_X1, index=["centerX"])
                st.dataframe(center_x.round(6), use_container_width=True)
            with c2:
                center_y = pd.DataFrame(cca.X2_mean.reshape(1, -1), columns=cca.feature_names_X2, index=["centerY"])
                st.dataframe(center_y.round(6), use_container_width=True)
            st.caption("Preview centered data (5 dòng đầu):")
            st.dataframe(_df_mat(cca.X1_centered[:5], columns=cca.feature_names_X1).round(4), use_container_width=True)
            st.dataframe(_df_mat(cca.X2_centered[:5], columns=cca.feature_names_X2).round(4), use_container_width=True)
        
        with st.expander("2️⃣ Ma trận hiệp phương sai — Sigma11, Sigma22, Sigma12"):
            st.markdown("**Sigma11** = cov(X1), **Sigma22** = cov(X2), **Sigma12** = cross-cov(X1,X2).")
            st.write("**Sigma11** (X1):")
            st.dataframe(_df_mat(cca.Sigma11, cca.feature_names_X1, cca.feature_names_X1).round(6), use_container_width=True)
            st.write("**Sigma22** (X2):")
            st.dataframe(_df_mat(cca.Sigma22, cca.feature_names_X2, cca.feature_names_X2).round(6), use_container_width=True)
            st.write("**Sigma12** (cross-covariance):")
            st.dataframe(_df_mat(cca.Sigma12, cca.feature_names_X1, cca.feature_names_X2).round(6), use_container_width=True)
        
        with st.expander("3️⃣ Cholesky — U1, U2 (Sigma11 = U1ᵀU1, Sigma22 = U2ᵀU2)"):
            st.markdown("Phân tích Cholesky (upper): **Sigma11 = U1ᵀ U1**, **Sigma22 = U2ᵀ U2**.")
            st.write("**U1**:")
            st.dataframe(_df_mat(cca.U1, cca.feature_names_X1, cca.feature_names_X1).round(6), use_container_width=True)
            st.write("**U2**:")
            st.dataframe(_df_mat(cca.U2, cca.feature_names_X2, cca.feature_names_X2).round(6), use_container_width=True)
        
        with st.expander("4️⃣ Ma trận K = (U1⁻¹)ᵀ Sigma12 (U2⁻¹)"):
            st.markdown("**K** dùng cho SVD bước sau.")
            st.dataframe(_df_mat(cca.K, cca.feature_names_X1, cca.feature_names_X2).round(6), use_container_width=True)
        
        with st.expander("5️⃣ SVD của K — U_hat, Λ (rho), V_hat"):
            st.markdown("**K = U_hat · Λ · V_hatᵀ**. **ρ (rho)** = giá trị kỳ dị = canonical correlations.")
            st.write("**U_hat** (trái):")
            st.dataframe(_df_mat(cca.U_hat, cca.feature_names_X1, [f"CC{i+1}" for i in range(cca.n_components)]).round(6), use_container_width=True)
            st.write("**ρ (Lambda)** — canonical correlations:")
            st.dataframe(pd.DataFrame({"Component": [f"CC{i+1}" for i in range(cca.n_components)], "ρ": cca.canonical_correlations}).round(6), use_container_width=True)
            st.write("**V_hat** (phải):")
            st.dataframe(_df_mat(cca.V_hat, cca.feature_names_X2, [f"CC{i+1}" for i in range(cca.n_components)]).round(6), use_container_width=True)
        
        with st.expander("6️⃣ Vectơ canonical — a (weights X1), b (weights X2)"):
            st.markdown("**a = U1⁻¹ U_hat**, **b = U2⁻¹ V_hat**. Đây là trọng số canonical (canonical weights).")
            st.write("**a** (x_weights):")
            st.dataframe(_df_mat(cca.x_weights, cca.feature_names_X1, [f"CC{i+1}" for i in range(cca.n_components)]).round(6), use_container_width=True)
            st.write("**b** (y_weights):")
            st.dataframe(_df_mat(cca.y_weights, cca.feature_names_X2, [f"CC{i+1}" for i in range(cca.n_components)]).round(6), use_container_width=True)
        
        with st.expander("7️⃣ Canonical correlations ρ (p)"):
            st.markdown("**ρ** = correlation giữa U_i và V_i (cặp canonical variate).")
            rho_df = pd.DataFrame({
                "Component": [f"CC{i+1}" for i in range(cca.n_components)],
                "ρ (p)": cca.canonical_correlations,
                "ρ²": cca.canonical_correlations ** 2,
            })
            rho_df_show = rho_df.copy()
            rho_df_show[["ρ (p)", "ρ²"]] = rho_df_show[["ρ (p)", "ρ²"]].round(6)
            st.dataframe(rho_df_show, use_container_width=True)
        
        # Visualizations
        st.markdown("---")
        st.subheader("📊 Visualizations")
        
        # Tabs for different visualizations
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "Canonical Correlations",
            "Canonical Variates",
            "Weights",
            "Loadings",
            "Statistical Tests",
            "📐 Geometric meaning",
        ])
        
        with tab1:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            
            # Bar plot
            ax1.bar(range(1, cca.n_components + 1), cca.canonical_correlations, 
                   color='steelblue', alpha=0.7)
            ax1.set_xlabel('Component', fontsize=12)
            ax1.set_ylabel('Canonical Correlation', fontsize=12)
            ax1.set_title('Canonical Correlations', fontsize=14, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            
            # Variance explained
            variance_explained = cca.canonical_correlations ** 2
            cumulative = np.cumsum(variance_explained) / np.sum(variance_explained) * 100
            x_pos = range(1, len(variance_explained) + 1)
            
            ax2.bar(x_pos, variance_explained, alpha=0.7, label='Individual', color='steelblue')
            ax2_twin = ax2.twinx()
            ax2_twin.plot(x_pos, cumulative, 'ro-', linewidth=2, markersize=8, label='Cumulative %')
            ax2_twin.set_ylabel('Cumulative % Variance', fontsize=12)
            ax2.set_xlabel('Component', fontsize=12)
            ax2.set_ylabel('R²', fontsize=12)
            ax2.set_title('Variance Explained', fontsize=14, fontweight='bold')
            ax2.grid(True, alpha=0.3)
            ax2.legend(loc='upper left')
            ax2_twin.legend(loc='upper right')
            
            plt.tight_layout()
            st.pyplot(fig)
        
        with tab2:
            n_display = min(3, cca.n_components)
            cols = st.columns(n_display)
            
            for i in range(n_display):
                with cols[i]:
                    fig, ax = plt.subplots(figsize=(5, 5))
                    ax.scatter(cca.x_scores[:, i], cca.y_scores[:, i], 
                             alpha=0.5, s=10, color='steelblue')
                    ax.set_xlabel(f'X1 - CC{i+1}', fontsize=10)
                    ax.set_ylabel(f'X2 - CC{i+1}', fontsize=10)
                    ax.set_title(f'CC{i+1}\n(r = {cca.canonical_correlations[i]:.4f})', 
                               fontsize=11, fontweight='bold')
                    ax.grid(True, alpha=0.3)
                    
                    # Add diagonal line
                    lims = [
                        np.min([ax.get_xlim()[0], ax.get_ylim()[0]]),
                        np.max([ax.get_xlim()[1], ax.get_ylim()[1]]),
                    ]
                    ax.plot(lims, lims, 'r--', alpha=0.5, linewidth=1)
                    
                    plt.tight_layout()
                    st.pyplot(fig)
        
        with tab3:
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**X1 Canonical Weights:**")
                weights_x1 = pd.DataFrame(
                    cca.x_weights,
                    index=st.session_state.X1.columns,
                    columns=[f"CC{i+1}" for i in range(cca.n_components)]
                )
                st.dataframe(weights_x1.round(4), use_container_width=True)
                
                fig, ax = plt.subplots(figsize=(6, 5))
                sns.heatmap(weights_x1, annot=True, fmt='.3f', cmap='RdBu_r', 
                          center=0, ax=ax, cbar_kws={'label': 'Weight'})
                ax.set_title('X1 Canonical Weights', fontweight='bold')
                plt.tight_layout()
                st.pyplot(fig)
            
            with col2:
                st.write("**X2 Canonical Weights:**")
                weights_x2 = pd.DataFrame(
                    cca.y_weights,
                    index=st.session_state.X2.columns,
                    columns=[f"CC{i+1}" for i in range(cca.n_components)]
                )
                st.dataframe(weights_x2.round(4), use_container_width=True)
                
                fig, ax = plt.subplots(figsize=(6, 5))
                sns.heatmap(weights_x2, annot=True, fmt='.3f', cmap='RdBu_r',
                          center=0, ax=ax, cbar_kws={'label': 'Weight'})
                ax.set_title('X2 Canonical Weights', fontweight='bold')
                plt.tight_layout()
                st.pyplot(fig)
        
        with tab4:
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**X1 Loadings:**")
                loadings_x1 = pd.DataFrame(
                    cca.x_loadings,
                    index=st.session_state.X1.columns,
                    columns=[f"CC{i+1}" for i in range(cca.n_components)]
                )
                st.dataframe(loadings_x1.round(4), use_container_width=True)
                
                fig, ax = plt.subplots(figsize=(6, 5))
                sns.heatmap(loadings_x1, annot=True, fmt='.3f', cmap='RdBu_r',
                          center=0, ax=ax, cbar_kws={'label': 'Loading'})
                ax.set_title('X1 Loadings', fontweight='bold')
                plt.tight_layout()
                st.pyplot(fig)
            
            with col2:
                st.write("**X2 Loadings:**")
                loadings_x2 = pd.DataFrame(
                    cca.y_loadings,
                    index=st.session_state.X2.columns,
                    columns=[f"CC{i+1}" for i in range(cca.n_components)]
                )
                st.dataframe(loadings_x2.round(4), use_container_width=True)
                
                fig, ax = plt.subplots(figsize=(6, 5))
                sns.heatmap(loadings_x2, annot=True, fmt='.3f', cmap='RdBu_r',
                          center=0, ax=ax, cbar_kws={'label': 'Loading'})
                ax.set_title('X2 Loadings', fontweight='bold')
                plt.tight_layout()
                st.pyplot(fig)

        with tab5:
            st.write("**1. Wilks' Lambda & Overall Significance**")
            rc_sq = cca.canonical_correlations ** 2
            wilks_lambda = np.prod(1 - rc_sq)
            n = len(st.session_state.X1_proc)
            p = st.session_state.X1_proc.shape[1]
            q = st.session_state.X2_proc.shape[1]

            chi2_stat = -(n - 1 - (p + q + 1) / 2) * np.log(wilks_lambda)
            df_chi2 = p * q
            p_value = 1 - chi2.cdf(chi2_stat, df_chi2)

            st.write(f"- Wilks' Lambda: {wilks_lambda:.4f}")
            st.write(f"- Chi-square statistic: {chi2_stat:.4f}")
            st.write(f"- P-value: {p_value:.4e}")

            st.write("**2. Redundancy Analysis (X2)**")
            y_variance_explained = np.mean(cca.y_loadings ** 2, axis=0)
            redundancy = y_variance_explained * rc_sq

            for i, red in enumerate(redundancy):
                st.write(f"- CC{i+1} giải thích {red*100:.2f}% phương sai của tập X2.")
            st.write(f"**Tổng phương sai X2 được giải thích: {np.sum(redundancy)*100:.2f}%**")

            st.write("**3. Permutation Test (CC1)**")
            if st.button("Run Permutation Test (1000 iterations)"):
                with st.spinner("Running..."):
                    count_greater = 0
                    n_perm = 1000
                    for _ in range(n_perm):
                        X2_perm = np.random.permutation(st.session_state.X2_proc)
                        cca_perm = SklearnCCA(n_components=1)
                        cca_perm.fit(st.session_state.X1_proc, X2_perm)
                        X_c_perm, Y_c_perm = cca_perm.transform(st.session_state.X1_proc, X2_perm)
                        rc_perm = np.corrcoef(X_c_perm[:, 0], Y_c_perm[:, 0])[0, 1]
                        if rc_perm >= cca.canonical_correlations[0]:
                            count_greater += 1
                    st.success(f"P-value từ hoán vị: {count_greater / n_perm:.4f}")
        
        with tab6:
            st.markdown("### Geometric meaning of CCA")
            st.markdown(get_geometry_description(cca, language="en"))
            st.markdown("---")
            st.markdown("#### Textbook-style geometry view")
            st.caption(
                "This figure is a conceptual schematic like the reference image: two skewed planes, basis vectors x1/x2 and y1/y2, "
                "canonical directions v_x and v_y, the connector e between their tips, and the angle phi = arccos(r)."
            )
            px = int(cca.X1_centered.shape[1])
            py = int(cca.X2_centered.shape[1])
            with st.expander("Tùy chọn cặp CC & hai biến tạo “mặt phẳng”", expanded=False):
                cc_idx = st.selectbox(
                    "Cặp canonical",
                    options=list(range(cca.n_components)),
                    format_func=lambda i: f"CC{i+1} (r={cca.canonical_correlations[i]:.4f})",
                    key="geom_cc_idx",
                )
                gx1, gx2, gy1, gy2 = st.columns(4)
                with gx1:
                    ix0 = st.number_input("X — cột 1", 0, max(0, px - 1), 0, key="geom_ix0")
                with gx2:
                    ix1 = st.number_input("X — cột 2", 0, max(0, px - 1), min(1, px - 1) if px > 1 else 0, key="geom_ix1")
                with gy1:
                    iy0 = st.number_input("Y — cột 1", 0, max(0, py - 1), 0, key="geom_iy0")
                with gy2:
                    iy1 = st.number_input("Y — cột 2", 0, max(0, py - 1), min(1, py - 1) if py > 1 else 0, key="geom_iy1")

            # Controls for point overlay (like geo.html)
            st.markdown("#### Overlay điểm dataset trên mặt phẳng")
            show_points = st.checkbox("Hiển thị điểm", value=True)
            pt_size = st.slider("Cỡ điểm", min_value=2, max_value=12, value=5, step=1)
            pt_alpha = st.slider("Opacity điểm (%)", min_value=20, max_value=100, value=75, step=5)
            cca._geo_show_points = bool(show_points)
            cca._geo_pt_size = int(pt_size)
            cca._geo_pt_alpha = float(pt_alpha)
            try:
                fig_spaces = plot_cca_variable_spaces_canonical(
                    cca,
                    component=int(cc_idx),
                    ix=(int(ix0), int(ix1)),
                    iy=(int(iy0), int(iy1)),
                    language="vi",
                )
            except Exception as ex:
                st.warning(str(ex))
                fig_spaces = plot_cca_variable_spaces_canonical(cca, component=0, language="vi")
            st.pyplot(fig_spaces)
            plt.close(fig_spaces)
            st.markdown("---")
            st.markdown("#### Extra plots")
            fig_r_angle = plot_correlation_angle(cca)
            st.pyplot(fig_r_angle)
            plt.close(fig_r_angle)
            st.markdown("**Scatter U₁ vs V₁ (CC1):** points close to the diagonal U=V correspond to high correlation (small angle θ).")
            fig_scatter_angle = plot_first_pair_scatter_with_angle(cca)
            st.pyplot(fig_scatter_angle)
            plt.close(fig_scatter_angle)
        
        # Download results
        st.markdown("---")
        st.subheader("💾 Download Results")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            csv_corr = corr_df.to_csv(index=False)
            st.download_button(
                "📥 Correlations",
                csv_corr,
                "canonical_correlations.csv",
                "text/csv",
                use_container_width=True
            )
        
        with col2:
            csv_weights_x1 = weights_x1.to_csv()
            st.download_button(
                "📥 X1 Weights",
                csv_weights_x1,
                "weights_X1.csv",
                "text/csv",
                use_container_width=True
            )
        
        with col3:
            csv_weights_x2 = weights_x2.to_csv()
            st.download_button(
                "📥 X2 Weights",
                csv_weights_x2,
                "weights_X2.csv",
                "text/csv",
                use_container_width=True
            )
        
        with col4:
            # Canonical variates
            variates_x1 = pd.DataFrame(
                cca.x_scores,
                columns=[f"U_CC{i+1}" for i in range(cca.n_components)]
            )
            csv_variates = variates_x1.to_csv(index=False)
            st.download_button(
                "📥 Variates X1",
                csv_variates,
                "variates_X1.csv",
                "text/csv",
                use_container_width=True
            )

except Exception as e:
    st.error(f"⚠️ An error occurred: {str(e)}")
    st.exception(e)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; padding: 1rem;'>
    <p>🎓 Canonical Correlation Analysis Demo App</p>
    <p style='font-size: 0.8rem;'>Built with Streamlit | MSA Project</p>
</div>
""", unsafe_allow_html=True)


if __name__ == "__main__":
    # This allows the app to run with: streamlit run demo_app.py
    pass
