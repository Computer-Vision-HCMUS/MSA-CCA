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


# File upload section
st.sidebar.subheader("📁 Data Upload")

uploaded_file_x1 = st.sidebar.file_uploader("Upload X1 Dataset (CSV)", type=['csv'], key='x1')
uploaded_file_x2 = st.sidebar.file_uploader("Upload X2 Dataset (CSV)", type=['csv'], key='x2')

use_default = st.sidebar.checkbox("Use default files (AQ_X1.csv, AQ_X2.csv)", value=True)

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
    # Load data
    if use_default:
        try:
            X1 = read_csv("AQ_X1.csv")
            X2 = read_csv("AQ_X2.csv")
            st.sidebar.success("✓ Default files loaded")
        except Exception as e:
            st.sidebar.error(f"✗ Error loading default files: {str(e)}")
            st.info("Please upload your CSV files using the sidebar.")
            st.stop()
    else:
        if uploaded_file_x1 is not None and uploaded_file_x2 is not None:
            X1 = pd.read_csv(uploaded_file_x1)
            X2 = pd.read_csv(uploaded_file_x2)
            
            # Clean data
            X1 = X1.loc[:, ~X1.columns.str.contains('^Unnamed')]
            X1 = X1.dropna(axis=1, how='all').dropna()
            X2 = X2.loc[:, ~X2.columns.str.contains('^Unnamed')]
            X2 = X2.dropna(axis=1, how='all').dropna()
            
            st.sidebar.success("✓ Files uploaded successfully")
        else:
            st.info("👈 Please upload both X1 and X2 datasets using the sidebar, or check 'Use default files'.")
            st.stop()
    
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
        
        st.dataframe(corr_df.style.format({
            'Correlation': '{:.6f}',
            'R²': '{:.6f}',
            'Variance Explained %': '{:.2f}%'
        }), use_container_width=True)
        
        # Visualizations
        st.markdown("---")
        st.subheader("📊 Visualizations")
        
        # Tabs for different visualizations
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Canonical Correlations",
            "Canonical Variates",
            "Weights",
            "Loadings",
            "Statistical Tests"
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
                st.dataframe(weights_x1.style.format('{:.4f}'), use_container_width=True)
                
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
                st.dataframe(weights_x2.style.format('{:.4f}'), use_container_width=True)
                
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
                st.dataframe(loadings_x1.style.format('{:.4f}'), use_container_width=True)
                
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
                st.dataframe(loadings_x2.style.format('{:.4f}'), use_container_width=True)
                
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
