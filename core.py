"""
Canonical Correlation Analysis (CCA) Implementation via Covariance Matrices
Based on Algorithm 1: CCA via Covariance Matrices using Cholesky Decomposition
"""
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import linalg


class CCA:
    """
    Canonical Correlation Analysis via Covariance Matrices
    
    CCA finds linear combinations of variables from two datasets
    that have maximum correlation with each other.
    
    Algorithm: CCA via Covariance Matrices using Cholesky Decomposition
    Steps:
    1. Cholesky Decomposition: Sigma11 = U1^T * U1 and Sigma22 = U2^T * U2
    2. Form the matrix K = (U1^-1)^T * Sigma12 * (U2^-1)
    3. Singular Value Decomposition (SVD): K = U_hat * Lambda * V_hat^T
    4. Recover Canonical Variables: rho = diag(Lambda), a = U1^-1 * U_hat and b = U2^-1 * V_hat
    """
    
    def __init__(self, n_components: int = None):
        """
        Initialize CCA
        
        Args:
            n_components: Number of canonical components to compute.
                         If None, uses min(n_features_X1, n_features_X2)
        """
        self.n_components = n_components
        self.canonical_correlations = None
        self.x_weights = None  # Canonical weights for X1 (a)
        self.y_weights = None  # Canonical weights for X2 (b)
        self.x_scores = None   # Canonical variates for X1
        self.y_scores = None   # Canonical variates for X2
        self.x_loadings = None # Loadings for X1
        self.y_loadings = None # Loadings for X2
        self.feature_names_X1 = None
        self.feature_names_X2 = None
        self.X1_mean = None
        self.X2_mean = None
        
        # Internal matrices
        self.Sigma11 = None  # Covariance matrix of X1
        self.Sigma22 = None  # Covariance matrix of X2
        self.Sigma12 = None  # Cross-covariance matrix
        self.U1 = None       # Cholesky factor for Sigma11
        self.U2 = None       # Cholesky factor for Sigma22
        
    def fit(self, X1: np.ndarray, X2: np.ndarray) -> 'CCA':
        """
        Fit CCA model using covariance matrices and Cholesky decomposition
        
        Algorithm Steps:
        1. Compute covariance matrices Sigma11, Sigma22, Sigma12
        2. Cholesky Decomposition: Sigma11 = U1^T * U1 and Sigma22 = U2^T * U2
        3. Form matrix K = (U1^-1)^T * Sigma12 * (U2^-1)
        4. Compute SVD of K: K = U_hat * Lambda * V_hat^T
        5. Recover canonical vectors: a = U1^-1 * U_hat and b = U2^-1 * V_hat
        
        Args:
            X1: First dataset, shape (n_samples, n_features_X1)
            X2: Second dataset, shape (n_samples, n_features_X2)
            
        Returns:
            self
        """
        # Store feature names
        if isinstance(X1, pd.DataFrame):
            self.feature_names_X1 = X1.columns.tolist()
            X1_array = X1.values
        else:
            X1_array = X1
            self.feature_names_X1 = [f"X1_{i+1}" for i in range(X1_array.shape[1])]
            
        if isinstance(X2, pd.DataFrame):
            self.feature_names_X2 = X2.columns.tolist()
            X2_array = X2.values
        else:
            X2_array = X2
            self.feature_names_X2 = [f"X2_{i+1}" for i in range(X2_array.shape[1])]
        
        n_samples, p = X1_array.shape
        _, q = X2_array.shape
        
        # Determine number of components
        if self.n_components is None:
            self.n_components = min(p, q)
        else:
            self.n_components = min(self.n_components, p, q)
        
        # Store means for centering
        self.X1_mean = np.mean(X1_array, axis=0)
        self.X2_mean = np.mean(X2_array, axis=0)
        
        # Center the data
        X1_centered = X1_array - self.X1_mean
        X2_centered = X2_array - self.X2_mean
        
        print("\n" + "="*70)
        print("ALGORITHM 1: CCA via Covariance Matrices")
        print("="*70)
        
        # Step 1: Compute covariance matrices
        print("\n/* Computing Covariance Matrices */")
        self.Sigma11 = (X1_centered.T @ X1_centered) / (n_samples - 1)
        self.Sigma22 = (X2_centered.T @ X2_centered) / (n_samples - 1)
        self.Sigma12 = (X1_centered.T @ X2_centered) / (n_samples - 1)
        print(f"  Sigma11 shape: {self.Sigma11.shape}")
        print(f"  Sigma22 shape: {self.Sigma22.shape}")
        print(f"  Sigma12 shape: {self.Sigma12.shape}")
        
        # Step 2: Cholesky Decomposition
        print("\n/* Step 1: Cholesky Decomposition */")
        print("  Cholesky factors: Sigma11 = U1^T * U1 and Sigma22 = U2^T * U2")
        
        # Add small regularization for numerical stability
        reg = 1e-6
        self.U1 = linalg.cholesky(self.Sigma11 + reg * np.eye(p), lower=False)
        self.U2 = linalg.cholesky(self.Sigma22 + reg * np.eye(q), lower=False)
        print(f"  U1 shape: {self.U1.shape}")
        print(f"  U2 shape: {self.U2.shape}")
        
        # Step 3: Form the matrix K
        print("\n/* Step 2: Form the matrix K */")
        print("  Compute K = (U1^-1)^T * Sigma12 * (U2^-1)")
        
        U1_inv = linalg.inv(self.U1)
        U2_inv = linalg.inv(self.U2)
        K = U1_inv.T @ self.Sigma12 @ U2_inv
        print(f"  K shape: {K.shape}")
        
        # Step 4: Singular Value Decomposition (SVD)
        print("\n/* Step 3: Singular Value Decomposition (SVD) */")
        print("  Compute SVD of K: K = U_hat * Lambda * V_hat^T")
        
        U_hat, Lambda, Vt_hat = linalg.svd(K, full_matrices=False)
        V_hat = Vt_hat.T
        
        # Take only the first n_components
        U_hat = U_hat[:, :self.n_components]
        Lambda = Lambda[:self.n_components]
        V_hat = V_hat[:, :self.n_components]
        
        print(f"  U_hat shape: {U_hat.shape}")
        print(f"  Lambda shape: {Lambda.shape}")
        print(f"  V_hat shape: {V_hat.shape}")
        
        # Step 5: Recover Canonical Variables
        print("\n/* Step 4: Recover Canonical Variables */")
        print("  Set canonical correlations: rho = diag(Lambda)")
        self.canonical_correlations = Lambda
        print(f"  rho = {self.canonical_correlations}")
        
        print("  Solve for vectors: a = U1^-1 * U_hat and b = U2^-1 * V_hat")
        self.x_weights = U1_inv @ U_hat  # a
        self.y_weights = U2_inv @ V_hat  # b
        print(f"  a shape: {self.x_weights.shape}")
        print(f"  b shape: {self.y_weights.shape}")
        
        # Compute canonical variates (scores)
        self.x_scores = X1_centered @ self.x_weights
        self.y_scores = X2_centered @ self.y_weights
        
        # Verify the canonical correlations
        print("\n/* Verification */")
        computed_corr = np.zeros(self.n_components)
        for i in range(self.n_components):
            computed_corr[i] = np.corrcoef(
                self.x_scores[:, i], 
                self.y_scores[:, i]
            )[0, 1]
        print(f"  Computed correlations: {computed_corr}")
        
        # Compute loadings (correlations between original variables and canonical variates)
        self.x_loadings = np.zeros((p, self.n_components))
        self.y_loadings = np.zeros((q, self.n_components))
        
        for i in range(self.n_components):
            for j in range(p):
                self.x_loadings[j, i] = np.corrcoef(
                    X1_centered[:, j], 
                    self.x_scores[:, i]
                )[0, 1]
            for j in range(q):
                self.y_loadings[j, i] = np.corrcoef(
                    X2_centered[:, j], 
                    self.y_scores[:, i]
                )[0, 1]
        
        print("\n" + "="*70)
        print("✓ CCA fitted successfully")
        print(f"  Dataset shapes: X1={X1_array.shape}, X2={X2_array.shape}")
        print(f"  Components: {self.n_components}")
        print(f"  Canonical correlations: {self.canonical_correlations}")
        print("="*70)
        
        return self
    
    def transform(self, X1: np.ndarray = None, X2: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform data to canonical space
        
        Args:
            X1: First dataset (optional, if None uses fitted data)
            X2: Second dataset (optional, if None uses fitted data)
            
        Returns:
            Tuple of transformed datasets
        """
        if X1 is not None:
            X1_array = X1.values if isinstance(X1, pd.DataFrame) else X1
            X1_centered = X1_array - self.X1_mean
            scores_X1 = X1_centered @ self.x_weights
        else:
            scores_X1 = self.x_scores
            
        if X2 is not None:
            X2_array = X2.values if isinstance(X2, pd.DataFrame) else X2
            X2_centered = X2_array - self.X2_mean
            scores_X2 = X2_centered @ self.y_weights
        else:
            scores_X2 = self.y_scores
            
        return scores_X1, scores_X2
    
    def fit_transform(self, X1: np.ndarray, X2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit CCA and transform data
        
        Args:
            X1: First dataset
            X2: Second dataset
            
        Returns:
            Tuple of (X1_transformed, X2_transformed)
        """
        self.fit(X1, X2)
        return self.x_scores, self.y_scores
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get summary of CCA results
        
        Returns:
            Dictionary containing CCA results
        """
        summary = {
            'n_components': self.n_components,
            'canonical_correlations': self.canonical_correlations,
            'x_weights': self.x_weights,
            'y_weights': self.y_weights,
            'x_scores': self.x_scores,
            'y_scores': self.y_scores,
            'x_loadings': self.x_loadings,
            'y_loadings': self.y_loadings,
            'feature_names_X1': self.feature_names_X1,
            'feature_names_X2': self.feature_names_X2,
        }
        return summary
    
    def print_summary(self) -> None:
        """
        Print detailed summary of CCA results
        """
        print("\n" + "="*70)
        print("CANONICAL CORRELATION ANALYSIS SUMMARY")
        print("="*70)
        
        print(f"\nNumber of components: {self.n_components}")
        
        print("\n--- Canonical Correlations ---")
        for i, corr in enumerate(self.canonical_correlations):
            print(f"  Component {i+1}: {corr:.6f} (r² = {corr**2:.6f})")
        
        print("\n--- Canonical Weights for X1 ---")
        weights_df_x1 = pd.DataFrame(
            self.x_weights,
            index=self.feature_names_X1,
            columns=[f"CC{i+1}" for i in range(self.n_components)]
        )
        print(weights_df_x1)
        
        print("\n--- Canonical Weights for X2 ---")
        weights_df_x2 = pd.DataFrame(
            self.y_weights,
            index=self.feature_names_X2,
            columns=[f"CC{i+1}" for i in range(self.n_components)]
        )
        print(weights_df_x2)
        
        print("\n--- Loadings for X1 ---")
        loadings_df_x1 = pd.DataFrame(
            self.x_loadings,
            index=self.feature_names_X1,
            columns=[f"CC{i+1}" for i in range(self.n_components)]
        )
        print(loadings_df_x1)
        
        print("\n--- Loadings for X2 ---")
        loadings_df_x2 = pd.DataFrame(
            self.y_loadings,
            index=self.feature_names_X2,
            columns=[f"CC{i+1}" for i in range(self.n_components)]
        )
        print(loadings_df_x2)
        
        print("="*70 + "\n")
    
    def plot_results(self) -> None:
        """
        Visualize CCA results (flexible for any number of components)
        """
        n_comp = self.n_components
        
        # Determine layout based on number of components
        if n_comp <= 3:
            n_scatter = n_comp
            fig = plt.figure(figsize=(18, 12))
            layout_rows, layout_cols = 3, 3
        else:
            n_scatter = min(6, n_comp)  # Show up to 6 scatter plots
            fig = plt.figure(figsize=(20, 14))
            layout_rows, layout_cols = 4, 3
        
        # 1. Canonical correlations bar plot
        plt.subplot(layout_rows, layout_cols, 1)
        plt.bar(range(1, self.n_components + 1), self.canonical_correlations, 
                color='steelblue', alpha=0.7)
        plt.xlabel('Component', fontweight='bold')
        plt.ylabel('Canonical Correlation', fontweight='bold')
        plt.title('Canonical Correlations', fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.xticks(range(1, min(n_comp+1, 11)))  # Limit x-axis labels if many components
        
        # 2. Variance explained
        plt.subplot(layout_rows, layout_cols, 2)
        variance_explained = self.canonical_correlations ** 2
        cumulative = np.cumsum(variance_explained) / np.sum(variance_explained) * 100
        x_pos = range(1, len(variance_explained) + 1)
        plt.bar(x_pos, variance_explained, alpha=0.7, label='Individual', color='steelblue')
        plt.plot(x_pos, cumulative, 'ro-', linewidth=2, markersize=6, label='Cumulative %')
        plt.xlabel('Component', fontweight='bold')
        plt.ylabel('R² / Cumulative %', fontweight='bold')
        plt.title('Variance Explained', fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 3. Top loadings comparison
        plt.subplot(layout_rows, layout_cols, 3)
        top_loadings_x1 = np.abs(self.x_loadings[:, 0])
        top_loadings_x2 = np.abs(self.y_loadings[:, 0])
        x_pos_load = np.arange(len(top_loadings_x1))
        width = 0.35
        plt.barh(x_pos_load, top_loadings_x1, width, label='X1', alpha=0.7)
        plt.barh(x_pos_load + width, top_loadings_x2[:len(top_loadings_x1)] if len(top_loadings_x2) >= len(top_loadings_x1) else list(top_loadings_x2) + [0]*(len(top_loadings_x1)-len(top_loadings_x2)), 
                 width, label='X2', alpha=0.7)
        plt.yticks(x_pos_load + width/2, self.feature_names_X1, fontsize=8)
        plt.xlabel('|Loading| for CC1', fontweight='bold')
        plt.title('Top Loadings (CC1)', fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3, axis='x')
        
        # 4. Scatter plots of canonical variates
        scatter_positions = [4, 5, 6, 7, 8, 9] if layout_rows == 3 else [4, 5, 6, 7, 8, 9, 10, 11, 12]
        for i in range(min(n_scatter, len(scatter_positions))):
            if i >= self.n_components:
                break
            plt.subplot(layout_rows, layout_cols, scatter_positions[i])
            plt.scatter(self.x_scores[:, i], self.y_scores[:, i], 
                       alpha=0.4, s=8, color='steelblue')
            plt.xlabel(f'X1 - CC{i+1}', fontweight='bold')
            plt.ylabel(f'X2 - CC{i+1}', fontweight='bold')
            plt.title(f'CC{i+1} (r = {self.canonical_correlations[i]:.4f})', 
                     fontweight='bold')
            plt.grid(True, alpha=0.3)
            # Add diagonal reference line
            lims = [
                np.min([plt.xlim()[0], plt.ylim()[0]]),
                np.max([plt.xlim()[1], plt.ylim()[1]]),
            ]
            plt.plot(lims, lims, 'r--', alpha=0.4, linewidth=1)
        
        plt.suptitle(f'Canonical Correlation Analysis Results ({n_comp} components)', 
                    fontsize=16, y=0.998, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        # Additional detailed heatmaps if needed
        if n_comp > 3:
            self._plot_detailed_heatmaps()
    
    def _plot_detailed_heatmaps(self) -> None:
        """Plot detailed heatmaps for weights and loadings"""
        n_comp_display = min(10, self.n_components)  # Display up to 10 components in heatmaps
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # X1 Loadings
        sns.heatmap(self.x_loadings[:, :n_comp_display], 
                   annot=True if n_comp_display <= 5 else False, 
                   fmt='.2f', cmap='RdBu_r', center=0, 
                   cbar_kws={'label': 'Loading'},
                   yticklabels=self.feature_names_X1,
                   xticklabels=[f'CC{i+1}' for i in range(n_comp_display)],
                   ax=axes[0, 0])
        axes[0, 0].set_title('X1 Loadings', fontweight='bold', fontsize=12)
        axes[0, 0].set_ylabel('X1 Variables', fontweight='bold')
        
        # X2 Loadings
        sns.heatmap(self.y_loadings[:, :n_comp_display], 
                   annot=True if n_comp_display <= 5 else False,
                   fmt='.2f', cmap='RdBu_r', center=0,
                   cbar_kws={'label': 'Loading'},
                   yticklabels=self.feature_names_X2,
                   xticklabels=[f'CC{i+1}' for i in range(n_comp_display)],
                   ax=axes[0, 1])
        axes[0, 1].set_title('X2 Loadings', fontweight='bold', fontsize=12)
        axes[0, 1].set_ylabel('X2 Variables', fontweight='bold')
        
        # X1 Weights
        sns.heatmap(self.x_weights[:, :n_comp_display], 
                   annot=True if n_comp_display <= 5 else False,
                   fmt='.2f', cmap='RdBu_r', center=0,
                   cbar_kws={'label': 'Weight'},
                   yticklabels=self.feature_names_X1,
                   xticklabels=[f'CC{i+1}' for i in range(n_comp_display)],
                   ax=axes[1, 0])
        axes[1, 0].set_title('X1 Canonical Weights', fontweight='bold', fontsize=12)
        axes[1, 0].set_ylabel('X1 Variables', fontweight='bold')
        
        # X2 Weights
        sns.heatmap(self.y_weights[:, :n_comp_display], 
                   annot=True if n_comp_display <= 5 else False,
                   fmt='.2f', cmap='RdBu_r', center=0,
                   cbar_kws={'label': 'Weight'},
                   yticklabels=self.feature_names_X2,
                   xticklabels=[f'CC{i+1}' for i in range(n_comp_display)],
                   ax=axes[1, 1])
        axes[1, 1].set_title('X2 Canonical Weights', fontweight='bold', fontsize=12)
        axes[1, 1].set_ylabel('X2 Variables', fontweight='bold')
        
        plt.suptitle('Detailed Weights and Loadings', fontsize=14, y=0.995, fontweight='bold')
        plt.tight_layout()
        plt.show()
