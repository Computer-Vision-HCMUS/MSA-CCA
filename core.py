"""
Canonical Correlation Analysis (CCA) Implementation using sklearn
"""
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cross_decomposition import CCA as SklearnCCA


class CCA:
    """
    Canonical Correlation Analysis wrapper using sklearn
    
    CCA finds linear combinations of variables from two datasets
    that have maximum correlation with each other.
    """
    
    def __init__(self, n_components: int = None):
        """
        Initialize CCA
        
        Args:
            n_components: Number of canonical components to compute.
                         If None, uses min(n_features_X1, n_features_X2)
        """
        self.n_components = n_components
        self.sklearn_cca = None
        self.canonical_correlations = None
        self.x_weights = None  # Canonical weights for X1
        self.y_weights = None  # Canonical weights for X2
        self.x_scores = None   # Canonical variates for X1
        self.y_scores = None   # Canonical variates for X2
        self.x_loadings = None # Loadings for X1
        self.y_loadings = None # Loadings for X2
        self.feature_names_X1 = None
        self.feature_names_X2 = None
        self.X1_mean = None
        self.X2_mean = None
        
    def fit(self, X1: np.ndarray, X2: np.ndarray) -> 'CCA':
        """
        Fit CCA model using sklearn
        
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
        
        # Store means for later use
        self.X1_mean = np.mean(X1_array, axis=0)
        self.X2_mean = np.mean(X2_array, axis=0)
        
        # Fit sklearn CCA
        self.sklearn_cca = SklearnCCA(n_components=self.n_components, max_iter=1000)
        self.sklearn_cca.fit(X1_array, X2_array)
        
        # Transform to get canonical variates
        self.x_scores, self.y_scores = self.sklearn_cca.transform(X1_array, X2_array)
        
        # Get canonical weights (coefficients)
        self.x_weights = self.sklearn_cca.x_weights_
        self.y_weights = self.sklearn_cca.y_weights_
        
        # Compute canonical correlations
        self.canonical_correlations = np.zeros(self.n_components)
        for i in range(self.n_components):
            self.canonical_correlations[i] = np.corrcoef(
                self.x_scores[:, i], 
                self.y_scores[:, i]
            )[0, 1]
        
        # Compute loadings (correlations between original variables and canonical variates)
        X1_centered = X1_array - self.X1_mean
        X2_centered = X2_array - self.X2_mean
        
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
        
        print(f"✓ CCA fitted successfully")
        print(f"  Dataset shapes: X1={X1_array.shape}, X2={X2_array.shape}")
        print(f"  Components: {self.n_components}")
        print(f"  Canonical correlations: {self.canonical_correlations}")
        
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
        if X1 is not None or X2 is not None:
            X1_array = X1.values if isinstance(X1, pd.DataFrame) else X1
            X2_array = X2.values if isinstance(X2, pd.DataFrame) else X2
            
            if X1_array is not None and X2_array is not None:
                scores_X1, scores_X2 = self.sklearn_cca.transform(X1_array, X2_array)
            elif X1_array is not None:
                scores_X1 = self.sklearn_cca.transform(X1_array)[0]
                scores_X2 = self.y_scores
            else:
                scores_X1 = self.x_scores
                scores_X2 = self.sklearn_cca.transform(X2_array)[1]
        else:
            scores_X1 = self.x_scores
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
        
        # 4+. Scatter plots of canonical variates
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
