"""
Main script for Canonical Correlation Analysis
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import read_csv, write_csv, visualize_data, analyze_data, standardize_data
from core import CCA


def main():
    """
    Main function to run CCA analysis
    """
    print("\n" + "="*70)
    print(" CANONICAL CORRELATION ANALYSIS (CCA) ")
    print("="*70 + "\n")
    
    # Get user input for file paths
    print("Enter input data files:")
    print("-" * 50)
    
    # Default files in current directory
    default_x1 = "AQ_X1.csv"
    default_x2 = "AQ_X2.csv"
    
    x1_path = input(f"Path to X1 dataset (default: {default_x1}): ").strip()
    if not x1_path:
        x1_path = default_x1
    
    x2_path = input(f"Path to X2 dataset (default: {default_x2}): ").strip()
    if not x2_path:
        x2_path = default_x2
    
    # Check if files exist
    if not os.path.exists(x1_path):
        print(f"✗ Error: File '{x1_path}' not found!")
        return
    
    if not os.path.exists(x2_path):
        print(f"✗ Error: File '{x2_path}' not found!")
        return
    
    print("\n" + "="*70)
    print("STEP 1: LOADING DATA")
    print("="*70)
    
    # Read data
    X1 = read_csv(x1_path)
    X2 = read_csv(x2_path)
    
    # Check if datasets have the same number of samples
    if len(X1) != len(X2):
        print(f"✗ Error: Datasets have different number of samples!")
        print(f"  X1: {len(X1)} samples")
        print(f"  X2: {len(X2)} samples")
        return
    
    print(f"\n✓ Both datasets loaded successfully with {len(X1)} samples")
    
    # Ask if user wants to see data analysis
    show_analysis = input("\nShow data analysis? (y/n, default: y): ").strip().lower()
    if show_analysis != 'n':
        print("\n" + "="*70)
        print("STEP 2: DATA ANALYSIS - X1")
        print("="*70)
        analysis_x1 = analyze_data(X1)
        
        print("\n" + "="*70)
        print("STEP 2: DATA ANALYSIS - X2")
        print("="*70)
        analysis_x2 = analyze_data(X2)
    
    # Ask if user wants to visualize data
    show_viz = input("\nVisualize data? (y/n, default: n): ").strip().lower()
    if show_viz == 'y':
        print("\nGenerating visualizations...")
        visualize_data(X1, "X1 Dataset Visualization")
        visualize_data(X2, "X2 Dataset Visualization")
    
    # Ask for number of components
    max_components = min(X1.shape[1], X2.shape[1])
    print(f"\nMaximum number of components: {max_components}")
    n_comp_input = input(f"Number of components for CCA (default: {max_components}): ").strip()
    
    if n_comp_input:
        try:
            n_components = int(n_comp_input)
            n_components = min(n_components, max_components)
        except ValueError:
            print(f"Invalid input. Using default: {max_components}")
            n_components = max_components
    else:
        n_components = max_components
    
    # Standardize data (recommended for CCA)
    standardize = input("\nStandardize data? (y/n, default: y): ").strip().lower()
    if standardize != 'n':
        print("\nStandardizing data...")
        X1_std, means1, stds1 = standardize_data(X1)
        X2_std, means2, stds2 = standardize_data(X2)
    else:
        X1_std = X1
        X2_std = X2
    
    # Run CCA
    print("\n" + "="*70)
    print("STEP 3: PERFORMING CANONICAL CORRELATION ANALYSIS")
    print("="*70 + "\n")
    
    cca = CCA(n_components=n_components)
    cca.fit(X1_std, X2_std)
    
    # Get canonical variates
    U, V = cca.transform()
    
    # Print detailed results
    cca.print_summary()
    
    # Sample output
    print("\n" + "="*70)
    print("STEP 4: SAMPLE OUTPUT")
    print("="*70 + "\n")
    
    print("First 10 canonical variates for X1:")
    print(pd.DataFrame(
        U[:10],
        columns=[f"CC{i+1}" for i in range(n_components)]
    ))
    
    print("\nFirst 10 canonical variates for X2:")
    print(pd.DataFrame(
        V[:10],
        columns=[f"CC{i+1}" for i in range(n_components)]
    ))
    
    # Verify canonical correlations
    print("\n--- Verification of Canonical Correlations ---")
    for i in range(n_components):
        corr = np.corrcoef(U[:, i], V[:, i])[0, 1]
        print(f"  CC{i+1}: {corr:.6f} (expected: {cca.canonical_correlations[i]:.6f})")
    
    # Save results
    save_results = input("\nSave results to CSV files? (y/n, default: n): ").strip().lower()
    if save_results == 'y':
        # Save canonical variates
        U_df = pd.DataFrame(
            U,
            columns=[f"U_CC{i+1}" for i in range(n_components)]
        )
        V_df = pd.DataFrame(
            V,
            columns=[f"V_CC{i+1}" for i in range(n_components)]
        )
        
        write_csv(U_df, "canonical_variates_X1.csv")
        write_csv(V_df, "canonical_variates_X2.csv")
        
        # Save weights
        weights_x1 = pd.DataFrame(
            cca.x_weights,
            index=X1.columns,
            columns=[f"CC{i+1}" for i in range(n_components)]
        )
        weights_x2 = pd.DataFrame(
            cca.y_weights,
            index=X2.columns,
            columns=[f"CC{i+1}" for i in range(n_components)]
        )
        
        write_csv(weights_x1, "canonical_weights_X1.csv")
        write_csv(weights_x2, "canonical_weights_X2.csv")
        
        # Save correlations
        correlations_df = pd.DataFrame({
            'Component': [f"CC{i+1}" for i in range(n_components)],
            'Canonical_Correlation': cca.canonical_correlations,
            'R_squared': cca.canonical_correlations ** 2
        })
        write_csv(correlations_df, "canonical_correlations.csv")
        
        print("\n✓ Results saved successfully!")
    
    # Visualize results
    show_plots = input("\nShow visualization plots? (y/n, default: y): ").strip().lower()
    if show_plots != 'n':
        print("\nGenerating CCA result plots...")
        cca.plot_results()
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
