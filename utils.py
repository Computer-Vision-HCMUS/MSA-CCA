"""
Utility functions for data processing and analysis
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, Any


def read_csv(file_path: str) -> pd.DataFrame:
    """
    Read CSV file and return a clean DataFrame
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        pd.DataFrame: Cleaned dataframe with only non-empty columns
    """
    try:
        # Read CSV file
        df = pd.read_csv(file_path)
        
        # Remove columns that are entirely empty or unnamed
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        df = df.dropna(axis=1, how='all')
        
        # Remove rows with any missing values
        df = df.dropna()
        
        print(f"✓ Successfully read {file_path}")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
        
        return df
    except Exception as e:
        print(f"✗ Error reading {file_path}: {str(e)}")
        raise


def write_csv(df: pd.DataFrame, file_path: str) -> None:
    """
    Write DataFrame to CSV file
    
    Args:
        df: DataFrame to write
        file_path: Output file path
    """
    try:
        df.to_csv(file_path, index=False)
        print(f"✓ Successfully wrote to {file_path}")
    except Exception as e:
        print(f"✗ Error writing {file_path}: {str(e)}")
        raise


def visualize_data(df: pd.DataFrame, title: str = "Data Visualization") -> None:
    """
    Visualize data with multiple plots
    
    Args:
        df: DataFrame to visualize
        title: Title for the visualization
    """
    n_cols = len(df.columns)
    
    # Create subplots dynamically based on number of columns
    fig = plt.figure(figsize=(15, 10))
    
    # 1. Histograms for each variable
    for i, col in enumerate(df.columns):
        plt.subplot(3, n_cols, i + 1)
        plt.hist(df[col], bins=30, edgecolor='black', alpha=0.7)
        plt.title(f'{col} Distribution')
        plt.xlabel(col)
        plt.ylabel('Frequency')
    
    # 2. Box plots
    for i, col in enumerate(df.columns):
        plt.subplot(3, n_cols, n_cols + i + 1)
        plt.boxplot(df[col])
        plt.title(f'{col} Box Plot')
        plt.ylabel(col)
    
    # 3. Time series plots
    for i, col in enumerate(df.columns):
        plt.subplot(3, n_cols, 2 * n_cols + i + 1)
        plt.plot(df[col], linewidth=0.5, alpha=0.7)
        plt.title(f'{col} Over Time')
        plt.xlabel('Index')
        plt.ylabel(col)
    
    plt.suptitle(title, fontsize=16, y=1.00)
    plt.tight_layout()
    plt.show()
    
    # Correlation heatmap if multiple columns
    if n_cols > 1:
        plt.figure(figsize=(8, 6))
        sns.heatmap(df.corr(), annot=True, cmap='coolwarm', center=0, 
                    square=True, linewidths=1)
        plt.title(f'{title} - Correlation Matrix')
        plt.tight_layout()
        plt.show()


def analyze_data(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Perform statistical analysis on the data
    
    Args:
        df: DataFrame to analyze
        
    Returns:
        Dictionary containing statistical measures
    """
    analysis = {
        'shape': df.shape,
        'columns': list(df.columns),
        'mean': df.mean().to_dict(),
        'variance': df.var().to_dict(),
        'std': df.std().to_dict(),
        'min': df.min().to_dict(),
        'max': df.max().to_dict(),
        'median': df.median().to_dict(),
        'covariance_matrix': df.cov(),
        'correlation_matrix': df.corr(),
    }
    
    # Print summary
    print("\n" + "="*60)
    print("DATA ANALYSIS SUMMARY")
    print("="*60)
    print(f"\nShape: {analysis['shape']}")
    print(f"Columns: {analysis['columns']}")
    
    print("\n--- Mean ---")
    for col, val in analysis['mean'].items():
        print(f"  {col:15s}: {val:.4f}")
    
    print("\n--- Standard Deviation ---")
    for col, val in analysis['std'].items():
        print(f"  {col:15s}: {val:.4f}")
    
    print("\n--- Variance ---")
    for col, val in analysis['variance'].items():
        print(f"  {col:15s}: {val:.4f}")
    
    print("\n--- Min / Max ---")
    for col in analysis['columns']:
        print(f"  {col:15s}: [{analysis['min'][col]:.4f}, {analysis['max'][col]:.4f}]")
    
    print("\n--- Covariance Matrix ---")
    print(analysis['covariance_matrix'])
    
    print("\n--- Correlation Matrix ---")
    print(analysis['correlation_matrix'])
    print("="*60 + "\n")
    
    return analysis


def standardize_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, float], Dict[str, float]]:
    """
    Standardize data (zero mean, unit variance)
    
    Args:
        df: DataFrame to standardize
        
    Returns:
        Tuple of (standardized DataFrame, means dict, stds dict)
    """
    means = df.mean().to_dict()
    stds = df.std().to_dict()
    
    df_std = (df - df.mean()) / df.std()
    
    print(f"✓ Data standardized: mean=0, std=1")
    
    return df_std, means, stds
