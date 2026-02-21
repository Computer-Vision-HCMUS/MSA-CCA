# MSA-CCA: Canonical Correlation Analysis

A comprehensive Python implementation of Canonical Correlation Analysis (CCA) using scikit-learn for multivariate statistical analysis.

## Overview

This project provides a complete toolkit for performing Canonical Correlation Analysis on multivariate datasets. CCA is a statistical technique that finds linear relationships between two sets of variables by identifying linear combinations that have maximum correlation. The implementation uses scikit-learn's robust CCA algorithm and provides extensive visualization and analysis tools.

## Features

- **Sklearn-based CCA**: Reliable implementation using scikit-learn's cross_decomposition module
- **Flexible Components**: Automatically handles any number of canonical components
- **Data Utilities**: Functions for reading, writing, visualizing, and analyzing data
- **Statistical Analysis**: Comprehensive statistics including mean, variance, covariance, and correlation
- **Interactive Demo**: Streamlit web application for easy exploration
- **Adaptive Visualization**: Rich visualizations that scale with the number of components
- **Export Results**: Save canonical correlations, weights, loadings, and variates
- **General Purpose**: Works with any paired datasets, not limited to specific examples

## Project Structure

```
MSA-CCA/
│
├── utils.py           # Utility functions (read_csv, write_csv, visualize_data, analyze_data)
├── core.py            # CCA implementation using sklearn
├── main.py            # Command-line interface for CCA
├── demo_app.py        # Streamlit web application
├── requirements.txt   # Python dependencies
├── .env.example       # Environment configuration template (optional)
├── AQ_X1.csv         # Sample dataset 1 (Temperature, Humidity, etc.)
├── AQ_X2.csv         # Sample dataset 2 (Air Quality measurements)
└── README.md         # This file
```

## Installation

1. **Clone or download this repository**

2. **Install required packages**:
   ```bash
   pip install -r requirements.txt
   ```

   Or install manually:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn streamlit
   ```

3. **(Optional) Configure environment settings**:
   ```bash
   cp .env.example .env
   # Edit .env file if you need custom configurations
   ```
   
   Note: The project works out-of-the-box without any environment configuration. The `.env.example` file is provided as a template if you want to customize data paths, output directories, or other optional settings.

## Usage

### Method 1: Command Line Interface (main.py)

Run the interactive command-line interface:

```bash
python main.py
```

Follow the prompts to:
- Specify input data files (default: AQ_X1.csv and AQ_X2.csv)
- View data analysis and visualizations
- Set number of canonical components
- Choose data standardization
- Save results to CSV files

### Method 2: Streamlit Web App (demo_app.py)

Launch the interactive web application:

```bash
streamlit run demo_app.py
```

Features:
- Upload custom CSV files or use default datasets
- Interactive parameter configuration
- Real-time visualization
- Download results as CSV files

### Method 3: Python Script

Use the modules in your own Python code:

```python
from utils import read_csv, analyze_data, standardize_data
from core import CCA

# Load data
X1 = read_csv("AQ_X1.csv")
X2 = read_csv("AQ_X2.csv")

# Analyze data
analyze_data(X1)
analyze_data(X2)

# Standardize (optional but recommended)
X1_std, _, _ = standardize_data(X1)
X2_std, _, _ = standardize_data(X2)

# Perform CCA
cca = CCA(n_components=3)
cca.fit(X1_std, X2_std)

# Print results
cca.print_summary()

# Visualize results
cca.plot_results()

# Get canonical variates
U, V = cca.transform()
```

## Sample Datasets

The project includes two sample datasets from air quality measurements:

- **AQ_X1.csv**: Environmental variables
  - T: Temperature
  - RH: Relative Humidity
  - AH: Absolute Humidity

- **AQ_X2.csv**: Air quality pollutants
  - CO(GT): Carbon Monoxide
  - NMHC(GT): Non-Methane Hydrocarbons
  - C6H6(GT): Benzene
  - NOx(GT): Nitrogen Oxides
  - NO2(GT): Nitrogen Dioxide

## CCA Output

The analysis provides:

1. **Canonical Correlations**: Correlation coefficients between canonical variates
2. **Canonical Weights**: Linear combination coefficients for each variable
3. **Canonical Loadings**: Correlations between original variables and canonical variates
4. **Canonical Variates**: Transformed data in canonical space
5. **Visualizations**:
   - Correlation bar plots
   - Scatter plots of canonical variates
   - Heatmaps of weights and loadings
   - Variance explained plots

## Module Documentation

### utils.py

- `read_csv(file_path)`: Read and clean CSV files
- `write_csv(df, file_path)`: Write DataFrame to CSV
- `visualize_data(df, title)`: Create comprehensive visualizations
- `analyze_data(df)`: Perform statistical analysis
- `standardize_data(df)`: Standardize data (mean=0, std=1)

### core.py

- `CCA(n_components)`: CCA wrapper class using sklearn
  - `fit(X1, X2)`: Fit CCA model using sklearn's algorithm
  - `transform(X1, X2)`: Transform data to canonical space
  - `fit_transform(X1, X2)`: Fit and transform in one step
  - `print_summary()`: Print detailed results
  - `plot_results()`: Visualize CCA results (adaptive to number of components)

## Visualization Features

The visualization system automatically adapts to the number of components:

- **Few components (≤3)**: Detailed view with all plots
- **Many components (>3)**: 
  - Scatter plots for top 6 components
  - Separate detailed heatmaps showing up to 10 components
  - Annotations in heatmaps when ≤5 components
  - Simplified view when >5 components

## Technical Details

- **Algorithm**: Uses sklearn's CCA based on PLS (Partial Least Squares) algorithm
- **Numerical Stability**: sklearn handles regularization automatically
- **Scalability**: Efficient for large datasets with many features
- **Components**: Automatically computes up to min(p, q) components
- **Standardization**: Recommended for interpretability (enabled by default)

## Requirements

- Python 3.7 or higher
- NumPy >= 1.21.0
- Pandas >= 1.3.0
- Matplotlib >= 3.4.0
- Seaborn >= 0.11.0
- Scikit-learn >= 1.0.0
- Streamlit >= 1.20.0

## Theory

Canonical Correlation Analysis (CCA) finds two sets of basis vectors, one for each dataset, such that the correlations between the projections of the datasets onto these basis vectors are maximized.

Given two datasets X₁ (n×p) and X₂ (n×q):
- Find weight vectors a and b
- Maximize: corr(X₁a, X₂b)
- Subject to: var(X₁a) = var(X₂b) = 1

The solution involves:
1. Computing covariance matrices C₁₁, C₂₂, C₁₂
2. Solving the generalized eigenvalue problem
3. Extracting canonical correlations and weights

