"""
Chart Components Module for Pairs Trading Visualization.

This module provides plotting functions using Plotly for visualizing pairs trading analysis.
All charts follow consistent styling with 'plotly_white' template and clear axis labels.

Responsibilities:
- Receive processed data from business logic layer
- Generate interactive Plotly figures
- NO business logic or calculations (Pure presentation layer)
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_raw_price(
    df1: pd.DataFrame, 
    df2: pd.DataFrame, 
    sym1: str, 
    sym2: str
) -> go.Figure:
    """
    Plot raw closing prices of two assets on dual Y-axes.
    
    Visualizes the actual price movements of two assets with different price levels
    using separate Y-axes to handle scale differences.
    
    Args:
        df1: DataFrame for first asset with 'C' (Close) column and datetime index
        df2: DataFrame for second asset with 'C' (Close) column and datetime index
        sym1: Symbol name for first asset (e.g., 'VNM')
        sym2: Symbol name for second asset (e.g., 'VIC')
    
    Returns:
        go.Figure: Plotly figure with dual Y-axes line chart
        
    Raises:
        ValueError: If DataFrames are empty or missing 'C' column
    
    Example:
        >>> fig = plot_raw_price(df_vnm, df_vic, 'VNM', 'VIC')
        >>> fig.show()
    """
    # Validate inputs
    if df1.empty or df2.empty:
        raise ValueError("Input DataFrames cannot be empty")
    
    if 'C' not in df1.columns or 'C' not in df2.columns:
        raise ValueError("DataFrames must contain 'C' (Close) column")
    
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add trace for first asset (primary y-axis)
    fig.add_trace(
        go.Scatter(
            x=df1.index,
            y=df1['C'],
            mode='lines',
            name=sym1,
            line=dict(color='#1f77b4', width=2),
            hovertemplate=f'{sym1}: %{{y:,.2f}}<extra></extra>'
        ),
        secondary_y=False
    )
    
    # Add trace for second asset (secondary y-axis)
    fig.add_trace(
        go.Scatter(
            x=df2.index,
            y=df2['C'],
            mode='lines',
            name=sym2,
            line=dict(color='#ff7f0e', width=2),
            hovertemplate=f'{sym2}: %{{y:,.2f}}<extra></extra>'
        ),
        secondary_y=True
    )
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f'Raw Price Comparison: {sym1} vs {sym2}',
            font=dict(size=20, family='Arial, sans-serif')
        ),
        template='plotly_white',
        hovermode='x unified',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        ),
        height=500
    )
    
    # Set axis titles
    fig.update_xaxes(title_text='Trading Date', showgrid=True)
    fig.update_yaxes(
        title_text=f'{sym1} Price (VND)',
        secondary_y=False,
        showgrid=True
    )
    fig.update_yaxes(
        title_text=f'{sym2} Price (VND)',
        secondary_y=True,
        showgrid=False
    )
    
    return fig


def plot_normalized_price(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    sym1: str,
    sym2: str
) -> go.Figure:
    """
    Plot normalized prices (base 1.0) to compare relative growth.
    
    Normalizes both price series to start at 1.0, enabling direct comparison
    of percentage changes and relative performance.
    
    Args:
        df1: DataFrame for first asset with 'C' (Close) column and datetime index
        df2: DataFrame for second asset with 'C' (Close) column and datetime index
        sym1: Symbol name for first asset
        sym2: Symbol name for second asset
    
    Returns:
        go.Figure: Plotly figure with normalized price comparison
        
    Raises:
        ValueError: If DataFrames are empty, missing 'C' column, or first price is zero
    
    Example:
        >>> fig = plot_normalized_price(df_vnm, df_vic, 'VNM', 'VIC')
        >>> fig.show()
        # Shows both series starting at 1.0
    """
    # Validate inputs
    if df1.empty or df2.empty:
        raise ValueError("Input DataFrames cannot be empty")
    
    if 'C' not in df1.columns or 'C' not in df2.columns:
        raise ValueError("DataFrames must contain 'C' (Close) column")
    
    # Handle division by zero: find first non-zero price
    # This handles edge cases where first price might be 0 due to data errors
    first_nonzero_idx1 = (df1['C'] != 0).idxmax() if (df1['C'] != 0).any() else None
    first_nonzero_idx2 = (df2['C'] != 0).idxmax() if (df2['C'] != 0).any() else None
    
    if first_nonzero_idx1 is None or first_nonzero_idx2 is None:
        raise ValueError("Cannot normalize: all price values are zero")
    
    base_price1 = df1.loc[first_nonzero_idx1, 'C']
    base_price2 = df2.loc[first_nonzero_idx2, 'C']
    
    # Normalize prices to base 1.0 using first non-zero values
    norm1 = df1['C'] / base_price1
    norm2 = df2['C'] / base_price2
    
    # Create figure
    fig = go.Figure()
    
    # Add normalized price traces
    fig.add_trace(
        go.Scatter(
            x=df1.index,
            y=norm1,
            mode='lines',
            name=sym1,
            line=dict(color='#1f77b4', width=2),
            hovertemplate=f'{sym1}: %{{y:.4f}}x<extra></extra>'
        )
    )
    
    fig.add_trace(
        go.Scatter(
            x=df2.index,
            y=norm2,
            mode='lines',
            name=sym2,
            line=dict(color='#ff7f0e', width=2),
            hovertemplate=f'{sym2}: %{{y:.4f}}x<extra></extra>'
        )
    )
    
    # Add horizontal line at y=1.0 (baseline)
    fig.add_hline(
        y=1.0,
        line_dash='dot',
        line_color='gray',
        annotation_text='Baseline',
        annotation_position='right'
    )
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f'Normalized Price (Base 1.0): {sym1} vs {sym2}',
            font=dict(size=20, family='Arial, sans-serif')
        ),
        xaxis_title='Trading Date',
        yaxis_title='Normalized Price (Base = 1.0)',
        template='plotly_white',
        hovermode='x unified',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        ),
        height=500
    )
    
    return fig


def plot_spread_bollinger(
    spread: pd.Series,
    window: int
) -> go.Figure:
    """
    Plot spread with Bollinger Bands (Mean ± 2σ).
    
    Visualizes the CCA-based spread with rolling statistics bands.
    Bollinger Bands indicate potential mean-reversion trading signals:
    - Price touches upper band → Consider SHORT
    - Price touches lower band → Consider LONG
    
    Args:
        spread: Series of spread values with datetime index
        window: Rolling window size for calculating mean and std
    
    Returns:
        go.Figure: Plotly figure with spread line and Bollinger Bands
        
    Raises:
        ValueError: If spread is empty or window is invalid
    
    Example:
        >>> fig = plot_spread_bollinger(spread_series, window=20)
        >>> fig.show()
        # Shows spread with shaded bands
    """
    # Validate inputs
    if spread.empty:
        raise ValueError("Spread series cannot be empty")
    
    if window < 2:
        raise ValueError(f"Window must be >= 2, got {window}")
    
    if len(spread) < window:
        raise ValueError(
            f"Spread length ({len(spread)}) must be >= window ({window})"
        )
    
    # Calculate Bollinger Bands
    rolling_mean = spread.rolling(window=window).mean()
    rolling_std = spread.rolling(window=window).std()
    
    upper_band = rolling_mean + 2 * rolling_std
    lower_band = rolling_mean - 2 * rolling_std
    
    # Create figure
    fig = go.Figure()
    
    # Add upper band (plotted first for fill order)
    fig.add_trace(
        go.Scatter(
            x=spread.index,
            y=upper_band,
            mode='lines',
            name='Upper Band (+2σ)',
            line=dict(color='rgba(255, 0, 0, 0.5)', width=1, dash='dash'),
            hovertemplate='Upper: %{y:.6f}<extra></extra>'
        )
    )
    
    # Add lower band with fill to upper band
    fig.add_trace(
        go.Scatter(
            x=spread.index,
            y=lower_band,
            mode='lines',
            name='Lower Band (-2σ)',
            line=dict(color='rgba(255, 0, 0, 0.5)', width=1, dash='dash'),
            fill='tonexty',  # Fill to previous trace (upper band)
            fillcolor='rgba(255, 0, 0, 0.1)',
            hovertemplate='Lower: %{y:.6f}<extra></extra>'
        )
    )
    
    # Add rolling mean
    fig.add_trace(
        go.Scatter(
            x=spread.index,
            y=rolling_mean,
            mode='lines',
            name='Rolling Mean',
            line=dict(color='orange', width=2, dash='dot'),
            hovertemplate='Mean: %{y:.6f}<extra></extra>'
        )
    )
    
    # Add spread line (on top)
    fig.add_trace(
        go.Scatter(
            x=spread.index,
            y=spread,
            mode='lines',
            name='Spread',
            line=dict(color='#2ca02c', width=2),
            hovertemplate='Spread: %{y:.6f}<extra></extra>'
        )
    )
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f'CCA Spread with Bollinger Bands (Window={window})',
            font=dict(size=20, family='Arial, sans-serif')
        ),
        xaxis_title='Trading Date',
        yaxis_title='Spread Value',
        template='plotly_white',
        hovermode='x unified',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        ),
        height=500
    )
    
    return fig


def plot_zscore(
    zscore: pd.Series,
    threshold: float = 2.0
) -> go.Figure:
    """
    Plot Z-score with trading signal thresholds.
    
    Visualizes standardized spread (Z-score) with horizontal lines at ±threshold
    indicating trading signals. Z-score represents how many standard deviations
    the spread is from its mean.
    
    Args:
        zscore: Series of Z-score values with datetime index
        threshold: Z-score threshold for trading signals (default: 2.0)
    
    Returns:
        go.Figure: Plotly figure with Z-score line and threshold bands
        
    Raises:
        ValueError: If zscore is empty or threshold is non-positive
    
    Example:
        >>> fig = plot_zscore(zscore_series, threshold=2.0)
        >>> fig.show()
        # Shows when |z| > 2 (potential trades)
    """
    # Validate inputs
    if zscore.empty:
        raise ValueError("Z-score series cannot be empty")
    
    if threshold <= 0:
        raise ValueError(f"Threshold must be positive, got {threshold}")
    
    # Create figure
    fig = go.Figure()
    
    # Add Z-score line
    fig.add_trace(
        go.Scatter(
            x=zscore.index,
            y=zscore,
            mode='lines',
            name='Z-Score',
            line=dict(color='#1f77b4', width=2),
            hovertemplate='Z-Score: %{y:.3f}<extra></extra>'
        )
    )
    
    # Add threshold lines
    fig.add_hline(
        y=threshold,
        line_dash='dash',
        line_color='red',
        annotation_text=f'Short Signal (+{threshold}σ)',
        annotation_position='right'
    )
    
    fig.add_hline(
        y=-threshold,
        line_dash='dash',
        line_color='green',
        annotation_text=f'Long Signal (-{threshold}σ)',
        annotation_position='right'
    )
    
    fig.add_hline(
        y=0,
        line_dash='dot',
        line_color='gray',
        annotation_text='Mean',
        annotation_position='right'
    )
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f'Z-Score with Trading Signals (Threshold=±{threshold})',
            font=dict(size=20, family='Arial, sans-serif')
        ),
        xaxis_title='Trading Date',
        yaxis_title='Z-Score (Standard Deviations)',
        template='plotly_white',
        hovermode='x unified',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        ),
        height=500
    )
    
    return fig


def plot_cca_weights(
    w_x: np.ndarray,
    w_y: np.ndarray,
    sym1: str,
    sym2: str,
    feature_names: list[str] = None
) -> go.Figure:
    """
    Plot CCA canonical weights as grouped bar chart.
    
    Visualizes the weights (loadings) that CCA assigns to each feature.
    For current implementation (Close price only), displays single weight per asset.
    Future-proof design supports multi-feature CCA.
    
    Args:
        w_x: Array of CCA weights for first asset (shape: (n_features,))
        w_y: Array of CCA weights for second asset (shape: (n_features,))
        sym1: Symbol name for first asset
        sym2: Symbol name for second asset
        feature_names: Optional list of feature names. 
                      If None, uses ['Feature_1', 'Feature_2', ...] or ['Close'] for single feature
    
    Returns:
        go.Figure: Plotly figure with grouped bar chart
        
    Raises:
        ValueError: If weight arrays are empty or have different lengths
    
    Example:
        >>> fig = plot_cca_weights(w_x, w_y, 'VNM', 'VIC', ['Close'])
        >>> fig.show()
    """
    # Validate inputs
    if w_x.size == 0 or w_y.size == 0:
        raise ValueError("Weight arrays cannot be empty")
    
    if w_x.shape != w_y.shape:
        raise ValueError(
            f"Weight arrays must have same shape. Got w_x={w_x.shape}, w_y={w_y.shape}"
        )
    
    # Prepare feature names
    n_features = w_x.size
    if feature_names is None:
        if n_features == 1:
            feature_names = ['Close']
        else:
            feature_names = [f'Feature_{i+1}' for i in range(n_features)]
    
    if len(feature_names) != n_features:
        raise ValueError(
            f"Number of feature names ({len(feature_names)}) must match "
            f"number of weights ({n_features})"
        )
    
    # Flatten arrays to 1D
    w_x_flat = w_x.flatten()
    w_y_flat = w_y.flatten()
    
    # Create figure
    fig = go.Figure()
    
    # Add bar for first asset
    fig.add_trace(
        go.Bar(
            x=feature_names,
            y=w_x_flat,
            name=sym1,
            marker_color='#1f77b4',
            text=[f'{w:.4f}' for w in w_x_flat],
            textposition='outside',
            hovertemplate='%{x}: %{y:.6f}<extra></extra>'
        )
    )
    
    # Add bar for second asset
    fig.add_trace(
        go.Bar(
            x=feature_names,
            y=w_y_flat,
            name=sym2,
            marker_color='#ff7f0e',
            text=[f'{w:.4f}' for w in w_y_flat],
            textposition='outside',
            hovertemplate='%{x}: %{y:.6f}<extra></extra>'
        )
    )
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f'CCA Canonical Weights: {sym1} vs {sym2}',
            font=dict(size=20, family='Arial, sans-serif')
        ),
        xaxis_title='Features',
        yaxis_title='CCA Weight (Loading)',
        template='plotly_white',
        barmode='group',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        ),
        height=500
    )
    
    return fig
