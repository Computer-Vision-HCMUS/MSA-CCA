"""
Correlation Analyzer Module for Pairs Trading Strategy.

This module implements Canonical Correlation Analysis (CCA) based spread calculation
and Z-Score normalization for statistical arbitrage strategies.

Why CCA?
--------
Traditional pairs trading uses simple price ratios or OLS regression. CCA finds optimal
linear combinations of two time series that maximize their correlation, producing a more
stationary spread for mean-reversion trading.

Mathematical Foundation:
- Log Returns: Stabilizes variance and makes returns comparable across different price levels
- CCA Weights: Canonical loadings that define the optimal hedge ratio
- Spread: Linear combination of returns using CCA weights (mean-reverting property)
- Z-Score: Standardized distance from mean, used for entry/exit signals
"""

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import CCA


class CorrelationAnalyzer:
    """
    Analyzer for computing CCA-based spread and Z-scores for pairs trading.
    
    This class implements a statistical framework for identifying trading opportunities
    in correlated asset pairs using Canonical Correlation Analysis.
    """
    
    def __init__(self) -> None:
        """Initialize the Correlation Analyzer."""
        pass
    
    def calculate_cca_zscore(
        self,
        df_x: pd.DataFrame,
        df_y: pd.DataFrame,
        window: int = 20
    ) -> tuple[pd.Series, pd.Series, np.ndarray, np.ndarray]:
        """
        Calculate CCA-based spread and Z-score for two price series.
        
        This method performs a 5-step process to generate trading signals:
        1. Temporal alignment of two time series
        2. Log return transformation (stabilizes variance)
        3. CCA fitting to find optimal linear combination
        4. Spread construction using CCA weights
        5. Z-score calculation with rolling statistics
        
        Args:
            df_x: DataFrame for first asset with OHLCV columns and datetime index
            df_y: DataFrame for second asset with OHLCV columns and datetime index
            window: Rolling window size for Z-score calculation (default: 20 periods)
                   Must be >= 2 for statistical validity
        
        Returns:
            tuple containing:
                - spread (pd.Series): CCA-weighted spread series with datetime index
                - zscore (pd.Series): Standardized Z-scores with datetime index
                - x_weights (np.ndarray): CCA canonical weights for asset X (1D array)
                - y_weights (np.ndarray): CCA canonical weights for asset Y (1D array)
        
        Raises:
            ValueError: If input validation fails:
                - DataFrames are empty after alignment
                - Insufficient data points (< window size)
                - Rolling standard deviation equals zero (non-varying spread)
                - Window size is less than 2
        
        Example:
            >>> analyzer = CorrelationAnalyzer()
            >>> spread, zscore, wx, wy = analyzer.calculate_cca_zscore(df_vnm, df_vic, window=20)
            >>> # Trade when |zscore| > 2.0 (mean reversion signal)
        """
        # Input validation
        if window < 2:
            raise ValueError(f"Window size must be >= 2, got {window}")
        
        # STEP 1: Align DataFrames by index (TradingDate)
        # Why: Remove mismatched trading days (holidays, suspensions, IPO dates)
        # Using .align() ensures robust handling of time series with different trading calendars
        
        # CRITICAL FIX: Use ALL 5 OHLCV columns, not just Close
        # Required columns for CCA analysis
        required_cols = ['O', 'H', 'L', 'C', 'V']
        
        # Validate that both DataFrames have all required columns
        for col in required_cols:
            if col not in df_x.columns:
                raise ValueError(f"df_x is missing required column: {col}")
            if col not in df_y.columns:
                raise ValueError(f"df_y is missing required column: {col}")
        
        df_x_ohlcv = df_x[required_cols].copy()
        df_y_ohlcv = df_y[required_cols].copy()
        
        # CRITICAL: align() with join='inner' removes non-overlapping dates
        # This prevents errors when combining series with different holiday schedules
        aligned_x, aligned_y = df_x_ohlcv.align(df_y_ohlcv, join='inner')
        
        if aligned_x.empty or aligned_y.empty:
            raise ValueError(
                "No overlapping dates between the two DataFrames. "
                "Check if the time series have any common trading days."
            )
        
        # STEP 2: Calculate Log Returns FOR ALL 5 COLUMNS
        # Why: Log returns are time-additive and normalize price differences
        # Formula: r_t = ln(P_t / P_{t-1}) for each of O, H, L, C, V
        
        # Calculate log returns for each column
        returns_x = pd.DataFrame(index=aligned_x.index)
        returns_y = pd.DataFrame(index=aligned_y.index)
        
        for col in required_cols:
            returns_x[col] = np.log(aligned_x[col] / aligned_x[col].shift(1))
            returns_y[col] = np.log(aligned_y[col] / aligned_y[col].shift(1))
        
        # Combine returns for both assets
        # This creates a DataFrame with 10 columns: 5 for X, 5 for Y
        returns_combined = pd.concat([returns_x, returns_y], axis=1, keys=['X', 'Y'])
        
        # Remove first row (NaN from shift) and any other NaN/Inf values
        returns_combined = returns_combined.replace([np.inf, -np.inf], np.nan).dropna()
        
        if returns_combined.empty:
            raise ValueError(
                "All log returns are NaN or Inf. "
                "Check if price data contains zeros or invalid values."
            )
        
        if len(returns_combined) < window:
            raise ValueError(
                f"Insufficient data points after log return calculation. "
                f"Got {len(returns_combined)} rows, need at least {window} for rolling window."
            )
        
        # STEP 3: Fit CCA Model WITH ALL 5 FEATURES
        # Why: CCA finds linear combinations that maximize correlation between two sets
        # This creates a more stationary spread than simple price differences
        # X and Y now have shape (n_samples, 5) instead of (n_samples, 1)
        
        X = returns_combined['X'].values  # Shape: (n, 5) - all 5 OHLCV log returns for asset X
        Y = returns_combined['Y'].values  # Shape: (n, 5) - all 5 OHLCV log returns for asset Y
        
        print(f"[DEBUG] CCA input shapes: X={X.shape}, Y={Y.shape}")
        
        cca = CCA(n_components=1)
        
        try:
            cca.fit(X, Y)
        except Exception as e:
            raise ValueError(f"CCA fitting failed: {str(e)}")
        
        # Extract canonical weights (loadings)
        # These represent the optimal weights for each of the 5 features
        # x_weights and y_weights now have shape (5,) instead of (1,)
        x_weights = cca.x_weights_.flatten()  # Shape: (5,) - weights for [O, H, L, C, V]
        y_weights = cca.y_weights_.flatten()  # Shape: (5,) - weights for [O, H, L, C, V]
        
        print(f"[DEBUG] CCA weights shapes: x_weights={x_weights.shape}, y_weights={y_weights.shape}")
        print(f"[DEBUG] X weights (O,H,L,C,V): {x_weights}")
        print(f"[DEBUG] Y weights (O,H,L,C,V): {y_weights}")
        
        # STEP 4: Calculate Spread using dot product
        # Why: Spread = weighted combination of returns with CCA weights
        # Formula: S_t = w_x · r_x,t - w_y · r_y,t (where · is dot product)
        # This properly combines all 5 features according to their CCA weights
        
        spread = np.dot(X, x_weights) - np.dot(Y, y_weights)
        
        spread_series = pd.Series(spread, index=returns_combined.index, name='spread')
        
        # STEP 5: Calculate Z-Score with Rolling Statistics
        # Why: Z-score measures how many standard deviations spread is from its mean
        # Signals: |z| > 2 suggests mean reversion opportunity
        rolling_mean = spread_series.rolling(window=window).mean()
        rolling_std = spread_series.rolling(window=window).std()
        
        # Check for zero standard deviation (constant spread - no trading opportunity)
        if (rolling_std == 0).any():
            # Find first occurrence
            zero_std_dates = rolling_std[rolling_std == 0].index
            raise ValueError(
                f"Rolling standard deviation is zero at {len(zero_std_dates)} time point(s). "
                f"First occurrence: {zero_std_dates[0]}. "
                "This indicates a non-varying spread (no volatility to trade)."
            )
        
        # Calculate Z-score: (X - μ) / σ
        zscore = (spread_series - rolling_mean) / rolling_std
        
        # Remove NaN values from the start (insufficient window data)
        zscore = zscore.dropna()
        
        if zscore.empty:
            raise ValueError(
                f"Z-score calculation resulted in empty series. "
                f"Check if spread series length ({len(spread_series)}) >= window ({window})."
            )
        
        # Validate output - no Inf values should exist
        if np.isinf(zscore).any():
            raise ValueError(
                "Z-score contains infinite values. "
                "This suggests division by near-zero standard deviation."
            )
        
        # Align spread series with zscore index (remove initial window-1 rows)
        spread_aligned = spread_series.loc[zscore.index]
        
        return spread_aligned, zscore, x_weights, y_weights
