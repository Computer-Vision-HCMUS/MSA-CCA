"""
Unit tests for Correlation Analyzer.

Tests cover CCA calculation, Z-score computation, error handling, and edge cases.
All tests use synthetic data - no real market data required.
"""

import unittest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path to import core module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.analyzer import CorrelationAnalyzer


class TestCorrelationAnalyzer(unittest.TestCase):
    """Test cases for CorrelationAnalyzer class."""
    
    def setUp(self):
        """Set up test fixtures before each test."""
        self.analyzer = CorrelationAnalyzer()
        
        # Create base date range for synthetic data
        self.dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    
    def _create_mock_dataframe(self, base_price: float, volatility: float, seed: int = 42) -> pd.DataFrame:
        """
        Create a synthetic OHLCV DataFrame for testing.
        
        Args:
            base_price: Starting price level
            volatility: Daily return volatility
            seed: Random seed for reproducibility
        
        Returns:
            DataFrame with columns ['O', 'H', 'L', 'C', 'V'] and datetime index
        """
        np.random.seed(seed)
        
        # Generate geometric Brownian motion for close prices
        returns = np.random.normal(0, volatility, len(self.dates))
        prices = base_price * np.exp(np.cumsum(returns))
        
        # Generate OHLC with realistic relationships
        df = pd.DataFrame(index=self.dates)
        df['C'] = prices
        df['O'] = prices * (1 + np.random.uniform(-0.005, 0.005, len(self.dates)))
        df['H'] = np.maximum(df['O'], df['C']) * (1 + np.random.uniform(0, 0.01, len(self.dates)))
        df['L'] = np.minimum(df['O'], df['C']) * (1 - np.random.uniform(0, 0.01, len(self.dates)))
        df['V'] = np.random.uniform(1e6, 5e6, len(self.dates))
        
        return df
    
    def test_initialization(self):
        """Test analyzer initialization."""
        analyzer = CorrelationAnalyzer()
        self.assertIsInstance(analyzer, CorrelationAnalyzer)
    
    def test_calculate_cca_zscore_success(self):
        """Test successful CCA Z-score calculation with correlated assets."""
        # Create two correlated synthetic price series
        df_x = self._create_mock_dataframe(base_price=100.0, volatility=0.02, seed=42)
        df_y = self._create_mock_dataframe(base_price=150.0, volatility=0.02, seed=43)
        
        # Calculate CCA Z-score
        spread, zscore, x_weights, y_weights = self.analyzer.calculate_cca_zscore(
            df_x, df_y, window=20
        )
        
        # Assertions on output types
        self.assertIsInstance(spread, pd.Series, "Spread should be a pandas Series")
        self.assertIsInstance(zscore, pd.Series, "Z-score should be a pandas Series")
        self.assertIsInstance(x_weights, np.ndarray, "X weights should be a numpy array")
        self.assertIsInstance(y_weights, np.ndarray, "Y weights should be a numpy array")
        
        # Assertions on output shapes
        self.assertEqual(len(spread), len(zscore), "Spread and Z-score should have same length")
        self.assertEqual(x_weights.shape, (5,), "X weights should be 1D array with 5 elements (O,H,L,C,V)")
        self.assertEqual(y_weights.shape, (5,), "Y weights should be 1D array with 5 elements (O,H,L,C,V)")
        
        # Assertions on data validity
        self.assertFalse(zscore.isna().any(), "Z-score should not contain NaN values")
        self.assertFalse(np.isinf(zscore).any(), "Z-score should not contain Inf values")
        self.assertFalse(spread.isna().any(), "Spread should not contain NaN values")
        self.assertFalse(np.isinf(spread).any(), "Spread should not contain Inf values")
        
        # Assertions on index
        self.assertIsInstance(spread.index, pd.DatetimeIndex, "Spread should have datetime index")
        self.assertIsInstance(zscore.index, pd.DatetimeIndex, "Z-score should have datetime index")
        
        # Check that we have reasonable number of data points (less than original due to rolling window)
        self.assertGreater(len(zscore), 50, "Should have substantial data points after rolling window")
        self.assertLess(len(zscore), 100, "Should lose some points due to rolling window and log returns")
    
    def test_cca_weights_are_nonzero(self):
        """Test that CCA weights are non-zero (meaningful hedge ratio)."""
        df_x = self._create_mock_dataframe(base_price=100.0, volatility=0.02, seed=42)
        df_y = self._create_mock_dataframe(base_price=150.0, volatility=0.02, seed=43)
        
        _, _, x_weights, y_weights = self.analyzer.calculate_cca_zscore(df_x, df_y, window=20)
        
        # Check that weights are arrays of length 5
        self.assertEqual(len(x_weights), 5, "X weights should have 5 elements")
        self.assertEqual(len(y_weights), 5, "Y weights should have 5 elements")
        
        # Check that at least some weights are non-zero (not all zero)
        self.assertTrue(np.any(x_weights != 0.0), "At least some X weights should be non-zero")
        self.assertTrue(np.any(y_weights != 0.0), "At least some Y weights should be non-zero")
        
        # Check no NaN values in weights
        self.assertFalse(np.any(np.isnan(x_weights)), "X weights should not contain NaN")
        self.assertFalse(np.any(np.isnan(y_weights)), "Y weights should not contain NaN")
    
    def test_zscore_statistical_properties(self):
        """Test that Z-score has expected statistical properties."""
        df_x = self._create_mock_dataframe(base_price=100.0, volatility=0.02, seed=42)
        df_y = self._create_mock_dataframe(base_price=150.0, volatility=0.02, seed=43)
        
        _, zscore, _, _ = self.analyzer.calculate_cca_zscore(df_x, df_y, window=20)
        
        # Z-score should roughly have mean ~0 and std ~1 (for recent window)
        # Note: This is approximate due to rolling window nature
        recent_zscore = zscore.tail(50)  # Check last 50 points
        
        self.assertLess(abs(recent_zscore.mean()), 1.0, 
                       "Z-score mean should be close to 0")
        self.assertGreater(recent_zscore.std(), 0.5, 
                          "Z-score should have reasonable volatility")
        self.assertLess(recent_zscore.std(), 2.0, 
                       "Z-score std should not be too extreme")
    
    def test_empty_dataframe_after_alignment_error(self):
        """Test error when DataFrames have no overlapping dates."""
        # Create DataFrames with non-overlapping date ranges
        dates_x = pd.date_range(start='2024-01-01', periods=50, freq='D')
        dates_y = pd.date_range(start='2024-03-01', periods=50, freq='D')
        
        df_x = pd.DataFrame(
            {'O': 100, 'H': 101, 'L': 99, 'C': 100.5, 'V': 1e6},
            index=dates_x
        )
        df_y = pd.DataFrame(
            {'O': 150, 'H': 151, 'L': 149, 'C': 150.5, 'V': 1e6},
            index=dates_y
        )
        
        with self.assertRaises(ValueError) as context:
            self.analyzer.calculate_cca_zscore(df_x, df_y, window=20)
        
        self.assertIn("No overlapping dates", str(context.exception))
    
    def test_insufficient_data_for_window_error(self):
        """Test error when data points < window size."""
        # Create small DataFrames
        small_dates = pd.date_range(start='2024-01-01', periods=10, freq='D')
        
        df_x = pd.DataFrame(
            {'O': 100, 'H': 101, 'L': 99, 'C': 100.5, 'V': 1e6},
            index=small_dates
        )
        df_y = pd.DataFrame(
            {'O': 150, 'H': 151, 'L': 149, 'C': 150.5, 'V': 1e6},
            index=small_dates
        )
        
        # Try with window=20 (larger than data points)
        with self.assertRaises(ValueError) as context:
            self.analyzer.calculate_cca_zscore(df_x, df_y, window=20)
        
        self.assertIn("Insufficient data points", str(context.exception))
    
    def test_zero_standard_deviation_error(self):
        """Test error when spread has zero standard deviation (constant spread)."""
        # Create perfectly correlated series with identical returns
        dates = pd.date_range(start='2024-01-01', periods=50, freq='D')
        
        # Constant prices (zero returns)
        df_x = pd.DataFrame(
            {'O': 100, 'H': 100, 'L': 100, 'C': 100, 'V': 1e6},
            index=dates
        )
        df_y = pd.DataFrame(
            {'O': 150, 'H': 150, 'L': 150, 'C': 150, 'V': 1e6},
            index=dates
        )
        
        with self.assertRaises(ValueError) as context:
            self.analyzer.calculate_cca_zscore(df_x, df_y, window=20)
        
        # Should fail early due to log returns being NaN (log(1) = 0, but shift creates issues)
        # Or could fail at rolling std = 0 check
        self.assertTrue(
            "NaN" in str(context.exception) or 
            "standard deviation is zero" in str(context.exception) or
            "Insufficient data" in str(context.exception)
        )
    
    def test_window_size_validation(self):
        """Test that window size < 2 raises ValueError."""
        df_x = self._create_mock_dataframe(base_price=100.0, volatility=0.02, seed=42)
        df_y = self._create_mock_dataframe(base_price=150.0, volatility=0.02, seed=43)
        
        with self.assertRaises(ValueError) as context:
            self.analyzer.calculate_cca_zscore(df_x, df_y, window=1)
        
        self.assertIn("Window size must be >= 2", str(context.exception))
    
    def test_alignment_removes_mismatched_dates(self):
        """Test that alignment properly handles missing dates in one series."""
        # Create series with some missing dates
        dates_x = pd.date_range(start='2024-01-01', periods=100, freq='D')
        dates_y = pd.date_range(start='2024-01-01', periods=100, freq='D')
        
        # Remove some dates from df_y (simulate trading suspensions)
        dates_y_filtered = dates_y.drop(dates_y[10:15])  # Remove 5 days
        
        df_x = self._create_mock_dataframe(base_price=100.0, volatility=0.02, seed=42)
        df_y = self._create_mock_dataframe(base_price=150.0, volatility=0.02, seed=43)
        df_y = df_y.loc[dates_y_filtered]  # Use filtered dates
        
        # Should not raise error, just use common dates
        spread, zscore, _, _ = self.analyzer.calculate_cca_zscore(df_x, df_y, window=20)
        
        self.assertIsInstance(spread, pd.Series)
        self.assertIsInstance(zscore, pd.Series)
        self.assertGreater(len(zscore), 50)  # Should still have substantial data
    
    def test_returns_calculation_handles_price_changes(self):
        """Test that log returns are calculated correctly for trending prices."""
        # Create upward trending series
        dates = pd.date_range(start='2024-01-01', periods=60, freq='D')
        trend = np.linspace(100, 120, 60)  # Linear upward trend
        
        df_x = pd.DataFrame({
            'O': trend * 0.99,
            'H': trend * 1.01,
            'L': trend * 0.98,
            'C': trend,
            'V': 1e6
        }, index=dates)
        
        df_y = pd.DataFrame({
            'O': trend * 1.5 * 0.99,
            'H': trend * 1.5 * 1.01,
            'L': trend * 1.5 * 0.98,
            'C': trend * 1.5,
            'V': 1e6
        }, index=dates)
        
        spread, zscore, x_weights, y_weights = self.analyzer.calculate_cca_zscore(
            df_x, df_y, window=20
        )
        
        # Should successfully calculate without errors
        self.assertIsInstance(zscore, pd.Series)
        self.assertGreater(len(zscore), 30)
        self.assertFalse(zscore.isna().any())
    
    def test_different_window_sizes(self):
        """Test that different window sizes produce different Z-scores."""
        df_x = self._create_mock_dataframe(base_price=100.0, volatility=0.02, seed=42)
        df_y = self._create_mock_dataframe(base_price=150.0, volatility=0.02, seed=43)
        
        _, zscore_10, _, _ = self.analyzer.calculate_cca_zscore(df_x, df_y, window=10)
        _, zscore_30, _, _ = self.analyzer.calculate_cca_zscore(df_x, df_y, window=30)
        
        # Different windows should give different z-scores (different smoothing)
        self.assertNotEqual(len(zscore_10), len(zscore_30), 
                           "Different windows should result in different series lengths")
        self.assertGreater(len(zscore_10), len(zscore_30),
                          "Shorter window should preserve more data points")


class TestCorrelationAnalyzerEdgeCases(unittest.TestCase):
    """Additional edge case tests for robustness."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.analyzer = CorrelationAnalyzer()
    
    def test_single_column_dataframe(self):
        """Test with DataFrame containing only Close column - should raise ValueError."""
        dates = pd.date_range(start='2024-01-01', periods=60, freq='D')
        
        # DataFrames with only 'C' column (missing O, H, L, V)
        df_x = pd.DataFrame({
            'C': np.random.randn(60).cumsum() + 100
        }, index=dates)
        
        df_y = pd.DataFrame({
            'C': np.random.randn(60).cumsum() + 150
        }, index=dates)
        
        # Should raise ValueError because CCA now requires all 5 OHLCV columns
        with self.assertRaises(ValueError) as context:
            self.analyzer.calculate_cca_zscore(df_x, df_y, window=20)
        
        self.assertIn("missing required column", str(context.exception))
    
    def test_very_large_window(self):
        """Test with window size close to data size."""
        dates = pd.date_range(start='2024-01-01', periods=50, freq='D')
        
        df_x = pd.DataFrame({
            'C': np.random.randn(50).cumsum() + 100,
            'O': 100, 'H': 101, 'L': 99, 'V': 1e6
        }, index=dates)
        
        df_y = pd.DataFrame({
            'C': np.random.randn(50).cumsum() + 150,
            'O': 150, 'H': 151, 'L': 149, 'V': 1e6
        }, index=dates)
        
        # Window=40 is large but should still work
        spread, zscore, _, _ = self.analyzer.calculate_cca_zscore(df_x, df_y, window=40)
        
        self.assertIsInstance(zscore, pd.Series)
        # Should have very few data points remaining
        self.assertGreater(len(zscore), 0)
        self.assertLess(len(zscore), 15)


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
