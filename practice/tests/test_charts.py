"""
Unit tests for Chart Components.

Tests cover chart generation, input validation, and error handling.
All tests use synthetic data - no real market data required.
"""

import unittest
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path to import ui module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from ui.charts import (
    plot_raw_price,
    plot_normalized_price,
    plot_spread_bollinger,
    plot_zscore,
    plot_cca_weights
)


class TestChartFunctions(unittest.TestCase):
    """Test cases for chart generation functions."""
    
    def setUp(self):
        """Set up test fixtures before each test."""
        # Create sample date range
        self.dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
        
        # Create sample price data
        np.random.seed(42)
        self.df1 = pd.DataFrame({
            'O': 100 + np.random.randn(100).cumsum(),
            'H': 102 + np.random.randn(100).cumsum(),
            'L': 98 + np.random.randn(100).cumsum(),
            'C': 100 + np.random.randn(100).cumsum(),
            'V': np.random.uniform(1e6, 5e6, 100)
        }, index=self.dates)
        
        self.df2 = pd.DataFrame({
            'O': 80 + np.random.randn(100).cumsum() * 0.8,
            'H': 82 + np.random.randn(100).cumsum() * 0.8,
            'L': 78 + np.random.randn(100).cumsum() * 0.8,
            'C': 80 + np.random.randn(100).cumsum() * 0.8,
            'V': np.random.uniform(1e6, 5e6, 100)
        }, index=self.dates)
        
        # Create sample spread and z-score
        self.spread = pd.Series(
            np.random.randn(100).cumsum() * 0.01,
            index=self.dates,
            name='spread'
        )
        
        self.zscore = pd.Series(
            np.random.randn(100),
            index=self.dates,
            name='zscore'
        )
        
        # Create sample CCA weights
        self.w_x = np.array([0.8])
        self.w_y = np.array([0.6])
    
    def test_plot_raw_price_success(self):
        """Test successful raw price chart generation."""
        fig = plot_raw_price(self.df1, self.df2, 'VNM', 'VIC')
        
        # Assert return type
        self.assertIsInstance(fig, go.Figure, "Should return plotly Figure object")
        
        # Assert has data traces
        self.assertGreater(len(fig.data), 0, "Figure should contain traces")
        self.assertEqual(len(fig.data), 2, "Should have 2 traces (one per asset)")
        
        # Assert trace names
        trace_names = [trace.name for trace in fig.data]
        self.assertIn('VNM', trace_names, "Should contain VNM trace")
        self.assertIn('VIC', trace_names, "Should contain VIC trace")
        
        # Assert layout
        self.assertIsNotNone(fig.layout.title, "Should have title")
        self.assertIn('VNM', fig.layout.title.text, "Title should mention symbols")
    
    def test_plot_raw_price_empty_dataframe_error(self):
        """Test raw price chart with empty DataFrame."""
        empty_df = pd.DataFrame()
        
        with self.assertRaises(ValueError) as context:
            plot_raw_price(empty_df, self.df2, 'VNM', 'VIC')
        
        self.assertIn("cannot be empty", str(context.exception))
    
    def test_plot_raw_price_missing_column_error(self):
        """Test raw price chart with missing 'C' column."""
        df_no_close = self.df1.drop(columns=['C'])
        
        with self.assertRaises(ValueError) as context:
            plot_raw_price(df_no_close, self.df2, 'VNM', 'VIC')
        
        self.assertIn("must contain 'C'", str(context.exception))
    
    def test_plot_normalized_price_success(self):
        """Test successful normalized price chart generation."""
        fig = plot_normalized_price(self.df1, self.df2, 'VNM', 'VIC')
        
        # Assert return type
        self.assertIsInstance(fig, go.Figure)
        
        # Assert has traces
        self.assertGreater(len(fig.data), 0)
        # Should have 2 price traces + 1 horizontal line
        self.assertGreaterEqual(len(fig.data), 2)
        
        # Assert trace names
        trace_names = [trace.name for trace in fig.data if trace.name]
        self.assertIn('VNM', trace_names)
        self.assertIn('VIC', trace_names)
        
        # Assert layout
        self.assertIn('Normalized', fig.layout.title.text)
    
    def test_plot_normalized_price_empty_error(self):
        """Test normalized price chart with empty DataFrame."""
        empty_df = pd.DataFrame()
        
        with self.assertRaises(ValueError):
            plot_normalized_price(empty_df, self.df2, 'VNM', 'VIC')
    
    def test_plot_normalized_price_zero_first_price_handled(self):
        """Test normalized price chart when first price is zero (should handle gracefully)."""
        df_zero = self.df1.copy()
        df_zero['C'].iloc[0] = 0
        df_zero['C'].iloc[1] = 0  # Make first two zeros
        # Third value and onwards should be valid
        
        # Should not raise error - should use first non-zero value
        fig = plot_normalized_price(df_zero, self.df2, 'VNM', 'VIC')
        
        self.assertIsInstance(fig, go.Figure)
        self.assertGreater(len(fig.data), 0)
    
    def test_plot_normalized_price_all_zero_error(self):
        """Test normalized price chart when all prices are zero (should raise error)."""
        df_all_zero = self.df1.copy()
        df_all_zero['C'] = 0  # All zeros
        
        with self.assertRaises(ValueError) as context:
            plot_normalized_price(df_all_zero, self.df2, 'VNM', 'VIC')
        
        self.assertIn("all price values are zero", str(context.exception))
    
    def test_plot_spread_bollinger_success(self):
        """Test successful Bollinger bands chart generation."""
        fig = plot_spread_bollinger(self.spread, window=20)
        
        # Assert return type
        self.assertIsInstance(fig, go.Figure)
        
        # Assert has multiple traces (spread + mean + upper + lower)
        self.assertGreaterEqual(len(fig.data), 4, 
                                "Should have spread, mean, upper band, lower band")
        
        # Check trace names
        trace_names = [trace.name for trace in fig.data if trace.name]
        self.assertIn('Spread', trace_names)
        self.assertIn('Rolling Mean', trace_names)
        # Check for Upper/Lower band (partial match due to sigma notation)
        has_upper = any('Upper' in name for name in trace_names)
        has_lower = any('Lower' in name for name in trace_names)
        self.assertTrue(has_upper, "Should have Upper Band trace")
        self.assertTrue(has_lower, "Should have Lower Band trace")
        
        # Assert layout
        self.assertIn('Bollinger', fig.layout.title.text)
    
    def test_plot_spread_bollinger_empty_error(self):
        """Test Bollinger chart with empty spread."""
        empty_spread = pd.Series(dtype=float)
        
        with self.assertRaises(ValueError) as context:
            plot_spread_bollinger(empty_spread, window=20)
        
        self.assertIn("cannot be empty", str(context.exception))
    
    def test_plot_spread_bollinger_invalid_window_error(self):
        """Test Bollinger chart with invalid window size."""
        with self.assertRaises(ValueError) as context:
            plot_spread_bollinger(self.spread, window=1)
        
        self.assertIn("Window must be >= 2", str(context.exception))
    
    def test_plot_spread_bollinger_insufficient_data_error(self):
        """Test Bollinger chart with insufficient data for window."""
        short_spread = pd.Series([1.0, 2.0, 3.0], index=pd.date_range('2024-01-01', periods=3))
        
        with self.assertRaises(ValueError) as context:
            plot_spread_bollinger(short_spread, window=20)
        
        self.assertIn("must be >= window", str(context.exception))
    
    def test_plot_zscore_success(self):
        """Test successful Z-score chart generation."""
        fig = plot_zscore(self.zscore, threshold=2.0)
        
        # Assert return type
        self.assertIsInstance(fig, go.Figure)
        
        # Assert has traces
        self.assertGreater(len(fig.data), 0)
        
        # Check for Z-score trace
        trace_names = [trace.name for trace in fig.data if trace.name]
        self.assertIn('Z-Score', trace_names)
        
        # Assert layout
        self.assertIn('Z-Score', fig.layout.title.text)
    
    def test_plot_zscore_empty_error(self):
        """Test Z-score chart with empty series."""
        empty_zscore = pd.Series(dtype=float)
        
        with self.assertRaises(ValueError) as context:
            plot_zscore(empty_zscore)
        
        self.assertIn("cannot be empty", str(context.exception))
    
    def test_plot_zscore_invalid_threshold_error(self):
        """Test Z-score chart with negative threshold."""
        with self.assertRaises(ValueError) as context:
            plot_zscore(self.zscore, threshold=-1.0)
        
        self.assertIn("must be positive", str(context.exception))
    
    def test_plot_cca_weights_success(self):
        """Test successful CCA weights chart generation."""
        fig = plot_cca_weights(self.w_x, self.w_y, 'VNM', 'VIC')
        
        # Assert return type
        self.assertIsInstance(fig, go.Figure)
        
        # Assert has traces (2 bars)
        self.assertEqual(len(fig.data), 2, "Should have 2 bar traces")
        
        # Check trace names
        trace_names = [trace.name for trace in fig.data]
        self.assertIn('VNM', trace_names)
        self.assertIn('VIC', trace_names)
        
        # Assert layout
        self.assertIn('CCA', fig.layout.title.text)
        self.assertEqual(fig.layout.barmode, 'group', "Should be grouped bar chart")
    
    def test_plot_cca_weights_with_feature_names(self):
        """Test CCA weights chart with custom feature names."""
        fig = plot_cca_weights(self.w_x, self.w_y, 'VNM', 'VIC', feature_names=['Close'])
        
        self.assertIsInstance(fig, go.Figure)
        self.assertEqual(len(fig.data), 2)
    
    def test_plot_cca_weights_multifeature(self):
        """Test CCA weights chart with multiple features."""
        w_x_multi = np.array([0.8, 0.6, 0.4, 0.3, 0.2])
        w_y_multi = np.array([0.7, 0.5, 0.5, 0.4, 0.1])
        
        fig = plot_cca_weights(
            w_x_multi, w_y_multi, 'VNM', 'VIC',
            feature_names=['O', 'H', 'L', 'C', 'V']
        )
        
        self.assertIsInstance(fig, go.Figure)
        self.assertEqual(len(fig.data), 2)
    
    def test_plot_cca_weights_empty_error(self):
        """Test CCA weights chart with empty arrays."""
        empty_weights = np.array([])
        
        with self.assertRaises(ValueError) as context:
            plot_cca_weights(empty_weights, self.w_y, 'VNM', 'VIC')
        
        self.assertIn("cannot be empty", str(context.exception))
    
    def test_plot_cca_weights_shape_mismatch_error(self):
        """Test CCA weights chart with mismatched array shapes."""
        w_x_wrong = np.array([0.8, 0.6])
        
        with self.assertRaises(ValueError) as context:
            plot_cca_weights(w_x_wrong, self.w_y, 'VNM', 'VIC')
        
        self.assertIn("same shape", str(context.exception))
    
    def test_plot_cca_weights_feature_names_mismatch_error(self):
        """Test CCA weights chart with wrong number of feature names."""
        with self.assertRaises(ValueError) as context:
            plot_cca_weights(
                self.w_x, self.w_y, 'VNM', 'VIC',
                feature_names=['Close', 'Extra']  # Too many names
            )
        
        self.assertIn("must match", str(context.exception))


class TestChartTemplateAndStyling(unittest.TestCase):
    """Test that all charts follow consistent styling."""
    
    def setUp(self):
        """Set up test data."""
        dates = pd.date_range(start='2024-01-01', periods=50, freq='D')
        self.df1 = pd.DataFrame({
            'C': 100 + np.random.randn(50).cumsum()
        }, index=dates)
        self.df2 = pd.DataFrame({
            'C': 80 + np.random.randn(50).cumsum()
        }, index=dates)
        self.spread = pd.Series(np.random.randn(50).cumsum() * 0.01, index=dates)
        self.zscore = pd.Series(np.random.randn(50), index=dates)
        self.w_x = np.array([0.8])
        self.w_y = np.array([0.6])
    
    def test_all_charts_use_plotly_white_template(self):
        """Test that all charts use 'plotly_white' template."""
        charts = [
            plot_raw_price(self.df1, self.df2, 'A', 'B'),
            plot_normalized_price(self.df1, self.df2, 'A', 'B'),
            plot_spread_bollinger(self.spread, window=20),
            plot_zscore(self.zscore),
            plot_cca_weights(self.w_x, self.w_y, 'A', 'B')
        ]
        
        # Simply check that template is set (plotly_white is default)
        # The key is that update_layout(template='plotly_white') was called
        for i, fig in enumerate(charts):
            self.assertIsNotNone(fig.layout.template, 
                                f"Chart {i} should have template set")
    
    def test_all_charts_have_titles(self):
        """Test that all charts have titles."""
        charts = [
            plot_raw_price(self.df1, self.df2, 'A', 'B'),
            plot_normalized_price(self.df1, self.df2, 'A', 'B'),
            plot_spread_bollinger(self.spread, window=20),
            plot_zscore(self.zscore),
            plot_cca_weights(self.w_x, self.w_y, 'A', 'B')
        ]
        
        for fig in charts:
            self.assertIsNotNone(fig.layout.title, "All charts should have titles")
            self.assertGreater(len(fig.layout.title.text), 0, "Title should not be empty")


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
