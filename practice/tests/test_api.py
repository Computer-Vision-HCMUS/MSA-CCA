"""
Unit tests for SSI API Client.

Tests cover authentication, data fetching, error handling, and edge cases.
All tests use mocked responses - NO real API calls are made.
"""

import unittest
from unittest.mock import Mock, patch
import pandas as pd
from requests.exceptions import Timeout, ConnectionError, HTTPError
import sys
import os

# Add parent directory to path to import data module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data.api_client import SSIAPIClient


class TestSSIAPIClient(unittest.TestCase):
    """Test cases for SSIAPIClient class."""
    
    def setUp(self):
        """Set up test fixtures before each test."""
        self.consumer_id = "test_consumer_id"
        self.consumer_secret = "test_consumer_secret"
        self.base_url = "https://api.ssi.com.vn"
        self.client = SSIAPIClient(
            consumer_id=self.consumer_id,
            consumer_secret=self.consumer_secret,
            base_url=self.base_url
        )
    
    def test_initialization_success(self):
        """Test successful client initialization."""
        self.assertEqual(self.client.consumer_id, self.consumer_id)
        self.assertEqual(self.client.consumer_secret, self.consumer_secret)
        self.assertEqual(self.client.base_url, self.base_url)
        self.assertEqual(self.client._access_token, "")
    
    def test_initialization_empty_parameters(self):
        """Test initialization with empty parameters raises ValueError."""
        with self.assertRaises(ValueError):
            SSIAPIClient("", "secret", "url")
        
        with self.assertRaises(ValueError):
            SSIAPIClient("id", "", "url")
            
        with self.assertRaises(ValueError):
            SSIAPIClient("id", "secret", "")
    
    @patch('data.api_client.requests.post')
    def test_authenticate_success(self, mock_post):
        """Test successful authentication - returns access token."""
        # Mock successful authentication response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'data': {
                'accessToken': 'test_access_token_12345'
            }
        }
        mock_response.raise_for_status = Mock()  # Does nothing on success
        mock_post.return_value = mock_response
        
        # Call authenticate
        token = self.client._authenticate()
        
        # Assertions
        self.assertEqual(token, 'test_access_token_12345')
        self.assertEqual(self.client._access_token, 'test_access_token_12345')
        
        # Verify request was made correctly
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        self.assertIn('api/v2/Market/AccessToken', call_args[0][0])
        self.assertEqual(call_args[1]['json']['consumerID'], self.consumer_id)
        self.assertEqual(call_args[1]['json']['consumerSecret'], self.consumer_secret)
    
    @patch('data.api_client.requests.post')
    def test_authenticate_timeout_error(self, mock_post):
        """Test authentication with timeout error - raises Timeout."""
        # Mock timeout
        mock_post.side_effect = Timeout("Connection timeout")
        
        # Assert that Timeout is raised
        with self.assertRaises(Timeout) as context:
            self.client._authenticate()
        
        self.assertIn("Authentication request timed out", str(context.exception))
    
    @patch('data.api_client.requests.post')
    def test_authenticate_connection_error(self, mock_post):
        """Test authentication with connection error - raises ConnectionError."""
        # Mock connection error
        mock_post.side_effect = ConnectionError("Failed to connect")
        
        # Assert that ConnectionError is raised
        with self.assertRaises(ConnectionError) as context:
            self.client._authenticate()
        
        self.assertIn("Failed to connect to SSI API", str(context.exception))
    
    @patch('data.api_client.requests.post')
    def test_authenticate_http_error(self, mock_post):
        """Test authentication with HTTP error - raises HTTPError."""
        # Mock HTTP error response
        mock_response = Mock()
        mock_response.status_code = 401
        mock_response.raise_for_status.side_effect = HTTPError("401 Unauthorized")
        mock_post.return_value = mock_response
        
        # Assert that HTTPError is raised
        with self.assertRaises(HTTPError) as context:
            self.client._authenticate()
        
        self.assertIn("HTTP error during authentication", str(context.exception))
    
    @patch('data.api_client.requests.post')
    def test_authenticate_invalid_response(self, mock_post):
        """Test authentication with invalid response structure - raises ValueError."""
        # Mock invalid response (missing accessToken)
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'data': {}}  # Missing accessToken
        mock_response.raise_for_status = Mock()
        mock_post.return_value = mock_response
        
        # Assert that ValueError is raised
        with self.assertRaises(ValueError) as context:
            self.client._authenticate()
        
        self.assertIn("Invalid authentication response", str(context.exception))
    
    @patch('data.api_client.requests.get')
    @patch('data.api_client.requests.post')
    def test_fetch_ohlcv_success(self, mock_post, mock_get):
        """Test successful OHLCV data fetching - returns properly formatted DataFrame."""
        # Mock authentication
        mock_auth_response = Mock()
        mock_auth_response.status_code = 200
        mock_auth_response.json.return_value = {
            'data': {'accessToken': 'test_token_abc123'}
        }
        mock_auth_response.raise_for_status = Mock()
        mock_post.return_value = mock_auth_response
        
        # Mock OHLCV data response (note: dates are intentionally out of order)
        mock_data_response = Mock()
        mock_data_response.status_code = 200
        mock_data_response.json.return_value = {
            'data': [
                {
                    'tradingDate': '2024-01-03',
                    'open': 105.5,
                    'high': 108.0,
                    'low': 104.0,
                    'close': 107.0,
                    'volume': 1200000
                },
                {
                    'tradingDate': '2024-01-01',
                    'open': 98.0,
                    'high': 101.0,
                    'low': 97.5,
                    'close': 100.5,
                    'volume': 950000
                },
                {
                    'tradingDate': '2024-01-02',
                    'open': 100.5,
                    'high': 105.0,
                    'low': 99.0,
                    'close': 103.0,
                    'volume': 1000000
                }
            ]
        }
        mock_data_response.raise_for_status = Mock()
        mock_get.return_value = mock_data_response
        
        # Fetch data
        df = self.client.fetch_ohlcv('VNM', '01/01/2024', '03/01/2024')
        
        # Assertions on DataFrame structure
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 3)
        self.assertListEqual(list(df.columns), ['O', 'H', 'L', 'C', 'V'])
        self.assertEqual(df.index.name, 'tradingDate')
        
        # Check data types - all should be float
        for col in df.columns:
            self.assertTrue(pd.api.types.is_float_dtype(df[col]), 
                          f"Column {col} should be float but is {df[col].dtype}")
        
        # Check sorting (past to present - ascending)
        self.assertTrue(df.index.is_monotonic_increasing, 
                       "Index should be sorted in ascending order")
        self.assertEqual(df.index[0], pd.Timestamp('2024-01-01'))
        self.assertEqual(df.index[1], pd.Timestamp('2024-01-02'))
        self.assertEqual(df.index[2], pd.Timestamp('2024-01-03'))
        
        # Check specific values
        self.assertAlmostEqual(df.iloc[0]['O'], 98.0)
        self.assertAlmostEqual(df.iloc[0]['C'], 100.5)
        self.assertAlmostEqual(df.iloc[0]['V'], 950000.0)
        
        self.assertAlmostEqual(df.iloc[2]['O'], 105.5)
        self.assertAlmostEqual(df.iloc[2]['H'], 108.0)
        self.assertAlmostEqual(df.iloc[2]['C'], 107.0)
        
        # Verify GET request was made correctly
        mock_get.assert_called_once()
        call_args = mock_get.call_args
        self.assertIn('api/v2/Market/DailyOhlc', call_args[0][0])
        self.assertEqual(call_args[1]['params']['symbol'], 'VNM')
    
    @patch('data.api_client.requests.get')
    @patch('data.api_client.requests.post')
    def test_fetch_ohlcv_dd_mm_yyyy_format(self, mock_post, mock_get):
        """Test successful OHLCV fetching with DD/MM/YYYY date format."""
        # Mock authentication
        mock_auth_response = Mock()
        mock_auth_response.status_code = 200
        mock_auth_response.json.return_value = {
            'data': {'accessToken': 'test_token_abc123'}
        }
        mock_auth_response.raise_for_status = Mock()
        mock_post.return_value = mock_auth_response
        
        # Mock OHLCV data response with DD/MM/YYYY format (SSI API style)
        mock_data_response = Mock()
        mock_data_response.status_code = 200
        mock_data_response.json.return_value = {
            'data': [
                {
                    'tradingDate': '27/02/2026',  # DD/MM/YYYY format
                    'open': 125.0,
                    'high': 128.5,
                    'low': 124.0,
                    'close': 127.5,
                    'volume': 1500000
                },
                {
                    'tradingDate': '26/02/2026',  # DD/MM/YYYY format
                    'open': 120.0,
                    'high': 125.5,
                    'low': 119.5,
                    'close': 124.5,
                    'volume': 1300000
                }
            ]
        }
        mock_data_response.raise_for_status = Mock()
        mock_get.return_value = mock_data_response
        
        # Fetch data
        df = self.client.fetch_ohlcv('MBB', '26/02/2026', '27/02/2026')
        
        # Assertions
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 2)
        self.assertListEqual(list(df.columns), ['O', 'H', 'L', 'C', 'V'])
        
        # Check dates parsed correctly
        self.assertEqual(df.index[0], pd.Timestamp('2026-02-26'))
        self.assertEqual(df.index[1], pd.Timestamp('2026-02-27'))
        
        # Check values
        self.assertAlmostEqual(df.iloc[0]['C'], 124.5)
        self.assertAlmostEqual(df.iloc[1]['C'], 127.5)
    
    @patch('data.api_client.requests.get')
    @patch('data.api_client.requests.post')
    def test_fetch_ohlcv_empty_response_error(self, mock_post, mock_get):
        """Test OHLCV fetching with empty data response - raises ValueError."""
        # Mock authentication
        mock_auth_response = Mock()
        mock_auth_response.status_code = 200
        mock_auth_response.json.return_value = {
            'data': {'accessToken': 'test_token'}
        }
        mock_auth_response.raise_for_status = Mock()
        mock_post.return_value = mock_auth_response
        
        # Mock empty data response
        mock_data_response = Mock()
        mock_data_response.status_code = 200
        mock_data_response.json.return_value = {'data': []}
        mock_data_response.raise_for_status = Mock()
        mock_get.return_value = mock_data_response
        
        # Should raise ValueError for empty data
        with self.assertRaises(ValueError) as context:
            self.client.fetch_ohlcv('INVALID', '01/01/2024', '03/01/2024')
        
        self.assertIn("No data returned", str(context.exception))
    
    @patch('data.api_client.requests.get')
    @patch('data.api_client.requests.post')
    def test_fetch_ohlcv_http_error(self, mock_post, mock_get):
        """Test OHLCV fetching with HTTP error - raises HTTPError."""
        # Mock authentication
        mock_auth_response = Mock()
        mock_auth_response.status_code = 200
        mock_auth_response.json.return_value = {
            'data': {'accessToken': 'test_token'}
        }
        mock_auth_response.raise_for_status = Mock()
        mock_post.return_value = mock_auth_response
        
        # Mock HTTP error
        mock_data_response = Mock()
        mock_data_response.status_code = 500
        mock_data_response.raise_for_status.side_effect = HTTPError("500 Server Error")
        mock_get.return_value = mock_data_response
        
        # Should raise HTTPError
        with self.assertRaises(HTTPError) as context:
            self.client.fetch_ohlcv('VNM', '01/01/2024', '03/01/2024')
        
        self.assertIn("HTTP error fetching data", str(context.exception))
    
    @patch('data.api_client.requests.get')
    @patch('data.api_client.requests.post')
    def test_fetch_ohlcv_timeout_error(self, mock_post, mock_get):
        """Test OHLCV fetching with timeout error - raises Timeout."""
        # Mock authentication
        mock_auth_response = Mock()
        mock_auth_response.status_code = 200
        mock_auth_response.json.return_value = {
            'data': {'accessToken': 'test_token'}
        }
        mock_auth_response.raise_for_status = Mock()
        mock_post.return_value = mock_auth_response
        
        # Mock timeout
        mock_get.side_effect = Timeout("Request timeout")
        
        # Should raise Timeout
        with self.assertRaises(Timeout) as context:
            self.client.fetch_ohlcv('VNM', '01/01/2024', '03/01/2024')
        
        self.assertIn("Data request timed out", str(context.exception))
    
    @patch('data.api_client.requests.get')
    @patch('data.api_client.requests.post')
    def test_fetch_ohlcv_connection_error(self, mock_post, mock_get):
        """Test OHLCV fetching with connection error - raises ConnectionError."""
        # Mock authentication
        mock_auth_response = Mock()
        mock_auth_response.status_code = 200
        mock_auth_response.json.return_value = {
            'data': {'accessToken': 'test_token'}
        }
        mock_auth_response.raise_for_status = Mock()
        mock_post.return_value = mock_auth_response
        
        # Mock connection error
        mock_get.side_effect = ConnectionError("Connection failed")
        
        # Should raise ConnectionError
        with self.assertRaises(ConnectionError) as context:
            self.client.fetch_ohlcv('VNM', '01/01/2024', '03/01/2024')
        
        self.assertIn("Failed to connect to SSI API", str(context.exception))
    
    @patch('data.api_client.requests.get')
    @patch('data.api_client.requests.post')
    def test_fetch_ohlcv_missing_columns_error(self, mock_post, mock_get):
        """Test OHLCV fetching with missing required columns - raises ValueError."""
        # Mock authentication
        mock_auth_response = Mock()
        mock_auth_response.status_code = 200
        mock_auth_response.json.return_value = {
            'data': {'accessToken': 'test_token'}
        }
        mock_auth_response.raise_for_status = Mock()
        mock_post.return_value = mock_auth_response
        
        # Mock incomplete data response (missing high, low, volume)
        mock_data_response = Mock()
        mock_data_response.status_code = 200
        mock_data_response.json.return_value = {
            'data': [
                {
                    'tradingDate': '2024-01-01',
                    'open': 100.5,
                    'close': 103.0
                }
            ]
        }
        mock_data_response.raise_for_status = Mock()
        mock_get.return_value = mock_data_response
        
        # Should raise ValueError due to missing columns
        with self.assertRaises(ValueError) as context:
            self.client.fetch_ohlcv('VNM', '01/01/2024', '03/01/2024')
        
        self.assertIn("Missing required columns", str(context.exception))
    
    def test_fetch_ohlcv_empty_symbol_error(self):
        """Test fetch_ohlcv with empty symbol - raises ValueError."""
        with self.assertRaises(ValueError) as context:
            self.client.fetch_ohlcv('', '01/01/2024', '03/01/2024')
        
        self.assertIn("Symbol must not be empty", str(context.exception))
    
    def test_fetch_ohlcv_empty_dates_error(self):
        """Test fetch_ohlcv with empty dates - raises ValueError."""
        with self.assertRaises(ValueError) as context:
            self.client.fetch_ohlcv('VNM', '', '03/01/2024')
        
        self.assertIn("start_date and end_date must not be empty", str(context.exception))


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
