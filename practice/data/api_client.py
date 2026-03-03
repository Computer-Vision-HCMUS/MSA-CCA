"""
SSI API Client Module

This module provides a client for interacting with the SSI (Securities Services Inc.) API.
It handles authentication and data retrieval for market data.
"""

import requests
import pandas as pd
import time
from requests.exceptions import Timeout, ConnectionError, HTTPError


class SSIAPIClient:
    """
    Client for SSI API communication.
    
    Handles authentication and data fetching from SSI market data endpoints.
    Implements error handling for network issues and data validation.
    """
    
    def __init__(self, consumer_id: str, consumer_secret: str, base_url: str) -> None:
        """
        Initialize SSI API Client.
        
        Args:
            consumer_id: SSI API consumer ID for authentication
            consumer_secret: SSI API consumer secret for authentication
            base_url: Base URL of the SSI API endpoint
            
        Raises:
            ValueError: If any parameter is empty or None
        """
        if not consumer_id or not consumer_secret or not base_url:
            raise ValueError("consumer_id, consumer_secret, and base_url must not be empty")
            
        self.consumer_id = consumer_id
        self.consumer_secret = consumer_secret
        self.base_url = base_url.rstrip('/')  # Remove trailing slash if present
        self._access_token: str = ""
        self._last_request_time: float = 0  # Track last API request for rate limiting
        
    def _authenticate(self) -> str:
        """
        Authenticate with SSI API and retrieve access token.
        
        Makes a POST request to the AccessToken endpoint with consumer credentials.
        
        Returns:
            str: Access token for subsequent API requests
            
        Raises:
            ConnectionError: If unable to connect to the API
            Timeout: If the request times out
            HTTPError: If the API returns an error status code
            ValueError: If authentication fails or returns invalid data
        """
        auth_url = f"{self.base_url}/api/v2/Market/AccessToken"
        
        payload = {
            "consumerID": self.consumer_id,
            "consumerSecret": self.consumer_secret
        }
        
        try:
            response = requests.post(
                auth_url,
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=10  # 10 seconds timeout
            )
            
            # DEBUG LOGGING: Always log response details (safely handle Mock objects)
            status_code = getattr(response, 'status_code', 'N/A')
            response_text = getattr(response, 'text', '')
            print(f"[DEBUG] Authentication Response Status: {status_code}")
            if response_text and isinstance(response_text, str):
                print(f"[DEBUG] Authentication Response Body: {response_text[:500]}")  # First 500 chars
            
            response.raise_for_status()
            
            data = response.json()
            
            # Check status in JSON body (SSI API may return status field)
            if isinstance(data, dict) and "status" in data:
                # Accept 200, "200", "Success" as valid statuses
                valid_statuses = [200, "200", "Success"]
                if data["status"] not in valid_statuses:
                    error_msg = data.get("message", "Unknown error")
                    raise ValueError(
                        f"API returned error status: {data['status']}. "
                        f"Message: {error_msg}"
                    )
            
            # Validate response structure
            if not data or "data" not in data or "accessToken" not in data["data"]:
                raise ValueError(
                    f"Invalid authentication response: missing access token. "
                    f"Response structure: {list(data.keys()) if isinstance(data, dict) else type(data)}"
                )
                
            self._access_token = data["data"]["accessToken"]
            
            if not self._access_token:
                raise ValueError("Authentication failed: empty access token received")
            
            print(f"[DEBUG] Authentication successful. Token: {self._access_token[:20]}...")
            return self._access_token
            
        except Timeout as e:
            raise Timeout(f"Authentication request timed out: {str(e)}") from e
        except ConnectionError as e:
            raise ConnectionError(f"Failed to connect to SSI API: {str(e)}") from e
        except HTTPError as e:
            # Enhanced error logging
            status_code = e.response.status_code if hasattr(e, 'response') and e.response else 'N/A'
            response_text = getattr(e.response, 'text', 'N/A') if hasattr(e, 'response') and e.response else 'N/A'
            
            print(f"[ERROR] HTTP Error during authentication")
            print(f"[ERROR] Status Code: {status_code}")
            print(f"[ERROR] Response Body: {response_text}")
            
            raise HTTPError(
                f"HTTP error during authentication: {str(e)}"
            ) from e
        except requests.exceptions.RequestException as e:
            print(f"[ERROR] Request exception during authentication: {str(e)}")
            raise Exception(f"Unexpected error during authentication: {str(e)}") from e
            
    def fetch_ohlcv(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch OHLCV (Open, High, Low, Close, Volume) data for a symbol.
        
        Retrieves daily market data from SSI API for the specified symbol and date range.
        Automatically handles authentication if token is not available.
        
        Args:
            symbol: Stock symbol (e.g., 'VNM', 'VIC')
            start_date: Start date in format 'DD/MM/YYYY'
            end_date: End date in format 'DD/MM/YYYY'
            
        Returns:
            pd.DataFrame: DataFrame with TradingDate as index and columns ['O', 'H', 'L', 'C', 'V']
                         All values are float type, sorted ascending by date
                         
        Raises:
            ValueError: If parameters are invalid or API returns empty data
            ConnectionError: If unable to connect to the API
            Timeout: If the request times out
            HTTPError: If the API returns an error status code
        """
        if not symbol:
            raise ValueError("Symbol must not be empty")
        if not start_date or not end_date:
            raise ValueError("start_date and end_date must not be empty")
            
        # Authenticate if no token exists
        if not self._access_token:
            self._authenticate()
            
        data_url = f"{self.base_url}/api/v2/Market/DailyOhlc"
        
        headers = {
            "Authorization": f"Bearer {self._access_token}",
            "Content-Type": "application/json"
        }
        
        # PAGINATION: Fetch all pages of data
        all_data = []
        page_index = 1
        page_size = 1000  # Maximum page size recommended by API
        
        print(f"[DEBUG] Fetching OHLCV for {symbol} from {start_date} to {end_date}")
        print(f"[DEBUG] Using pagination: pageSize={page_size}")
        
        try:
            while True:
                params = {
                    "symbol": symbol,
                    "fromDate": start_date,
                    "toDate": end_date,
                    "pageSize": page_size,
                    "pageIndex": page_index
                }
                
                print(f"[DEBUG] Fetching page {page_index}...")
                print(f"[DEBUG] Request URL: {data_url}")
                print(f"[DEBUG] Request Params: {params}")
                
                # Rate limiting: SSI API allows max 1 request per second
                time_since_last_request = time.time() - self._last_request_time
                if time_since_last_request < 1.0:
                    sleep_time = 1.0 - time_since_last_request
                    print(f"[DEBUG] Rate limiting: sleeping for {sleep_time:.2f}s")
                    time.sleep(sleep_time)
                
                response = requests.get(
                    data_url,
                    params=params,
                    headers=headers,
                    timeout=30  # 30 seconds timeout for data retrieval
                )
                
                # Update last request time for rate limiting
                self._last_request_time = time.time()
                
                # DEBUG LOGGING: Always log response details (safely handle Mock objects)
                status_code = getattr(response, 'status_code', 'N/A')
                response_text = getattr(response, 'text', '')
                print(f"[DEBUG] Page {page_index} Response Status: {status_code}")
                if response_text and isinstance(response_text, str):
                    print(f"[DEBUG] Response Body (first 500 chars): {response_text[:500]}")
                
                response.raise_for_status()
                
                result = response.json()
                
                # Check status in JSON body (SSI API may return status field)
                if isinstance(result, dict) and "status" in result:
                    # Accept 200, "200", "Success" as valid statuses
                    valid_statuses = [200, "200", "Success"]
                    if result["status"] not in valid_statuses:
                        error_msg = result.get("message", "Unknown error")
                        response_text = getattr(response, 'text', str(result))
                        print(f"[ERROR] API returned error status: {result['status']}")
                        print(f"[ERROR] Error message: {error_msg}")
                        print(f"[ERROR] Full response: {response_text}")
                        raise ValueError(
                            f"API returned error status: {result['status']}. "
                            f"Message: {error_msg}. "
                            f"Symbol: {symbol}, Date range: {start_date} to {end_date}"
                        )
                
                # Validate response structure
                if not result or "data" not in result:
                    response_text = getattr(response, 'text', str(result))
                    print(f"[ERROR] Invalid response structure. Keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
                    print(f"[ERROR] Full response: {response_text}")
                    raise ValueError(
                        f"Invalid API response for symbol {symbol}: missing data field. "
                        f"Available keys: {list(result.keys()) if isinstance(result, dict) else 'N/A'}"
                    )
                
                page_data = result["data"]
                
                # Check if this is the last page
                if not page_data or len(page_data) == 0:
                    print(f"[DEBUG] Page {page_index}: No more data, stopping pagination")
                    break
                
                # Add page data to collection
                all_data.extend(page_data)
                print(f"[DEBUG] Page {page_index}: Fetched {len(page_data)} records (Total: {len(all_data)})")
                
                # If page has fewer records than pageSize, it's the last page
                if len(page_data) < page_size:
                    print(f"[DEBUG] Page {page_index}: Got {len(page_data)} < {page_size}, last page reached")
                    break
                
                # Move to next page
                page_index += 1
                
                # Sleep between pages to avoid rate limiting (0.2s as per requirements)
                # This is in addition to the 1s rate limit check at start of loop
                time.sleep(0.2)
            
            # Check if we got any data at all
            if not all_data or len(all_data) == 0:
                raise ValueError(
                    f"No data returned for symbol {symbol} "
                    f"from {start_date} to {end_date}"
                )
            
            print(f"[DEBUG] Pagination complete: {len(all_data)} total records from {page_index} page(s)")
            
            # Convert all collected data to DataFrame
            df = pd.DataFrame(all_data)
            
            # Create case-insensitive column mapping
            # API may return columns in different cases or formats
            col_mapping = {col.lower(): col for col in df.columns}
            
            # Try multiple possible column names (case-insensitive)
            # SSI API may use different naming conventions
            def find_column(aliases):
                """Find first matching column from list of aliases."""
                for alias in aliases:
                    if alias.lower() in col_mapping:
                        return col_mapping[alias.lower()]
                return None
            
            # Map required columns with aliases
            column_names = {
                "date": find_column(["tradingDate", "TradingDate", "date", "Date", "ngayGiaoDich"]),
                "open": find_column(["open", "Open", "openPrice", "OpenPrice", "giaM"]),
                "high": find_column(["high", "High", "highPrice", "HighPrice", "giaCaoNhat"]),
                "low": find_column(["low", "Low", "lowPrice", "LowPrice", "giaThapNhat"]),
                "close": find_column(["close", "Close", "closePrice", "ClosePrice", "giaDong"]),
                "volume": find_column(["volume", "Volume", "totalVolume", "TotalVolume", "khoiLuong"])
            }
            
            # Check for missing columns
            missing_cols = [k for k, v in column_names.items() if v is None]
            if missing_cols:
                raise ValueError(
                    f"Missing required columns in API response: {missing_cols}. "
                    f"Available columns: {list(df.columns)}. "
                    f"Please check API documentation or run debug_api.py to inspect response structure."
                )
            
            # Rename columns to standard format
            df = df.rename(columns={
                column_names["open"]: "O",
                column_names["high"]: "H",
                column_names["low"]: "L",
                column_names["close"]: "C",
                column_names["volume"]: "V",
                column_names["date"]: "tradingDate"
            })
            
            # Select only OHLCV columns
            df = df[["tradingDate", "O", "H", "L", "C", "V"]]
            
            # Convert trading date to datetime (handle multiple date formats)
            # API may return dates in different formats: "YYYY-MM-DD" or "DD/MM/YYYY"
            try:
                # Try ISO format first (YYYY-MM-DD)
                df["tradingDate"] = pd.to_datetime(df["tradingDate"], format="%Y-%m-%d")
            except (ValueError, TypeError):
                try:
                    # Try DD/MM/YYYY format
                    df["tradingDate"] = pd.to_datetime(df["tradingDate"], format="%d/%m/%Y")
                except (ValueError, TypeError):
                    # Let pandas infer the format
                    df["tradingDate"] = pd.to_datetime(df["tradingDate"], format="mixed", dayfirst=True)
            
            # Set trading date as index
            df = df.set_index("tradingDate")
            
            # Sort by date ascending (oldest to newest)
            df = df.sort_index(ascending=True)
            
            # Convert all OHLCV columns to float
            for col in ["O", "H", "L", "C", "V"]:
                df[col] = df[col].astype(float)
            
            print(f"[DEBUG] Successfully fetched {len(df)} records for {symbol}")
            return df
            
        except Timeout as e:
            print(f"[ERROR] Timeout for symbol {symbol}")
            print(f"[ERROR] Details: {str(e)}")
            raise Timeout(f"Data request timed out for symbol {symbol}: {str(e)}") from e
        except ConnectionError as e:
            print(f"[ERROR] Connection error for symbol {symbol}")
            print(f"[ERROR] Details: {str(e)}")
            raise ConnectionError(f"Failed to connect to SSI API for symbol {symbol}: {str(e)}") from e
        except HTTPError as e:
            # Enhanced error logging
            status_code = e.response.status_code if hasattr(e, 'response') and e.response else 'N/A'
            response_text = getattr(e.response, 'text', 'N/A') if hasattr(e, 'response') and e.response else 'N/A'
            
            print(f"[ERROR] HTTP Error for symbol {symbol}")
            print(f"[ERROR] Status Code: {status_code}")
            print(f"[ERROR] Response Body: {response_text}")
            
            # Try re-authenticating once if we get 401 Unauthorized
            if hasattr(e, 'response') and e.response is not None and e.response.status_code == 401:
                print("[INFO] Received 401 Unauthorized - attempting re-authentication")
                self._access_token = ""
                self._authenticate()
                # Retry the request once
                print("[INFO] Retrying fetch_ohlcv after re-authentication")
                return self.fetch_ohlcv(symbol, start_date, end_date)
            
            raise HTTPError(
                f"HTTP error fetching data for symbol {symbol}: {str(e)}"
            ) from e
        except requests.exceptions.RequestException as e:
            print(f"[ERROR] Request exception for symbol {symbol}")
            print(f"[ERROR] Details: {str(e)}")
            raise Exception(f"Unexpected error fetching data for symbol {symbol}: {str(e)}") from e
        except Exception as e:
            # Catch pandas or other processing errors
            if isinstance(e, (ValueError, Timeout, ConnectionError, HTTPError)):
                raise
            print(f"[ERROR] Data processing error for symbol {symbol}")
            print(f"[ERROR] Error type: {type(e).__name__}")
            print(f"[ERROR] Details: {str(e)}")
            raise ValueError(f"Error processing data for symbol {symbol}: {str(e)}") from e
