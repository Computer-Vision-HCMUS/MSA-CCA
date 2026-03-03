"""
Debug script to test SSI API connection and inspect response.

This script helps debug API issues by:
1. Testing authentication
2. Fetching sample data
3. Printing actual response structure and columns

Usage:
    python debug_api.py
"""

import os
import sys
import json

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Try to load dotenv if available, otherwise proceed without it
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("ℹ️  python-dotenv not installed, reading from environment only")
    print()

from data.api_client import SSIAPIClient


def main():
    """Test SSI API connection and inspect response."""
    
    print("=" * 70)
    print("SSI API DEBUG TOOL")
    print("=" * 70)
    print()
    
    # Read credentials
    consumer_id = os.getenv("SSI_CONSUMER_ID")
    consumer_secret = os.getenv("SSI_CONSUMER_SECRET")
    base_url = os.getenv("SSI_BASE_URL", "https://fc-data.ssi.com.vn/")
    
    # Check credentials
    if not consumer_id or not consumer_secret:
        print("❌ ERROR: Missing credentials in .env file!")
        print()
        print("Please ensure your .env file contains:")
        print("  SSI_CONSUMER_ID=your_id")
        print("  SSI_CONSUMER_SECRET=your_secret")
        print("  SSI_BASE_URL=https://fc-data.ssi.com.vn/")
        sys.exit(1)
    
    print("✓ Credentials loaded from .env")
    print(f"  Consumer ID: {consumer_id[:10]}..." if len(consumer_id) > 10 else consumer_id)
    print(f"  Base URL: {base_url}")
    print()
    
    # Test authentication
    print("-" * 70)
    print("STEP 1: Testing Authentication")
    print("-" * 70)
    
    try:
        client = SSIAPIClient(
            consumer_id=consumer_id,
            consumer_secret=consumer_secret,
            base_url=base_url
        )
        
        token = client._authenticate()
        print(f"✓ Authentication successful!")
        print(f"  Access Token: {token[:20]}..." if len(token) > 20 else token)
        print()
        
    except Exception as e:
        print(f"❌ Authentication failed: {e}")
        sys.exit(1)
    
    # Test data fetching
    print("-" * 70)
    print("STEP 2: Fetching Sample Data (VNM, last 5 days)")
    print("-" * 70)
    
    from datetime import datetime, timedelta
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5)
    
    start_str = start_date.strftime("%d/%m/%Y")
    end_str = end_date.strftime("%d/%m/%Y")
    
    print(f"  Symbol: VNM")
    print(f"  Date Range: {start_str} to {end_str}")
    print()
    
    try:
        # Use internal method to get raw response first
        import requests
        
        data_url = f"{base_url}/api/v2/Market/DailyOhlc"
        params = {
            "symbol": "VNM",
            "fromDate": start_str,
            "toDate": end_str
        }
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json"
        }
        
        response = requests.get(data_url, params=params, headers=headers, timeout=30)
        response.raise_for_status()
        result = response.json()
        
        print("✓ Raw API Response:")
        print(f"  Status: {result.get('status', 'N/A')}")
        print(f"  Message: {result.get('message', 'N/A')}")
        
        if "data" in result and result["data"]:
            data_list = result["data"]
            print(f"  Records: {len(data_list)}")
            print()
            
            # Show first record structure
            if len(data_list) > 0:
                first_record = data_list[0]
                print("✓ First Record Structure:")
                print(json.dumps(first_record, indent=2, ensure_ascii=False))
                print()
                
                print("✓ Available Columns & Sample Values:")
                for col, val in first_record.items():
                    val_type = type(val).__name__
                    if col.lower() in ['tradingdate', 'date', 'ngaygiaodich']:
                        print(f"  - {col}: {val} (type: {val_type}) ← DATE COLUMN")
                    else:
                        print(f"  - {col}: {val} (type: {val_type})")
                print()
        else:
            print("  ⚠️  No data returned")
            print()
        
    except Exception as e:
        print(f"❌ Failed to fetch raw data: {e}")
        print()
    
    # Test with SSIAPIClient
    print("-" * 70)
    print("STEP 3: Testing SSIAPIClient.fetch_ohlcv()")
    print("-" * 70)
    
    try:
        df = client.fetch_ohlcv(
            symbol="VNM",
            start_date=start_str,
            end_date=end_str
        )
        
        print("✓ DataFrame created successfully!")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
        print(f"  Index: {df.index.name}")
        print(f"  Date Range: {df.index.min()} to {df.index.max()}")
        print()
        
        print("✓ Sample Data (last 3 rows):")
        print(df.tail(3))
        print()
        
        print("✓ Data Types:")
        print(df.dtypes)
        print()
        
        print("=" * 70)
        print("✓✓✓ ALL TESTS PASSED!")
        print("=" * 70)
        
    except Exception as e:
        print(f"❌ Failed to create DataFrame: {e}")
        print()
        import traceback
        print("Full traceback:")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
