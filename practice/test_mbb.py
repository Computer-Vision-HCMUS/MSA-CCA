"""Quick test for MBB data fetch"""
from dotenv import load_dotenv
import os
from data.api_client import SSIAPIClient

load_dotenv()

client = SSIAPIClient(
    os.getenv('SSI_CONSUMER_ID'),
    os.getenv('SSI_CONSUMER_SECRET'),
    os.getenv('SSI_BASE_URL')
)

print("Fetching MBB data from 02/09/2025 to 01/03/2026...")
df = client.fetch_ohlcv('MBB', '02/09/2025', '01/03/2026')

print(f"\n✓ SUCCESS! Fetched {len(df)} records for MBB")
print(f"Date range: {df.index.min()} to {df.index.max()}")
print(f"\nFirst 5 rows:")
print(df.head())
print(f"\nLast 5 rows:")
print(df.tail())
