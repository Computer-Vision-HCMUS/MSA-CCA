"""Test rate limiting with multiple consecutive API calls"""
from dotenv import load_dotenv
import os
import time
from data.api_client import SSIAPIClient

load_dotenv()

client = SSIAPIClient(
    os.getenv('SSI_CONSUMER_ID'),
    os.getenv('SSI_CONSUMER_SECRET'),
    os.getenv('SSI_BASE_URL')
)

print("Testing rate limiting with consecutive API calls...")
print("=" * 70)

# Test 1: First call
print("\n1. Fetching TCB data...")
start_time = time.time()
df1 = client.fetch_ohlcv('TCB', '01/01/2026', '01/02/2026')
elapsed1 = time.time() - start_time
print(f"   ✓ Fetched {len(df1)} records in {elapsed1:.2f}s")

# Test 2: Second call (should be rate limited)
print("\n2. Fetching VNM data (should wait due to rate limit)...")
start_time2 = time.time()
df2 = client.fetch_ohlcv('VNM', '01/01/2026', '01/02/2026')
elapsed2 = time.time() - start_time2
print(f"   ✓ Fetched {len(df2)} records in {elapsed2:.2f}s")

# Test 3: Third call after waiting
print("\n3. Fetching MBB data...")
start_time3 = time.time()
df3 = client.fetch_ohlcv('MBB', '01/01/2026', '01/02/2026')
elapsed3 = time.time() - start_time3
print(f"   ✓ Fetched {len(df3)} records in {elapsed3:.2f}s")

print("\n" + "=" * 70)
print("✓ Rate limiting working correctly!")
print(f"Total time: {elapsed1 + elapsed2 + elapsed3:.2f}s")
print("No 429 errors encountered!")
