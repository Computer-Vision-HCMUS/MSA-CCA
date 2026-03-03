"""Test với date range thực tế của user"""
from dotenv import load_dotenv
import os
from data.api_client import SSIAPIClient
from core.analyzer import CorrelationAnalyzer

load_dotenv()

print("="*70)
print("TEST VỚI DATE RANGE CỦA USER")
print("="*70)

client = SSIAPIClient(
    os.getenv('SSI_CONSUMER_ID'),
    os.getenv('SSI_CONSUMER_SECRET'),
    os.getenv('SSI_BASE_URL')
)

# Date range của user
start_date = "02/09/2025"
end_date = "01/03/2026"

print(f"\n📅 Date Range: {start_date} to {end_date}")
print("   (Expected: ~6 tháng = 120+ ngày giao dịch)")
print()

# Load data
print("🔄 Đang tải dữ liệu...")
try:
    df1 = client.fetch_ohlcv('MBB', start_date, end_date)
    df2 = client.fetch_ohlcv('TCB', start_date, end_date)
    
    print(f"✅ Đã tải dữ liệu:")
    print(f"   - MBB: {len(df1)} ngày giao dịch")
    print(f"   - TCB: {len(df2)} ngày giao dịch")
    print(f"   - Minimum: {min(len(df1), len(df2))} ngày")
    
    # Check validation
    rolling_window = 20
    min_required = rolling_window + 1  # 21
    actual = min(len(df1), len(df2))
    
    print(f"\n📊 Validation Check:")
    print(f"   - Rolling window: {rolling_window}")
    print(f"   - Required: {min_required}+ ngày giao dịch")
    print(f"   - Actual: {actual} ngày giao dịch")
    print(f"   - After log returns: {actual - 1} rows")
    
    if actual < min_required:
        print(f"\n❌ VALIDATION FAILED!")
        print(f"   Cần thêm {min_required - actual} ngày nữa")
        print(f"\n💡 Giải pháp:")
        print(f"   1. Giảm rolling window xuống ≤ {actual - 1}")
        print(f"   2. Hoặc tăng date range")
    else:
        print(f"\n✅ VALIDATION PASSED! Đủ dữ liệu")
        
        # Try CCA analysis
        print(f"\n🧮 Đang chạy CCA analysis...")
        analyzer = CorrelationAnalyzer()
        zscore, spread, w_x, w_y = analyzer.calculate_cca_zscore(
            df1, df2, window=rolling_window
        )
        
        print(f"✅ CCA Analysis thành công!")
        print(f"   - Z-score values: {len(zscore)}")
        print(f"   - Latest Z-score: {zscore.iloc[-1]:.2f}")
        print(f"   - Latest Spread: {spread.iloc[-1]:.6f}")
        
        # Trading signal
        if zscore.iloc[-1] > 2:
            signal = "🔴 Bán MBB Mua TCB"
        elif zscore.iloc[-1] < -2:
            signal = "🟢 Mua MBB Bán TCB"
        else:
            signal = "🟡 Chờ đợi"
        print(f"   - Signal: {signal}")
        
        print(f"\n✅ KẾT LUẬN: App nên chạy OK với data này!")
        
except Exception as e:
    print(f"\n❌ LỖI: {str(e)}")
    print(f"\n🔍 Chi tiết lỗi:")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
