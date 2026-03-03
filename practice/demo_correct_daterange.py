"""Demo với date range ĐÚNG - gần đây"""
from dotenv import load_dotenv
import os
from data.api_client import SSIAPIClient
from core.analyzer import CorrelationAnalyzer
from datetime import datetime, timedelta

load_dotenv()

print("="*70)
print("DEMO - DATE RANGE GẦN ĐÂY (ĐÚNG)")
print("="*70)

client = SSIAPIClient(
    os.getenv('SSI_CONSUMER_ID'),
    os.getenv('SSI_CONSUMER_SECRET'),
    os.getenv('SSI_BASE_URL')
)

# Date range đúng: 60 ngày gần nhất
today = datetime.now()
start = today - timedelta(days=60)

start_date = start.strftime("%d/%m/%Y")
end_date = today.strftime("%d/%m/%Y")

print(f"\n📅 Date Range: {start_date} to {end_date}")
print(f"   (60 ngày gần nhất - trong giới hạn của SSI API)")
print()

# Load data
print("🔄 Đang tải dữ liệu...")
try:
    df1 = client.fetch_ohlcv('MBB', start_date, end_date)
    df2 = client.fetch_ohlcv('TCB', start_date, end_date)
    
    print(f"\n✅ Đã tải dữ liệu:")
    print(f"   - MBB: {len(df1)} ngày giao dịch")
    print(f"   - TCB: {len(df2)} ngày giao dịch")
    print(f"   - Minimum: {min(len(df1), len(df2))} ngày")
    
    # Check validation for rolling window 20
    rolling_window = 20
    min_required = rolling_window + 1  # 21
    actual = min(len(df1), len(df2))
    
    print(f"\n📊 Validation Check:")
    print(f"   - Rolling window: {rolling_window}")
    print(f"   - Required: {min_required}+ ngày")
    print(f"   - Actual: {actual} ngày")
    
    if actual < min_required:
        print(f"\n❌ VẪN KHÔNG ĐỦ!")
        print(f"   Giảm rolling window xuống ≤ {actual - 1}")
    else:
        print(f"\n✅ ĐỦ DỮ LIỆU! Chạy CCA analysis...")
        
        analyzer = CorrelationAnalyzer()
        zscore, spread, w_x, w_y = analyzer.calculate_cca_zscore(
            df1, df2, window=rolling_window
        )
        
        print(f"\n🎉 CCA ANALYSIS THÀNH CÔNG!")
        print(f"   - Z-score values: {len(zscore)}")
        print(f"   - Latest Z-score: {zscore.iloc[-1]:.2f}")
        print(f"   - Latest Spread: {spread.iloc[-1]:.6f}")
        
        # Trading signal
        z = zscore.iloc[-1]
        if z > 2:
            signal = "🔴 BÁN MBB - MUA TCB"
            explanation = f"Spread quá cao ({z:.2f}σ), kỳ vọng giảm"
        elif z < -2:
            signal = "🟢 MUA MBB - BÁN TCB"
            explanation = f"Spread quá thấp ({z:.2f}σ), kỳ vọng tăng"
        else:
            signal = "🟡 CHỜ ĐỢI"
            explanation = f"Spread trong vùng an toàn ({z:.2f}σ)"
        
        print(f"\n📈 TRADING SIGNAL:")
        print(f"   {signal}")
        print(f"   {explanation}")
        
        print(f"\n💡 TÓM TẮT:")
        print(f"   ✅ Chọn date range 60 ngày gần nhất")
        print(f"   ✅ API trả về {actual} ngày (đủ cho rolling window {rolling_window})")
        print(f"   ✅ CCA analysis thành công")
        print(f"   ✅ Có trading signal rõ ràng")
        
except Exception as e:
    print(f"\n❌ LỖI: {str(e)}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
print("💡 KHUYẾN NGHỊ:")
print("   - Luôn chọn Start Date trong vòng 30-60 ngày gần nhất")
print("   - End Date = hôm nay (hoặc gần nhất)")
print("   - Tránh chọn date quá xa (> 90 ngày)")
print("="*70)
