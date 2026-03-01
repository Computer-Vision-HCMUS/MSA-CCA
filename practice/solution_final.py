"""Test với rolling window PHÙ HỢP với data có sẵn"""
from dotenv import load_dotenv
import os
from data.api_client import SSIAPIClient
from core.analyzer import CorrelationAnalyzer
from datetime import datetime, timedelta

load_dotenv()

print("="*70)
print("GIẢI PHÁP CUỐI CÙNG - ROLLING WINDOW PHÙ HỢP")
print("="*70)

client = SSIAPIClient(
    os.getenv('SSI_CONSUMER_ID'),
    os.getenv('SSI_CONSUMER_SECRET'),
    os.getenv('SSI_BASE_URL')
)

# Date range 60 ngày gần nhất (nhưng API sẽ chỉ trả về ~10 ngày)
today = datetime.now()
start = today - timedelta(days=60)

start_date = start.strftime("%d/%m/%Y")
end_date = today.strftime("%d/%m/%Y")

print(f"\n📅 Date Range: {start_date} to {end_date}")
print()

# Load data
print("🔄 Đang tải dữ liệu...")
try:
    df1 = client.fetch_ohlcv('MBB', start_date, end_date)
    df2 = client.fetch_ohlcv('TCB', start_date, end_date)
    
    actual = min(len(df1), len(df2))
    print(f"\n✅ Đã tải: {actual} ngày giao dịch")
    print(f"   (API chỉ trả về data gần nhất)")
    
    # Sử dụng rolling window phù hợp
    rolling_window = actual - 1  # Để đủ cho log returns
    if rolling_window < 5:
        rolling_window = 5
    if rolling_window > 10:
        rolling_window = 10
        
    print(f"\n📊 Chọn rolling window: {rolling_window}")
    print(f"   (Tự động điều chỉnh theo data có sẵn)")
    
    # Run CCA
    print(f"\n🧮 Đang chạy CCA analysis...")
    analyzer = CorrelationAnalyzer()
    zscore, spread, w_x, w_y = analyzer.calculate_cca_zscore(
        df1, df2, window=rolling_window
    )
    
    print(f"\n🎉 THÀNH CÔNG!")
    print(f"="*70)
    print(f"\n📈 KẾT QUẢ PHÂN TÍCH:")
    print(f"   - Data points: {actual} ngày")
    print(f"   - Rolling window: {rolling_window}")
    print(f"   - Z-score values: {len(zscore)}")
    print(f"   - Latest Z-score: {zscore.iloc[-1]:.2f}")
    print(f"   - Latest Spread: {spread.iloc[-1]:.6f}")
    
    # Trading signal
    z = zscore.iloc[-1]
    if z > 2:
        signal = "🔴 BÁN MBB - MUA TCB"
        explanation = f"Spread quá cao ({z:.2f}σ)"
    elif z < -2:
        signal = "🟢 MUA MBB - BÁN TCB"
        explanation = f"Spread quá thấp ({z:.2f}σ)"
    else:
        signal = "🟡 CHỜ ĐỢI"
        explanation = f"Spread bình thường ({z:.2f}σ)"
    
    print(f"\n🎯 TRADING SIGNAL:")
    print(f"   {signal}")
    print(f"   {explanation}")
    
    print(f"\n" + "="*70)
    print(f"✅ CÁCH SỬ DỤNG APP:")
    print(f"   1. Mở Streamlit app: streamlit run app.py")
    print(f"   2. Sidebar → Rolling Window: chọn {rolling_window} (hoặc nhỏ hơn)")
    print(f"   3. Date Range: giữ mặc định (60 ngày gần nhất)")
    print(f"   4. Click 🚀 Phân tích")
    print(f"="*70)
        
except Exception as e:
    print(f"\n❌ LỖI: {str(e)}")
    import traceback
    traceback.print_exc()
