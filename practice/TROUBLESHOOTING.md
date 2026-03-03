# Troubleshooting Guide - API Issues

## Lỗi: "Missing required columns in API response"

### Nguyên nhân

Lỗi này xảy ra khi API SSI trả về dữ liệu với tên cột khác với expected format. SSI API có thể thay đổi naming convention hoặc sử dụng tên cột khác nhau.

### Giải pháp

#### Bước 1: Debug API Response

Chạy script debug để xem chính xác API response:

```bash
# Cài đặt dependencies (nếu chưa có)
pip install -r requirements.txt

# Chạy debug script
python debug_api.py
```

Script sẽ hiển thị:
- ✅ Authentication status
- ✅ Raw API response structure
- ✅ Available column names
- ✅ Sample data

#### Bước 2: Xem kết quả Debug

Script sẽ in ra thông tin như:

```
Available Columns:
  - tradingDate: str
  - open: float
  - high: float
  - low: float
  - close: float
  - volume: int
```

hoặc có thể là:

```
Available Columns:
  - TradingDate: str
  - openPrice: float
  - highPrice: float
  - lowPrice: float
  - closePrice: float
  - totalVolume: int
```

#### Bước 3: Cập nhật Column Mapping (Nếu cần)

Nếu API sử dụng tên cột **hoàn toàn khác** với các aliases đã định nghĩa, bạn cần cập nhật file `data/api_client.py`:

Tìm phần `find_column()` aliases và thêm tên cột mới:

```python
# Map required columns with aliases
column_names = {
    "date": find_column([
        "tradingDate", "TradingDate", "date", "Date", 
        "ngayGiaoDich",
        "YOUR_NEW_DATE_COLUMN_NAME"  # ← Thêm tên mới ở đây
    ]),
    "open": find_column([
        "open", "Open", "openPrice", "OpenPrice", 
        "giaM",
        "YOUR_NEW_OPEN_COLUMN_NAME"  # ← Thêm tên mới ở đây
    ]),
    # ... tương tự cho các cột khác
}
```

#### Bước 4: Test lại

```bash
# Test với unittest
python -m unittest tests.test_api -v

# Hoặc chạy debug script lại
python debug_api.py

# Hoặc chạy app
streamlit run app.py
```

---

## Các Lỗi API Khác

### Lỗi: Date format parsing error

**Triệu chứng:**
```
time data "27/02/2026" doesn't match format "%Y-%m-%d", at position 0.
```

**Nguyên nhân:**
API trả về date format "DD/MM/YYYY" (ví dụ: 27/02/2026) thay vì "YYYY-MM-DD" (2026-02-27).

**Giải pháp:**
✅ **ĐÃ FIX**: Code hiện tại tự động xử lý cả hai format:
- ISO format: `2026-02-27` (YYYY-MM-DD)
- Vietnamese format: `27/02/2026` (DD/MM/YYYY)
- Mixed formats trong cùng dataset

Nếu vẫn gặp lỗi:
1. Cập nhật code mới nhất từ Git
2. Chạy test: `python -m unittest tests.test_api.TestSSIAPIClient.test_fetch_ohlcv_dd_mm_yyyy_format -v`
3. Nếu vẫn lỗi, chạy debug script để xem format thực tế:
   ```bash
   python debug_api.py
   ```

### Lỗi: "Authentication failed"

**Triệu chứng:**
```
❌ Lỗi xác thực API
Consumer ID hoặc Consumer Secret không chính xác.
```

**Giải pháp:**

1. Kiểm tra file `.env`:
   ```bash
   # Windows
   type .env
   
   # Linux/Mac
   cat .env
   ```

2. Đảm bảo credentials chính xác:
   - Không có khoảng trắng thừa
   - Không có quotes (`"`)
   - Format đúng:
     ```env
     SSI_CONSUMER_ID=actual_id_here
     SSI_CONSUMER_SECRET=actual_secret_here
     ```

3. Test credentials trực tiếp:
   ```bash
   python debug_api.py
   ```

### Lỗi: "Không có dữ liệu"

**Triệu chứng:**
```
❌ Không có dữ liệu
Không tìm thấy dữ liệu cho mã cổ phiếu...
```

**Nguyên nhân:**
- Mã cổ phiếu không đúng
- Khoảng thời gian không có dữ liệu giao dịch (ngày lễ, cuối tuần)
- Symbol chưa được niêm yết trong khoảng thời gian đã chọn

**Giải pháp:**

1. Kiểm tra mã cổ phiếu:
   - Phải là mã HOSE, HNX, UPCOM (VD: MBB, TCB, VNM, VIC)
   - Viết hoa toàn bộ (app tự động upper nhưng nên kiểm tra)

2. Chọn khoảng thời gian phù hợp:
   - Tránh ngày lễ, Tết
   - Tránh cuối tuần
   - Khuyến nghị: 3-6 tháng gần nhất

3. Test với mã phổ biến:
   ```
   Symbol 1: VNM
   Symbol 2: VIC
   Date: 3 tháng gần nhất
   ```

### Lỗi: "Connection timeout"

**Triệu chứng:**
```
❌ Lỗi kết nối
Không thể kết nối tới API SSI. Vui lòng kiểm tra kết nối internet.
```

**Giải pháp:**

1. Kiểm tra internet connection
2. Kiểm tra firewall/proxy settings
3. Thử tăng timeout trong `api_client.py`:
   ```python
   # Line ~139
   response = requests.post(
       auth_url,
       json=payload,
       headers={"Content-Type": "application/json"},
       timeout=30  # ← Tăng từ 10 lên 30 giây
   )
   ```

4. Kiểm tra trạng thái API SSI:
   - Truy cập: https://fc-data.ssi.com.vn/
   - Xem có bảo trì hay downtime không

---

## Debug Nâng Cao

### Chạy Python REPL để test trực tiếp

```python
# Mở Python REPL
python

# Import và test
>>> from data.api_client import SSIAPIClient
>>> import os
>>> from dotenv import load_dotenv
>>> load_dotenv()

>>> # Khởi tạo client
>>> client = SSIAPIClient(
...     consumer_id=os.getenv("SSI_CONSUMER_ID"),
...     consumer_secret=os.getenv("SSI_CONSUMER_SECRET"),
...     base_url=os.getenv("SSI_BASE_URL")
... )

>>> # Test authentication
>>> token = client._authenticate()
>>> print(f"Token: {token[:20]}...")

>>> # Test fetch data
>>> df = client.fetch_ohlcv("VNM", "01/01/2026", "01/03/2026")
>>> print(df.head())
>>> print(df.columns)
>>> print(df.dtypes)
```

### Kiểm tra raw HTTP response

```python
import requests
import os
from dotenv import load_dotenv

load_dotenv()

# Authenticate
auth_url = "https://fc-data.ssi.com.vn/api/v2/Market/AccessToken"
payload = {
    "consumerID": os.getenv("SSI_CONSUMER_ID"),
    "consumerSecret": os.getenv("SSI_CONSUMER_SECRET")
}
response = requests.post(auth_url, json=payload, timeout=10)
token = response.json()["data"]["accessToken"]
print(f"Token: {token[:20]}...")

# Fetch data
data_url = "https://fc-data.ssi.com.vn/api/v2/Market/DailyOhlc"
params = {"symbol": "VNM", "fromDate": "01/01/2026", "toDate": "01/03/2026"}
headers = {"Authorization": f"Bearer {token}"}
response = requests.get(data_url, params=params, headers=headers, timeout=30)

# Print response
import json
print(json.dumps(response.json(), indent=2))
```

---

## Lỗi: "Insufficient data points after log return calculation"

### Nguyên nhân

Lỗi này xảy ra khi khoảng thời gian được chọn quá ngắn so với rolling window, **HOẶC** API không trả về đủ dữ liệu:

```
Got 9 rows, need at least 20 for rolling window
```

#### Hai trường hợp phổ biến:

**1. Date range quá ngắn:**
- Chọn 10 ngày nhưng rolling window = 20 → Lỗi

**2. SSI API giới hạn dữ liệu lịch sử:** ⚠️ **QUAN TRỌNG**
- SSI API chỉ cung cấp dữ liệu gần đây (thường 30-90 ngày)
- Mặc dù bạn chọn 6 tháng (VD: 02/09/2025-01/03/2026), API chỉ trả về ~10 ngày
- Nguyên nhân: API không có dữ liệu lịch sử xa

**Vì sao?**
- Tính log returns mất đi 1 dòng đầu tiên
- Rolling window 20 cần ít nhất 20 data points
- Nên cần tối thiểu 21 ngày giao dịch thô

### Giải pháp

#### ✅ Option 1: Chọn date range GẦN ĐÂY (Khuyến nghị cao)

⚠️ **SSI API chỉ có dữ liệu gần đây (30-90 ngày), KHÔNG có dữ liệu lịch sử xa!**

```
Khuyến nghị date range:
- Tốt nhất: 30-60 ngày gần nhất (VD: 1 tháng đến hôm nay)
- Tránh: Chọn date range cách đây > 90 ngày
```

**Ví dụ đúng (ngày hôm nay: 01/03/2026):**
- Start: 01/01/2026 (cách đây 60 ngày) ✅
- End: 01/03/2026 (hôm nay) ✅

**Ví dụ sai:**
- Start: 02/09/2025 (cách đây 180 ngày) ❌ → API chỉ trả về ~10 ngày
- Kết quả: Không đủ dữ liệu cho rolling window 20

**Trong Streamlit App:**
1. Mở sidebar
2. Chọn Start Date trong vòng 30-60 ngày gần nhất
3. End Date = hôm nay (hoặc gần nhất)
4. Click "🚀 Phân tích"

#### Option 2: Giảm Rolling Window

**Trong Streamlit App:**
1. Mở sidebar
2. Giảm "Rolling Window" slider xuống (VD: từ 20 → 10)
3. Click "🚀 Phân tích" lại

### Cảnh báo tự động

Ứng dụng sẽ hiển thị cảnh báo trong sidebar khi:
- ⚠️ **Warning màu vàng**: Dữ liệu có thể không đủ
- ℹ️ **Info màu xanh**: Dữ liệu vừa đủ nhưng nên thêm
- ✅ **Success màu xanh lá**: Dữ liệu tốt

### Ví dụ

```python
# ❌ Không đủ dữ liệu
# 10 ngày giao dịch, rolling window 20 → LỖI

# ✅ Đủ dữ liệu
# 120 ngày giao dịch (6 tháng), rolling window 20 → OK
```

---

## Liên hệ Support

Nếu vấn đề vẫn chưa được giải quyết:

1. **Check logs**: Xem chi tiết lỗi trong terminal
2. **Run debug_api.py**: Gửi output để được support
3. **Check SSI docs**: https://docs.ssi.com.vn/
4. **GitHub Issues**: Tạo issue với full error output

---

**Last Updated**: March 2026
