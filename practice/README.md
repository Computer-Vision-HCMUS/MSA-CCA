# Pairs Trading CCA - Ứng dụng phân tích giao dịch cặp cổ phiếu

## 📋 Giới thiệu

Hệ thống phân tích giao dịch cặp cổ phiếu sử dụng Canonical Correlation Analysis (CCA) để xác định cơ hội mean-reversion trading. Ứng dụng tích hợp với API SSI để lấy dữ liệu lịch sử và tính toán các chỉ báo giao dịch.

## 🚀 Cài đặt nhanh

### Bước 1: Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### Bước 2: Cấu hình API credentials (BẮT BUỘC)

```bash
# Copy file mẫu
copy .env.example .env    # Windows
# hoặc
cp .env.example .env      # Linux/Mac
```

### Bước 3: Điền thông tin xác thực vào file .env

Mở file `.env` và điền thông tin API từ SSI:

```env
SSI_CONSUMER_ID=your_actual_consumer_id_here
SSI_CONSUMER_SECRET=your_actual_consumer_secret_here
SSI_BASE_URL=https://fc-data.ssi.com.vn/
```

⚠️ **CHÚ Ý QUAN TRỌNG:**
- File `.env` chứa thông tin nhạy cảm và **ĐÃ ĐƯỢC GHI VÀO .gitignore**
- **TUYỆT ĐỐI KHÔNG commit file .env** lên Git
- Chỉ chia sẻ file `.env.example` với team members

### Bước 4: Chạy ứng dụng

```bash
streamlit run app.py
```

Ứng dụng sẽ mở tại: `http://localhost:8501`

## 🔐 Bảo mật

### Strict Mode - Environment Variables Only

Ứng dụng hoạt động ở chế độ **STRICT MODE**:
- ✅ Credentials **CHỈ** được đọc từ file `.env`
- ✅ **KHÔNG** có ô nhập API key trên giao diện web
- ✅ **Fail-fast**: Dừng ngay nếu thiếu credentials
- ✅ File `.env` tự động bị Git ignore

### Cách kiểm tra credentials

```bash
# Windows PowerShell
Get-Content .env

# Linux/Mac
cat .env
```

## 📂 Cấu trúc dự án

```
practice/
├── app.py                  # Entry point - Streamlit UI
├── .env                    # API credentials (GIT IGNORED)
├── .env.example            # Template for credentials
├── .gitignore              # Git ignore config
├── requirements.txt        # Python dependencies
├── core/
│   └── analyzer.py         # CCA calculation logic
├── data/
│   └── api_client.py       # SSI API client
├── ui/
│   └── charts.py           # Plotly visualization
└── tests/
    ├── test_analyzer.py
    ├── test_api.py
    └── test_charts.py
```

## 🎯 Hướng dẫn sử dụng

### 1. Sidebar - Cấu hình phân tích

#### 📊 Trading Pair
- **Symbol 1**: Mã cổ phiếu thứ nhất (VD: MBB)
- **Symbol 2**: Mã cổ phiếu thứ hai (VD: TCB)

#### 📅 Date Range
- **Start Date**: Ngày bắt đầu lấy dữ liệu
- **End Date**: Ngày kết thúc lấy dữ liệu
- ⚠️ **Lưu ý quan trọng:** SSI API chỉ cung cấp dữ liệu gần đây (thường 30-90 ngày)
- **Khuyến nghị:** 30-60 ngày gần nhất để có đủ dữ liệu

#### ⚙️ Analysis Parameters
- **Rolling Window**: Cửa sổ tính toán Bollinger Bands và Z-score (10-60 ngày)
- Mặc định: 20 ngày

### 2. Main Dashboard

#### Metrics (Chỉ số hiện tại)
- **Z-Score**: Độ lệch chuẩn hóa của spread
- **Spread**: Giá trị spread hiện tại
- **Tín hiệu**: Gợi ý giao dịch dựa trên Z-score

#### Charts (Biểu đồ phân tích)
1. **Raw Price**: So sánh giá gốc 2 cổ phiếu
2. **Normalized Price**: Giá chuẩn hóa base 1.0
3. **Spread with Bollinger Bands**: Spread với các dải Bollinger
4. **Z-Score**: Z-score với ngưỡng giao dịch ±2σ
5. **CCA Weights**: Trọng số tối ưu cho từng cổ phiếu

### 3. Trading Signals

| Z-Score | Tín hiệu | Giải thích |
|---------|----------|------------|
| > +2σ   | 🔴 Bán 1 Mua 2 | Spread quá cao, kỳ vọng giảm |
| < -2σ   | 🟢 Mua 1 Bán 2 | Spread quá thấp, kỳ vọng tăng |
| ∈ [-2σ, +2σ] | 🟡 Chờ đợi | Spread trong vùng an toàn |

## 🧪 Chạy tests

```bash
# Run all tests
python -m unittest discover tests -v

# Run specific test file
python -m unittest tests.test_charts -v
python -m unittest tests.test_analyzer -v
python -m unittest tests.test_api -v
```

## 🐛 Troubleshooting

Gặp vấn đề? Xem hướng dẫn chi tiết tại: **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)**

### Quick Fixes

#### Lỗi: "Missing required columns in API response"

```bash
# Chạy debug script để xem API response structure
python debug_api.py
```

Script sẽ hiển thị chính xác column names từ API.

#### Lỗi: "Thiếu cấu hình xác thực"

**Nguyên nhân**: File `.env` không tồn tại hoặc thiếu thông tin

**Giải pháp**:
```bash
# 1. Kiểm tra file .env có tồn tại không
ls .env  # Linux/Mac
dir .env # Windows

# 2. Nếu chưa có, copy từ template
cp .env.example .env

# 3. Mở .env và điền thông tin
code .env
```

### Lỗi: "Import dotenv could not be resolved"

**Nguyên nhân**: Chưa cài python-dotenv

**Giải pháp**:
```bash
pip install python-dotenv
# hoặc
pip install -r requirements.txt
```

### Lỗi: "Lỗi xác thực API"

**Nguyên nhân**: Consumer ID/Secret không đúng

**Giải pháp**:
- Kiểm tra lại thông tin trong file `.env`
- Đảm bảo không có khoảng trắng thừa
- Kiểm tra credentials trên dashboard SSI

### Lỗi: "API calls quota exceeded! maximum admitted 1 per 1s" (429)

**Nguyên nhân**: SSI API giới hạn 1 request/giây

**Giải pháp**: ✅ **Đã tự động xử lý**
- API client tự động rate limiting (chờ 1 giây giữa các request)
- Bạn không cần làm gì, hệ thống xử lý tự động
- Log sẽ hiển thị: `[DEBUG] Rate limiting: sleeping for 0.XXs`

### Lỗi: "Không có dữ liệu"

**Nguyên nhân**: Mã cổ phiếu không hợp lệ hoặc ngày nghỉ lễ

**Giải pháp**:
- Đảm bảo mã cổ phiếu đúng (VD: MBB, TCB, VNM)
- Chọn khoảng thời gian có dữ liệu giao dịch
- Tránh ngày lễ, cuối tuần

## � Tính năng kỹ thuật

### Rate Limiting (Tự động)
- ✅ Tự động chờ 1 giây giữa các API request
- ✅ Tránh lỗi 429 "API calls quota exceeded"
- ✅ Không yêu cầu cấu hình thêm

### Error Handling
- ✅ Comprehensive logging với [DEBUG], [INFO], [ERROR]
- ✅ Retry logic cho 401 Unauthorized
- ✅ Graceful degradation cho network errors

### Data Validation
- ✅ Flexible column mapping (case-insensitive)
- ✅ Multi-format date parsing (DD/MM/YYYY, YYYY-MM-DD)
- ✅ Validates JSON body status field

## �📚 Tech Stack

- **Frontend**: Streamlit 1.54.0
- **Visualization**: Plotly 5.17.0
- **Data Processing**: Pandas 2.0.0, NumPy 1.24.0
- **Machine Learning**: scikit-learn 1.3.0 (CCA)
- **API Client**: Requests 2.31.0
- **Environment**: python-dotenv 1.0.0

## 📄 License

Educational and research purposes only. Not financial advice.

## 👥 Contributors

CCA Pairs Trading Team - March 2026

---

**⚠️ Disclaimer**: This tool is for educational and research purposes only. It does not constitute financial advice. Always do your own research before making investment decisions.
