"""
Pairs Trading CCA - Streamlit Application

This is the main entry point for the Canonical Correlation Analysis (CCA) 
based Pairs Trading web application. It serves as the Presentation Layer,
orchestrating data flow between UI, business logic, and data layers.

Architecture:
- Presentation Layer (This file): User interface and workflow orchestration
- Business Logic Layer (core.analyzer): CCA calculations and Z-score computation
- Data Layer (data.api_client): API communication with SSI market data
- Visualization Layer (ui.charts): Plotly chart generation

Author: CCA Pairs Trading Team
Date: March 2026
"""

import os
import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Internal modules
from data.api_client import SSIAPIClient
from core.analyzer import CorrelationAnalyzer
from ui.charts import (
    plot_raw_price,
    plot_normalized_price,
    plot_spread_bollinger,
    plot_zscore,
    plot_cca_weights
)

# Load environment variables from .env file (if exists)
# This allows secure credential management without hardcoding
load_dotenv()


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Pairs Trading CCA",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================================
# FAIL-FAST VALIDATION: CHECK ENVIRONMENT VARIABLES
# ============================================================================

# Read credentials from environment variables (STRICT MODE)
SSI_CONSUMER_ID = os.getenv("SSI_CONSUMER_ID")
SSI_CONSUMER_SECRET = os.getenv("SSI_CONSUMER_SECRET")
SSI_BASE_URL = os.getenv("SSI_BASE_URL", "https://fc-data.ssi.com.vn/")

# Fail-fast: Stop app if credentials are missing
if not SSI_CONSUMER_ID or not SSI_CONSUMER_SECRET:
    st.error(
        "❌ **Thiếu cấu hình xác thực!**\n\n"
        "Ứng dụng không thể hoạt động mà không có API credentials. "
        "Vui lòng kiểm tra file `.env` trong thư mục gốc dự án.\n\n"
        "**Hướng dẫn:**\n"
        "1. Copy file `.env.example` thành `.env`\n"
        "2. Điền thông tin `SSI_CONSUMER_ID` và `SSI_CONSUMER_SECRET`\n"
        "3. Khởi động lại ứng dụng"
    )
    st.info(
        "📝 **Cấu trúc file .env mẫu:**\n\n"
        "```\n"
        "SSI_CONSUMER_ID=your_consumer_id_here\n"
        "SSI_CONSUMER_SECRET=your_consumer_secret_here\n"
        "SSI_BASE_URL=https://fc-data.ssi.com.vn/\n"
        "```"
    )
    st.stop()


# ============================================================================
# CACHED DATA LOADING FUNCTION
# ============================================================================

@st.cache_data(ttl=3600, show_spinner=False)
def load_data(
    sym1: str,
    sym2: str,
    start: str,
    end: str,
    cid: str,
    csecret: str,
    url: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load OHLCV data for two symbols from SSI API with caching.
    
    This function fetches historical price data for a pair of symbols
    and caches the results for 1 hour to avoid redundant API calls.
    
    Security Note:
    - Credentials (cid, csecret) are hashed by Streamlit before caching
    - show_spinner=False to use custom spinner in main flow
    
    Args:
        sym1: First symbol code (e.g., 'MBB')
        sym2: Second symbol code (e.g., 'TCB')
        start: Start date in format 'dd/MM/yyyy'
        end: End date in format 'dd/MM/yyyy'
        cid: SSI API consumer ID
        csecret: SSI API consumer secret
        url: SSI API base URL
    
    Returns:
        tuple: (df1, df2) where each DataFrame contains OHLCV data
               with columns ['O', 'H', 'L', 'C', 'V'] and datetime index
    
    Raises:
        Exception: If API authentication fails or data fetch fails
    """
    client = SSIAPIClient(
        consumer_id=cid,
        consumer_secret=csecret,
        base_url=url
    )
    
    df1 = client.fetch_ohlcv(symbol=sym1, start_date=start, end_date=end)
    df2 = client.fetch_ohlcv(symbol=sym2, start_date=start, end_date=end)
    
    return df1, df2


# ============================================================================
# SIDEBAR - ANALYSIS CONFIGURATION (NO CREDENTIALS)
# ============================================================================

st.sidebar.title("⚙️ Cấu hình phân tích")
st.sidebar.markdown("---")

# Trading Pair Selection
st.sidebar.subheader("📊 Trading Pair")
symbol_1 = st.sidebar.text_input(
    "Symbol 1",
    value="MBB",
    help="First symbol in the pair (e.g., MBB)"
).upper()

symbol_2 = st.sidebar.text_input(
    "Symbol 2",
    value="TCB",
    help="Second symbol in the pair (e.g., TCB)"
).upper()

st.sidebar.markdown("---")

# Date Range Selection
st.sidebar.subheader("📅 Date Range")

# Default date range: last 60 days (SSI API has limited historical data)
default_end = datetime.now()
default_start = default_end - timedelta(days=60)

start_date = st.sidebar.date_input(
    "Start Date",
    value=default_start,
    help="Start date for historical data (⚠️ SSI API chỉ có data gần đây, khuyến nghị: 30-60 ngày)",
    max_value=default_end
)

end_date = st.sidebar.date_input(
    "End Date",
    value=default_end,
    help="End date for historical data (tối đa: hôm nay)",
    max_value=default_end
)

# Convert dates to dd/MM/yyyy format for API
start_date_str = start_date.strftime("%d/%m/%Y")
end_date_str = end_date.strftime("%d/%m/%Y")

# Warning about SSI API data availability
days_from_now = (datetime.now().date() - start_date).days
if days_from_now > 90:
    st.sidebar.error(
        f"⚠️ **Cảnh báo:** Date range quá xa!\n\n"
        f"Start date cách đây {days_from_now} ngày. "
        f"SSI API có thể không có dữ liệu lịch sử xa như vậy.\n\n"
        f"**Khuyến nghị:** Chọn date gần đây hơn (< 90 ngày)."
    )
elif days_from_now > 60:
    st.sidebar.warning(
        f"⚠️ Date range khá xa ({days_from_now} ngày). "
        f"Có thể không có đủ dữ liệu từ API."
    )

st.sidebar.markdown("---")

# Analysis Parameters
st.sidebar.subheader("⚙️ Analysis Parameters")
rolling_window = st.sidebar.slider(
    "Rolling Window",
    min_value=5,
    max_value=30,
    value=10,
    step=1,
    help="Window size for rolling statistics (Bollinger Bands, Z-score). SSI API có giới hạn data nên khuyến nghị ≤ 10"
)

# Show data requirement warning
date_range_days = (end_date - start_date).days
min_required_data = rolling_window + 1
estimated_trading_days = int(date_range_days * 0.7)  # Rough estimate: 70% of calendar days

if estimated_trading_days < min_required_data:
    st.sidebar.warning(
        f"⚠️ **Cảnh báo dữ liệu**\n\n"
        f"Khoảng thời gian hiện tại (~{estimated_trading_days} ngày giao dịch) "
        f"có thể không đủ cho rolling window {rolling_window}.\n\n"
        f"**Cần:** {min_required_data}+ ngày giao dịch\n\n"
        f"💡 Tăng khoảng thời gian hoặc giảm rolling window."
    )
elif estimated_trading_days < min_required_data + 10:
    st.sidebar.info(
        f"ℹ️ Dữ liệu vừa đủ (~{estimated_trading_days} ngày). "
        f"Khuyến nghị thêm dữ liệu để kết quả chính xác hơn."
    )
else:
    st.sidebar.success(
        f"✅ Khoảng dữ liệu tốt (~{estimated_trading_days} ngày)"
    )

st.sidebar.markdown("---")

# Analysis Trigger Button
analyze_button = st.sidebar.button(
    "🚀 Phân tích",
    type="primary",
    use_container_width=True
)


# ============================================================================
# MAIN AREA - DASHBOARD
# ============================================================================

st.title("📈 Pairs Trading - Canonical Correlation Analysis")
st.markdown(
    """
    **Hệ thống phân tích giao dịch cặp cổ phiếu sử dụng CCA (Canonical Correlation Analysis)**
    
    Ứng dụng này giúp xác định cơ hội giao dịch mean-reversion giữa hai cổ phiếu có tương quan cao.
    """
)

st.markdown("---")

# Show instructions if button not clicked yet
if not analyze_button:
    st.info(
        """
        👈 **Hướng dẫn sử dụng:**
        
        1. Nhập **API Credentials** của bạn từ SSI
        2. Chọn **hai mã cổ phiếu** muốn phân tích (ví dụ: MBB và TCB)
        3. Chọn **khoảng thời gian** phân tích
        4. Điều chỉnh **Rolling Window** nếu cần
        5. Nhấn nút **"Phân tích"** để bắt đầu
        
        ---
        
        **Giải thích tín hiệu:**
        - 🔴 **Bán 1 Mua 2**: Z-score > +2σ → Spread quá cao, kỳ vọng hồi về trung bình
        - 🟢 **Mua 1 Bán 2**: Z-score < -2σ → Spread quá thấp, kỳ vọng hồi về trung bình
        - 🟡 **Chờ đợi**: |Z-score| < 2σ → Chưa có tín hiệu rõ ràng
        """
    )
    st.stop()


# ============================================================================
# MAIN PROCESSING LOGIC WITH ERROR HANDLING
# ============================================================================

try:
    # Validation checks
    if symbol_1 == symbol_2:
        st.error("❌ Hai mã cổ phiếu phải khác nhau")
        st.stop()
    
    if start_date >= end_date:
        st.error("❌ Ngày bắt đầu phải trước ngày kết thúc")
        st.stop()
    
    # ------------------------------------------------------------------------
    # STEP 1: Load Data from API
    # ------------------------------------------------------------------------
    with st.spinner(f"📡 Đang tải dữ liệu cho {symbol_1} và {symbol_2}..."):
        df1, df2 = load_data(
            sym1=symbol_1,
            sym2=symbol_2,
            start=start_date_str,
            end=end_date_str,
            cid=SSI_CONSUMER_ID,
            csecret=SSI_CONSUMER_SECRET,
            url=SSI_BASE_URL
        )
    
    # Check if data is valid
    if df1.empty or df2.empty:
        st.error(
            f"❌ Không lấy được dữ liệu cho {symbol_1} hoặc {symbol_2}. "
            f"Vui lòng kiểm tra mã cổ phiếu và khoảng thời gian."
        )
        st.stop()
    
    st.success(f"✅ Đã tải {len(df1)} ngày giao dịch cho {symbol_1} và {len(df2)} ngày cho {symbol_2}")
    
    # ------------------------------------------------------------------------
    # DATA VALIDATION: Check if data is sufficient for rolling window
    # ------------------------------------------------------------------------
    # After log return calculation, we lose 1 row
    # So we need at least (rolling_window + 1) raw data points
    min_required_days = rolling_window + 1
    actual_days = min(len(df1), len(df2))
    
    if actual_days < min_required_days:
        st.error(
            f"❌ **Dữ liệu không đủ cho cửa sổ rolling {rolling_window} ngày**\n\n"
            f"📊 **Hiện tại:**\n"
            f"- Dữ liệu thực tế từ API: {actual_days} ngày giao dịch\n"
            f"- Sau tính log return: {actual_days - 1} dòng\n\n"
            f"✅ **Cần thiết:**\n"
            f"- Tối thiểu: {min_required_days} ngày giao dịch\n"
            f"- Hoặc giảm Rolling Window xuống ≤ {actual_days - 1} ngày\n\n"
            f"💡 **Giải pháp:**\n"
            f"1. **Giảm Rolling Window** trong sidebar xuống ≤ {actual_days - 1}\n"
            f"2. **Chọn date range gần đây hơn** (SSI API chỉ có dữ liệu vài tuần gần nhất)\n"
            f"3. **Chọn khoảng thời gian ngắn hơn** (VD: 30-60 ngày)"
        )
        st.warning(
            f"⚠️ **Lưu ý về SSI API:**\n\n"
            f"SSI API có giới hạn dữ liệu lịch sử. Mặc dù bạn chọn date range "
            f"từ {start_date_str} đến {end_date_str}, API chỉ trả về {actual_days} ngày giao dịch.\n\n"
            f"**Khuyến nghị:** Chọn date range trong vòng 30-60 ngày gần đây để có đủ dữ liệu."
        )
        st.stop()
    
    # ------------------------------------------------------------------------
    # STEP 2: Calculate CCA Spread and Z-Score
    # ------------------------------------------------------------------------
    with st.spinner("🧮 Đang tính toán CCA spread và Z-score..."):
        analyzer = CorrelationAnalyzer()
        zscore, spread, w_x, w_y = analyzer.calculate_cca_zscore(
            df_x=df1,
            df_y=df2,
            window=rolling_window
        )
    
    # Check if calculation produced valid results
    if zscore.empty or spread.empty:
        st.error("❌ Không thể tính toán spread. Dữ liệu có thể không đủ hoặc không phù hợp.")
        st.stop()
    
    st.success("✅ Hoàn thành tính toán CCA")
    
    # ------------------------------------------------------------------------
    # STEP 3: Display Key Metrics
    # ------------------------------------------------------------------------
    st.markdown("### 📊 Chỉ số hiện tại")
    
    # Get latest values
    current_zscore = zscore.iloc[-1]
    current_spread = spread.iloc[-1]
    
    # Determine trading signal and colors
    if current_zscore > 2.0:
        signal = "Bán 1 Mua 2"
        signal_color = "🔴"
        signal_delta = f"+{current_zscore:.2f}σ"
        zscore_delta_color = "inverse"  # Red for extreme positive
    elif current_zscore < -2.0:
        signal = "Mua 1 Bán 2"
        signal_color = "🟢"
        signal_delta = f"{current_zscore:.2f}σ"
        zscore_delta_color = "inverse"  # Red for extreme negative
    else:
        signal = "Chờ đợi"
        signal_color = "🟡"
        signal_delta = f"{current_zscore:.2f}σ"
        zscore_delta_color = "off"  # Neutral when within range
    
    # Display metrics in 3 columns
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="Z-Score",
            value=f"{current_zscore:.2f}",
            delta=f"{abs(current_zscore):.2f}σ từ trung bình",
            delta_color=zscore_delta_color,
            help="Độ lệch chuẩn hóa của spread. |Z| > 2 cho tín hiệu giao dịch mạnh"
        )
    
    with col2:
        st.metric(
            label="Spread",
            value=f"{current_spread:.6f}",
            help="Giá trị spread hiện tại (linear combination của log returns)"
        )
    
    with col3:
        st.metric(
            label=f"{signal_color} Tín hiệu",
            value=signal,
            delta=signal_delta,
            delta_color="off",  # Don't show arrow, just display info
            help="Tín hiệu dựa trên Z-score: >+2σ Bán 1 Mua 2, <-2σ Mua 1 Bán 2"
        )
    
    st.markdown("---")
    
    # ------------------------------------------------------------------------
    # STEP 4: Display Charts
    # ------------------------------------------------------------------------
    st.markdown("### 📈 Biểu đồ phân tích")
    
    # Chart 1: Raw Price Comparison (Full Width)
    st.markdown(f"#### 1. So sánh giá gốc: {symbol_1} vs {symbol_2}")
    with st.spinner("Đang vẽ biểu đồ giá gốc..."):
        fig_raw = plot_raw_price(df1, df2, symbol_1, symbol_2)
        st.plotly_chart(fig_raw, use_container_width=True)
    
    # Chart 2: Normalized Price (Full Width)
    st.markdown(f"#### 2. Giá chuẩn hóa (Base 1.0)")
    with st.spinner("Đang vẽ biểu đồ giá chuẩn hóa..."):
        fig_norm = plot_normalized_price(df1, df2, symbol_1, symbol_2)
        st.plotly_chart(fig_norm, use_container_width=True)
    
    # Charts 3 & 4: Spread and Z-Score (Side by Side)
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.markdown(f"#### 3. Spread với Bollinger Bands")
        with st.spinner("Đang vẽ biểu đồ spread..."):
            fig_spread = plot_spread_bollinger(spread, window=rolling_window)
            st.plotly_chart(fig_spread, use_container_width=True)
    
    with col_right:
        st.markdown(f"#### 4. Z-Score với ngưỡng giao dịch")
        with st.spinner("Đang vẽ biểu đồ Z-score..."):
            fig_zscore = plot_zscore(zscore, threshold=2.0)
            st.plotly_chart(fig_zscore, use_container_width=True)
    
    # Chart 5: CCA Weights (Full Width)
    st.markdown("#### 5. Trọng số CCA (Canonical Weights)")
    with st.spinner("Đang vẽ biểu đồ trọng số CCA..."):
        fig_weights = plot_cca_weights(w_x, w_y, symbol_1, symbol_2, feature_names=['O', 'H', 'L', 'C', 'V'])
        st.plotly_chart(fig_weights, use_container_width=True)
    
    st.markdown("---")
    
    # ------------------------------------------------------------------------
    # STEP 5: Additional Information
    # ------------------------------------------------------------------------
    with st.expander("📋 Thông tin chi tiết"):
        st.markdown(f"""
        **Thông số phân tích:**
        - Cặp giao dịch: **{symbol_1}** - **{symbol_2}**
        - Khoảng thời gian: **{start_date_str}** đến **{end_date_str}**
        - Số ngày giao dịch: **{len(df1)}** ngày ({symbol_1}), **{len(df2)}** ngày ({symbol_2})
        - Rolling Window: **{rolling_window}** ngày
        - CCA Weight {symbol_1}: **{w_x[0]:.6f}**
        - CCA Weight {symbol_2}: **{w_y[0]:.6f}**
        
        **Giải thích:**
        - **Z-Score** đo lường độ lệch của spread so với trung bình, tính theo đơn vị độ lệch chuẩn (σ)
        - **Spread** là tổ hợp tuyến tính của log returns, được tối ưu hóa bởi CCA để tối đa hóa tương quan
        - **Bollinger Bands** hiển thị biên độ ±2σ xung quanh trung bình động
        - **Tín hiệu giao dịch** dựa trên giả định mean-reversion: spread sẽ quay về trung bình
        """)
    
    # Show raw data option
    with st.expander("🔍 Xem dữ liệu gốc"):
        tab1, tab2, tab3, tab4 = st.tabs([
            f"📊 {symbol_1} OHLCV",
            f"📊 {symbol_2} OHLCV",
            "📈 Spread",
            "📉 Z-Score"
        ])
        
        with tab1:
            st.dataframe(df1.tail(50), use_container_width=True)
        
        with tab2:
            st.dataframe(df2.tail(50), use_container_width=True)
        
        with tab3:
            st.dataframe(spread.tail(50).to_frame(name='Spread'), use_container_width=True)
        
        with tab4:
            st.dataframe(zscore.tail(50).to_frame(name='Z-Score'), use_container_width=True)

except Exception as e:
    # Clean error message for end users (no technical traceback)
    error_message = str(e)
    
    # Provide user-friendly error messages based on error type
    if "authentication" in error_message.lower() or "401" in error_message:
        st.error(
            "❌ **Lỗi xác thực API**\n\n"
            "Consumer ID hoặc Consumer Secret không chính xác. "
            "Vui lòng kiểm tra lại thông tin đăng nhập của bạn."
        )
    elif "connection" in error_message.lower() or "timeout" in error_message.lower():
        st.error(
            "❌ **Lỗi kết nối**\n\n"
            "Không thể kết nối tới API SSI. Vui lòng kiểm tra kết nối internet "
            "và thử lại sau."
        )
    elif "không lấy được dữ liệu" in error_message.lower() or "empty" in error_message.lower():
        st.error(
            "❌ **Không có dữ liệu**\n\n"
            "Không tìm thấy dữ liệu cho mã cổ phiếu hoặc khoảng thời gian đã chọn. "
            "Vui lòng kiểm tra lại mã cổ phiếu và ngày tháng."
        )
    else:
        st.error(f"❌ **Đã xảy ra lỗi**\n\n{error_message}")
    
    st.info(
        """
        **💡 Gợi ý xử lý:**
        - ✓ Kiểm tra lại Consumer ID và Consumer Secret
        - ✓ Đảm bảo mã cổ phiếu hợp lệ (ví dụ: MBB, TCB, VNM)
        - ✓ Chọn khoảng thời gian có dữ liệu giao dịch (tránh ngày lễ, cuối tuần)
        - ✓ Thử giảm khoảng thời gian xuống 3-6 tháng
        - ✓ Kiểm tra kết nối internet và trạng thái API SSI
        """
    )


# ============================================================================
# FOOTER
# ============================================================================

st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; font-size: 0.9em;'>
        <p>📈 Pairs Trading CCA System | Built with Streamlit & Plotly</p>
        <p>⚠️ Disclaimer: This tool is for educational and research purposes only. 
        Not financial advice.</p>
    </div>
    """,
    unsafe_allow_html=True
)
