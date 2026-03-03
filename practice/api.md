# YÊU CẦU NÂNG CAO: XỬ LÝ DỮ LIỆU LỚN VÀ PHÂN TRANG (PAGINATION)
Trong trường hợp người dùng yêu cầu khoảng thời gian dài (ví dụ 10 năm), dữ liệu trả về sẽ vượt quá giới hạn an toàn của `pageSize` (khuyến nghị pageSize=1000). Hàm `fetch_ohlcv` bắt buộc phải tự động xử lý phân trang.

1. Khởi tạo mảng trống `all_data = []` và biến `page_index = 1`.
2. Sử dụng vòng lặp `while True:` để gọi API liên tục.
3. Điều kiện dừng vòng lặp (Break conditions):
   - Mảng `data` trong response trả về rỗng.
   - Hoặc số lượng phần tử trong `data` nhỏ hơn `pageSize` (tức là đã đến trang cuối cùng).
4. Sau mỗi lần gọi thành công, dùng `all_data.extend(response['data'])` để gom dữ liệu và tăng `page_index += 1`.
5. Tối ưu Network: Import thư viện `time` và thêm `time.sleep(0.2)` giữa các vòng lặp để tránh bị server block do gọi API quá nhanh (Rate Limit).
6. Sau khi vòng lặp kết thúc, mới chuyển đổi mảng `all_data` thành Pandas DataFrame.