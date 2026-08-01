# 🏍️ ĐỒ ÁN TỐT NGHIỆP DATA SCIENCE: DỰ ĐOÁN GIÁ VÀ PHÁT HIỆN BẤT THƯỜNG XE MÁY TRÊN CHỢ TỐT

**Môn học:** Khoa học Dữ liệu & Học máy (Data Science & Machine Learning)  
**Thực hiện bởi:** Phan Phúc Lộc - Nguyễn Nhật Trường  

---

## 🎯 1. MỤC TIÊU ĐỒ ÁN

Dự án áp dụng quy trình Khoa học Dữ liệu (Data Science Process) để giải quyết 2 bài toán kinh doanh thực tiễn trên nền tảng thương mại điện tử Chợ Tốt:
1. **Dự đoán giá (Price Prediction):** Xây dựng hệ thống AI định giá xe máy cũ sát với giá trị thực tế của thị trường dựa trên các đặc trưng vật lý và thông số hao mòn (Tuổi đời, ODO, Hãng, Dòng xe, Phụ kiện...).
2. **Phát hiện bất thường (Anomaly Detection):** Xây dựng cơ chế chấm điểm rủi ro (Ensemble Anomaly Score) để tự động nhận diện tin đăng ảo, lừa đảo cọc hoặc hét giá quá cao.

---

## 🔄 2. QUY TRÌNH THỰC HIỆN

Dự án được triển khai bài bản và hợp nhất việc tính toán trên toàn bộ tập dữ liệu (Unified Pipeline) thay vì chia nhỏ phân khúc, bao gồm các bước:
1. **Business Understanding:** Phân tích bài toán, xác định các nguyên nhân gây nhiễu giá trên thị trường xe cũ.
2. **EDA & Data Prep:** Khám phá dữ liệu, nội suy các giá trị thiếu, trích xuất đặc trưng từ văn bản tự do (NLP Flags: xe zin, HQCN, ABS...) và loại bỏ nhiễu (Deep Cleaning / IQR Filter).
3. **Feature Engineering:** Mã hóa các biến phân loại và chuẩn hóa toàn bộ dữ liệu để đưa vào không gian tính toán chung (Unified Model), giúp mô hình tổng quát hóa tốt hơn.
4. **Regression Models (Bài toán 1):** Thực nghiệm và so sánh hiệu năng 5 thuật toán Học máy (Random Forest, XGBoost, Ridge, Linear, SVR) để tìm ra mô hình dự đoán giá chính xác nhất.
5. **Anomaly Detection (Bài toán 2):** Xây dựng 4 cờ tín hiệu phát hiện bất thường kết hợp giữa Thống kê mô tả (Robust Z-Score, Giới hạn phân vị) và Học máy không giám sát (Isolation Forest).
6. **Deployment (XAI):** Đóng gói mô hình và triển khai lên Giao diện Web (Streamlit) tích hợp công nghệ AI giải thích được (Explainable AI - XAI) minh bạch hóa điểm số cho Quản trị viên.

---

## 📂 3. CẤU TRÚC THƯ MỤC (DIRECTORY STRUCTURE)

Dự án được tổ chức gọn gàng trong thư mục gốc `DL07_k314_27_phanphucloc_nguyennhattruong/` với các thành phần như sau:

```text
DL07_k314_27_phanphucloc_nguyennhattruong/
│
├── catboost_info/   # Chứa các tệp log và thông tin khởi tạo/huấn luyện tạm thời.
├── Data/            # Chứa tập dữ liệu thô ban đầu và dữ liệu đã qua tiền xử lý (data_ready_v3.csv).
├── Models/          # Thư mục xuất và lưu trữ các tệp mô hình đã huấn luyện (.pkl).
├── Notebooks/       # Chứa file mã nguồn Jupyter Notebook thực thi toàn bộ quy trình.
├── Reports/         # Nơi xuất các biểu đồ trực quan hóa (EDA, WordCloud) và báo cáo.
└── README.md        # Tài liệu hướng dẫn (File này).

Chào bạn, việc đưa trực tiếp các con số đánh giá hiệu năng (Metrics) vào file `README.md` là một bước đi rất đúng đắn. Nó giúp hội đồng chấm thi hoặc nhà tuyển dụng nhìn vào là thấy ngay thành quả tính toán và chiều sâu kỹ thuật của đồ án.

Dựa trên bảng kết quả chạy thực tế mà bạn đã cung cấp trước đó, tôi đã thiết kế lại và bổ sung **Mục số 5 (Đánh giá & Lựa chọn thuật toán)** bằng một bảng so sánh trực quan kèm theo những lời nhận định mang tính chuyên gia.

Bạn hãy sao chép toàn bộ nội dung dưới đây và lưu đè lên file `README.md` trên GitHub nhé:

---

```markdown
# 🏍️ ĐỒ ÁN TỐT NGHIỆP DATA SCIENCE: DỰ ĐOÁN GIÁ VÀ PHÁT HIỆN BẤT THƯỜNG XE MÁY TRÊN CHỢ TỐT

**Môn học:** Khoa học Dữ liệu & Học máy (Data Science & Machine Learning)  
**Thực hiện bởi:** Phan Phúc Lộc - Nguyễn Nhật Trường  

---

## 🎯 1. MỤC TIÊU ĐỒ ÁN

Dự án áp dụng quy trình Khoa học Dữ liệu (Data Science Process) để giải quyết 2 bài toán kinh doanh thực tiễn trên nền tảng thương mại điện tử Chợ Tốt:
1. **Dự đoán giá (Price Prediction):** Xây dựng hệ thống AI định giá xe máy cũ sát với giá trị thực tế của thị trường dựa trên các đặc trưng vật lý và thông số hao mòn (Tuổi đời, ODO, Hãng, Dòng xe, Phụ kiện...).
2. **Phát hiện bất thường (Anomaly Detection):** Xây dựng cơ chế chấm điểm rủi ro (Ensemble Anomaly Score) để tự động nhận diện tin đăng ảo, lừa đảo cọc hoặc hét giá quá cao.

---

## 🔄 2. QUY TRÌNH THỰC HIỆN

Dự án được triển khai bài bản và hợp nhất việc tính toán trên toàn bộ tập dữ liệu (Unified Pipeline) thay vì chia nhỏ phân khúc, bao gồm các bước:
1. **Business Understanding:** Phân tích bài toán, xác định các nguyên nhân gây nhiễu giá trên thị trường xe cũ.
2. **EDA & Data Prep:** Khám phá dữ liệu, nội suy các giá trị thiếu, trích xuất đặc trưng từ văn bản tự do (NLP Flags: xe zin, HQCN, ABS...) và loại bỏ nhiễu (Deep Cleaning / IQR Filter).
3. **Feature Engineering:** Mã hóa các biến phân loại và chuẩn hóa toàn bộ dữ liệu để đưa vào không gian tính toán chung (Unified Model), giúp mô hình tổng quát hóa tốt hơn.
4. **Regression Models (Bài toán 1):** Thực nghiệm và so sánh hiệu năng 5 thuật toán Học máy (Random Forest, XGBoost, Ridge, Linear, SVR) để tìm ra mô hình dự đoán giá chính xác nhất.
5. **Anomaly Detection (Bài toán 2):** Xây dựng 4 cờ tín hiệu phát hiện bất thường kết hợp giữa Thống kê mô tả (Robust Z-Score, Giới hạn phân vị) và Học máy không giám sát (Isolation Forest).
6. **Deployment (XAI):** Đóng gói mô hình và triển khai lên Giao diện Web (Streamlit) tích hợp công nghệ AI giải thích được (Explainable AI - XAI) minh bạch hóa điểm số cho Quản trị viên.

---

## 📂 3. CẤU TRÚC THƯ MỤC (DIRECTORY STRUCTURE)

Dự án được tổ chức gọn gàng trong thư mục gốc `DL07_k314_27_phanphucloc_nguyennhattruong/` với các thành phần như sau:

```text
DL07_k314_27_phanphucloc_nguyennhattruong/
│
├── catboost_info/   # Chứa các tệp log và thông tin khởi tạo/huấn luyện tạm thời.
├── Data/            # Chứa tập dữ liệu thô ban đầu và dữ liệu đã qua tiền xử lý (data_ready_v3.csv).
├── Models/          # Thư mục xuất và lưu trữ các tệp mô hình đã huấn luyện (.pkl).
├── Notebooks/       # Chứa file mã nguồn Jupyter Notebook thực thi toàn bộ quy trình.
├── Reports/         # Nơi xuất các biểu đồ trực quan hóa (EDA, WordCloud) và báo cáo.
└── README.md        # Tài liệu hướng dẫn (File này).

```

---

## 🚀 4. HƯỚNG DẪN ĐỌC CODE & THỨ TỰ THỰC THI

Toàn bộ quy trình tính toán từ xử lý dữ liệu thô đến việc huấn luyện mô hình được **hợp nhất vào một file Notebook duy nhất** nhằm đảm bảo tính liền mạch của luồng dữ liệu (Data Pipeline).

**📁 File mã nguồn chính:** `Notebooks/DL07_k314_27_Price_Prediction_AnomalyDetection.ipynb`

Vui lòng mở file Notebook trên Google Colab hoặc Jupyter và thực thi tuần tự từ trên xuống dưới theo các phân vùng chính:

* **Phần 1 - EDA & Data Preprocessing:** Khởi tạo môi trường, xử lý Missing values, trích xuất NLP (Biển VIP, Smartkey, Nhập Ý...), và làm sạch dữ liệu chuyên sâu (Lọc bỏ ODO ảo).
* **Phần 2 - Xây dựng Mô hình Dự đoán Giá (Unified Model):** Chuẩn bị không gian vector đặc trưng, chia tập Train/Test, huấn luyện 5 thuật toán Machine Learning đồng thời và tìm ra Quán quân định giá. Các file model sẽ tự động xuất vào thư mục `Models/`.
* **Phần 3 - Cơ chế Phát hiện Bất thường:** Tính toán sai số (Residuals) từ mô hình định giá, xây dựng bộ cảnh báo đa lớp bằng Isolation Forest và Z-Robust.
* **Phần 4 - Đánh giá & Báo cáo:** Tổng hợp chỉ số hiệu năng (RMSE, R-squared), so sánh thực tế vs dự đoán.

*(Lưu ý: Bạn có thể chọn `Kernel` -> `Restart & Run All` để hệ thống tự động chạy toàn bộ quy trình từ A đến Z).*

---

## ⚖️ 5. BẢNG ĐÁNH GIÁ & LỰA CHỌN MÔ HÌNH

Hệ thống đã tiến hành thực nghiệm song song trên 5 thuật toán Học máy. Bảng kết quả dưới đây phản ánh hiệu năng đo lường được trên tập kiểm thử (Test set) sau khi dữ liệu đã được áp dụng bộ lọc nhiễu chuyên sâu (Deep Cleaning):

| Thuật toán | MAE (VNĐ) | RMSE (VNĐ) | R-squared | Thời gian huấn luyện |
| --- | --- | --- | --- | --- |
| **Random Forest** | 6,040,502 | **12,482,572** | **0.76** | 5.05s |
| **XGBoost** | **5,869,234** | 12,958,564 | 0.75 | 0.65s |
| **Ridge Regression** | 9,118,619 | 16,133,663 | 0.61 | 0.29s |
| **Linear Regression** | 9,198,710 | 16,448,681 | 0.59 | 0.27s |
| **SVR** | 12,444,478 | 24,138,921 | 0.12 | 10.74s |

**💡 Nhận định Chuyên môn:**

* **Lựa chọn Random Forest làm Unified Model:** Mặc dù XGBoost có sai số trung bình tuyệt đối (MAE) thấp hơn đôi chút và tốc độ cực nhanh, nhưng Random Forest lại sở hữu sai số RMSE thấp nhất (12.48 triệu VNĐ) và điểm $R^2$ cao nhất (0.76). Vì RMSE nhạy cảm với các dự đoán lệch chuẩn lớn, việc Random Forest chiến thắng ở chỉ số này chứng tỏ mô hình có độ ổn định (Robustness) xuất sắc, ít bị uốn nắn bởi các điểm dữ liệu dị biệt trên thị trường xe máy cũ.
* **Kiểm duyệt Bất thường (Isolation Forest):** Kế thừa sai số thực tế từ Random Forest, hệ thống ứng dụng Isolation Forest đóng vai trò làm màng lọc AI không giám sát. Thuật toán này kết hợp cùng Khoảng tin cậy thống kê giúp hệ thống dễ dàng cô lập và cảnh báo các tin đăng lừa đảo có cấu trúc dữ liệu dị thường.

---

## ⚙️ 6. KIẾN TRÚC TRIỂN KHAI THỰC TẾ (APP.PY)

Nhóm đã đóng gói lõi thuật toán thành một ứng dụng Web tương tác (Streamlit) để mô phỏng luồng kiểm duyệt thực tế:

1. **Phía Người bán:** Điền thông tin xe -> Hệ thống báo độ lệch chuẩn và gợi ý Vùng giá thanh khoản an toàn.
2. **Phía AI Chấm điểm:** Tin đăng có dấu hiệu vi phạm lập tức bị hệ thống gắn `Anomaly_Score = 100` và treo ở hàng đợi chờ duyệt.
3. **Phía Quản trị viên (Admin View):** Giao diện hiển thị chi tiết nguyên nhân vi phạm dựa trên 4 cờ (S1, S2, S3, S4), giúp nhân viên sàn duyệt tin bằng bằng chứng thuật toán thay vì cảm tính.

```

```