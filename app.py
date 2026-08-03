# ======================================================================================
# ------------------------------------------------------
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import datetime
import time

# ==========================================
# 1. CẤU HÌNH GIAO DIỆN & CSS (BẢN KHÓA CỨNG SIDEBAR & BANNER CHUẨN)
# ==========================================
st.set_page_config(page_title="Sàn Xe Máy AI", page_icon="🏍️", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    /* 1. Ẩn header mặc định */
    header {visibility: hidden;}
    
    /* 2. KHÓA CỨNG SIDEBAR (HARD-LOCK) */
    [data-testid="stSidebarCollapseButton"] { display: none !important; }
    [data-testid="stSidebar"] {
        transform: translateX(0px) !important; 
        visibility: visible !important; 
        position: relative !important;
    }
    
    /* 3. Đẩy khung nội dung xuống để nhường chỗ cho Banner 3 dòng */
    .block-container {padding-top: 165px !important;}
    
    /* 4. Top Banner Cố định (Tăng height lên 145px) */
    .top-banner {
        position: fixed; top: 0; left: 0; width: 100%; height: 145px;
        background: linear-gradient(rgba(0, 0, 0, 0.7), rgba(0, 0, 0, 0.8)), 
                    url('https://images.unsplash.com/photo-1558981403-c5f9899a28bc?q=80&w=2070&auto=format&fit=crop') center/cover;
        color: white; z-index: 99999; text-align: center;
        display: flex; flex-direction: column; justify-content: center; align-items: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.4);
        border-bottom: 4px solid #FFBA00;
    }
    
    /* Định dạng 3 tầng chữ cho Banner */
    .banner-school {
        font-size: 14px; font-weight: 600; color: #D1D5DB; 
        text-transform: uppercase; letter-spacing: 1.5px; margin-bottom: 6px;
    }
    .banner-title { 
        margin: 0; font-size: 34px; font-weight: 900; color: #FFBA00; 
        line-height: 1.1; text-transform: uppercase; 
        text-shadow: 2px 2px 4px rgba(0,0,0,0.6); 
    }
    .banner-subtitle { 
        margin: 8px 0 0 0; font-size: 20px; font-weight: 500; color: #F9FAFB; 
        letter-spacing: 0.5px;
    }
    
    /* 5. Nút bấm Style Chợ Tốt */
    div.stButton > button:first-child {
        background-color: #FFBA00; color: #111111; font-weight: bold; border: none; padding: 10px 24px; border-radius: 8px;
    }
    div.stButton > button:first-child:hover {
        background-color: #e5a700; color: #000000; box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }
    
    /* 6. Sidebar Footer */
    .sidebar-footer-box {
        background-color: #FFFBEB; padding: 20px; border-radius: 10px;
        border: 2px solid #FDE68A; margin-top: 30px; margin-bottom: 20px;
    }
    .footer-title { color: #B45309; font-weight: 900; font-size: 16px; margin-bottom: 5px; text-transform: uppercase; }
    .footer-text { font-weight: bold; font-size: 15px; color: #374151; margin-bottom: 5px; }
    
    /* 7. Headers */
    .main-header { font-size: 2.2rem; font-weight: 800; color: #27272A; margin-bottom: 0px;}
    .sub-header { font-size: 1.1rem; color: #6B7280; margin-bottom: 25px;}

    /* 8. Custom Box cho AI Insight */
    .ai-insight-box {
        background-color: #F8FAFC; border: 2px dashed #94A3B8; border-radius: 10px; padding: 15px; margin-top: 15px;
    }
    </style>
    
    <div class="top-banner">
        <div class="banner-school">TRUNG TÂM TIN HỌC - ĐẠI HỌC KHOA HỌC TỰ NHIÊN TPHCM</div>
        <div class="banner-title">ĐỒ ÁN TỐT NGHIỆP DATA SCIENCE</div>
        <div class="banner-subtitle">Price Prediction & Anomaly Detection</div>
    </div>
    """, unsafe_allow_html=True)

# ==========================================
# 2. HỆ THỐNG LOAD DATA & MÔ HÌNH
# ==========================================
@st.cache_resource
def load_all_systems():
    try:
        t1_model = joblib.load('Models/unified_model.pkl')
        t1_schema = joblib.load('Models/feature_schema.pkl')
        
        data_path = 'Data/data_ready_v3.csv' if os.path.exists('Data/data_ready_v3.csv') else 'data_ready_v3.csv'
        df_raw = pd.read_csv(data_path)
        
        dynamic_mapping = {}
        for brand in df_raw['Thương_hiệu'].dropna().unique():
            brand_df = df_raw[df_raw['Thương_hiệu'] == brand]
            dynamic_mapping[brand] = {}
            for model_name in brand_df['Dòng_xe'].dropna().unique():
                type_val = brand_df[brand_df['Dòng_xe'] == model_name]['Loại_xe'].mode()[0]
                dynamic_mapping[brand][model_name] = type_val

        try:
            iso_model = joblib.load('Models/isolation_forest_model.pkl')
            iso_prep = joblib.load('Models/anomaly_preprocessor.pkl')
            df_history = pd.read_csv('Data/task2_anomaly_data.csv')
            has_task2 = True
        except Exception as e2:
            iso_model, iso_prep, df_history, has_task2 = None, None, None, False
            
        return t1_model, t1_schema, dynamic_mapping, iso_model, iso_prep, df_history, has_task2
        
    except Exception as e:
        st.error(f"🚨 LỖI HỆ THỐNG CHI TIẾT: {e}")
        return None, None, None, None, None, None, False

t1_model, t1_schema, dynamic_mapping, iso_model, iso_prep, df_history, has_task2 = load_all_systems()

DB_FILE = "Data/submitted_posts.csv"
def load_db():
    if os.path.exists(DB_FILE): return pd.read_csv(DB_FILE)
    cols = ["ID", "Thời gian", "Người đăng", "Thương hiệu", "Dòng xe", "Giá mong muốn", "Giá AI dự đoán", 
            "Z_Robust", "S1_Residual_Z", "S2_Price_Limit", "S3_Confidence", "S4_Unsupervised", "Anomaly_Score", 
            "Loại_Bất_Thường", "Phần_Trăm_Lệch", "Giá_Min_Khuyên_Dùng", "Giá_Max_Khuyên_Dùng", "Trạng thái Duyệt"]
    return pd.DataFrame(columns=cols)

def save_to_db(df):
    os.makedirs('Data', exist_ok=True)
    df.to_csv(DB_FILE, index=False, encoding='utf-8-sig')

# ==========================================
# 3. LÕI CHẤM ĐIỂM & TÍNH KHOẢNG GIÁ KHUYÊN DÙNG
# ==========================================
def detect_anomaly_real(input_dict, gia_user, gia_ai):
    if gia_ai <= 0 or np.isnan(gia_ai): 
        return 0.0, 0, 0, 0, 0, 0, "Lỗi AI không thể định giá", 0, 0, 0
        
    residual = gia_user - gia_ai
    lech_percent = round(((gia_user - gia_ai) / gia_ai) * 100, 1)
    
    if has_task2:
        segment = f"{input_dict['Thương_hiệu']}_{input_dict['Dòng_xe']}"
        df_history['Segment'] = df_history['Thương_hiệu'] + "_" + df_history['Dòng_xe']
        segment_data = df_history[df_history['Segment'] == segment]
        if len(segment_data) < 5: segment_data = df_history
            
        median_resid = segment_data['Residuals'].median()
        mad_resid = np.median(np.abs(segment_data['Residuals'] - median_resid))
        mad_resid = mad_resid if mad_resid > 0 else 1e-6
        z_robust = (residual - median_resid) / (1.4826 * mad_resid)
        
        p1, p10, p90, p99 = np.percentile(segment_data['Giá'], [1, 10, 90, 99])
        
        s1 = 1 if abs(z_robust) >= 3 else 0
        s2 = 1 if gia_user < p1 or gia_user > p99 else 0
        s3 = 1 if gia_user < p10 or gia_user > p90 else 0
        
        row_if = pd.DataFrame([{
            'Thương_hiệu': input_dict['Thương_hiệu'], 'Dòng_xe': input_dict['Dòng_xe'], 'Loại_xe': input_dict['Loại_xe'],
            'Tuổi_đời_xe': input_dict['Tuổi_đời_xe'], 'Số_Km_đã_đi': input_dict['Số_Km_đã_đi'],
            'Giá': gia_user, 'Giá_Dự_Đoán': gia_ai, 'Residuals': residual, 'Z_Robust': z_robust
        }])
        try:
            X_unsup = iso_prep.transform(row_if)
            if_pred = iso_model.predict(X_unsup)[0]
            s4 = 1 if if_pred == -1 else 0
        except:
            s4 = 0
            
        score = 100 if (s1 == 1 or s2 == 1 or s4 == 1) else 0
        
        # Tính khoảng giá khuyên dùng dựa trên Giá AI +- 1.5 MAD
        gia_min_rec = max(int(gia_ai - 1.5 * mad_resid), int(gia_ai * 0.85))
        gia_max_rec = int(gia_ai + 1.5 * mad_resid)
    else:
        z_robust = round(lech_percent / 10, 2) 
        s1 = 1 if abs(z_robust) > 3.0 else 0
        s2 = 1 if gia_user > gia_ai * 1.4 or gia_user < gia_ai * 0.6 else 0
        s3, s4 = 0, 0
        score = 100 if (s1 == 1 or s2 == 1) else 0
        gia_min_rec = int(gia_ai * 0.85)
        gia_max_rec = int(gia_ai * 1.15)

    loai_bt = "Bình thường"
    if score == 100:
        loai_bt = "Giá cao bất thường (Khả năng xe độ/sưu tầm)" if z_robust > 0 else "Giá thấp bất thường (Khả năng lừa đảo/cọc ảo)"
        
    return round(z_robust, 2), s1, s2, s3, s4, score, loai_bt, lech_percent, gia_min_rec, gia_max_rec

# ==========================================
# 4. SIDEBAR MENU
# ==========================================
with st.sidebar:
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    st.image("https://cdn-icons-png.flaticon.com/512/2933/2933900.png", width=80)
    st.markdown("### QUẢN TRỊ HỆ THỐNG")
    
    menu = st.radio("ĐIỀU HƯỚNG", ["📈 Câu chuyện Dữ liệu", "📊 Đánh giá thuật toán", "📝 Dự đoán & Đăng tin", "🛡️ Quản trị viên (Admin)"])
    
    st.markdown("""
        <div class="sidebar-footer-box">
            <div class="footer-title">Giảng viên hướng dẫn:</div>
            <div class="footer-text">👨‍🏫 Cô. Khuất Thùy Phương</div>
            <hr style="margin: 12px 0px; border-color: #D1D5DB;">
            <div class="footer-title">Học viên thực hiện:</div>
            <div class="footer-text">👨‍🎓 Phan Phúc Lộc</div>
            <div class="footer-text">👨‍🎓 Nguyễn Nhật Trường</div>
        </div>
        """, unsafe_allow_html=True)

if t1_model is None: 
    st.stop()

# ==========================================
# GIAO DIỆN CÁC TRANG (PAGES)
# ==========================================
elif menu == "📈 Câu chuyện Dữ liệu":
    st.markdown('<p class="main-header">Câu chuyện Dữ liệu & Tầm nhìn Triển khai</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Giải quyết bài toán định giá và kiểm duyệt tin đăng tự động cho Sàn thương mại điện tử</p>', unsafe_allow_html=True)
    
    st.markdown("### 🛒 1. Bối cảnh Nghiệp vụ (Business Objective)")
    st.info("**Chợ Tốt** là nền tảng mua bán trực tuyến hàng đầu tại Việt Nam, kết nối hàng triệu người dùng với đa dạng các hạng mục như nhà cửa, ô tô, xe máy, đồ điện tử... Trong khuôn khổ đồ án này, chúng tôi tập trung khai thác và giải quyết bài toán riêng cho phân khúc **Thị trường Xe máy cũ**.")
    
    st.markdown("### ❓ 2. Bài toán đặt ra (The Problem)")
    st.markdown("> 💡 *Giả sử Chợ Tốt hiện tại chưa có các công cụ tự động để hỗ trợ người dùng định giá bán hợp lý, cũng như chưa có hệ thống tự động quét các tin đăng có mức giá ảo/lừa đảo. Là những Kỹ sư Dữ liệu, chúng ta sẽ giải quyết bài toán này như thế nào?*")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style="background-color: #F0F9FF; padding: 20px; border-radius: 10px; border-top: 5px solid #0EA5E9; height: 100%;">
            <h3 style="color: #0284C7; margin-top: 0;">📈 BÀI TOÁN 1:<br>Dự đoán giá (Price Prediction)</h3>
            <p>Đây là một trong những ứng dụng phổ biến và quan trọng nhất trong lĩnh vực Khoa học Dữ liệu (DS) và Học máy (ML).</p>
            <ul>
                <li><b>Mục tiêu chính:</b> Xây dựng một mô hình Trí tuệ nhân tạo (AI) có khả năng dự đoán giá trị của một sản phẩm, dịch vụ hoặc tài sản (ở đây là xe máy cũ).</li>
                <li><b>Giá trị mang lại:</b> Giúp người bán không bị "hớ" khi định giá quá thấp, và người mua không bị mua đắt. Mô hình đóng vai trò như một chuyên gia thẩm định giá dựa trên ODO, Tuổi đời, Dòng xe, và Tình trạng xe.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
        <div style="background-color: #FEF2F2; padding: 20px; border-radius: 10px; border-top: 5px solid #EF4444; height: 100%;">
            <h3 style="color: #B91C1C; margin-bottom: 5px; margin-top: 0;">🚨 BÀI TOÁN 2:<br>Phát hiện bất thường (Anomaly Detection)</h3>
            <p>Mục tiêu là tìm ra các điểm dữ liệu khác biệt đáng kể so với phần lớn dữ liệu còn lại (gọi là <i>anomalies, outliers</i>).</p>
            <b>Trong ngữ cảnh giá cả thị trường xe máy:</b>
            <ul>
                <li>📉 <b>Giá quá thấp:</b> Một chiếc xe rao bán rẻ bất thường có thể do lỗi nhập liệu, hoặc nguy hiểm hơn là <b>sản phẩm giả, lừa đảo chiếm đoạt tiền cọc</b>.</li>
                <li>📈 <b>Giá quá cao:</b> Có thể do người bán cố tình thổi phồng ngáo giá, hoặc đây là một <b>sản phẩm hiếm, xe độ kiểng, biển số VIP</b> có giá trị sưu tầm cao.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br><hr>", unsafe_allow_html=True)
    st.success("🎯 **KẾT LUẬN:** Đồ án này tích hợp cả 2 bài toán trên vào cùng một hệ thống. Giá dự đoán từ **Bài toán 1** sẽ làm cơ sở tham chiếu (Baseline) vững chắc để **Bài toán 2** quét qua và bắt gọn các tin đăng dị thường, tạo ra một môi trường mua bán minh bạch và an toàn.")

elif menu == "📊 Đánh giá thuật toán":
    st.markdown('<p class="main-header">Đánh giá Hiệu năng & Logic Thuật toán</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Phân tích kết quả dự đoán giá và cơ chế phát hiện tin đăng bất thường</p>', unsafe_allow_html=True)
    
    st.markdown("### 🏆 1. Đánh giá Mô hình Dự đoán Giá (Sau khi Deep Cleaning)")
    st.markdown("Hệ thống đã tiến hành thực nghiệm trên 5 thuật toán Machine Learning để tìm ra mô hình tối ưu nhất. Lưu ý: Bảng kết quả dưới đây phản ánh hiệu năng **sau khi dữ liệu đã được lọc nhiễu chuyên sâu (Loại bỏ ODO ảo và Outliers)**.")
    
    results_data = {
        "Thuật toán": ["Random Forest", "XGBoost", "Ridge Regression", "Linear Regression", "Support Vector Machine (SVR)"],
        "MAE (VNĐ)": ["6,040,502", "5,869,234", "9,118,619", "9,198,710", "12,444,478"],
        "RMSE (VNĐ)": ["12,482,572", "12,958,564", "16,133,663", "16,448,681", "24,138,921"],
        "R-squared": ["0.76", "0.75", "0.61", "0.59", "0.12"],
        "Thời gian (s)": ["5.05", "0.65", "0.29", "0.27", "10.74"]
    }
    df_results = pd.DataFrame(results_data)
    
    st.dataframe(df_results, use_container_width=True, hide_index=True)
    
    st.info("""
    **💡 Nhận định & Lựa chọn Mô hình:**
    * **Random Forest (Tân Quán Quân):** Sau khi loại bỏ các tin đăng rác, Random Forest đã vươn lên dẫn đầu với $R^2 = 0.76$ và sai số RMSE thấp nhất (~12.48 triệu VNĐ). Cơ chế *Bagging* (tổng hợp từ nhiều cây quyết định độc lập) giúp mô hình đạt độ ổn định cao, tránh được hiện tượng học vẹt (Overfitting) trên một tập dữ liệu đã được tinh chuẩn.
    * **XGBoost (Á quân Tốc độ):** Mặc dù lùi xuống vị trí thứ 2 về $R^2$ (0.75), XGBoost lại gây ấn tượng mạnh với sai số MAE thấp nhất (~5.86 triệu VNĐ) và tốc độ huấn luyện nhanh như chớp (0.65s). 
    * **Nhóm Tuyến tính (Ridge/Linear):** Chỉ đạt $R^2$ quanh mốc 0.60. Việc loại bỏ các giá trị dị biệt đã làm giảm phương sai tổng thể, bộc lộ rõ giới hạn của các đường thẳng trong việc nắm bắt xu hướng giá phức tạp.
    * **SVR (Kém nhất):** Vẫn đuối sức với $R^2 = 0.12$ và thời gian chạy lâu nhất (10.74s) do không xử lý tốt ma trận thưa thớt sinh ra từ One-Hot Encoding.
    
    👉 **Kết luận:** Nhóm quyết định lựa chọn **Random Forest** làm thuật toán lõi (Unified Model) nhờ khả năng kiểm soát sai số phạt (RMSE) xuất sắc nhất, mang lại độ tin cậy cao nhất làm đầu vào cho bài toán kiểm duyệt bất thường.
    """)

    st.markdown("<hr>", unsafe_allow_html=True)

    st.markdown("### 🚨 2. Cơ chế tính toán Phát hiện Bất thường (Anomaly Detection)")
    st.markdown("Thay vì sử dụng luật If/Else cứng nhắc, hệ thống kết hợp Thống kê mô tả mạnh (Robust Statistics) và Học máy không giám sát (Unsupervised Learning) thành **Kiến trúc chấm điểm tổ hợp (Ensemble Scoring)** qua 4 tín hiệu:")

    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div style="background-color: #F3F4F6; padding: 15px; border-radius: 8px; border-left: 5px solid #6366F1; height: 100%; margin-bottom: 10px;">
            <h4 style="color: #4338CA; margin-top: 0;">S1: Phần dư chuẩn hóa (Z-Robust)</h4>
            <p style="font-size: 14px;">Thay vì dùng Trung bình (Mean), hệ thống dùng Trung vị (Median) và Độ lệch tuyệt đối trung vị (MAD) theo từng phân khúc xe để chống nhiễu.</p>
            <code style="font-size: 12px; color: #E11D48;">Z_Robust = (Residual - Median) / (1.4826 * MAD)</code>
            <p style="font-size: 14px; margin-top: 5px;"><b>Quy tắc:</b> Gắn cờ vi phạm nếu giá trị tuyệt đối $|Z| \ge 3$.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="background-color: #F3F4F6; padding: 15px; border-radius: 8px; border-left: 5px solid #10B981; height: 100%;">
            <h4 style="color: #047857; margin-top: 0;">S2 & S3: Giới hạn giá & Khoảng tin cậy</h4>
            <p style="font-size: 14px;">Đánh giá dựa trên ranh giới tuyệt đối của từng phân khúc cụ thể (Ví dụ: So sánh SH với tập dữ liệu SH).</p>
            <ul style="font-size: 14px;">
                <li><b>S2 (Vi phạm Max/Min):</b> Giá vượt ra ngoài vùng 1% rẻ nhất hoặc 1% đắt nhất (P1 - P99).</li>
                <li><b>S3 (Khoảng tin cậy):</b> Cảnh báo nhẹ khi giá nằm ngoài vùng phổ biến (P10 - P90).</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style="background-color: #F3F4F6; padding: 15px; border-radius: 8px; border-left: 5px solid #F59E0B; height: 100%; margin-bottom: 10px;">
            <h4 style="color: #B45309; margin-top: 0;">S4: Isolation Forest (AI Không giám sát)</h4>
            <p style="font-size: 14px;">Đưa toàn bộ vector đặc trưng (ODO, Tuổi đời, Giá, Residual) vào mô hình rừng cây cô lập.</p>
            <p style="font-size: 14px;"><b>Cơ chế:</b> Thuật toán phân chia không gian dữ liệu ngẫu nhiên. Chiếc xe nào có thông số phi logic (Ví dụ: 10 năm tuổi, đi 100,000km nhưng giá cao ngang xe mới) sẽ bị cô lập rất nhanh trên cây và bị gán nhãn Bất thường (-1).</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="background-color: #FEF2F2; padding: 15px; border-radius: 8px; border-left: 5px solid #EF4444; height: 100%;">
            <h4 style="color: #B91C1C; margin-top: 0;">Chấm điểm Tổng hợp (Composite Score)</h4>
            <p style="font-size: 14px;">Quy đổi 4 tín hiệu trên thành thang điểm 100. Các tin đăng đạt 100 điểm sẽ được phân loại:</p>
            <ul style="font-size: 14px;">
                <li><b>Z < 0:</b> Giá rẻ bất thường (Nghi ngờ lừa đảo cọc ảo).</li>
                <li><b>Z > 0:</b> Giá đắt bất thường (Nghi ngờ ngáo giá, xe độ, biển VIP).</li>
            </ul>
            <p style="font-size: 14px;"><b>Hành động:</b> Chuyển vào danh sách đỏ để Admin duyệt thủ công.</p>
        </div>
        """, unsafe_allow_html=True)

elif menu == "📝 Dự đoán & Đăng tin":
    st.markdown('<p class="main-header">Đăng tin & Thẩm định Giá tự động</p>', unsafe_allow_html=True)
    if not has_task2: 
        st.warning("⚠️ Cảnh báo: Chưa tìm thấy mô hình Isolation Forest (Bài 2). Chấm điểm bằng mô phỏng dự phòng.")
    
    st.markdown("#### 1. Thông số Kỹ thuật & Phân loại")
    c1, c2, c3 = st.columns(3)
    with c1:
        nguoi_dang = st.text_input("Người bán", value="Bán xe máy cũ chính chủ")
        thuong_hieu = st.selectbox("Thương hiệu", sorted(list(dynamic_mapping.keys())), index=sorted(list(dynamic_mapping.keys())).index('Honda') if 'Honda' in dynamic_mapping else 0)
        dong_xe_options = sorted(list(dynamic_mapping[thuong_hieu].keys()))
        dong_xe = st.selectbox("Dòng xe", dong_xe_options)
        loai_xe_auto = dynamic_mapping[thuong_hieu][dong_xe]
        st.text_input("Loại xe (AI tự nhận diện)", value=loai_xe_auto, disabled=True)
        loai_xe = loai_xe_auto

    with c2:
        dung_tich_sach = [x for x in t1_schema['catalogs']['Dung_tích_xe'] if x not in ['Nhật Bản', 'Bảo hành hãng', 'Không rõ']]
        dung_tich = st.selectbox("Dung tích xi lanh", sorted(dung_tich_sach))
        xuat_xu = st.selectbox("Xuất xứ", sorted(t1_schema['catalogs']['Xuất_xứ']))
        tuoi_doi = st.number_input("Tuổi đời xe (Năm)", 0, 50, 1)
        odo = st.number_input("Số Km đã đi (ODO)", 0, 500000, 10000, step=1000)
        
    with c3:
        st.markdown("**Phụ kiện & Trạng thái**")
        is_zin = st.checkbox("Xe nguyên bản (Zin)", value=True)
        is_chinh_chu = st.checkbox("Chính chủ", value=True)
        is_hqcn = st.checkbox("Hải quan chính ngạch (HQCN)", value=False)
        is_nhap_y = st.checkbox("Nhập Ý", value=False)
        is_abs = st.checkbox("Phanh ABS", value=False)
        is_smartkey = st.checkbox("Khóa Smartkey", value=False)
        is_vip_plate = st.checkbox("Biển số VIP", value=False)
        
    st.markdown("#### 2. Mức giá & Thẩm định")
    gia_muon_ban = st.number_input("Giá bạn mong muốn bán (VNĐ)", min_value=1000000, value=35000000, step=1000000)
    
    st.markdown("<br>", unsafe_allow_html=True)
    submit_btn = st.button("Lấy Định Giá & Nhận Định Từ AI", type="primary", use_container_width=True)

    if submit_btn:
        input_data = {
            'Số_Km_đã_đi': odo, 'Tuổi_đời_xe': tuoi_doi, 'Km_per_Year': odo / (tuoi_doi if tuoi_doi > 0 else 1),
            'is_hqcn': int(is_hqcn), 'is_nhap_y': int(is_nhap_y), 'is_abs': int(is_abs),
            'is_smartkey': int(is_smartkey), 'is_vip_plate': int(is_vip_plate), 'is_zin': int(is_zin), 'is_chinh_chu': int(is_chinh_chu),
            'Thương_hiệu': thuong_hieu, 'Dòng_xe': dong_xe, 'Xuất_xứ': xuat_xu, 'Dung_tích_xe': dung_tich, 'Loại_xe': loai_xe
        }
        try:
            df_input = pd.DataFrame([input_data])[t1_schema['num_cols'] + t1_schema['cat_cols']]
            predicted_price = float(t1_model.predict(df_input)[0])
        except Exception as e:
            st.error(f"Lỗi AI dự đoán: {e}")
            predicted_price = 0
            
        z_rb, s1, s2, s3, s4, a_score, loai_bt, lech_pct, g_min, g_max = detect_anomaly_real(input_data, gia_muon_ban, predicted_price)
        
        st.markdown("---")
        c1, c2 = st.columns(2)
        c1.info(f"🤖 **Hệ thống AI định giá chiếc xe này:** {predicted_price:,.0f} VNĐ")
        c2.success(f"💵 **Giá bạn đang muốn đăng bán:** {gia_muon_ban:,.0f} VNĐ")
        
        if a_score == 100:
            st.error(f"🔍 **Nhận định từ hệ thống:** 🔴 **GIÁ BẤT THƯỜNG!** {loai_bt}.")
            
            st.markdown(f"""
            <div class="ai-insight-box" style="border-color: #EF4444; background-color: #FEF2F2;">
                <h4 style="color: #991B1B; margin-top:0;">📊 PHÂN TÍCH CHI TIẾT TỪ HỆ THỐNG AI (CƠ CHẾ GIẢI THÍCH):</h4>
                <ul>
                    <li><b>Độ chênh lệch giá:</b> Giá bạn mong muốn đang <b>{'CAO' if lech_pct > 0 else 'THẤP'} HƠN {abs(lech_pct)}%</b> so với giá AI định giá chuẩn.</li>
                    <li><b>Chỉ số Robust Z-Score:</b> <code>Z = {z_rb}</code> (Vượt quá ngưỡng an toàn [-3.0 đến +3.0]).</li>
                    <li><b>Khoảng giá an toàn gợi ý cho phân khúc này:</b> từ <b style="color: #059669;">{g_min:,.0f} VNĐ</b> đến <b style="color: #059669;">{g_max:,.0f} VNĐ</b>.</li>
                </ul>
                <p style="margin-bottom:0; font-style: italic; color: #7F1D1D;">⚠️ <b>Lưu ý:</b> Tin đăng của bạn đã được chuyển đến <b>Quản trị viên (Admin)</b> để duyệt thủ công nhằm bảo đảm tính minh bạch cho sàn.</p>
            </div>
            """, unsafe_allow_html=True)
            
        elif predicted_price == 0:
             st.error("Hệ thống không thể định giá do lỗi dữ liệu.")
        else:
            st.success("🔍 **Nhận định từ hệ thống:** 🟢 **HỢP LÝ!** Mức giá an toàn và sát thị trường.")
            st.markdown(f"""
            <div class="ai-insight-box" style="border-color: #10B981; background-color: #ECFDF5;">
                <h4 style="color: #065F46; margin-top:0;">📊 PHÂN TÍCH CHI TIẾT TỪ HỆ THỐNG AI:</h4>
                <ul>
                    <li><b>Độ chênh lệch giá:</b> Giá bán nằm trong biên độ an toàn (Chênh lệch <b>{lech_pct}%</b>).</li>
                    <li><b>Chỉ số Robust Z-Score:</b> <code>Z = {z_rb}</code> (Nằm trong vùng phân phối chuẩn).</li>
                    <li><b>Đánh giá thanh khoản:</b> Tin đăng sẽ được ưu tiên hiển thị ở vị trí top sàn mua bán.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
            
        db = load_db()
        new_row = pd.DataFrame([{
            "ID": f"POST_{len(db)+1:04d}", "Thời gian": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
            "Người đăng": nguoi_dang, "Thương hiệu": thuong_hieu, "Dòng xe": dong_xe, 
            "Giá mong muốn": int(gia_muon_ban), "Giá AI dự đoán": int(predicted_price), 
            "Z_Robust": z_rb, "S1_Residual_Z": int(s1), "S2_Price_Limit": int(s2), "S3_Confidence": int(s3), "S4_Unsupervised": int(s4), 
            "Anomaly_Score": int(a_score), "Loại_Bất_Thường": loai_bt, "Phần_Trăm_Lệch": lech_pct,
            "Giá_Min_Khuyên_Dùng": g_min, "Giá_Max_Khuyên_Dùng": g_max, "Trạng thái Duyệt": "Chờ duyệt ⏳"
        }])
        save_to_db(pd.concat([db, new_row], ignore_index=True))

elif menu == "🛡️ Quản trị viên (Admin)":
    st.markdown('<p class="main-header">Hệ thống Giám sát & Duyệt tin</p>', unsafe_allow_html=True)
    
    # Tạo 2 Tab để chia luồng Quản trị Thủ công và Hàng loạt
    tab_manual, tab_batch = st.tabs(["📑 Duyệt thủ công (Cá nhân)", "📁 Duyệt hàng loạt (Cửa hàng / Đối tác)"])
    
    # ==========================================
    # TAB 1: DUYỆT THỦ CÔNG (LUỒNG CŨ ĐÃ VÁ LỖI MAP)
    # ==========================================
    with tab_manual:
        db = load_db()
        if db.empty: 
            st.info("Chưa có tin đăng cá nhân nào được ghi nhận trên hệ thống.")
        else:
            st.markdown("#### 📋 Bảng Phân Tích Điểm Bất Thường (Anomaly Scoring Dashboard)")
            
            display_cols = ["ID", "Thương hiệu", "Dòng xe", "Giá mong muốn", "Giá AI dự đoán", "Phần_Trăm_Lệch", "Z_Robust", "S1_Residual_Z", "S2_Price_Limit", "S3_Confidence", "S4_Unsupervised", "Anomaly_Score", "Loại_Bất_Thường", "Trạng thái Duyệt"]
            df_display = db[display_cols].copy()
            
            df_display["Giá mong muốn"] = df_display["Giá mong muốn"].apply(lambda x: f"{int(x):,}" if pd.notnull(x) else "0")
            df_display["Giá AI dự đoán"] = df_display["Giá AI dự đoán"].apply(lambda x: f"{int(x):,}" if pd.notnull(x) else "0")
            df_display["Phần_Trăm_Lệch"] = df_display["Phần_Trăm_Lệch"].apply(lambda x: f"{x}%" if pd.notnull(x) else "0%")
            
            for col in ["S1_Residual_Z", "S2_Price_Limit", "S3_Confidence", "S4_Unsupervised", "Anomaly_Score"]:
                df_display[col] = df_display[col].apply(lambda x: f"{int(x)}" if pd.notnull(x) else "0")
            
            def highlight_anomaly(val):
                color = '#FEF2F2' if str(val) == '100' else 'transparent'
                font_color = '#B91C1C' if str(val) == '100' else 'black'
                font_weight = 'bold' if str(val) == '100' else 'normal'
                return f'background-color: {color}; color: {font_color}; font-weight: {font_weight}'

            try:
                styled_df = df_display.style.map(highlight_anomaly, subset=['Anomaly_Score'])
            except AttributeError:
                styled_df = df_display.style.map(highlight_anomaly, subset=['Anomaly_Score'])
                
            st.dataframe(styled_df, use_container_width=True)
            
            st.markdown("---")
            st.markdown("#### ⚖️ Quyết định Duyệt tin & Bằng chứng AI (Dành cho Admin)")
            
            pending_posts = db[db["Trạng thái Duyệt"] == "Chờ duyệt ⏳"]
            if pending_posts.empty:
                st.success("🎉 Tất cả tin đăng đã được xử lý xong!")
            else:
                for index, row in pending_posts.iterrows():
                    with st.container():
                        col1, col2, col3 = st.columns([4, 2, 2])
                        with col1:
                            st.write(f"**Mã tin:** `{row['ID']}` | **Xe:** {row['Thương hiệu']} {row['Dòng xe']}")
                            st.write(f"💵 **Giá đăng:** {int(row['Giá mong muốn']):,} VNĐ | 🤖 **AI Định giá:** {int(row['Giá AI dự đoán']):,} VNĐ (Lệch: **{row['Phần_Trăm_Lệch']}%**)")
                            
                            if int(row['Anomaly_Score']) == 100:
                                st.error(f"🔴 **CẢNH BÁO VI PHẠM:** {row['Loại_Bất_Thường']}")
                                st.caption(f"🛡️ **Các cờ AI kích hoạt:** S1(Z-Robust={row['Z_Robust']})={row['S1_Residual_Z']} | S2(Max/Min)={row['S2_Price_Limit']} | S3(Conf)={row['S3_Confidence']} | S4(IsolationForest)={row['S4_Unsupervised']}")
                            else:
                                st.success("🟢 Tin đăng an toàn. Không phát hiện cờ vi phạm.")
                        with col2:
                            if st.button("✅ Cho phép đăng", key=f"ok_{row['ID']}", use_container_width=True):
                                db.at[index, "Trạng thái Duyệt"] = "Đã duyệt ✅"
                                save_to_db(db); st.rerun()
                        with col3:
                            if st.button("❌ Từ chối tin", key=f"ban_{row['ID']}", type="primary", use_container_width=True):
                                db.at[index, "Trạng thái Duyệt"] = "Từ chối ❌"
                                save_to_db(db); st.rerun()
                        st.markdown("<hr style='margin: 10px 0px;'>", unsafe_allow_html=True)

    # ==========================================
    # TAB 2: MODULE DUYỆT TIN HÀNG LOẠT & XUẤT FILE 
    # ==========================================
    with tab_batch:
        st.markdown("### 📁 Kiểm Duyệt Tin Đăng Hàng Loạt (Batch Processing)")
        st.caption("Tải lên tệp dữ liệu (CSV/Excel) từ các cửa hàng đối tác để AI thẩm định tự động qua Pipeline.")
        
        uploaded_file = st.file_uploader("Chọn file dữ liệu (Hỗ trợ: .csv, .xlsx)", type=['csv', 'xlsx'])

        if uploaded_file is not None:
            try:
                # 1. Đọc tệp dữ liệu
                if uploaded_file.name.endswith('.csv'):
                    df_batch = pd.read_csv(uploaded_file)
                else:
                    df_batch = pd.read_excel(uploaded_file)
                    
                st.success(f"✅ Tải lên thành công {len(df_batch):,} dòng dữ liệu. Hệ thống đang tiến hành xử lý NLP và định giá...")
                
                # 2. Xử lý qua Pipeline AI tái sử dụng 
                with st.spinner('🤖 Đang gọi mô hình AI để thẩm định hàng loạt...'):
                    results = []
                    for idx, row in df_batch.iterrows():
                        mo_ta = str(row.get('Mo_Ta_Chi_Tiet', '')).lower()
                        
                        input_data = {
                            'Số_Km_đã_đi': row.get('ODO_Km', 0),
                            'Tuổi_đời_xe': row.get('Tuoi_Doi_Nam', 1),
                            'Km_per_Year': row.get('ODO_Km', 0) / (row.get('Tuoi_Doi_Nam', 1) if row.get('Tuoi_Doi_Nam', 1) > 0 else 1),
                            'is_hqcn': 1 if 'hqcn' in mo_ta or 'hải quan' in mo_ta else 0,
                            'is_nhap_y': 1 if 'nhập ý' in mo_ta else 0,
                            'is_abs': 1 if 'abs' in mo_ta else 0,
                            'is_smartkey': 1 if 'smartkey' in mo_ta else 0,
                            'is_vip_plate': 1 if 'vip' in mo_ta or 'ngũ quý' in mo_ta or 'tứ quý' in mo_ta else 0,
                            'is_zin': 1 if 'zin' in mo_ta or 'nguyên bản' in mo_ta else 0,
                            'is_chinh_chu': 1 if 'chính chủ' in mo_ta else 0,
                            'Thương_hiệu': row.get('Thuong_Hieu', 'Không rõ'),
                            'Dòng_xe': row.get('Dong_Xe', 'Không rõ'),
                            'Xuất_xứ': row.get('Xuat_Xu', 'Việt Nam'),
                            'Dung_tích_xe': row.get('Dung_Tich', '100-175cc'),
                            'Loại_xe': row.get('Loai_Xe', 'Tay ga')
                        }
                        
                        try:
                            df_input = pd.DataFrame([input_data])[t1_schema['num_cols'] + t1_schema['cat_cols']]
                            pred_price = float(t1_model.predict(df_input)[0])
                        except Exception:
                            pred_price = 0
                            
                        gia_muon_ban = row.get('Gia_Mong_Muon_VND', 0)
                        z_rb, s1, s2, s3, s4, a_score, loai_bt, lech_pct, g_min, g_max = detect_anomaly_real(input_data, gia_muon_ban, pred_price)
                        
                        row_result = row.copy()
                        row_result['Gia_AI_Du_Doan'] = pred_price
                        row_result['Z_Score'] = z_rb
                        row_result['Anomaly_Score'] = a_score
                        row_result['Loai_Bat_Thuong'] = loai_bt
                        results.append(row_result)
                    
                    df_evaluated = pd.DataFrame(results)

                # 3. Thuật toán Phân luồng (Triage)
                df_normal = df_evaluated[(df_evaluated['Z_Score'] >= -3.0) & (df_evaluated['Z_Score'] <= 3.0)].copy()
                df_high = df_evaluated[df_evaluated['Z_Score'] > 3.0].copy()
                df_low = df_evaluated[df_evaluated['Z_Score'] < -3.0].copy()
                
                st.markdown("---")
                st.markdown("### 📊 KẾT QUẢ PHÂN LOẠI TỪ HỆ THỐNG AI")
                
                # 4. Hiển thị Giao diện xử lý hàng loạt
                t1, t2, t3 = st.tabs([
                    f"🟢 BÌNH THƯỜNG ({len(df_normal)})", 
                    f"🔴 GIÁ QUÁ CAO ({len(df_high)})", 
                    f"🟠 GIÁ QUÁ THẤP ({len(df_low)})"
                ])
                
                # --- NHÓM BÌNH THƯỜNG ---
                with t1:
                    st.info("Nhóm tin đăng có mức giá hợp lý, nằm trong vùng phân phối chuẩn an toàn [-3.0 đến +3.0].")
                    if not df_normal.empty:
                        df_normal.insert(0, "Duyệt_Tin", True)
                        edited_normal = st.data_editor(
                            df_normal[['Duyệt_Tin', 'Ma_Tin', 'Thuong_Hieu', 'Dong_Xe', 'Gia_Mong_Muon_VND', 'Gia_AI_Du_Doan', 'Z_Score']],
                            hide_index=True, use_container_width=True,
                            disabled=['Ma_Tin', 'Thuong_Hieu', 'Dong_Xe', 'Gia_Mong_Muon_VND', 'Gia_AI_Du_Doan', 'Z_Score']
                        )
                        
                        # Thêm công cụ Hành động và Xuất File nằm ngang
                        col_action_1, col_download_1 = st.columns(2)
                        with col_action_1:
                            if st.button("✅ Thực thi Quyết định (Bình Thường)", use_container_width=True):
                                so_luong_duyet = edited_normal['Duyệt_Tin'].sum()
                                st.success(f"Đã cập nhật: Phê duyệt thành công {so_luong_duyet} tin hợp lệ!")
                        with col_download_1:
                            csv_normal = df_normal.to_csv(index=False).encode('utf-8-sig') # Dùng utf-8-sig để Excel không lỗi font
                            st.download_button(
                                label="📥 Tải danh sách Bình Thường (.csv)",
                                data=csv_normal,
                                file_name=f"DS_Binh_Thuong_{datetime.datetime.now().strftime('%Y%m%d')}.csv",
                                mime="text/csv",
                                use_container_width=True
                            )

                # --- NHÓM GIÁ CAO ---
                with t2:
                    st.error("Cảnh báo: Nhóm xe có giá bán vượt mức trần an toàn (Z-Score > 3.0). Nguy cơ ngáo giá hoặc xe độ/sưu tầm.")
                    if not df_high.empty:
                        df_high.insert(0, "Tu_Choi", True)
                        edited_high = st.data_editor(
                            df_high[['Tu_Choi', 'Ma_Tin', 'Thuong_Hieu', 'Dong_Xe', 'Mo_Ta_Chi_Tiet', 'Gia_Mong_Muon_VND', 'Gia_AI_Du_Doan', 'Z_Score']],
                            hide_index=True, use_container_width=True,
                            disabled=['Ma_Tin', 'Thuong_Hieu', 'Dong_Xe', 'Mo_Ta_Chi_Tiet', 'Gia_Mong_Muon_VND', 'Gia_AI_Du_Doan', 'Z_Score']
                        )
                        
                        col_action_2, col_download_2 = st.columns(2)
                        with col_action_2:
                            if st.button("❌ Thực thi Quyết định (Nhóm Giá Cao)", use_container_width=True):
                                so_luong_tu_choi = edited_high['Tu_Choi'].sum()
                                st.warning(f"Đã cập nhật: Từ chối và chặn hiển thị {so_luong_tu_choi} tin vi phạm!")
                        with col_download_2:
                            csv_high = df_high.to_csv(index=False).encode('utf-8-sig')
                            st.download_button(
                                label="📥 Tải danh sách Giá Cao (.csv)",
                                data=csv_high,
                                file_name=f"DS_Gia_Cao_{datetime.datetime.now().strftime('%Y%m%d')}.csv",
                                mime="text/csv",
                                use_container_width=True
                            )

                # --- NHÓM GIÁ THẤP ---
                with t3:
                    st.warning("Cảnh báo: Nhóm xe có giá bán quá rẻ so với thị trường (Z-Score < -3.0). Nguy cơ lừa đảo tiền cọc.")
                    if not df_low.empty:
                        df_low.insert(0, "Tu_Choi", True)
                        edited_low = st.data_editor(
                            df_low[['Tu_Choi', 'Ma_Tin', 'Thuong_Hieu', 'Dong_Xe', 'Gia_Mong_Muon_VND', 'Gia_AI_Du_Doan', 'Z_Score']],
                            hide_index=True, use_container_width=True,
                            disabled=['Ma_Tin', 'Thuong_Hieu', 'Dong_Xe', 'Gia_Mong_Muon_VND', 'Gia_AI_Du_Doan', 'Z_Score']
                        )
                        
                        col_action_3, col_download_3 = st.columns(2)
                        with col_action_3:
                            if st.button("❌ Thực thi Quyết định (Nhóm Giá Thấp)", use_container_width=True):
                                so_luong_tu_choi = edited_low['Tu_Choi'].sum()
                                st.warning(f"Đã cập nhật: Khóa vĩnh viễn {so_luong_tu_choi} tin có dấu hiệu lừa đảo!")
                        with col_download_3:
                            csv_low = df_low.to_csv(index=False).encode('utf-8-sig')
                            st.download_button(
                                label="📥 Tải danh sách Giá Thấp (.csv)",
                                data=csv_low,
                                file_name=f"DS_Gia_Thap_{datetime.datetime.now().strftime('%Y%m%d')}.csv",
                                mime="text/csv",
                                use_container_width=True
                            )

            except Exception as e:
                st.error(f"Lỗi hệ thống khi đọc hoặc phân tích file: {e}. Vui lòng kiểm tra lại cấu trúc file dữ liệu.")
