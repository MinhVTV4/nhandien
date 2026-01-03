import streamlit as st
import face_recognition
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
import os
from PIL import Image

# Cấu hình trang Web
st.set_page_config(page_title="AI Attendance 2026", page_icon="🛡️", layout="wide")

st.title("🛡️ Hệ thống Nhận diện & Điểm danh Thông minh")
st.markdown("---")

# --- BƯỚC 1: HÀM HỖ TRỢ ---
def load_and_encode_faces(path='faces'):
    known_encodings = []
    known_names = []
    if not os.path.exists(path):
        os.makedirs(path)
        return known_encodings, known_names
        
    for file in os.listdir(path):
        if file.endswith((".jpg", ".png", ".jpeg")):
            name = os.path.splitext(file)[0]
            try:
                img = face_recognition.load_image_file(f"{path}/{file}")
                encoding = face_recognition.face_encodings(img)[0]
                known_encodings.append(encoding)
                known_names.append(name)
            except Exception as e:
                st.error(f"Lỗi khi học ảnh {file}: {e}")
    return known_encodings, known_names

# --- BƯỚC 2: HỌC MẪU (LƯU VÀO SESSION STATE) ---
if 'known_encodings' not in st.session_state:
    with st.spinner("Đang khởi tạo bộ não AI..."):
        encodings, names = load_and_encode_faces()
        st.session_state.known_encodings = encodings
        st.session_state.known_names = names
        st.session_state.logs = pd.DataFrame(columns=["Tên", "Thời gian", "Trạng thái"])

# --- BƯỚC 3: GIAO DIỆN CHÍNH ---
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📸 Camera Điểm danh")
    img_file = st.camera_input("Đưa mặt vào khung hình")

    if img_file:
        # Chuyển đổi ảnh từ camera
        input_img = Image.open(img_file)
        img_array = np.array(input_img)
        
        # Tìm mặt và mã hóa
        face_locs = face_recognition.face_locations(img_array)
        face_encods = face_recognition.face_encodings(img_array, face_locs)
        
        if not face_encods:
            st.warning("Không tìm thấy khuôn mặt. Vui lòng thử lại!")
        
        for encoding in face_encods:
            matches = face_recognition.compare_faces(st.session_state.known_encodings, encoding, tolerance=0.5)
            name = "KHÁCH"
            
            if True in matches:
                # Tìm người khớp nhất
                face_distances = face_recognition.face_distance(st.session_state.known_encodings, encoding)
                best_match_index = np.argmin(face_distances)
                name = st.session_state.known_names[best_match_index]
                
                # Ghi log điểm danh (không ghi trùng trong 1 phiên làm việc)
                now = datetime.now().strftime("%H:%M:%S - %d/%m/%Y")
                if name not in st.session_state.logs["Tên"].values:
                    new_entry = pd.DataFrame({"Tên": [name], "Thời gian": [now], "Trạng thái": ["Có mặt"]})
                    st.session_state.logs = pd.concat([st.session_state.logs, new_entry], ignore_index=True)
                    st.balloons() # Hiệu ứng chúc mừng
            
            if name == "KHÁCH":
                st.error(f"Phát hiện: {name}")
            else:
                st.success(f"Xin chào: {name}!")

with col2:
    st.subheader("📝 Nhật ký Hệ thống")
    st.dataframe(st.session_state.logs, use_container_width=True)
    
    # Nút tải báo cáo
    if not st.session_state.logs.empty:
        csv = st.session_state.logs.to_csv(index=False).encode('utf-8-sig')
        st.download_button(
            label="📥 Tải Báo cáo (.csv)",
            data=csv,
            file_name=f'diem_danh_{datetime.now().strftime("%Y%m%d")}.csv',
            mime='text/csv',
        )
    
    st.info(f"Tổng số người đã học: {len(st.session_state.known_names)}")
