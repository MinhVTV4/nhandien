import streamlit as st
import face_recognition
import numpy as np
from PIL import Image, ImageDraw

st.set_page_config(page_title="Nhận diện khuôn mặt", layout="centered")

st.title("📷 Ứng dụng Nhận diện Khuôn mặt")
st.write("Chụp ảnh bằng camera để phát hiện khuôn mặt")

# ==============================
# CAMERA INPUT
# ==============================
img_file = st.camera_input("Chụp ảnh")

if img_file is not None:
    # Load ảnh
    image = Image.open(img_file).convert("RGB")
    image_np = np.array(image)

    with st.spinner("🔍 Đang nhận diện khuôn mặt..."):
        face_locations = face_recognition.face_locations(image_np)

    # Vẽ khung khuôn mặt
    draw = ImageDraw.Draw(image)
    for top, right, bottom, left in face_locations:
        draw.rectangle(((left, top), (right, bottom)), outline="red", width=3)

    # Hiển thị kết quả
    st.success(f"✅ Phát hiện {len(face_locations)} khuôn mặt")
    st.image(image, caption="Kết quả nhận diện", use_container_width=True)

else:
    st.info("👆 Hãy bấm nút chụp ảnh để bắt đầu")
