import streamlit as st
import numpy as np
from PIL import Image
import cv2

# ==============================
# CONFIG
# ==============================
st.set_page_config(
    page_title="AI Image Demo",
    layout="centered"
)

st.title("🤖 AI Demo – Phân tích ảnh")
st.write("Upload ảnh → AI xử lý → Hiển thị kết quả")

# ==============================
# CACHE MODEL (GIẢ LẬP)
# ==============================
@st.cache_resource
def load_ai_model():
    # Giả lập model nặng
    return "dummy_model"

model = load_ai_model()

# ==============================
# UPLOAD IMAGE
# ==============================
uploaded_file = st.file_uploader(
    "📤 Upload ảnh",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file is None:
    st.info("👆 Vui lòng upload ảnh để bắt đầu")
    st.stop()

# ==============================
# LOAD & PREPROCESS
# ==============================
image = Image.open(uploaded_file).convert("RGB")
image_np = np.array(image)

st.image(image, caption="Ảnh gốc", use_container_width=True)

# ==============================
# AI PROCESSING
# ==============================
with st.spinner("🧠 AI đang phân tích..."):
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 100, 200)

# ==============================
# OUTPUT
# ==============================
st.success("✅ Phân tích xong")

st.subheader("🔍 Kết quả AI (Edge Detection)")
st.image(edges, use_container_width=True)

# ==============================
# EXPLAIN
# ==============================
st.markdown("""
### 📌 Giải thích
- Ảnh được chuyển sang **grayscale**
- AI phát hiện **đường biên (edges)**
- Đây là bước nền cho:
  - Nhận diện khuôn mặt
  - Phát hiện vật thể
  - OCR
""")
