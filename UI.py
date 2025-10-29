import torch
from torchvision import transforms
from PIL import Image
import logging
from pathlib import Path
import streamlit as st
import os
from typing import Tuple
from datetime import datetime  

# Import utilities
from utils.watermark import add_watermark
from utils.ecc_encrypt import encrypt_image
from utils.ecc_decrypt import decrypt_image

# Logging config
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# ===================== MODEL ANALYZER =====================
class XRayAnalyzer:
    CLASSES = ["Normal", "Pneumonia"]
    MODEL_PATH = "Model/model.pth"

    def __init__(self):
        self.model = self._load_model()
        self.transform = self._get_transforms()

    def _load_model(self) -> torch.nn.Module:
        try:
            model_path = Path(self.MODEL_PATH)
            if not model_path.exists():
                raise FileNotFoundError(f"Model file not found at {model_path}")

            model = torch.load(model_path, map_location=torch.device("cpu"))
            model.eval()
            logger.info("Model loaded successfully")
            return model
        except Exception as e:
            logger.error(f"Model loading error: {str(e)}")
            raise RuntimeError("Error loading model")

    @staticmethod
    def _get_transforms():
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def predict_with_torch(self, image_path: str) -> Tuple[str, float]:
        try:
            image = Image.open(image_path).convert("RGB")
            image_tensor = self.transform(image).unsqueeze(0)

            with torch.no_grad():
                outputs = self.model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)

            return self.CLASSES[predicted.item()], confidence.item()
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise RuntimeError("Error during model prediction.")

# ===================== MAIN APP =====================
def main():
    st.set_page_config(page_title="X-Ray Analyzer", page_icon="🩻", layout="wide")

    # ================= CUSTOM CSS =================
    st.markdown(
        """
        <style>
        /* Background */
        .stApp {
            background: linear-gradient(135deg, #dbeafe, #e0f2fe, #f0f9ff);
            font-family: 'Segoe UI', sans-serif;
        }

        /* Titles */
        .main-title {
            text-align: center;
            color: #0d47a1;
            font-size: 42px;
            font-weight: 900;
            margin-bottom: 5px;
            text-shadow: 1px 1px 4px rgba(0,0,0,0.2);
        }
        .sub-title {
            text-align: center;
            color: #37474f;
            font-size: 18px;
            margin-bottom: 35px;
        }

        /* Glassmorphism Cards */
        .glass-card {
            background: rgba(255,255,255,0.8);
            border-radius: 18px;
            padding: 25px;
            margin: 15px 0;
            box-shadow: 0px 6px 18px rgba(0,0,0,0.15);
            backdrop-filter: blur(12px);
            transition: transform 0.25s ease;
        }
        .glass-card:hover {
            transform: scale(1.02);
            box-shadow: 0px 10px 22px rgba(0,0,0,0.25);
        }

        /* File Upload Styling */
        .stFileUploader label {
            background: linear-gradient(90deg, #1a73e8, #42a5f5);
            color: white !important;
            padding: 10px 20px;
            border-radius: 10px;
            font-weight: 600;
            cursor: pointer;
        }
        .stFileUploader label:hover {
            background: linear-gradient(90deg, #0d47a1, #1565c0);
        }

        /* Footer */
        .footer {
            position: fixed;
            bottom: 0;
            left: 0;
            width: 100%;
            background: #0d47a1;
            color: white;
            text-align: center;
            padding: 12px;
            font-size: 14px;
            box-shadow: 0px -2px 8px rgba(0,0,0,0.3);
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # ================= SIDEBAR =================
    st.sidebar.image("https://img.icons8.com/fluency/96/medical-doctor.png", width=80)
    st.sidebar.title("🧭 Navigation")
    menu = st.sidebar.radio("Go to", ["🏠 Home", "📤 Upload X-Ray", "ℹ️ About Project"])

    # ================= HOME =================
    if menu == "🏠 Home":
        st.markdown("<h1 class='main-title'>🩻 Secure X-Ray Analyzer</h1>", unsafe_allow_html=True)
        st.markdown("<p class='sub-title'>AI-powered chest X-ray analysis with watermarking, encryption & secure transmission</p>", unsafe_allow_html=True)
        st.markdown(
            "<div class='glass-card'><h3>🚀 Features</h3>"
            "<ul>"
            "<li>🔒 End-to-End Secure Transmission (ECC Encryption)</li>"
            "<li>🖼️ Automatic Watermarking with Timestamp</li>"
            "<li>🤖 AI-based Disease Detection (Pneumonia)</li>"
            "<li>📊 Confidence Scores for Transparency</li>"
            "</ul></div>",
            unsafe_allow_html=True
        )

    # ================= UPLOAD & ANALYZE =================
    elif menu == "📤 Upload X-Ray":
        st.markdown("<h1 class='main-title'>📥 Upload & Analyze X-Ray</h1>", unsafe_allow_html=True)
        st.markdown("<p class='sub-title'>Upload a chest X-ray to begin secure analysis</p>", unsafe_allow_html=True)

        for folder in ["temp", "data/encrypted", "data/decrypted"]:
            os.makedirs(folder, exist_ok=True)

        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Upload Chest X-ray Image", type=["jpg", "jpeg", "png"])
        st.markdown('</div>', unsafe_allow_html=True)

        if uploaded_file is not None:
            # Save file
            image_path = os.path.join("temp", uploaded_file.name)
            with open(image_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            # ✅ Add watermark
            upload_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            watermark_text = f"SECURED | {upload_time}"
            watermarked_path = os.path.join("temp", "watermarked_" + uploaded_file.name)
            add_watermark(image_path, watermarked_path, watermark_text)

            # ✅ Encrypt & Decrypt
            encrypted_path = os.path.join("data/encrypted", uploaded_file.name + ".bin")
            ephemeral_pub = encrypt_image(watermarked_path, "keys/public_key.pem", encrypted_path)
            decrypted_path = os.path.join("data/decrypted", "dec_" + uploaded_file.name)
            decrypt_image(encrypted_path, "keys/private_key.pem", ephemeral_pub, decrypted_path)

            # ================= RESULTS =================
            st.markdown("<h2>📊 Results</h2>", unsafe_allow_html=True)
            col1, col2 = st.columns(2)

            with col1:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                st.image(watermarked_path, caption=f"🖼️ Watermarked (Uploaded {upload_time})", use_container_width=True)
                st.success("✅ Image secured with watermark & encryption.")
                st.markdown('</div>', unsafe_allow_html=True)

            analyzer = XRayAnalyzer()

            with col2:
                st.markdown('<div class="glass-card">', unsafe_allow_html=True)
                with st.spinner("🔍 Analyzing with AI model..."):
                    try:
                        prediction, confidence = analyzer.predict_with_torch(decrypted_path)
                        st.metric(label="Prediction", value=prediction)
                        st.metric(label="Confidence", value=f"{confidence*100:.2f}%")
                        if prediction == "Pneumonia":
                            st.error("⚠️ Pneumonia Detected. Please consult a doctor.")
                        else:
                            st.success("✅ Normal X-ray detected.")
                    except Exception as e:
                        st.error(f"Error during analysis: {str(e)}")
                st.markdown('</div>', unsafe_allow_html=True)

        else:
            st.info("📌 Please upload an image to continue.")

    # ================= ABOUT =================
    elif menu == "ℹ️ About Project":
        st.markdown("<h1 class='main-title'>ℹ️ About This Project</h1>", unsafe_allow_html=True)
        st.markdown(
            "<div class='glass-card'>"
            "<p><b>📌 Final Year Project</b> - Islamia College Peshawar</p>"
            "<p>This project ensures <b>secure transmission of medical images</b> "
            "using watermarking + ECC encryption, followed by <b>AI-powered Pneumonia detection</b>. "
            "It protects sensitive patient data while supporting early disease diagnosis.</p>"
            "</div>",
            unsafe_allow_html=True
        )

    # ================= FOOTER =================
    st.markdown(
        """
        <div class="footer">
            © 2025 X-Ray Analyzer | Developed by <b>Muhammad Usman</b> | Islamia College Peshawar
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()
