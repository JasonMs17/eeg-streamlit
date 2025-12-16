import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import os

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="EEG Eye State Classification",
    layout="centered"
)

st.title("🧠 EEG Eye State Classification")
st.markdown("Aplikasi untuk mengklasifikasikan kondisi mata (Terbuka/Tertutup) berdasarkan data EEG.")

# =========================
# MODEL SELECTION
# =========================
st.subheader("🔧 Pilih Model")

model_choice = st.selectbox(
    "Model:",
    [
        "LSTM (best_model.h5)",
        "CNN (best_cnn_model.h5)",
        "EEGNet (best_eegnet_model.h5)"
    ]
)

MODEL_MAP = {
    "LSTM (best_model.h5)": ("LSTM", "best_model.h5"),
    "CNN (best_cnn_model.h5)": ("CNN", "best_cnn_model.h5"),
    "EEGNet (best_eegnet_model.h5)": ("EEGNet", "best_eegnet_model.h5")
}

MODEL_TYPE, MODEL_PATH = MODEL_MAP[model_choice]

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_selected_model(path):
    """Memuat model Keras/TensorFlow dan menangani error jika file tidak ditemukan."""
    if not os.path.exists(path):
        st.error(f"❌ File model tidak ditemukan: {path}. Pastikan file ada di direktori yang sama.")
        return None
    try:
        # Menambahkan custom_objects jika ada lapisan khusus, misalnya Lambda
        # Tetapi untuk LSTM, CNN, Dense biasa, ini tidak diperlukan.
        return load_model(path)
    except Exception as e:
        st.error(f"❌ Gagal memuat model {path}. Mungkin ada ketidaksesuaian versi atau arsitektur. Error: {e}")
        return None

model = load_selected_model(MODEL_PATH)

if model is not None:
    st.success(f"✅ Model {MODEL_TYPE} berhasil dimuat dari **{MODEL_PATH}**.")

# =========================
# PREPROCESS FUNCTION
# =========================
def preprocess_eeg(data, model_type, target_len=128, target_ch=14):
    """
    Preprocessing data EEG menjadi bentuk yang sesuai untuk inference model.
    target_len: panjang waktu (timesteps) yang diharapkan (default 128)
    target_ch: jumlah channel yang diharapkan (default 14)
    """

    data = np.asarray(data, dtype=np.float32)

    # Pastikan 2D (waktu x channel)
    if data.ndim != 2:
        raise ValueError("CSV harus berbentuk tabel (2D).")

    # Anggap BARIS = waktu, jika jumlah baris lebih kecil dari kolom, transpos
    if data.shape[0] < data.shape[1]:
        data = data.T # Sekarang bentuk: (Timesteps, Channels)

    # ===== CHANNEL HANDLING (Pad/Cut Channels) =====
    if data.shape[1] >= target_ch:
        # Ambil hanya sejumlah channel yang diperlukan
        data = data[:, :target_ch]
    else:
        # Tambahkan padding channel jika kurang
        pad_ch = target_ch - data.shape[1]
        data = np.pad(data, ((0, 0), (0, pad_ch)))

    # ===== TIME HANDLING (Pad/Cut Timesteps) =====
    if data.shape[0] >= target_len:
        # Ambil hanya sejumlah timesteps yang diperlukan
        data = data[:target_len, :]
    else:
        # Tambahkan padding waktu jika kurang
        pad_t = target_len - data.shape[0]
        data = np.pad(data, ((0, pad_t), (0, 0)))
    
    # Normalisasi (penting jika model Anda dilatih dengan data ternormalisasi)
    # data = (data - np.mean(data, axis=0)) / (np.std(data, axis=0) + 1e-6)

    # ===== FINAL SHAPE (BATCH SIZE = 1) =====
    if model_type in ["LSTM", "CNN"]:
        # Bentuk untuk LSTM/CNN 3D: (Batch, Timesteps, Channels) -> (1, 128, 14)
        return data.reshape(1, target_len, target_ch)

    elif model_type == "EEGNet":
        # Bentuk untuk EEGNet 4D: (Batch, Channels, Timesteps, 1) -> (1, 14, 128, 1)
        return data.T.reshape(1, target_ch, target_len, 1)

# =========================
# CSV INPUT
# =========================
st.subheader("📥 Upload Data EEG (CSV)")
uploaded_file = st.file_uploader("Upload CSV EEG", type=["csv"])

# =========================
# PREDICTION
# =========================
if uploaded_file and model is not None:
    df = pd.read_csv(uploaded_file)

    st.write("📄 Preview Data (14 Channel, 128 Timesteps)")
    # Batasi preview untuk 128 baris pertama (jika lebih panjang) dan 14 kolom pertama
    preview_df = df.iloc[:128, :14]
    st.dataframe(preview_df)

    if st.button("🔍 Prediksi"):
        st.markdown("---")
        st.subheader("⏳ Sedang Memproses...")
        
        try:
            # 1. Preprocessing Data
            X_input = preprocess_eeg(df.values, MODEL_TYPE)

            # Debugging shape (Penting untuk mengatasi error 'unknown rank')
            st.info(f"☑️ Data berhasil diproses. Bentuk input model: **{X_input.shape}**")

            # 2. Prediction
            prediction = model.predict(X_input)

            # =========================
            # OUTPUT HANDLING
            # =========================
            labels = ["Eye Closed", "Eye Open"]

            if MODEL_TYPE == "EEGNet":
                # EEGNet biasanya menggunakan Softmax (2 output) dan sparse_categorical_crossentropy
                prob = prediction[0] 
            else:
                # LSTM/CNN biasanya menggunakan Sigmoid (1 output) dan binary_crossentropy
                # Output sigmoid adalah P(class 1). P(class 0) = 1 - P(class 1)
                prob_open = float(prediction[0][0])
                prob = [float(1 - prob_open), prob_open]
            
            prob = np.array(prob)
            
            # Mendapatkan kelas dengan probabilitas tertinggi
            result = labels[int(np.argmax(prob))]

            # =========================
            # RESULT TEXT
            # =========================
            st.subheader("✅ Hasil Prediksi")

            if result == "Eye Open":
                st.success(f"## 👁️ Hasil Klasifikasi: Mata TERBUKA")
            else:
                st.success(f"## 🙈 Hasil Klasifikasi: Mata TERTUTUP")

            # =========================
            # NUMERIC RESULT
            # =========================
            st.write("### Nilai Probabilitas")
            st.write(f"🔹 **{labels[0]}** : **{prob[0]:.4f}**")
            st.write(f"🔹 **{labels[1]}**  : **{prob[1]:.4f}**")

            # =========================
            # VISUALIZATION
            # =========================
            fig, ax = plt.subplots()
            
            # Urutkan dari tertinggi untuk visualisasi yang lebih baik
            sorted_indices = np.argsort(prob)[::-1]
            sorted_prob = prob[sorted_indices]
            sorted_labels = np.array(labels)[sorted_indices]

            bars = ax.bar(sorted_labels, sorted_prob, color=['#1f77b4', '#ff7f0e']) # Warna berbeda untuk Eye Open/Closed

            ax.set_ylim(0, 1)
            ax.set_ylabel("Probability")
            ax.set_title(f"Confidence Prediction ({MODEL_TYPE})")

            # Tambahkan label nilai di atas bar
            for bar, p in zip(bars, sorted_prob):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height(),
                    f"{p:.2f}",
                    ha="center",
                    va="bottom"
                )

            st.pyplot(fig)
            plt.close(fig) # Penting untuk membebaskan memori Matplotlib

        except Exception as e:
            # Memberikan pesan error yang lebih jelas di aplikasi Streamlit
            st.error("---")
            st.error("❌ **TERJADI KESALAHAN SAAT PREDIKSI**")
            st.code(f"Error Detail: {e}", language='python')
            st.warning("Penyebab umum: Bentuk input yang dihasilkan fungsi `preprocess_eeg` tidak sesuai dengan lapisan pertama model. Coba cek ulang *input_shape* model Anda saat pelatihan.")

# =========================
# FOOTER
# =========================
st.markdown("---")
st.caption("EEG Eye State Classification | LSTM • CNN • EEGNet")
st.caption("Pastikan file model `.h5` berada di lokasi yang benar.")