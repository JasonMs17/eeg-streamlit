import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import os
from eeg_info import show_eeg_info
from tutorial import show_tutorial

# =========================
# CONFIG
# =========================
st.set_page_config(
    page_title="EEG Eye State Classification",
    layout="centered"
)

# =========================
# PAGE STYLING
# =========================
st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 25%, #f093fb 50%, #4facfe 75%, #00f2fe 100%);
        background-size: 400% 400%;
        animation: gradient 15s ease infinite;
        min-height: 100vh;
    }
    
    @keyframes gradient {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    
    [data-testid="stMainBlockContainer"] {
        background-color: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        padding: 30px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
        margin-top: 80px;
        margin-bottom: 20px;
    }
    <style>
    /* Buat tab list full width */
    .stTabs [data-baseweb="tab-list"] {
        display: flex;
        justify-content: space-between;  /* Membagi rata tab */
        width: 100%;
    }
    /* Ganti ukuran teks tab */
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 16px; /* Adjust the font size as needed (e.g., 1.5rem, 24px, large) */
    }
    </style>
    """,
    unsafe_allow_html=True

    
)

# Display Image
st.image("images/image.png", width='content')

st.markdown("<p style='text-align: center'>Aplikasi untuk mengklasifikasikan kondisi mata (Terbuka/Tertutup) berdasarkan data Electroencephalography.<br><small>Model dilatih menggunakan dataset dari <a href='https://www.kaggle.com/datasets/robikscube/eye-state-classification-eeg-dataset/data' target='_blank'>Kaggle</a> dengan device Emotiv Epoc (14 channel EEG).</small></p>", unsafe_allow_html=True)

# =========================
# TABS
# =========================
tab1, tab2, tab3 = st.tabs(["Klasifikasi", "Tutorial Penggunaan", "Informasi tentang EEG"])

with tab3:
    show_eeg_info()

with tab2:
    show_tutorial()

with tab1:
    # =========================
    # MODEL SELECTION
    # =========================
    st.subheader("Pilih Model")

    model_choice = st.selectbox(
        "Model:",
        ["LSTM", "CNN", "EEGNet"]
    )

    MODEL_MAP = {
        "LSTM": ("LSTM", "model/lstm.h5"),
        "CNN": ("CNN", "model/cnn.h5"),
        "EEGNet": ("EEGNet", "model/eegnet.h5")
    }

    MODEL_TYPE, MODEL_PATH = MODEL_MAP[model_choice]

    # =========================
    # MODEL EXPLANATION
    # =========================
    if model_choice == "LSTM":
        st.info("LSTM (Long Short-Term Memory) adalah model yang baik untuk data sekuensial seperti time series EEG. Model ini mampu mengingat pola jangka panjang dan cocok untuk menangkap sinyal otak yang kompleks.")
    elif model_choice == "CNN":
        st.info("CNN (Convolutional Neural Network) adalah model yang fokus pada fitur lokal dalam data EEG. Model ini cepat dalam pemrosesan dan computationally efficient, serta cocok untuk mendeteksi pola spasial dan temporal.")
    elif model_choice == "EEGNet":
        st.info("EEGNet adalah model khusus yang didesain untuk data EEG. Model ini menggabungkan keunggulan CNN dan filter yang dioptimalkan untuk EEG, sehingga lebih akurat dan efisien untuk klasifikasi sinyal otak.")

    # =========================
    # LOAD MODEL
    # =========================
    @st.cache_resource
    def load_selected_model(path):
        """Memuat model Keras/TensorFlow dan menangani error jika file tidak ditemukan."""
        if not os.path.exists(path):
            st.error(f"❌ File model tidak ditemukan")
            return None
        try:
            # Menambahkan custom_objects jika ada lapisan khusus, misalnya Lambda
            # Tetapi untuk LSTM, CNN, Dense biasa, ini tidak diperlukan.
            return load_model(path)
        except Exception as e:
            st.error(f"❌ Gagal memuat model {path}. Mungkin ada ketidaksesuaian versi atau arsitektur. Error: {e}")
            return None

    model = load_selected_model(MODEL_PATH)

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
    st.subheader("Input Data EEG")
    
    input_method = st.radio(
        "Pilih metode input:",
        ["Upload File CSV", "Input Manual (Text)"],
        horizontal=True
    )
    
    df = None
    if input_method == "Upload File CSV":
        uploaded_file = st.file_uploader("Upload CSV EEG", type=["csv"])
        if uploaded_file:
            df = pd.read_csv(uploaded_file)
    else:
        st.write("**Format:** Masukkan data dengan format: `val1,val2,val3,...` (satu baris per timestep)")
        st.write("**Contoh:** `1.2,3.4,5.6,7.8,9.0,1.1,2.2,3.3,4.4,5.5,6.6,7.7,8.8,9.9`")
        
        st.info("""
        **Pastikan urutan data yang dimasukkan sebagai berikut:**
        ```
        AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8, AF4
        ```
        """)

        # Sample Data Expander
        with st.expander("📋 Sample Data EEG (Klik untuk melihat contoh data yang bisa di-copy)"):
            st.markdown("""
            **Berikut adalah contoh data EEG yang bisa Anda gunakan untuk testing:**
            
            ```text
            4329.23,4009.23,4289.23,4148.21,4350.26,4586.15,4096.92,4641.03,4222.05,4238.46,4211.28,4280.51,4635.9,4393.85
            4324.62,4004.62,4293.85,4148.72,4342.05,4586.67,4097.44,4638.97,4210.77,4226.67,4207.69,4279.49,4632.82,4384.1
            4327.69,4006.67,4295.38,4156.41,4336.92,4583.59,4096.92,4630.26,4207.69,4222.05,4206.67,4282.05,4628.72,4389.23
            4328.72,4011.79,4296.41,4155.9,4343.59,4582.56,4097.44,4630.77,4217.44,4235.38,4210.77,4287.69,4632.31,4396.41
            4326.15,4011.79,4292.31,4151.28,4347.69,4586.67,4095.9,4627.69,4210.77,4244.1,4212.82,4288.21,4632.82,4398.46
            ```
            """)
        
        
        text_input = st.text_area(
            "Masukkan data EEG (satu baris = satu timestep):",
            height=150,
            placeholder="1.2,3.4,5.6,7.8,9.0,1.1,2.2,3.3,4.4,5.5,6.6,7.7,8.8,9.9\n2.1,4.3,6.5,8.7,0.9,2.1,3.2,4.3,5.4,6.5,7.6,8.7,9.8,0.1\n..."
        )
        
        if text_input.strip():
            try:
                # Parse text input
                lines = [line.strip() for line in text_input.split('\n') if line.strip()]
                data_rows = []
                
                for line in lines:
                    # Split by comma and convert to float
                    values = [float(x.strip()) for x in line.split(',') if x.strip()]
                    if values:  # Only add non-empty rows
                        data_rows.append(values)
                
                if data_rows:
                    # Create DataFrame with appropriate column names
                    max_cols = max(len(row) for row in data_rows)
                    columns = [f'CH{i+1}' for i in range(max_cols)]
                    
                    # Pad shorter rows with NaN
                    for row in data_rows:
                        while len(row) < max_cols:
                            row.append(np.nan)
                    
                    df = pd.DataFrame(data_rows, columns=columns)
                    st.success(f"✅ Data berhasil diparse: {len(data_rows)} timesteps × {max_cols} channels")
                else:
                    st.warning("⚠️ Tidak ada data yang valid ditemukan")
                    
            except ValueError as e:
                st.error(f"❌ Error parsing data: {e}")
                st.info("Pastikan semua nilai adalah angka dan dipisahkan dengan koma")

    # =========================
    # PREDICTION
    # =========================
    if df is not None and model is not None:

        st.write("Preview Data (14 Channel, 128 Timesteps)")
        # Batasi preview untuk 128 baris pertama (jika lebih panjang) dan 14 kolom pertama
        preview_df = df.iloc[:128, :14]
        st.dataframe(preview_df)

        # =========================
        # EEG SIGNAL VISUALIZATION
        # =========================
        st.subheader("Visualisasi Sinyal EEG")
        fig, ax = plt.subplots(figsize=(12, 6))
        
        for i in range(min(14, preview_df.shape[1])):
            ax.plot(preview_df.index, preview_df.iloc[:, i], label=f'CH{i+1}', alpha=0.7)
        
        ax.set_xlabel('Timesteps')
        ax.set_ylabel('Amplitude')
        ax.set_title('EEG Signal Channels')
        ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1))
        ax.grid(True, alpha=0.3)
        
        st.pyplot(fig)
        plt.close(fig)

        if st.button("🔍 Prediksi"):
            st.markdown("---")
            
            try:
                # 1. Preprocessing Data
                X_input = preprocess_eeg(df.values, MODEL_TYPE)

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
                st.subheader("Hasil Prediksi")

                if result == "Eye Open":
                    st.success(f"### Mata Terbuka")
                else:
                    st.success(f"### Mata Tertutup")

                # =========================
                # NUMERIC RESULT
                # =========================
                st.write("### Nilai Probabilitas")
                st.write(f"🔹 **{labels[0]}** : **{prob[0]:.4f}**")
                st.write(f"🔹 **{labels[1]}**  : **{prob[1]:.4f}**")

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
    st.caption("Teknik Informatika - Universitas Padjadjaran © 2025")
