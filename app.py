import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import os
from scipy.signal import butter, filtfilt
from eeg_info import show_eeg_info
from tutorial import show_tutorial

# Konfigurasi halaman Streamlit
st.set_page_config(
    page_title="EEG Eye State Classification",
    layout="centered"
)

# Custom styling untuk tampilan yang lebih menarik
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

# Tampilkan logo/gambar header aplikasi
st.image("images/image.png", width='content')

st.markdown("<p style='text-align: center'>Aplikasi untuk mengklasifikasikan kondisi mata (Terbuka/Tertutup) berdasarkan data Electroencephalography.<br><small>Model dilatih menggunakan dataset dari <a href='https://www.kaggle.com/datasets/robikscube/eye-state-classification-eeg-dataset/data' target='_blank'>Kaggle</a> dengan device Emotiv Epoc (14 channel EEG).</small></p>", unsafe_allow_html=True)

# Buat 3 tab utama untuk navigasi aplikasi
tab1, tab2, tab3 = st.tabs(["Klasifikasi", "Tutorial Penggunaan", "Informasi tentang EEG"])

with tab3:
    show_eeg_info()

with tab2:
    show_tutorial()

with tab1:
    # User memilih model mana yang ingin digunakan
    st.subheader("Pilih Model")

    model_choice = st.selectbox(
        "Model:",
        ["LSTM", "CNN"]
    )

    MODEL_MAP = {
        "LSTM": ("LSTM", "model/lstm.h5"),
        "CNN": ("CNN", "model/cnn.h5")
    }

    MODEL_TYPE, MODEL_PATH = MODEL_MAP[model_choice]

    # Tampilkan penjelasan singkat tentang model pilihan
    if model_choice == "LSTM":
        st.info("LSTM (Long Short-Term Memory) adalah model yang baik untuk data sekuensial seperti time series EEG. Model ini mampu mengingat pola jangka panjang dan cocok untuk menangkap sinyal otak yang kompleks.")
    elif model_choice == "CNN":
        st.info("CNN (Convolutional Neural Network) adalah model yang fokus pada fitur lokal dalam data EEG. Model ini cepat dalam pemrosesan dan computationally efficient, serta cocok untuk mendeteksi pola spasial dan temporal.")

    # Load model yang sudah dilatih dari file H5
    @st.cache_resource
    def load_selected_model(path):
        """Muat model dari file, dengan error handling jika file tidak ada atau rusak"""
        if not os.path.exists(path):
            st.error(f"❌ File model tidak ditemukan")
            return None
        try:
            # Load model biasa, tidak perlu custom objects untuk LSTM/CNN standar
            return load_model(path)
        except Exception as e:
            st.error(f"❌ Gagal memuat model {path}. Mungkin ada ketidaksesuaian versi atau arsitektur. Error: {e}")
            return None

    model = load_selected_model(MODEL_PATH)

    # Bandpass filter untuk preprocessing EEG
    def bandpass_filter(data, lowcut=0.5, highcut=45.0, fs=128, order=5):
        """Terapkan bandpass filter pada sinyal dengan padlen yang aman"""
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype="band")
        # Gunakan padlen yang fleksibel sesuai panjang data
        padlen = min(len(data) - 1, 33)
        if padlen < 1:
            padlen = 1
        return filtfilt(b, a, data, padlen=padlen)

    # Fungsi untuk siapkan data EEG agar bisa diprediksi
    def preprocess_eeg(data, model_type, target_len=128, target_ch=14, fs=128):
        """
        Siapkan data EEG menjadi format yang sesuai untuk model.
        - target_len: jumlah timesteps yang diinginkan (default 128)
        - target_ch: jumlah channel yang diinginkan (default 14)
        - fs: sampling rate (default 128 Hz)
        """

        data = np.asarray(data, dtype=np.float32)

        # Harus bentuk 2D (baris=waktu, kolom=channel)
        if data.ndim != 2:
            raise ValueError("CSV harus berbentuk tabel (2D).")

        # Jika baris lebih sedikit dari kolom, anggap perlu di-transpose
        if data.shape[0] < data.shape[1]:
            data = data.T

        # Handle jumlah timesteps DULU - padding jika kurang supaya filter bisa bekerja
        if data.shape[0] < target_len:
            pad_t = target_len - data.shape[0]
            data = np.pad(data, ((0, pad_t), (0, 0)))

        # Handle jumlah channel - potong atau padding sesuai kebutuhan
        if data.shape[1] >= target_ch:
            data = data[:, :target_ch]
        else:
            pad_ch = target_ch - data.shape[1]
            data = np.pad(data, ((0, 0), (0, pad_ch)))

        # Terapkan bandpass filter pada setiap channel SETELAH data sudah cukup panjang
        for ch in range(data.shape[1]):
            data[:, ch] = bandpass_filter(
                data[:, ch],
                lowcut=0.5,
                highcut=45,
                fs=fs,
                order=5
            )

        # Potong ke target_len jika data lebih panjang
        if data.shape[0] > target_len:
            data = data[:target_len, :]

        # Reshape ke bentuk (1, 128, 14) - 1 batch, 128 timesteps, 14 channels
        return data.reshape(1, target_len, target_ch)

    # Section untuk input data (upload atau manual)
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

        # Ekspander dengan contoh data untuk user
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
        
        # Button untuk memproses data
        if st.button("Proses Data"):
            if text_input.strip():
                try:
                    # Parse setiap baris teks menjadi list angka
                    lines = [line.strip() for line in text_input.split('\n') if line.strip()]
                    data_rows = []
                    
                    for line in lines:
                        # Pisahkan dengan koma dan konversi ke float
                        values = [float(x.strip()) for x in line.split(',') if x.strip()]
                        if values:
                            data_rows.append(values)
                    
                    if data_rows:
                        # Buat DataFrame dengan nama kolom
                        max_cols = max(len(row) for row in data_rows)
                        columns = [f'CH{i+1}' for i in range(max_cols)]
                        
                        # Isi baris yang kurang kolom dengan NaN
                        for row in data_rows:
                            while len(row) < max_cols:
                                row.append(np.nan)
                        
                        df = pd.DataFrame(data_rows, columns=columns)
                        st.session_state.text_df = df
                        st.success(f"✅ Data berhasil diparse: {len(data_rows)} timesteps × {max_cols} channels")
                    else:
                        st.warning("⚠️ Tidak ada data yang valid ditemukan")
                        
                except ValueError as e:
                    st.error(f"❌ Error parsing data: {e}")
                    st.info("Pastikan semua nilai adalah angka dan dipisahkan dengan koma")
            else:
                st.warning("⚠️ Silakan masukkan data terlebih dahulu")

    # Ambil data dari session state jika ada
    if "text_df" in st.session_state:
        df = st.session_state.text_df
    else:
        df = None

    # Jalankan prediksi jika sudah ada data dan model berhasil dimuat
    if df is not None and model is not None:

        st.write("Preview Data (14 Channel, 128 Timesteps)")
        # Tampilkan preview data (max 128 baris, 14 kolom)
        preview_df = df.iloc[:128, :14]
        st.dataframe(preview_df)

        # Visualisasi semua channel EEG sebagai line plot
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
                # Siapkan data EEG untuk model
                X_input = preprocess_eeg(df.values, MODEL_TYPE)

                # Jalankan prediksi
                prediction = model.predict(X_input)

                # Decode hasil prediksi
                labels = ["Eye Closed", "Eye Open"]

                if MODEL_TYPE == "EEGNet":
                    # EEGNet pake softmax, langsung ambil dua output
                    prob = prediction[0] 
                else:
                    # LSTM/CNN pake sigmoid, output 1 nilai. Hitung P(closed) dan P(open)
                    prob_open = float(prediction[0][0])
                    prob = [float(1 - prob_open), prob_open]
                
                prob = np.array(prob)
                
                # Ambil kelas dengan probabilitas tertinggi
                result = labels[int(np.argmax(prob))]

                # Tampilkan hasil prediksi
                st.subheader("Hasil Prediksi")

                if result == "Eye Open":
                    st.success(f"### Mata Terbuka")
                else:
                    st.success(f"### Mata Tertutup")

                # Tampilkan nilai probabilitas untuk setiap kelas
                st.write("### Nilai Probabilitas")
                st.write(f"🔹 **{labels[0]}** : **{prob[0]:.4f}**")
                st.write(f"🔹 **{labels[1]}**  : **{prob[1]:.4f}**")

                # Buat bar chart untuk visualisasi probabilitas
                fig, ax = plt.subplots()
                
                # Urutkan dari probabilitas tertinggi ke terendah
                sorted_indices = np.argsort(prob)[::-1]
                sorted_prob = prob[sorted_indices]
                sorted_labels = np.array(labels)[sorted_indices]

                bars = ax.bar(sorted_labels, sorted_prob, color=['#1f77b4', '#ff7f0e'])

                ax.set_ylim(0, 1)
                ax.set_ylabel("Probability")
                ax.set_title(f"Confidence Prediction ({MODEL_TYPE})")

                # Tambah label nilai di atas setiap bar
                for bar, p in zip(bars, sorted_prob):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height(),
                        f"{p:.2f}",
                        ha="center",
                        va="bottom"
                    )

                st.pyplot(fig)
                plt.close(fig)

            except Exception as e:
                # Tampilkan error detail untuk debugging
                st.error("---")
                st.error("❌ **TERJADI KESALAHAN SAAT PREDIKSI**")
                st.code(f"Error Detail: {e}", language='python')
                st.warning("Kemungkinan penyebab: Bentuk input tidak sesuai dengan input layer model. Cek ulang input_shape saat training model.")

    # Footer
    st.markdown("---")
    st.caption("EEG Eye State Classification | LSTM • CNN • EEGNet")
    st.caption("Teknik Informatika - Universitas Padjadjaran © 2025")
