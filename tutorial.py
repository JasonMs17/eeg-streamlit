import streamlit as st

def show_tutorial():
    """Menampilkan tutorial lengkap cara menggunakan aplikasi."""

    st.header("Tutorial Penggunaan Aplikasi")

    st.markdown("""
    Selamat datang di aplikasi **EEG Eye State Classification**! Panduan lengkap ini akan membantu Anda menggunakan aplikasi ini dengan mudah.
    """)

    # Langkah 1: Pilih Model
    st.subheader("Langkah 1: Pilih Model")
    st.markdown("""
    **Tujuan:** Memilih model machine learning yang akan digunakan untuk klasifikasi.

    **Cara:**
    Pada dropdown **"Model:"**, pilih salah satu opsi:
       - **LSTM**
       - **CNN**

    **Catatan:** Model EEGNet tidak dimasukkan karena hasil akurasi kurang memuaskan.
    """)

    # Langkah 2: Input Data
    st.subheader("Langkah 2: Input Data EEG")
    st.markdown("""
    **Tujuan:** Memasukkan data EEG yang akan diklasifikasikan.

    **Opsi Input:**

    #### A. Upload File CSV
    1. Klik **"Upload File CSV"**
    2. Pilih file CSV dari komputer Anda
    3. Pastikan format data benar

    #### B. Input Manual (Text)
    1. Klik **"Input Manual (Text)"**
    2. Masukkan data dalam format: `val1,val2,val3,...`
    3. Satu baris = satu timestep
    4. Klik expander **"Sample Data EEG"** untuk contoh data
    """)

    # Format Data
    st.subheader("Format Data yang Benar")
    st.markdown("""
    **Spesifikasi Data:**
    - **14 Channel EEG** (wajib)
    - **Minimal 128 timesteps** (baris)
    - **Format:** CSV atau text dengan separator koma

    **Urutan Channel (Wajib):**
    ```
    AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8, AF4
    ```

    **Contoh Data:**
    ```text
    4329.23,4009.23,4289.23,4148.21,4350.26,4586.15,4096.92,4641.03,4222.05,4238.46,4211.28,4280.51,4635.9,4393.85
    4324.62,4004.62,4293.85,4148.72,4342.05,4586.67,4097.44,4638.97,4210.77,4226.67,4207.69,4279.49,4632.82,4384.1
    ```
    """)

    # Langkah 3: Prediksi
    st.subheader("Langkah 3: Lakukan Prediksi")
    st.markdown("""
    **Tujuan:** Mengklasifikasikan kondisi mata berdasarkan data EEG.

    **Cara:**
    1. Pastikan data sudah dimuat (akan muncul preview)
    2. Klik tombol **"🔍 Prediksi"**
    3. Tunggu proses selesai

    **Yang Terjadi:**
    - Data akan diproses dan dinormalisasi
    - Model akan memprediksi probabilitas
    - Hasil akan ditampilkan dengan visualisasi
    """)

    # Langkah 4: Interpretasi Hasil
    st.subheader("Langkah 4: Interpretasi Hasil")
    st.markdown("""
    **Memahami Output:**

    #### Hasil Klasifikasi
    - **Mata TERBUKA**: Probabilitas "Eye Open" > "Eye Closed"
    - **Mata TERTUTUP**: Probabilitas "Eye Closed" > "Eye Open"

    #### Nilai Probabilitas
    - **0.0 - 0.5**: Prediksi tidak yakin
    - **0.5 - 0.8**: Prediksi cukup yakin
    - **0.8 - 1.0**: Prediksi sangat yakin

    #### Visualisasi Chart
    - Bar chart menunjukkan confidence level
    - Nilai di atas bar menunjukkan probabilitas eksak
    """)

    