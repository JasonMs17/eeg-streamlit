import streamlit as st

def show_eeg_info():
    """Menampilkan informasi lengkap tentang EEG"""
    
    st.header("Apa itu EEG (Electroencephalography)?")
    
    st.subheader("Definisi")
    st.markdown("""
    **EEG (Electroencephalography)** adalah teknik untuk merekam aktivitas listrik otak menggunakan elektroda yang ditempatkan 
    di permukaan kulit kepala. Aktivitas listrik ini berasal dari jutaan neuron yang berkomunikasi satu sama lain.
    """)

    st.subheader("Bagaimana EEG Bekerja?")
    st.markdown("""
    1. **Penempatan Elektroda**: Elektroda non-invasif ditempatkan di berbagai posisi di kepala
    2. **Pendeteksian Sinyal**: Elektroda mendeteksi perbedaan potensial listrik minimal antara neuron-neuron
    3. **Amplifikasi**: Sinyal sangat lemah (mikrovolts) diamplifikasi agar dapat diukur
    4. **Perekaman**: Data dikonversi menjadi sinyal digital untuk analisis lebih lanjut
    """)
    
    st.subheader("Sistem Penempatan Standar 10-20")
    st.markdown("""
    Sistem 10-20 adalah standar internasional untuk menempatkan elektroda EEG:
    - **Fp**: Frontopolar
    - **F**: Frontal
    - **C**: Central
    - **P**: Parietal
    - **O**: Occipital
    - **T**: Temporal
    - **A**: Auricular (Reference)
    
    Angka ganjil (1,3,5,7) menunjukkan sisi **KIRI**, angka genap (2,4,6,8) menunjukkan sisi **KANAN**
    """)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image("images/area.png", caption="Ilustrasi Channel pada Alat Emotiv EPOC", width='content')
    
    st.subheader("Frekuensi Band EEG")
    st.markdown("""
    | Band | Frekuensi | Keadaan | Deskripsi |
    |------|-----------|--------|-----------|
    | **Delta (δ)** | 0.5-4 Hz | Tidur Dalam | Aktivitas lambat, pemulihan tubuh |
    | **Theta (θ)** | 4-8 Hz | Mengantuk | Meditasi, kreativitas |
    | **Alpha (α)** | 8-12 Hz | Rileks, Mata Tertutup | Relaksasi sadar |
    | **Beta (β)** | 12-30 Hz | Aktif, Terjaga | Konsentrasi, aktivitas mental |
    | **Gamma (γ)** | 30-100 Hz | Fokus Tinggi | Pemrosesan informasi kompleks |
    """)
    
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.image("images/brainwave.jpg", caption="Ilustrasi Gelombang EEG Berdasarkan Frekuensi", width='stretch')
    
    st.subheader("Deteksi Kondisi Mata (Eye State)")
    st.markdown("""
    **Tujuan Aplikasi Ini:**
    
    Aplikasi ini mengklasifikasikan apakah mata seseorang **TERBUKA** atau **TERTUTUP** berdasarkan pola EEG.
    
    **Pola EEG untuk Kondisi Mata:**
    - **Mata Tertutup**: Dominasi Alpha waves (8-12 Hz), sinyal lebih teratur
    - **Mata Terbuka**: Dominasi Beta waves (12-30 Hz), sinyal lebih bervariasi, visual processing aktif
    
    **Dataset:**
    - Diambil dari 117 subjek
    - 14 channel EEG (sesuai sistem 10-20)
    - 128 sample per recording (~1 detik pada sampling rate 128 Hz)
    """)
    
    st.subheader("Model Machine Learning")
    st.markdown("""
    Aplikasi ini menggunakan 3 model deep learning:
    
    1. **LSTM (Long Short-Term Memory)**
       - Cocok untuk data sekuensial time-series
       - Bisa mengingat pattern jangka panjang dalam EEG
    
    2. **CNN (Convolutional Neural Network)**
       - Ekstrak fitur spasial dari EEG channels
       - Efisien untuk pengenalan pola lokal
    
    3. **EEGNet**
       - Arsitektur khusus dirancang untuk data EEG
       - Menggabungkan spatial dan temporal convolutions
    """)
    