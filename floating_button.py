import streamlit as st

@st.dialog("Tutorial Cara Penggunaan", width="medium")
def show_how_to_use_dialog():
    """Dialog untuk menampilkan panduan cara menggunakan aplikasi."""
    st.markdown("""
    **1️⃣ Pilih Model**
    - Pilih salah satu model dari dropdown: LSTM, CNN, atau EEGNet
    - Setiap model memiliki kinerja yang berbeda
    
    **2️⃣ Persiapkan Data EEG**
    - Pastikan data dalam format CSV
    - Data harus berisi 14 channel (kolom) EEG
    - Data minimal 128 timesteps (baris)
    - Format: baris = waktu, kolom = channel
    
    **3️⃣ Upload File CSV**
    - Klik tombol "Upload CSV EEG"
    - Pilih file CSV dari komputer Anda
    - Aplikasi akan menampilkan preview data
    
    **4️⃣ Lakukan Prediksi**
    - Klik tombol "🔍 Prediksi"
    - Aplikasi akan memproses data Anda
    - Hasil akan ditampilkan dengan probabilitas
    
    **5️⃣ Interpretasi Hasil**
    - **Eye Open (Mata Terbuka)**: Nilai probabilitas lebih tinggi di kolom "Eye Open"
    - **Eye Closed (Mata Tertutup)**: Nilai probabilitas lebih tinggi di kolom "Eye Closed"
    - Semakin mendekati 1.00, semakin yakin prediksi
    
    **Format CSV:**
    ```
    CH1, CH2, CH3, ..., CH14
    val1, val2, val3, ..., val14
    ```
    
    **Tips:**
    - ✅ Gunakan data yang sudah dipreproses
    - ✅ Pastikan sampling rate konsisten
    - ⚠️ Hindari data dengan noise yang tinggi
    """)

def show_floating_button():
    """Menampilkan floating button dengan modal popup native Streamlit."""
    
    # Initialize session state untuk modal
    if "show_how_to_use" not in st.session_state:
        st.session_state.show_how_to_use = False
    
    # Button to trigger dialog
    if st.button("Lihat Tutorial Cara Penggunaan", key="how_to_use_btn", use_container_width=True):
        show_how_to_use_dialog()
    
    # Hide button with CSS
    st.markdown("""
        <style>
        [data-testid="stHorizontalBlock"] + div {
            display: none;
        }
        </style>
    """, unsafe_allow_html=True)
