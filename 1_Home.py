import streamlit as st

# New Line
def new_line(n=1):
    for i in range(n):
        st.write("\n")

def main():
    # Dataframe selection
    st.markdown("<h1 align='center'> <b>Analisis Perbandingan Kinerja Metode Triple Exponential Smoothing dan Autoregressive Integrated Moving Average pada Peramalan Harga Cryptocurrency</b></h1>", unsafe_allow_html=True)
    new_line(1)
    st.markdown("Selamat datang! Aplikasi ini dirancang untuk menganalisis kinerja metode Triple Exponential Smoothing (TES) dan Autoregressive Integrated Moving Average (ARIMA) dalam memprediksi harga cryptocurrency. Sistem ini diharapkan dapat menjadi landasan untuk strategi perdagangan yang lebih cerdas dan keputusan investasi yang lebih baik dalam cryptocurrency.", unsafe_allow_html=True)
    
    st.divider()
    
    # Overview
    new_line()
    st.markdown("<h2 style='text-align: center;'>🗺️ Gambaran Umum</h2>", unsafe_allow_html=True)
    new_line()
    
    st.markdown("""
    Dalam proses membangun model prediksi, ada serangkaian langkah yang harus diikuti. Berikut ini adalah langkah-langkah utama dalam proses Machine Learning:
    
    - **📦 Pengumpulan Data**: Proses pengumpulan data dari berbagai sumber seperti pustaka yfinance, file CSV, database, API, dll.<br> <br>
    - **🧹 Pembersihan Data**: Proses pembersihan data dengan menghapus duplikasi, menangani nilai yang hilang, dll. Langkah ini sangat penting karena seringkali data tidak bersih dan mengandung banyak nilai yang hilang dan outlier.<br> <br>
    - **⚙️ Pra-pemrosesan Data**: Proses mengubah data ke dalam format yang sesuai untuk analisis. Ini termasuk menangani fitur kategorikal, fitur numerik, penskalaan dan transformasi, dll.<br> <br>
    - **💡 Rekayasa Fitur**: Proses yang memanipulasi fitur itu sendiri. Terdiri dari beberapa langkah seperti ekstraksi fitur, transformasi fitur, dan pemilihan fitur.<br> <br>
    - **✂️ Pembagian Data**: Proses membagi data menjadi set pelatihan dan pengujian. Set pelatihan digunakan untuk melatih model dan set pengujian digunakan untuk mengevaluasi model.<br> <br>
    - **🧠 Membangun Model Pembelajaran Mesin**: Model yang digunakan pada aplikasi ini adalah Triple Exponential Smoothing (TES) dan Autoregressive Integrated Moving Average (ARIMA).<br> <br>
    - **⚖️ Evaluasi Model Pembelajaran Mesin**: Proses mengevaluasi model prediksi dengan menggunakan metrik seperti Mean Absolute Percentage Error (MAPE) dan Root Mean Squared Error (RMSE).<br> <br>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    Pada bagian membangun model, pengguna memasukkan nilai masing-masing hiperparameter untuk TES dan ARIMA. Hiperparameter adalah variabel yang secara signifikan mempengaruhi proses pelatihan model:
    
    - **Triple Exponential Smoothing (TES)**:
        - **α (Alpha)**: Tingkat pemulusan untuk level komponen (nilai antara 0 dan 1).<br> <br>
        - **β (Beta)**: Tingkat pemulusan untuk tren komponen (nilai antara 0 dan 1).<br> <br>
        - **γ (Gamma)**: Tingkat pemulusan untuk komponen musiman (nilai antara 0 dan 1).<br> <br>
        - **Musim**: Periode musiman, jumlah periode dalam satu siklus musiman.<br> <br>
    
    - **Autoregressive Integrated Moving Average (ARIMA)**:
        - **p (Auto-regressive order)**: Jumlah lag observasi yang dimasukkan ke dalam model (nilai integer).<br> <br>
        - **d (Differencing order)**: Jumlah kali perbedaan yang diperlukan untuk menjadikan seri waktu stasioner (nilai integer).<br> <br>
        - **q (Moving average order)**: Ukuran jendela rata-rata bergerak untuk model (nilai integer).<br> <br>
    """, unsafe_allow_html=True)
    new_line()
    
    # Source Code
    new_line()
    st.header("📂 Source Code")
    st.markdown("Untuk pengembangan aplikasi ini, source code tersedia di [**GitHub**](https://github.com/hayuraaa/Forecasting-LSTM-GRU.git). Jangan ragu untuk berkontribusi, memberikan feedback, atau menyesuaikan aplikasi agar sesuai dengan kebutuhan Anda.", unsafe_allow_html=True)
    new_line()
    
    # Contributors
    st.header("👤 Kontributor")
    st.markdown("Aplikasi dibuat untuk kebutuhan tugas akhir/skripsi, **Adi Prasetyo** (200170002) .", unsafe_allow_html=True)
    new_line()
    
    st.markdown("""Jika Anda memiliki pertanyaan atau saran, jangan ragu untuk menghubungi **adiprstyo11@gmail.com**. Kami siap membantu!

**Connect with us on social media:** 

<a href="https://www.linkedin.com/in/adiprrassetyo/" target="_blank">
  <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQe0adDoUGWVD3jGzfT8grK5Uhw0dLXSk3OWJwZaXI-t95suRZQ-wPF7-Az6KurXDVktV4&usqp=CAU" alt="LinkedIn" width="80" height="80" style="border-radius: 25%;">
</a>       
<a href="https://www.instagram.com/adiprrassetyo/" target="_blank">
  <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/e/e7/Instagram_logo_2016.svg/2048px-Instagram_logo_2016.svg.png" alt="Instagram" width="80" height="80" style="border-radius: 25%;">
</a>       
<a href="https://github.com/adiprrassetyo/" target="_blank">
  <img src="https://seeklogo.com/images/G/github-logo-5F384D0265-seeklogo.com.png" alt="GitHub" width="80" height="80" style="border-radius: 25%;">
</a>

<br>
<br>

Kami menantikan kabar dari Anda dan mendukung perjalanan Anda dalam pembelajaran mesin!
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
