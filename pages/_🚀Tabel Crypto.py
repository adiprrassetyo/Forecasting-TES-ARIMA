import streamlit as st
import pandas as pd
import yfinance as yf

def get_crypto_data(coins):
    # Menyiapkan data untuk ditampilkan dalam tabel
    crypto_list = []
    for coin in coins:
        crypto_info = yf.Ticker(coin)
        if crypto_info:
            crypto_name = crypto_info.info.get('longName', 'N/A')
            crypto_symbol = coin
            
            # Mendapatkan harga koin kripto
            crypto_price = get_crypto_price(crypto_info)
            
            # Tambahkan informasi ke dalam list
            crypto_list.append({
                'Nama': crypto_name,
                'Kode': crypto_symbol,
                'Harga': crypto_price,
            })

    return pd.DataFrame(crypto_list)

def get_crypto_price(crypto_info):
    try:
        # Mendapatkan harga koin kripto dari yfinance
        crypto_data = crypto_info.history(period="1d")
        if not crypto_data.empty:
            return crypto_data['Close'][0]
    except Exception as e:
        print(f"Error getting price for {crypto_info}: {e}")
    return None

def main():
    st.title('Table of Top 5 Cryptocurrency📊')
    st.write("""
        Dalam era digital ini, cryptocurrency telah menjadi salah satu aset investasi yang menarik perhatian banyak pihak. Volatilitas harga yang tinggi membuat prediksi harga cryptocurrency menjadi tantangan yang menarik untuk dieksplorasi.

        Pada penelitian ini, dilakukan analisis perbandingan kinerja dua metode peramalan, yaitu Triple Exponential Smoothing (TES) dan Autoregressive Integrated Moving Average (ARIMA), dalam memprediksi harga lima koin kripto teratas. Koin-koin tersebut adalah Bitcoin (BTC), Ethereum (ETH), Binance Coin (BNB), Solana (SOL), dan Ripple (XRP).

        Berikut adalah daftar harga terkini dari kelima koin kripto tersebut:
    """)

    # Koin yang ingin ditampilkan dalam tabel
    coins = ["BTC-USD", "ETH-USD", "BNB-USD", "SOL-USD", "XRP-USD"]

    # Mendapatkan data koin kripto
    crypto_df = get_crypto_data(coins)

    # Menampilkan tabel koin kripto
    st.table(crypto_df)
    
    st.write("""
        Data di atas menunjukkan harga terkini dari masing-masing koin kripto. Analisis lebih lanjut akan membandingkan kinerja metode TES dan ARIMA dalam memprediksi harga dari koin-koin tersebut. Untuk informasi lebih lengkap, kunjungi [CoinMarketCap](https://coinmarketcap.com/view/metaverse/).
    """)

if __name__ == '__main__':
    main()
