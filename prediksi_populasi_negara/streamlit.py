import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt

# Load model dan fitur
model = joblib.load("model_populasi.pkl")
features = joblib.load("model_features.pkl")

st.title("Prediksi Populasi Negara")
st.write("Masukkan negara dan rentang tahun untuk memprediksi populasi.")

# Input tahun
tahun_awal = st.number_input("Masukkan Tahun Awal", min_value=1950, max_value=2100, value=2024, step=1)
tahun_akhir = st.number_input("Masukkan Tahun Akhir", min_value=1950, max_value=2100, value=2030, step=1)

# Input negara
negara_list = [f.replace("Nama Negara_", "") for f in features if "Nama Negara_" in f]
negara = st.selectbox("Pilih Negara", negara_list)

if st.button("Prediksi"):
    tahun_range = list(range(tahun_awal, tahun_akhir + 1))
    prediksi_populasi = []

    for tahun in tahun_range:
        input_data = pd.DataFrame(np.zeros((1, len(features))), columns=features)
        input_data["Tahun"] = tahun

        negara_column = f"Nama Negara_{negara}"
        if negara_column in input_data.columns:
            input_data[negara_column] = 1

        prediksi = model.predict(input_data)[0]
        prediksi_populasi.append(prediksi)

    # Konversi ke juta jiwa
    prediksi_populasi_juta = [p / 1_000_000 for p in prediksi_populasi]

    # Buat dataframe hasil prediksi
    hasil_df = pd.DataFrame({"Tahun": tahun_range, "Prediksi Populasi (Juta Jiwa)": prediksi_populasi_juta})
    st.write(hasil_df)

    # Warna batang
    warna_batang = ["gold", "pink", "hotpink", "red", "orangered", "purple"]

    # Pastikan jumlah warna sesuai jumlah tahun yang dipilih
    while len(warna_batang) < len(tahun_range):
        warna_batang += warna_batang  # Duplikasi warna jika tidak cukup
    warna_batang = warna_batang[: len(tahun_range)]  # Potong sesuai jumlah tahun

    # Buat diagram batang
    fig, ax = plt.subplots()
    ax.bar(tahun_range, prediksi_populasi_juta, color=warna_batang, edgecolor="black")

    # Format sumbu Y supaya dalam juta jiwa tanpa notasi ilmiah
    ax.ticklabel_format(style="plain", axis="y")  # Hapus notasi 1e8
    ax.set_yticks(np.arange(0, max(prediksi_populasi_juta) + 50, 50))  # Atur skala lebih rapi
    
    ax.set_xticks(tahun_range)
    ax.set_xticklabels(tahun_range, rotation=0)

    # Tambahkan label dan judul
    ax.set_xlabel("Tahun", fontsize=12)
    ax.set_ylabel("Jumlah Penduduk (Juta Jiwa)", fontsize=12)
    ax.set_title(f"Prediksi Populasi {negara} ({tahun_awal}-{tahun_akhir})", fontsize=14)
    ax.grid(axis="y", linestyle="--", alpha=0.7)

    # Tampilkan angka di atas batang
    for i, v in enumerate(prediksi_populasi_juta):
        ax.text(tahun_range[i], v + 2, f"{v:.1f}", ha="center", fontsize=10, fontweight="bold", color="black")

    # Tampilkan diagram batang di Streamlit
    st.pyplot(fig)
