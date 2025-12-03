import streamlit as st
import pickle
import numpy as np

# =================================
# LOAD MODEL & SCALER
# =================================
model = pickle.load(open("model_xgb_stunting.pkl", "rb"))
scaler = pickle.load(open("scaler_stunting.pkl", "rb"))

st.title("Prediksi Stunting Menggunakan XGBoost")
st.write("Masukkan data anak untuk memprediksi status stunting")

# =================================
# INPUT FORM
# =================================

umur = st.number_input("Umur (bulan)", 0, 60, 24)
berat = st.number_input("Berat Badan (kg)", 0.0, 30.0, 10.0)
tinggi = st.number_input("Tinggi Badan (cm)", 0.0, 120.0, 80.0)

jenis_kelamin = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
asi = st.selectbox("ASI Eksklusif", ["Ya", "Tidak"])
pendidikan_ibu = st.selectbox("Pendidikan Ibu", ["SD", "SMP", "SMA", "Kuliah"])
penghasilan = st.number_input("Pendapatan Orang Tua (Rp)", 0, 10000000, 2000000)

# Encoding manual (sesuaikan dengan notebook-mu!)
jenis_kelamin_enc = 1 if jenis_kelamin == "Laki-laki" else 0
asi_enc = 1 if asi == "Ya" else 0

pendidikan_map = {
    "SD": 0,
    "SMP": 1,
    "SMA": 2,
    "Kuliah": 3
}
pendidikan_enc = pendidikan_map[pendidikan_ibu]

# =================================
# PREDIKSI
# =================================

if st.button("Prediksi"):
    # Urutan fitur harus SAMA dengan notebook-mu
    features = np.array([
        umur,
        berat,
        tinggi,
        jenis_kelamin_enc,
        asi_enc,
        pendidikan_enc,
        penghasilan
    ]).reshape(1, -1)

    # Scaling sesuai training
    scaled = scaler.transform(features)

    pred = model.predict(scaled)[0]

    label = "Tidak Stunting" if pred == 0 else "Stunting"

    st.subheader("Hasil Prediksi")
    st.write(f"**Status: {label}**")
