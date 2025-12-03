import streamlit as st
from catboost import CatBoostClassifier
import numpy as np

# =========================================
# LOAD MODEL CATBOOST (FORMAT .cbm)
# =========================================
model = CatBoostClassifier()
model.load_model("model_catboost_stunting.cbm")

st.title("Aplikasi Prediksi Stunting")
st.write("Masukkan data anak lalu klik tombol prediksi untuk mengetahui status stunting.")

# =========================================
# INPUT FORM
# =========================================

umur = st.number_input("Umur (bulan)", 0, 60, 24)
berat = st.number_input("Berat Badan (kg)", 0.0, 30.0, 10.0)
tinggi = st.number_input("Tinggi Badan (cm)", 0.0, 120.0, 80.0)

jenis_kelamin = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
asi = st.selectbox("ASI Eksklusif", ["Ya", "Tidak"])
pendidikan_ibu = st.selectbox("Pendidikan Ibu", ["SD", "SMP", "SMA", "Kuliah"])
pendapatan = st.number_input("Pendapatan Orangtua (Rp)", 0, 20000000, 2000000)

# =========================================
# ENCODING FITUR (HARUS SAMA DENGAN NOTEBOOK)
# =========================================

jk_enc = 1 if jenis_kelamin == "Laki-laki" else 0
asi_enc = 1 if asi == "Ya" else 0

pendidikan_map = {
    "SD": 0,
    "SMP": 1,
    "SMA": 2,
    "Kuliah": 3
}
pendidikan_enc = pendidikan_map[pendidikan_ibu]

# =========================================
# PREDIKSI
# =========================================

if st.button("Prediksi Stunting"):
    # Urutan fitur WAJIB sama dengan training CatBoost kamu
    fitur = np.array([
        umur,
        berat,
        tinggi,
        jk_enc,
        asi_enc,
        pendidikan_enc,
        pendapatan
    ]).reshape(1, -1)

    pred = model.predict(fitur)[0]

    hasil = "TIDAK STUNTING" if pred == 0 else "STUNTING"

    st.subheader("Hasil Prediksi")
    st.write(f"Status Anak: **{hasil}**")
