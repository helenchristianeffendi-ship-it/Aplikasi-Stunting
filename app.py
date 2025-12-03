# STREAMLIT APP - LOGISTIC REGRESSION
import streamlit as st
import numpy as np
import pickle

# Load model & scaler
model = pickle.load(open("logreg_model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("Prediksi Diabetes - Logistic Regression")

st.write("Masukkan data pasien untuk mengetahui apakah terindikasi diabetes atau tidak.")

preg = st.number_input("Jumlah Kehamilan", 0, 20)
glu = st.number_input("Glukosa", 0, 300)
bp = st.number_input("Tekanan Darah", 0, 200)
skin = st.number_input("Skin Thickness", 0, 100)
ins = st.number_input("Insulin", 0, 900)
bmi = st.number_input("BMI", 0.0, 70.0)
dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0)
age = st.number_input("Usia", 1, 120)

if st.button("Prediksi"):
    data = np.array([[preg, glu, bp, skin, ins, bmi, dpf, age]])
    scaled = scaler.transform(data)
    pred = model.predict(scaled)[0]

    if pred == 1:
        st.error("⚠️ Hasil: Pasien terindikasi Diabetes")
    else:
        st.success("✔️ Hasil: Pasien TIDAK terindikasi Diabetes")
