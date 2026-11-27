import requests
import streamlit as st

API_URL = "http://127.0.0.1:8000/predict"

FEATURES = {
    "alcohol": 13.0, "malic_acid": 2.3, "ash": 2.4, "alcalinity_of_ash": 19.5,
    "magnesium": 99.0, "total_phenols": 2.3, "flavanoids": 2.0,
    "nonflavanoid_phenols": 0.36, "proanthocyanins": 1.6,
    "color_intensity": 5.0, "hue": 1.0,
    "od280_od315_of_diluted_wines": 2.6, "proline": 750.0,
}

st.title("Wine Classifier")
st.write("Введите характеристики вина, чтобы получить класс.")

inputs = {}
cols = st.columns(2)
for i, (name, default) in enumerate(FEATURES.items()):
    inputs[name] = cols[i % 2].number_input(name, value=default)

if st.button("Predict"):
    try:
        r = requests.post(API_URL, json=inputs, timeout=5)
        r.raise_for_status()
        result = r.json()
        st.success(f"Класс: {result['class_name']} (prediction = {result['prediction']})")
    except Exception as e:
        st.error(f"Ошибка запроса к API: {e}")