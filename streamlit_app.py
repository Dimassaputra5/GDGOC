import streamlit as st
import joblib as jb
import pandas as pd
import numpy as np
import time
import re
from streamlit_option_menu import option_menu

df = pd.read_csv('df.csv')
rf_model = jb.load('rf__model.sav')

st.title("Prediksi Harga Handphone")
st.write('Data set yang digunakan: [Here](https://drive.google.com/file/d/1BEzYGaWuiFmXAXtrQLVbFmqCCJ9-GzHK/view?usp=sharing)')


# UI
st.markdown("Masukkan input data.")

col1, col2= st.columns(2)

with col1:
    Battery = st.number_input('Masukan Besar Battery(Mah):', 1900, 7000, value=1900, step=10)
    st.write(f'Minimal Battery {df["Battery"].min()}, Maksimal Battery: {7000}')
    Rating = st.number_input('Masukan Rating:', max_value=5.0, value=0.00, step=0.25, format='%.2f')
    st.write(f'Minimal Rating:{1}, Maksimal Rating: {5}')
    Display = st.number_input('Masukan Ukuran Handphone():', 4.5, 10.0, value=4.5, step=0.1)
    st.write(f'Minimal Display {df["Display"].min()}, Maksimal Display: {10}')

with col2:
    Ram = st.slider('Masukan angka Ram(GB):', df['Ram'].min(), df['Ram'].max())
    Spec_score = st.slider('Masukan Spec Skor', df['Spec_score'].min(), df['Spec_score'].max())
    Inbuilt_memory = st.radio('Masukan Ukuran Internal memory(GB):', df['Inbuilt_memory'].unique())       

InbuiltMemory_SpecScore = Inbuilt_memory * Spec_score
InbuiltMemory_Ram = Inbuilt_memory * Ram
Ram_SpecScore = Ram * Spec_score
Ram_squared = Ram **2
Spec_score_squared = Spec_score **2
rear_count = df['rear_count'].mode()
total_rear_mp = df['total_rear_mp'].mode()
front_mp = df['front_mp'].mode()
screen_heigt = df['screen_heigt'].mode()
screen_width = df['screen_width'].mode()
df = df.select_dtypes(include='object')

# Prediksi
if st.button("Prediksi"):
    input_data = np.array([[Battery, Rating, Display, Ram, Spec_score,
                            Inbuilt_memory, InbuiltMemory_SpecScore, InbuiltMemory_Ram,
                            Ram_SpecScore, Ram_squared, Spec_score_squared, rear_count, 
                            total_rear_mp, front_mp, screen_width, screen_heigt ]])
    hasil = rf_model.predict(input_data)
    st.success(f"Perkiraan Harga: {hasil[0]:,.2f}")
