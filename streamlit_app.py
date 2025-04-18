import streamlit as st
import pickle as pkl
import pandas as pd
import numpy as np
import time
import sklearn
df1 = pd.read_csv('mobile phone price prediction.csv')
df = pd.read_csv('df.csv')
with open ('rf_model.pkl', 'rb') as file:
    rf_model = pkl.load(file)

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
    Spec_score = st.slider('Masukan Spec Skor', df1['Spec_score'].min(), df1['Spec_score'].max())
    Inbuilt_memory = st.radio('Masukan Ukuran Internal memory(GB):', df['Inbuilt_memory'].unique())       

# Debugging: Periksa jumlah fitur yang diharapkan oleh model
st.write(f"Jumlah fitur yang diharapkan oleh model: {rf_model.n_features_in_}")

InbuiltMemory_SpecScore = Inbuilt_memory * Spec_score
InbuiltMemory_Ram = Inbuilt_memory * Ram
Ram_SpecScore = Ram * Spec_score
Ram_squared = Ram **2
Spec_score_squared = Spec_score **2
rear_count = df['rear_count'].mode()[0]
total_rear_mp = df['total_rear_mp'].mode()[0]
front_mp = df['front_mp'].mode()[0]
screen_heigt = df['screen_heigt'].mode()[0]
screen_width = df['screen_width'].mode()[0]


if 'No_of_sim' not in df1.columns or 'Processor_name' not in df1.columns:
    st.error("Kolom 'No_of_sim' atau 'Processor_name' tidak ditemukan di df1.")
else:
    df_1 = df1[['No_of_sim', 'Processor_name']].fillna(0)
    df_1 = df_1.drop_duplicates()
    df = pd.get_dummies(df_1, columns=['No_of_sim', 'Processor_name'], drop_first=True)
object_columns = df.select_dtypes(include='object').columns
st.write(len(df.columns))

# Gabungkan input pengguna dengan DataFrame
df_input = pd.DataFrame([{
    'Battery': Battery,
    'Rating': Rating,
    'Display': Display,
    'Ram': Ram,
    'Spec_score': Spec_score,
    'Inbuilt_memory': Inbuilt_memory,
    'InbuiltMemory_SpecScore': InbuiltMemory_SpecScore,
    'InbuiltMemory_Ram': InbuiltMemory_Ram,
    'Ram_SpecScore': Ram_SpecScore,
    'Ram_squared': Ram_squared,
    'Spec_score_squared': Spec_score_squared,
    'rear_count': rear_count,
    'total_rear_mp': total_rear_mp,
    'front_mp': front_mp,
    'screen_width': screen_width,
    'screen_heigt': screen_heigt
}])

# Gabungkan dengan fitur dari df (hasil pd.get_dummies)
df_combined = pd.concat([df_input, df], axis=1)

# Pastikan jumlah fitur sesuai dengan model
missing_cols = set(rf_model.feature_names_in_) - set(df_combined.columns)
for col in missing_cols:
    df_combined[col] = 0  # Tambahkan kolom yang hilang dengan nilai default

# Urutkan kolom agar sesuai dengan model
df_combined = df_combined[rf_model.feature_names_in_]

# Prediksi
if st.button("Prediksi"):
    with st.spinner('Sedang memproses...'):
        time.sleep(2)
    hasil = rf_model.predict(df_combined)
    st.success(f"Perkiraan Harga: {hasil[0]:,.2f}")
