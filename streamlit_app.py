import streamlit as st
import pickle as pkl
import pandas as pd
import numpy as np
import time

# Load data dan model
def load_data():
    df1 = pd.read_csv('mobile phone price prediction.csv')
    df = pd.read_csv('df.csv')
    with open('rf_model.pkl', 'rb') as file:
        rf_model = pkl.load(file)
    return df1, df, rf_model

# Fungsi untuk menampilkan input pengguna
def user_input(df, df1):
    st.markdown("Masukkan input data.")
    col1, col2 = st.columns(2)

    with col1:
        Battery = st.number_input('Masukan Besar Battery(Mah):', 1900, 7000, value=1900, step=10)
        Rating = st.number_input('Masukan Rating:', max_value=5.0, value=0.00, step=0.25, format='%.2f')
        Display = st.number_input('Masukan Ukuran Handphone():', 4.5, 10.0, value=4.5, step=0.1)

    with col2:
        Ram = st.slider('Masukan angka Ram(GB):', df['Ram'].min(), df['Ram'].max())
        Spec_score = st.slider('Masukan Spec Skor', df1['Spec_score'].min(), df1['Spec_score'].max())
        Inbuilt_memory = st.radio('Masukan Ukuran Internal memory(GB):', df['Inbuilt_memory'].unique())

    return Battery, Rating, Display, Ram, Spec_score, Inbuilt_memory

# Fungsi untuk memproses input pengguna
def process_input(Battery, Rating, Display, Ram, Spec_score, Inbuilt_memory, df, df1):
    InbuiltMemory_SpecScore = Inbuilt_memory * Spec_score
    InbuiltMemory_Ram = Inbuilt_memory * Ram
    Ram_SpecScore = Ram * Spec_score
    Ram_squared = Ram ** 2
    Spec_score_squared = Spec_score ** 2

    # Ambil nilai default dari mode
    rear_count = df['rear_count'].mode()[0]
    total_rear_mp = df['total_rear_mp'].mode()[0]
    front_mp = df['front_mp'].mode()[0]
    screen_heigt = df['screen_heigt'].mode()[0]
    screen_width = df['screen_width'].mode()[0]

    # Gabungkan input pengguna ke DataFrame
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

    # Proses fitur tambahan
    if 'No_of_sim' in df1.columns and 'Processor_name' in df1.columns:
        df_1 = df1[['No_of_sim', 'Processor_name']].fillna(0).drop_duplicates()
        df_dummies = pd.get_dummies(df_1, columns=['No_of_sim', 'Processor_name'], drop_first=True)
    else:
        st.error("Kolom 'No_of_sim' atau 'Processor_name' tidak ditemukan di df1.")
        df_dummies = pd.DataFrame()

    # Gabungkan input pengguna dengan fitur tambahan
    df_combined = pd.concat([df_input, df_dummies], axis=1)
    return df_combined

# Fungsi untuk memastikan fitur sesuai dengan model
def align_features(df_combined, rf_model):
    missing_cols = set(rf_model.feature_names_in_) - set(df_combined.columns)
    for col in missing_cols:
        df_combined[col] = 0  # Tambahkan kolom yang hilang dengan nilai default
    return df_combined[rf_model.feature_names_in_]

# Fungsi utama untuk prediksi
def predict_price(df_combined, rf_model):
    with st.spinner('Sedang memproses...'):
        time.sleep(2)
    hasil = rf_model.predict(df_combined)
    st.success(f"Perkiraan Harga: {hasil[0]:,.2f}")

# Main
def main():
    st.title("Prediksi Harga Handphone")
    st.write('Data set yang digunakan: [Here](https://drive.google.com/file/d/1BEzYGaWuiFmXAXtrQLVbFmqCCJ9-GzHK/view?usp=sharing)')

    df1, df, rf_model = load_data()
    Battery, Rating, Display, Ram, Spec_score, Inbuilt_memory = user_input(df, df1)
    df_combined = process_input(Battery, Rating, Display, Ram, Spec_score, Inbuilt_memory, df, df1)
    df_combined = align_features(df_combined, rf_model)

    if st.button("Prediksi"):
        predict_price(df_combined, rf_model)

if __name__ == "__main__":
    main()
