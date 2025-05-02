import streamlit as st
import pickle as pkl
import pandas as pd
import numpy as np
import time
import math

# Load data dan model
def load_data():
    df1 = pd.read_csv('mobile phone price prediction.csv')
    df = pd.read_csv('df.csv')
    with open('rf_model.pkl', 'rb') as file:
        rf_model = pkl.load(file)
    return df1, df, rf_model

# Fungsi untuk menampilkan input pengguna
def user_input(df, df1):
    st.markdown("### 🔧 Masukkan Spesifikasi Handphone")
    col1, col2 = st.columns(2)

    with col1:
        Battery = st.number_input('🔋 Battery (mAh):', 1900, 7000, value=1900, step=10)
        Rating = st.number_input('⭐ Rating (max 5.0):', max_value=5.0, value=0.00, step=0.25, format='%.2f')
        Display = st.number_input('📱 Ukuran Layar (inchi):', 4.5, 10.0, value=4.5, step=0.1)

    with col2:
        Ram = st.slider('💾 RAM (GB):', df['Ram'].min(), df['Ram'].max())
        Spec_score = st.slider('📊 Spec Score:', df1['Spec_score'].min(), df1['Spec_score'].max())
        Inbuilt_memory = st.radio('💽 Internal Memory (GB):', df['Inbuilt_memory'].unique())

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

# Fungsi untuk menyamakan fitur dengan model
def align_features(df_combined, rf_model):
    missing_cols = set(rf_model.feature_names_in_) - set(df_combined.columns)
    for col in missing_cols:
        df_combined[col] = 0
    return df_combined[rf_model.feature_names_in_]

# Fungsi utama prediksi
def predict_price(df_combined, rf_model):
    with st.spinner('🔄 Sedang memproses...'):
        time.sleep(2)
    hasil = rf_model.predict(df_combined)
    st.markdown(f"""
        <div style="text-align: center; padding: 20px; background-color: #eaf4ff; border-radius: 10px;">
            <h2 style="color: #3E64FF;">📈 Perkiraan Harga: <span style="font-size: 1.5em;">Rp {hasil[0]:,.2f}</span></h2>
        </div>
    """, unsafe_allow_html=True)

# Fungsi utama aplikasi
def main():
    st.set_page_config(page_title="Prediksi Harga Handphone", layout="centered")
    
    # CSS Kustom
    st.markdown("""
        <style>
            .main {
                background-color: #f9f9f9;
                padding: 20px;
                border-radius: 10px;
            }

            h1 {
                color: #3E64FF;
                text-align: center;
            }

            .stButton>button {
                background-color: #3E64FF;
                color: white;
                border-radius: 8px;
                height: 3em;
                width: 100%;
                font-size: 1.1em;
                font-weight: bold;
            }

            .stButton>button:hover {
                background-color: #254EDB;
                transition: 0.3s;
            }

            .stRadio, .stNumberInput, .stSlider {
                padding-bottom: 10px;
            }

            .stMarkdown {
                font-size: 16px;
            }
        </style>
    """, unsafe_allow_html=True)

    st.title("📱 Prediksi Harga Handphone")
    st.markdown("Gunakan aplikasi ini untuk memprediksi harga handphone berdasarkan spesifikasi utama.")

    df1, df, rf_model = load_data()
    Battery, Rating, Display, Ram, Spec_score, Inbuilt_memory = user_input(df, df1)
    df_combined = process_input(Battery, Rating, Display, Ram, Spec_score, Inbuilt_memory, df, df1)
    df_combined = align_features(df_combined, rf_model)

    st.markdown("---")
    if st.button("🔍 Prediksi Harga"):
        predict_price(df_combined, rf_model)

if __name__ == "__main__":
    main()
