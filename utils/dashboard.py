# dashboard.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

def create_dashboard(df):
    st.title('Dashboard Analisis Cuaca BMKG')

    # Sidebar
    st.sidebar.header('Filter')
    selected_city = st.sidebar.selectbox('Pilih Kota', df['city'].unique())

    filtered_data = df[df['city'] == selected_city]

    st.header(f'Data Cuaca untuk {selected_city}')
    
    # Statistik
    st.subheader('Statistik Cuaca')
    st.dataframe(filtered_data[['temperature', 'humidity', 'wind_speed', 'rainfall']].describe())

    # Tren Suhu
    st.subheader('Tren Suhu')
    st.line_chart(filtered_data.set_index('date')['temperature'])

    # Distribusi Kondisi Cuaca
    st.subheader('Distribusi Kondisi Cuaca')
    st.bar_chart(filtered_data['condition'].value_counts())

    # Data mentah
    if st.checkbox('Tampilkan data mentah'):
        st.write(filtered_data)
