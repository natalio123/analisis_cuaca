import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
    
def create_dashboard(df):
    """
    Membuat dashboard sederhana menggunakan Python dash/streamlit (kode template)
    Dalam implementasi nyata, ini akan menggunakan Dash atau Streamlit
    """
    print("\nTemplate kode untuk dashboard visualisasi data cuaca:")
    
    dashboard_code = """
    # Dashboard Analisis Cuaca BMKG
    # Implementasi dengan Streamlit
    
    import streamlit as st
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Load data
    df = pd.read_csv('data/weather_data.csv')
    df['date'] = pd.to_datetime(df['date'])
    
    st.title('Dashboard Analisis Cuaca BMKG')
    
    # Sidebar untuk filter
    st.sidebar.header('Filter')
    selected_city = st.sidebar.selectbox('Pilih Kota', df['city'].unique())
    
    # Filter data berdasarkan kota yang dipilih
    filtered_data = df[df['city'] == selected_city]
    
    # Tampilkan informasi umum
    st.header(f'Data Cuaca untuk {selected_city}')
    
    # Tampilkan statistik
    st.subheader('Statistik Cuaca')
    stats = filtered_data[['temperature', 'humidity', 'wind_speed', 'rainfall']].describe()
    st.dataframe(stats)
    
    # Plot tren suhu
    st.subheader('Tren Suhu')
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(filtered_data['date'], filtered_data['temperature'], marker='o')
    ax.set_xlabel('Tanggal')
    ax.set_ylabel('Suhu (°C)')
    st.pyplot(fig)
    
    # Plot kondisi cuaca
    st.subheader('Distribusi Kondisi Cuaca')
    condition_counts = filtered_data['condition'].value_counts()
    st.bar_chart(condition_counts)
    
    # Tampilkan data mentah jika diminta
    if st.checkbox('Tampilkan data mentah'):
        st.write(filtered_data)
    """
    
    print(dashboard_code)
