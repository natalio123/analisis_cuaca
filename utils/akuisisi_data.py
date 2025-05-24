import requests
from datetime import datetime, timedelta
import numpy as np

def fetch_bmkg_data():
    """
    Mengambil data cuaca dari API BMKG
    
    API BMKG menyediakan data cuaca dalam format JSON untuk berbagai provinsi di Indonesia
    """
    print("Mengambil data dari API BMKG...")
    
    # URL endpoint untuk data cuaca dari BMKG
    # API ini menyediakan ramalan cuaca untuk berbagai kota di Indonesia
    url = "https://data.bmkg.go.id/DataMKG/MEWS/DigitalForecast/DigitalForecast-Indonesia.xml"
    
    try:
        response = requests.get(url)
        # Untuk XML format, kita perlu mengkonversi ke format yang mudah diproses
        # Dalam contoh nyata, kita akan parse XML nya, tapi di sini kita asumsikan sudah dalam JSON
        
        # Simulasi data untuk keperluan demonstrasi
        # Dalam implementasi nyata, parse data XML dari response.text
        weather_data = simulate_weather_data()
        
        print(f"Data berhasil diambil: {len(weather_data)} rekaman")
        return weather_data
    except Exception as e:
        print(f"Error mengambil data: {e}")
        return None

def simulate_weather_data():
    """
    Mensimulasikan data cuaca untuk beberapa kota di Indonesia
    Dalam implementasi nyata, ini akan diganti dengan parsing data XML dari BMKG
    """
    cities = ["Jakarta", "Surabaya", "Bandung", "Medan", "Makassar", 
             "Semarang", "Palembang", "Yogyakarta", "Denpasar", "Manado"]
    
    data = []
    
    # Membuat data cuaca simulasi untuk 30 hari terakhir
    for city in cities:
        for i in range(21):
            date = (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d')
            
            # Simulasi nilai parameter cuaca dengan sedikit random variation
            temp = round(np.random.normal(30, 3), 1)  # temp dalam Celsius
            humidity = round(np.random.normal(75, 10))  # kelembaban dalam %
            wind_speed = round(np.random.normal(15, 5), 1)  # kec. angin km/h
            rainfall = max(0, round(np.random.normal(5, 10), 1))  # curah hujan mm
            
            # Menentukan kondisi cuaca berdasarkan parameter
            if rainfall > 20:
                condition = "Hujan Lebat"
            elif rainfall > 5:
                condition = "Hujan Ringan"
            elif humidity > 85:
                condition = "Berawan"
            else:
                condition = "Cerah"
                
            data.append({
                "city": city,
                "date": date,
                "temperature": temp,
                "humidity": humidity,
                "wind_speed": wind_speed,
                "rainfall": rainfall,
                "condition": condition
            })
    
    return data