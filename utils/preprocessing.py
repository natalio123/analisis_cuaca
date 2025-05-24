import pandas as pd
import os

def preprocess_data(data):
    """
    Melakukan preprocessing data cuaca
    """
    print("Melakukan preprocessing data...")
    
    if not data:
        print("Tidak ada data untuk diproses")
        return None
    
    # Konversi data ke DataFrame pandas
    df = pd.DataFrame(data)
    
    # Konversi kolom tanggal ke format datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Menambahkan fitur tambahan
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['day_of_week'] = df['date'].dt.dayofweek
    
    # Menangani missing values jika ada
    df = df.fillna({
        'temperature': df['temperature'].mean(),
        'humidity': df['humidity'].mean(),
        'wind_speed': df['wind_speed'].mean(),
        'rainfall': df['rainfall'].mean()
    })
    
    # Normalisasi beberapa fitur numerik
    df['temperature_norm'] = (df['temperature'] - df['temperature'].min()) / (df['temperature'].max() - df['temperature'].min())
    df['humidity_norm'] = (df['humidity'] - df['humidity'].min()) / (df['humidity'].max() - df['humidity'].min())
    
    print("Preprocessing selesai dengan hasil:")
    print(f"- Jumlah baris data: {len(df)}")
    print(f"- Kolom: {', '.join(df.columns)}")
    
    return df

def save_data(df, format="csv"):
    """
    Menyimpan data yang sudah dipreprocess
    """
    if df is None:
        return
    
    print(f"Menyimpan data dalam format {format}...")
    
    # Buat direktori untuk menyimpan data jika belum ada
    os.makedirs("data", exist_ok=True)
    
    if format == "csv":
        # Simpan sebagai CSV (untuk penggunaan sederhana)
        df.to_csv("data/weather_data.csv", index=False)
        print("Data disimpan sebagai CSV di data/weather_data.csv")
    
    elif format == "parquet":
        # Simpan sebagai Parquet (format kolumnar untuk big data)
        df.to_parquet("data/weather_data.parquet")
        print("Data disimpan sebagai Parquet di data/weather_data.parquet")
    
    elif format == "json":
        # Simpan sebagai JSON
        df.to_json("data/weather_data.json", orient="records")
        print("Data disimpan sebagai JSON di data/weather_data.json")