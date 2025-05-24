import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator

def initialize_spark():
    """
    Menginisialisasi Spark Session untuk analisis Big Data
    """
    print("Menginisialisasi Spark Session...")
    spark = SparkSession.builder \
        .appName("BMKG Weather Analysis") \
        .config("spark.driver.memory", "2g") \
        .getOrCreate()
    
    print("Spark session berhasil diinisialisasi")
    return spark

def load_data_to_spark(spark, file_path="data/weather_data.csv"):
    """
    Memuat data ke Spark DataFrame untuk analisis
    """
    print(f"Memuat data dari {file_path} ke Spark DataFrame...")
    
    if file_path.endswith(".csv"):
        spark_df = spark.read.csv(file_path, header=True, inferSchema=True)
    elif file_path.endswith(".parquet"):
        spark_df = spark.read.parquet(file_path)
    else:
        print(f"Format file {file_path} tidak didukung")
        return None
    
    print(f"Data berhasil dimuat: {spark_df.count()} baris")
    return spark_df

def analyze_weather_data(spark_df):
    """
    Melakukan analisis data cuaca menggunakan Spark
    """
    if spark_df is None:
        return
    
    print("Melakukan analisis data cuaca...")
    
    # Mengkonversi Spark DataFrame ke pandas DataFrame untuk exploratory analysis
    pdf = spark_df.toPandas()
    
    # 1. Analisis deskriptif
    print("\n== Statistik Deskriptif ==")
    desc_stats = pdf[['temperature', 'humidity', 'wind_speed', 'rainfall']].describe()
    print(desc_stats)
    
    # 2. Distribusi kondisi cuaca
    print("\n== Distribusi Kondisi Cuaca ==")
    weather_counts = pdf['condition'].value_counts()
    print(weather_counts)
    
    # 3. Statistik cuaca per kota
    print("\n== Rata-rata Parameter Cuaca per Kota ==")
    city_stats = pdf.groupby('city')[['temperature', 'humidity', 'rainfall']].mean()
    print(city_stats)
    
    return pdf

def predict_weather(spark, spark_df):
    """
    Melakukan prediksi cuaca menggunakan model machine learning
    """
    if spark_df is None:
        return
    
    print("\nMembangun model prediksi cuaca...")
    
    # Persiapkan data untuk pemodelan
    # Pilih fitur dan target
    feature_cols = ["temperature", "humidity", "wind_speed", "day_of_week", "month"]
    assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    data = assembler.transform(spark_df)
    
    # Split data menjadi training dan testing
    train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)
    
    # Membuat dan melatih model Random Forest untuk prediksi curah hujan
    rf = RandomForestRegressor(featuresCol="features", labelCol="rainfall", numTrees=20)
    model = rf.fit(train_data)
    
    # Evaluasi model
    predictions = model.transform(test_data)
    evaluator = RegressionEvaluator(labelCol="rainfall", predictionCol="prediction", metricName="rmse")
    rmse = evaluator.evaluate(predictions)
    
    print(f"Root Mean Squared Error (RMSE) pada data testing: {rmse:.4f}")
    
    # Ekstrak feature importance
    importance = model.featureImportances
    feature_importance = [(feature, float(importance[i])) for i, feature in enumerate(feature_cols)]
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    
    print("\nPentingnya fitur untuk prediksi curah hujan:")
    for feature, importance in feature_importance:
        print(f"- {feature}: {importance:.4f}")
    
    return model

def visualize_data(df):
    """
    Membuat visualisasi dari data cuaca
    """
    if df is None:
        return
    
    print("\nMembuat visualisasi data cuaca...")
    
    # Buat direktori untuk menyimpan plot jika belum ada
    os.makedirs("plots", exist_ok=True)
    
    # Set style
    plt.style.use('ggplot')
    
    # 1. Plot distribusi suhu berdasarkan kota
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='city', y='temperature', data=df)
    plt.title('Distribusi Suhu berdasarkan Kota')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('plots/temperature_by_city.png')
    
    # 2. Plot hubungan suhu, kelembaban, dan kondisi cuaca
    plt.figure(figsize=(10, 8))
    scatter = sns.scatterplot(x='temperature', y='humidity', hue='condition', 
                              palette='viridis', s=100, alpha=0.7, data=df)
    plt.title('Hubungan antara Suhu, Kelembaban, dan Kondisi Cuaca')
    plt.tight_layout()
    plt.savefig('plots/temp_humidity_condition.png')
    
    # 3. Plot tren curah hujan berdasarkan tanggal untuk Jakarta
    jakarta_data = df[df['city'] == 'Jakarta'].sort_values('date')
    
    plt.figure(figsize=(14, 6))
    plt.plot(jakarta_data['date'], jakarta_data['rainfall'], marker='o', linestyle='-')
    plt.title('Tren Curah Hujan di Jakarta')
    plt.xlabel('Tanggal')
    plt.ylabel('Curah Hujan (mm)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('plots/jakarta_rainfall_trend.png')
    
    # 4. Heatmap korelasi
    plt.figure(figsize=(10, 8))
    corr = df[['temperature', 'humidity', 'wind_speed', 'rainfall']].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=0.5)
    plt.title('Korelasi antar Parameter Cuaca')
    plt.tight_layout()
    plt.savefig('plots/weather_correlation.png')
    
    print("Visualisasi selesai dibuat dan disimpan di folder 'plots'")