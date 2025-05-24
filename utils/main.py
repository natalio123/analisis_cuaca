from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.evaluation import RegressionEvaluator

from akuisisi_data import fetch_bmkg_data
from preprocessing import preprocess_data, save_data
from visualisasi import visualize_data
from dashboard import create_dashboard

print("===== PROJECT ANALISIS CUACA BIG DATA DENGAN API BMKG =====\n")

# 1. Akuisisi Data
weather_data = fetch_bmkg_data()

# 2. Preprocessing dan Penyimpanan Data
df = preprocess_data(weather_data)
save_data(df, format="parquet")  # Gunakan format parquet untuk Big Data

# 3. Analisis dan Visualisasi Data dengan Spark
try:
    print("Menginisialisasi Spark Session...")
    spark = SparkSession.builder \
        .appName("BMKG Weather Analysis") \
        .config("spark.driver.memory", "2g") \
        .getOrCreate()
    print("Spark session berhasil diinisialisasi")

    file_path = "data/weather_data.parquet"
    print(f"Memuat data dari {file_path} ke Spark DataFrame...")

    if file_path.endswith(".csv"):
        spark_df = spark.read.csv(file_path, header=True, inferSchema=True)
    elif file_path.endswith(".parquet"):
        spark_df = spark.read.parquet(file_path)
    else:
        print(f"Format file {file_path} tidak didukung")
        spark_df = None

    if spark_df:
        print(f"Data berhasil dimuat: {spark_df.count()} baris")

        # Analisis deskriptif
        print("Melakukan analisis data cuaca...")
        pdf = spark_df.toPandas()

        print("\n== Statistik Deskriptif ==")
        desc_stats = pdf[['temperature', 'humidity', 'wind_speed', 'rainfall']].describe()
        print(desc_stats)

        print("\n== Distribusi Kondisi Cuaca ==")
        weather_counts = pdf['condition'].value_counts()
        print(weather_counts)

        print("\n== Rata-rata Parameter Cuaca per Kota ==")
        city_stats = pdf.groupby('city')[['temperature', 'humidity', 'rainfall']].mean()
        print(city_stats)

        # Pemodelan prediktif
        print("\nMembangun model prediksi cuaca...")
        feature_cols = ["temperature", "humidity", "wind_speed", "day_of_week", "month"]
        assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
        data = assembler.transform(spark_df)

        train_data, test_data = data.randomSplit([0.8, 0.2], seed=42)
        rf = RandomForestRegressor(featuresCol="features", labelCol="rainfall", numTrees=20)
        model = rf.fit(train_data)

        predictions = model.transform(test_data)
        evaluator = RegressionEvaluator(labelCol="rainfall", predictionCol="prediction", metricName="rmse")
        rmse = evaluator.evaluate(predictions)
        print(f"Root Mean Squared Error (RMSE) pada data testing: {rmse:.4f}")

        importance = model.featureImportances
        feature_importance = [(feature, float(importance[i])) for i, feature in enumerate(feature_cols)]
        feature_importance.sort(key=lambda x: x[1], reverse=True)

        print("\nPentingnya fitur untuk prediksi curah hujan:")
        for feature, importance in feature_importance:
            print(f"- {feature}: {importance:.4f}")

        # Visualisasi
        visualize_data(pdf)

        # Template dashboard
        create_dashboard(pdf)

        spark.stop()
    else:
        print("Data Spark tidak tersedia.")

except Exception as e:
    print(f"Error dalam analisis data: {e}")
    print("Melanjutkan dengan analisis tanpa Spark...")
    visualize_data(df)
