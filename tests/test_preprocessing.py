import unittest
import pandas as pd
import os
import tempfile
import shutil
from datetime import datetime, timedelta
import sys

# Add the utils directory to the path to import preprocessing module
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from preprocessing import preprocess_data, save_data


class TestPreprocessing(unittest.TestCase):
    
    def setUp(self):
        """Setup test data dan temporary directory untuk setiap test"""
        # Sample weather data untuk testing
        self.sample_data = [
            {
                'date': '2024-01-01',
                'temperature': 25.5,
                'humidity': 70.0,
                'wind_speed': 10.2,
                'rainfall': 0.0
            },
            {
                'date': '2024-01-02',
                'temperature': 27.8,
                'humidity': 65.5,
                'wind_speed': 8.7,
                'rainfall': 2.5
            },
            {
                'date': '2024-01-03',
                'temperature': 24.2,
                'humidity': 80.0,
                'wind_speed': 12.1,
                'rainfall': 5.2
            }
        ]
        
        # Data dengan missing values untuk testing
        self.data_with_missing = [
            {
                'date': '2024-01-01',
                'temperature': 25.5,
                'humidity': None,
                'wind_speed': 10.2,
                'rainfall': 0.0
            },
            {
                'date': '2024-01-02',
                'temperature': None,
                'humidity': 65.5,
                'wind_speed': 8.7,
                'rainfall': 2.5
            }
        ]
        
        # Buat temporary directory untuk testing file operations
        self.temp_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.temp_dir)
    
    def tearDown(self):
        """Cleanup setelah setiap test"""
        os.chdir(self.original_cwd)
        shutil.rmtree(self.temp_dir)
    
    def test_preprocess_data_normal(self):
        """Test preprocessing dengan data normal"""
        result = preprocess_data(self.sample_data)
        
        # Check apakah result adalah DataFrame
        self.assertIsInstance(result, pd.DataFrame)
        
        # Check jumlah baris
        self.assertEqual(len(result), 3)
        
        # Check apakah kolom tanggal dikonversi ke datetime
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(result['date']))
        
        # Check apakah kolom tambahan ditambahkan
        expected_columns = ['date', 'temperature', 'humidity', 'wind_speed', 'rainfall',
                           'month', 'day', 'day_of_week', 'temperature_norm', 'humidity_norm']
        for col in expected_columns:
            self.assertIn(col, result.columns)
        
        # Check nilai month, day, day_of_week
        self.assertEqual(result.iloc[0]['month'], 1)  # January
        self.assertEqual(result.iloc[0]['day'], 1)
        
        # Check normalisasi values (should be between 0 and 1)
        self.assertTrue(all(0 <= val <= 1 for val in result['temperature_norm']))
        self.assertTrue(all(0 <= val <= 1 for val in result['humidity_norm']))
    
    def test_preprocess_data_empty(self):
        """Test preprocessing dengan data kosong"""
        result = preprocess_data([])
        self.assertIsNone(result)
        
        result = preprocess_data(None)
        self.assertIsNone(result)
    
    def test_preprocess_data_with_missing_values(self):
        """Test preprocessing dengan missing values"""
        result = preprocess_data(self.data_with_missing)
        
        self.assertIsInstance(result, pd.DataFrame)
        
        # Check apakah tidak ada missing values setelah preprocessing
        self.assertFalse(result.isnull().any().any())
        
        # Check apakah missing values diganti dengan mean
        self.assertIsNotNone(result.iloc[0]['humidity'])
        self.assertIsNotNone(result.iloc[1]['temperature'])
    
    def test_save_data_csv(self):
        """Test menyimpan data sebagai CSV"""
        df = preprocess_data(self.sample_data)
        save_data(df, format="csv")
        
        # Check apakah file CSV dibuat
        self.assertTrue(os.path.exists("data/weather_data.csv"))
        
        # Check apakah data bisa dibaca kembali
        loaded_df = pd.read_csv("data/weather_data.csv")
        self.assertEqual(len(loaded_df), len(df))
    
    def test_save_data_json(self):
        """Test menyimpan data sebagai JSON"""
        df = preprocess_data(self.sample_data)
        save_data(df, format="json")
        
        # Check apakah file JSON dibuat
        self.assertTrue(os.path.exists("data/weather_data.json"))
        
        # Check apakah data bisa dibaca kembali
        loaded_df = pd.read_json("data/weather_data.json")
        self.assertEqual(len(loaded_df), len(df))
    
    def test_save_data_parquet(self):
        """Test menyimpan data sebagai Parquet"""
        df = preprocess_data(self.sample_data)
        
        try:
            save_data(df, format="parquet")
            # Check apakah file Parquet dibuat
            self.assertTrue(os.path.exists("data/weather_data.parquet"))
            
            # Check apakah data bisa dibaca kembali
            loaded_df = pd.read_parquet("data/weather_data.parquet")
            self.assertEqual(len(loaded_df), len(df))
        except ImportError:
            # Skip test jika pyarrow tidak terinstall
            self.skipTest("Pyarrow not installed, skipping parquet test")
    
    def test_save_data_none(self):
        """Test menyimpan data None (tidak seharusnya membuat file)"""
        save_data(None, format="csv")
        
        # Check apakah tidak ada file yang dibuat
        self.assertFalse(os.path.exists("data/weather_data.csv"))
    
    def test_data_types_after_preprocessing(self):
        """Test tipe data setelah preprocessing"""
        result = preprocess_data(self.sample_data)
        
        # Check tipe data kolom numerik
        numeric_columns = ['temperature', 'humidity', 'wind_speed', 'rainfall',
                          'temperature_norm', 'humidity_norm']
        for col in numeric_columns:
            self.assertTrue(pd.api.types.is_numeric_dtype(result[col]))
        
        # Check tipe data kolom integer
        integer_columns = ['month', 'day', 'day_of_week']
        for col in integer_columns:
            self.assertTrue(pd.api.types.is_integer_dtype(result[col]))
    
    def test_normalization_correctness(self):
        """Test apakah normalisasi dilakukan dengan benar"""
        result = preprocess_data(self.sample_data)
        
        # Check temperature normalization
        temp_min = result['temperature'].min()
        temp_max = result['temperature'].max()
        expected_norm_min = (temp_min - temp_min) / (temp_max - temp_min)  # Should be 0
        expected_norm_max = (temp_max - temp_min) / (temp_max - temp_min)  # Should be 1
        
        self.assertAlmostEqual(result['temperature_norm'].min(), expected_norm_min)
        self.assertAlmostEqual(result['temperature_norm'].max(), expected_norm_max)
        
        # Check humidity normalization
        humid_min = result['humidity'].min()
        humid_max = result['humidity'].max()
        expected_norm_min = (humid_min - humid_min) / (humid_max - humid_min)  # Should be 0
        expected_norm_max = (humid_max - humid_min) / (humid_max - humid_min)  # Should be 1
        
        self.assertAlmostEqual(result['humidity_norm'].min(), expected_norm_min)
        self.assertAlmostEqual(result['humidity_norm'].max(), expected_norm_max)


if __name__ == '__main__':
    # Menjalankan tests dengan output yang verbose
    unittest.main(verbosity=2)