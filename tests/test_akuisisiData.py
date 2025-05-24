"""
Unit Tests untuk Akuisisi Data BMKG
===================================

File ini berisi test cases untuk menguji fungsi-fungsi akuisisi data
dari API BMKG dan simulasi data cuaca.
"""

import unittest
from unittest.mock import patch, Mock
import requests
from datetime import datetime, timedelta
import sys
import os

# Tambahkan parent directory ke path untuk import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import fungsi yang akan ditest dari akuisisi_data.py
from akuisisi_data import fetch_bmkg_data, simulate_weather_data


class TestAkuisisiData(unittest.TestCase):
    """
    Test class untuk menguji fungsi akuisisi data BMKG
    """
    
    def setUp(self):
        """
        Setup yang dijalankan sebelum setiap test case
        """
        self.expected_cities = ["Jakarta", "Surabaya", "Bandung", "Medan", "Makassar", 
                               "Semarang", "Palembang", "Yogyakarta", "Denpasar", "Manado"]
        self.expected_days = 21
        self.total_expected_records = len(self.expected_cities) * self.expected_days
    
    def test_simulate_weather_data_structure(self):
        """
        Test struktur data yang dihasilkan oleh simulate_weather_data
        """
        print("Testing simulate_weather_data structure...")
        
        data = simulate_weather_data()
        
        # Test jumlah data yang dihasilkan
        self.assertEqual(len(data), self.total_expected_records, 
                        f"Jumlah data harus {self.total_expected_records}")
        
        # Test struktur data pertama
        first_record = data[0]
        expected_keys = ["city", "date", "temperature", "humidity", 
                        "wind_speed", "rainfall", "condition"]
        
        for key in expected_keys:
            self.assertIn(key, first_record, f"Key '{key}' harus ada dalam data")
        
        print("✓ Struktur data simulate_weather_data valid")
    
    def test_simulate_weather_data_values(self):
        """
        Test validitas nilai-nilai dalam data simulasi
        """
        print("Testing simulate_weather_data values...")
        
        data = simulate_weather_data()
        
        for record in data[:10]:  # Test 10 record pertama saja untuk efisiensi
            # Test city
            self.assertIn(record["city"], self.expected_cities, 
                         f"Kota {record['city']} tidak valid")
            
            # Test temperature range (realistis untuk Indonesia)
            self.assertGreaterEqual(record["temperature"], 15.0, 
                                   "Suhu terlalu rendah untuk Indonesia")
            self.assertLessEqual(record["temperature"], 45.0, 
                                "Suhu terlalu tinggi untuk Indonesia")
            
            # Test humidity range
            self.assertGreaterEqual(record["humidity"], 0, 
                                   "Kelembaban tidak boleh negatif")
            self.assertLessEqual(record["humidity"], 100, 
                                "Kelembaban tidak boleh lebih dari 100%")
            
            # Test wind_speed (tidak boleh negatif)
            self.assertGreaterEqual(record["wind_speed"], 0, 
                                   "Kecepatan angin tidak boleh negatif")
            
            # Test rainfall (tidak boleh negatif)
            self.assertGreaterEqual(record["rainfall"], 0, 
                                   "Curah hujan tidak boleh negatif")
            
            # Test condition values
            valid_conditions = ["Hujan Lebat", "Hujan Ringan", "Berawan", "Cerah"]
            self.assertIn(record["condition"], valid_conditions, 
                         f"Kondisi cuaca '{record['condition']}' tidak valid")
            
            # Test date format
            try:
                datetime.strptime(record["date"], '%Y-%m-%d')
            except ValueError:
                self.fail(f"Format tanggal '{record['date']}' tidak valid")
        
        print("✓ Nilai-nilai dalam simulate_weather_data valid")
    
    def test_simulate_weather_data_condition_logic(self):
        """
        Test logic penentuan kondisi cuaca berdasarkan parameter
        """
        print("Testing weather condition logic...")
        
        data = simulate_weather_data()
        
        for record in data[:20]:  # Test beberapa record
            rainfall = record["rainfall"]
            humidity = record["humidity"]
            condition = record["condition"]
            
            # Test logic kondisi cuaca
            if rainfall > 20:
                self.assertEqual(condition, "Hujan Lebat", 
                               f"Dengan curah hujan {rainfall}mm, kondisi harus 'Hujan Lebat'")
            elif rainfall > 5:
                self.assertEqual(condition, "Hujan Ringan", 
                               f"Dengan curah hujan {rainfall}mm, kondisi harus 'Hujan Ringan'")
            elif humidity > 85:
                self.assertEqual(condition, "Berawan", 
                               f"Dengan kelembaban {humidity}%, kondisi harus 'Berawan'")
            else:
                self.assertEqual(condition, "Cerah", 
                               f"Dengan kondisi normal, cuaca harus 'Cerah'")
        
        print("✓ Logic kondisi cuaca benar")
    
    def test_simulate_weather_data_date_range(self):
        """
        Test apakah tanggal dalam data sesuai dengan range yang diharapkan
        """
        print("Testing date range in simulated data...")
        
        data = simulate_weather_data()
        
        # Ambil semua tanggal unik
        dates = list(set([record["date"] for record in data]))
        dates.sort()
        
        # Test jumlah tanggal unik
        self.assertEqual(len(dates), self.expected_days, 
                        f"Harus ada {self.expected_days} tanggal unik")
        
        # Test apakah tanggal dalam range 21 hari terakhir
        today = datetime.now()
        oldest_expected = (today - timedelta(days=self.expected_days-1)).strftime('%Y-%m-%d')
        newest_expected = today.strftime('%Y-%m-%d')
        
        self.assertGreaterEqual(dates[0], oldest_expected, 
                               "Tanggal tertua tidak sesuai")
        self.assertLessEqual(dates[-1], newest_expected, 
                            "Tanggal terbaru tidak sesuai")
        
        print("✓ Range tanggal dalam data simulasi benar")
    
    @patch('akuisisi_data.requests.get')
    def test_fetch_bmkg_data_success(self, mock_get):
        """
        Test fetch_bmkg_data ketika API call berhasil
        """
        print("Testing fetch_bmkg_data success scenario...")
        
        # Mock successful response
        mock_response = Mock()
        mock_response.text = "<xml>mock data</xml>"
        mock_get.return_value = mock_response
        
        # Test function call
        result = fetch_bmkg_data()
        
        # Verify API was called
        mock_get.assert_called_once()
        
        # Verify result structure (karena menggunakan simulate_weather_data)
        self.assertIsNotNone(result, "Result tidak boleh None")
        self.assertEqual(len(result), self.total_expected_records, 
                        "Jumlah data tidak sesuai")
        
        print("✓ fetch_bmkg_data berhasil dengan API call sukses")
    
    @patch('akuisisi_data.requests.get')
    def test_fetch_bmkg_data_api_error(self, mock_get):
        """
        Test fetch_bmkg_data ketika API call gagal
        """
        print("Testing fetch_bmkg_data API error scenario...")
        
        # Mock API error
        mock_get.side_effect = requests.exceptions.RequestException("API Error")
        
        # Test function call
        result = fetch_bmkg_data()
        
        # Verify API was called
        mock_get.assert_called_once()
        
        # Verify result is None when error occurs
        self.assertIsNone(result, "Result harus None ketika API error")
        
        print("✓ fetch_bmkg_data menangani API error dengan benar")
    
    @patch('akuisisi_data.requests.get')
    def test_fetch_bmkg_data_timeout(self, mock_get):
        """
        Test fetch_bmkg_data ketika API timeout
        """
        print("Testing fetch_bmkg_data timeout scenario...")
        
        # Mock timeout
        mock_get.side_effect = requests.exceptions.Timeout("Request timeout")
        
        # Test function call
        result = fetch_bmkg_data()
        
        # Verify result is None when timeout occurs
        self.assertIsNone(result, "Result harus None ketika timeout")
        
        print("✓ fetch_bmkg_data menangani timeout dengan benar")
    
    def test_data_consistency_across_cities(self):
        """
        Test konsistensi data antar kota
        """
        print("Testing data consistency across cities...")
        
        data = simulate_weather_data()
        
        # Group data by city
        city_data = {}
        for record in data:
            city = record["city"]
            if city not in city_data:
                city_data[city] = []
            city_data[city].append(record)
        
        # Test setiap kota memiliki jumlah data yang sama
        for city, records in city_data.items():
            self.assertEqual(len(records), self.expected_days, 
                           f"Kota {city} harus memiliki {self.expected_days} record")
        
        # Test semua kota ada dalam data
        for expected_city in self.expected_cities:
            self.assertIn(expected_city, city_data.keys(), 
                         f"Kota {expected_city} tidak ditemukan dalam data")
        
        print("✓ Data konsisten antar semua kota")
    
    def test_numerical_data_types(self):
        """
        Test tipe data numerik dalam hasil simulasi
        """
        print("Testing numerical data types...")
        
        data = simulate_weather_data()
        sample_record = data[0]
        
        # Test tipe data
        self.assertIsInstance(sample_record["temperature"], (int, float), 
                             "Temperature harus berupa angka")
        self.assertIsInstance(sample_record["humidity"], (int, float), 
                             "Humidity harus berupa angka")
        self.assertIsInstance(sample_record["wind_speed"], (int, float), 
                             "Wind speed harus berupa angka")
        self.assertIsInstance(sample_record["rainfall"], (int, float), 
                             "Rainfall harus berupa angka")
        self.assertIsInstance(sample_record["city"], str, 
                             "City harus berupa string")
        self.assertIsInstance(sample_record["date"], str, 
                             "Date harus berupa string")
        self.assertIsInstance(sample_record["condition"], str, 
                             "Condition harus berupa string")
        
        print("✓ Tipe data numerik sesuai dengan expected")


class TestPerformance(unittest.TestCase):
    """
    Test class untuk menguji performa fungsi akuisisi data
    """
    
    def test_simulate_weather_data_performance(self):
        """
        Test performa generate data simulasi
        """
        print("Testing simulate_weather_data performance...")
        
        import time
        
        start_time = time.time()
        data = simulate_weather_data()
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        # Eksekusi harus selesai dalam waktu kurang dari 2 detik
        self.assertLess(execution_time, 2.0, 
                       f"Generate data terlalu lama: {execution_time:.2f} detik")
        
        # Verify data generated
        self.assertGreater(len(data), 0, "Data harus dihasilkan")
        
        print(f"✓ Generate data selesai dalam {execution_time:.3f} detik")


def run_tests():
    """
    Fungsi untuk menjalankan semua test cases
    """
    print("="*60)
    print("MENJALANKAN UNIT TESTS UNTUK AKUISISI DATA BMKG")
    print("="*60)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestAkuisisiData))
    suite.addTests(loader.loadTestsFromTestCase(TestPerformance))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*60)
    print("RINGKASAN HASIL TEST")
    print("="*60)
    print(f"Total Tests Run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success Rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback}")
    
    if result.errors:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    # Jalankan semua tests
    success = run_tests()
    
    if success:
        print("\n🎉 Semua tests berhasil!")
        exit(0)
    else:
        print("\n❌ Ada tests yang gagal!")
        exit(1)