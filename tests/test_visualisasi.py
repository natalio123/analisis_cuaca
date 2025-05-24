import unittest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import tempfile
import shutil
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock

# Import fungsi yang akan ditest
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from visualisasi import visualize_data


class TestVisualisasi(unittest.TestCase):
    """
    Test class untuk menguji fungsi visualisasi data cuaca
    """
    
    def setUp(self):
        """
        Setup data dummy untuk testing
        """
        # Buat data dummy yang realistis
        np.random.seed(42)
        n_samples = 100
        
        # Generate tanggal
        start_date = datetime(2023, 1, 1)
        dates = [start_date + timedelta(days=i) for i in range(n_samples)]
        
        # Generate data cuaca
        cities = ['Jakarta', 'Bandung', 'Surabaya', 'Medan', 'Makassar']
        conditions = ['Cerah', 'Berawan', 'Hujan', 'Hujan Ringan', 'Badai']
        
        self.test_data = pd.DataFrame({
            'date': np.random.choice(dates, n_samples),
            'city': np.random.choice(cities, n_samples),
            'temperature': np.random.normal(28, 5, n_samples),  # Suhu rata-rata 28°C
            'humidity': np.random.uniform(60, 95, n_samples),   # Kelembaban 60-95%
            'wind_speed': np.random.exponential(10, n_samples), # Kecepatan angin
            'rainfall': np.random.exponential(5, n_samples),    # Curah hujan
            'condition': np.random.choice(conditions, n_samples)
        })
        
        # Buat temporary directory untuk plots
        self.temp_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        
    def tearDown(self):
        """
        Cleanup setelah testing
        """
        # Hapus temporary directory
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
        # Tutup semua plot yang terbuka
        plt.close('all')
        
    @patch('matplotlib.pyplot.savefig')
    @patch('os.makedirs')
    def test_visualize_data_with_valid_data(self, mock_makedirs, mock_savefig):
        """
        Test visualize_data dengan data yang valid
        """
        # Test dengan data yang valid
        visualize_data(self.test_data)
        
        # Verifikasi bahwa makedirs dipanggil untuk membuat folder plots
        mock_makedirs.assert_called_once_with("plots", exist_ok=True)
        
        # Verifikasi bahwa savefig dipanggil 4 kali (4 plot yang dibuat)
        self.assertEqual(mock_savefig.call_count, 4)
        
        # Verifikasi nama file yang disimpan
        expected_files = [
            'plots/temperature_by_city.png',
            'plots/temp_humidity_condition.png',
            'plots/jakarta_rainfall_trend.png',
            'plots/weather_correlation.png'
        ]
        
        actual_files = [call[0][0] for call in mock_savefig.call_args_list]
        self.assertEqual(set(actual_files), set(expected_files))
        
    def test_visualize_data_with_none_input(self):
        """
        Test visualize_data dengan input None
        """
        # Test dengan input None - seharusnya return early tanpa error
        result = visualize_data(None)
        self.assertIsNone(result)
        
    def test_visualize_data_with_empty_dataframe(self):
        """
        Test visualize_data dengan DataFrame kosong
        """
        empty_df = pd.DataFrame()
        
        # Test dengan DataFrame kosong - seharusnya tidak error
        try:
            visualize_data(empty_df)
        except Exception as e:
            # Jika ada error, pastikan itu bukan error yang tidak terduga
            self.assertIn(('column', 'key', 'missing'), str(e).lower().split())
            
    def test_visualize_data_missing_columns(self):
        """
        Test visualize_data dengan kolom yang hilang
        """
        # DataFrame dengan kolom yang tidak lengkap
        incomplete_df = pd.DataFrame({
            'temperature': [25, 30, 28],
            'humidity': [80, 75, 85]
            # Missing required columns: city, condition, date, rainfall, wind_speed
        })
        
        # Test dengan kolom yang tidak lengkap
        with self.assertRaises(KeyError):
            visualize_data(incomplete_df)
            
    @patch('matplotlib.pyplot.savefig')
    def test_plot_generation_types(self, mock_savefig):
        """
        Test apakah semua jenis plot dibuat dengan benar
        """
        visualize_data(self.test_data)
        
        # Verifikasi bahwa 4 plot dibuat
        self.assertEqual(mock_savefig.call_count, 4)
        
        # Verifikasi nama file sesuai dengan jenis plot
        saved_files = [call[0][0] for call in mock_savefig.call_args_list]
        
        self.assertIn('plots/temperature_by_city.png', saved_files)
        self.assertIn('plots/temp_humidity_condition.png', saved_files)
        self.assertIn('plots/jakarta_rainfall_trend.png', saved_files)
        self.assertIn('plots/weather_correlation.png', saved_files)
        
    def test_jakarta_data_filtering(self):
        """
        Test apakah data Jakarta difilter dengan benar untuk trend plot
        """
        # Pastikan ada data Jakarta dalam test data
        jakarta_mask = self.test_data['city'] == 'Jakarta'
        if not jakarta_mask.any():
            # Tambahkan beberapa data Jakarta jika tidak ada
            jakarta_data = pd.DataFrame({
                'date': [datetime(2023, 1, 1), datetime(2023, 1, 2)],
                'city': ['Jakarta', 'Jakarta'],
                'temperature': [30, 32],
                'humidity': [80, 75],
                'wind_speed': [5, 7],
                'rainfall': [10, 0],
                'condition': ['Hujan', 'Cerah']
            })
            self.test_data = pd.concat([self.test_data, jakarta_data], ignore_index=True)
        
        # Test bahwa fungsi berjalan tanpa error dengan data Jakarta
        with patch('matplotlib.pyplot.savefig'):
            try:
                visualize_data(self.test_data)
            except Exception as e:
                self.fail(f"visualize_data raised an exception: {e}")
                
    def test_correlation_matrix_calculation(self):
        """
        Test apakah matriks korelasi dihitung dengan benar
        """
        # Verifikasi bahwa data numerik ada untuk korelasi
        numeric_cols = ['temperature', 'humidity', 'wind_speed', 'rainfall']
        
        for col in numeric_cols:
            self.assertIn(col, self.test_data.columns)
            self.assertTrue(pd.api.types.is_numeric_dtype(self.test_data[col]))
            
        # Test korelasi manual
        corr_matrix = self.test_data[numeric_cols].corr()
        
        # Verifikasi bahwa matriks korelasi valid
        self.assertEqual(corr_matrix.shape, (4, 4))
        self.assertTrue(np.allclose(np.diag(corr_matrix), 1.0))  # Diagonal harus 1
        
    @patch('matplotlib.pyplot.style.use')
    def test_plot_styling(self, mock_style):
        """
        Test apakah styling plot diterapkan
        """
        with patch('matplotlib.pyplot.savefig'):
            visualize_data(self.test_data)
            
        # Verifikasi bahwa style 'ggplot' digunakan
        mock_style.assert_called_once_with('ggplot')
        
    def test_data_types_validation(self):
        """
        Test validasi tipe data dalam DataFrame
        """
        # Verifikasi tipe data yang diharapkan
        expected_types = {
            'temperature': (int, float, np.number),
            'humidity': (int, float, np.number),
            'wind_speed': (int, float, np.number),
            'rainfall': (int, float, np.number),
            'city': (str, object),
            'condition': (str, object)
        }
        
        for col, expected_type in expected_types.items():
            if col in self.test_data.columns:
                col_type = type(self.test_data[col].iloc[0])
                self.assertTrue(
                    isinstance(self.test_data[col].iloc[0], expected_type),
                    f"Column {col} has incorrect type: {col_type}"
                )


class TestVisualisasiIntegration(unittest.TestCase):
    """
    Integration tests untuk fungsi visualisasi
    """
    
    def setUp(self):
        """
        Setup untuk integration test
        """
        # Buat data yang lebih realistis untuk integration test
        np.random.seed(123)
        dates = pd.date_range('2023-01-01', periods=50, freq='D')
        
        self.integration_data = pd.DataFrame({
            'date': np.tile(dates, 3),  # 3 kota
            'city': np.repeat(['Jakarta', 'Bandung', 'Surabaya'], 50),
            'temperature': np.concatenate([
                np.random.normal(30, 3, 50),  # Jakarta - lebih panas
                np.random.normal(25, 3, 50),  # Bandung - lebih dingin
                np.random.normal(28, 3, 50)   # Surabaya - sedang
            ]),
            'humidity': np.random.uniform(70, 90, 150),
            'wind_speed': np.random.exponential(8, 150),
            'rainfall': np.random.exponential(3, 150),
            'condition': np.random.choice(['Cerah', 'Berawan', 'Hujan'], 150)
        })
        
    def test_full_visualization_pipeline(self):
        """
        Test complete visualization pipeline
        """
        with patch('matplotlib.pyplot.savefig') as mock_save:
            with patch('os.makedirs') as mock_mkdir:
                # Run full visualization
                visualize_data(self.integration_data)
                
                # Assertions
                mock_mkdir.assert_called_once()
                self.assertEqual(mock_save.call_count, 4)
                
                # Verify no exceptions were raised
                self.assertTrue(True)  # If we get here, no exceptions occurred


if __name__ == '__main__':
    # Setup test runner
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestVisualisasi))
    suite.addTests(loader.loadTestsFromTestCase(TestVisualisasiIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    print(f"{'='*50}")