import unittest
import pandas as pd
import numpy as np
import io
import sys
from unittest.mock import patch, MagicMock, mock_open
from datetime import datetime, timedelta
import tempfile
import os

# Import fungsi yang akan ditest
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))
from dashboard import create_dashboard


class TestDashboard(unittest.TestCase):
    """
    Test class untuk menguji fungsi dashboard data cuaca
    """
    
    def setUp(self):
        """
        Setup data dummy untuk testing dashboard
        """
        # Buat data dummy yang realistis
        np.random.seed(42)
        n_samples = 50
        
        # Generate tanggal berurutan
        start_date = datetime(2023, 1, 1)
        dates = [start_date + timedelta(days=i) for i in range(n_samples)]
        
        # Generate data cuaca
        cities = ['Jakarta', 'Bandung', 'Surabaya', 'Medan', 'Makassar']
        conditions = ['Cerah', 'Berawan', 'Hujan', 'Hujan Ringan', 'Badai']
        
        self.test_data = pd.DataFrame({
            'date': dates,
            'city': np.random.choice(cities, n_samples),
            'temperature': np.random.normal(28, 5, n_samples),
            'humidity': np.random.uniform(60, 95, n_samples),
            'wind_speed': np.random.exponential(10, n_samples),
            'rainfall': np.random.exponential(5, n_samples),
            'condition': np.random.choice(conditions, n_samples)
        })
        
        # Convert date to proper datetime format
        self.test_data['date'] = pd.to_datetime(self.test_data['date'])
        
    def test_create_dashboard_with_valid_data(self):
        """
        Test create_dashboard dengan data yang valid
        """
        # Capture stdout untuk memverifikasi output
        captured_output = io.StringIO()
        sys.stdout = captured_output
        
        # Test fungsi dashboard
        result = create_dashboard(self.test_data)
        
        # Restore stdout
        sys.stdout = sys.__stdout__
        output = captured_output.getvalue()
        
        # Verifikasi bahwa output mengandung template kode dashboard
        self.assertIn("Template kode untuk dashboard visualisasi data cuaca:", output)
        self.assertIn("Dashboard Analisis Cuaca BMKG", output)
        self.assertIn("import streamlit as st", output)
        self.assertIn("st.title('Dashboard Analisis Cuaca BMKG')", output)
        
    def test_create_dashboard_with_none_input(self):
        """
        Test create_dashboard dengan input None
        """
        captured_output = io.StringIO()
        sys.stdout = captured_output
        
        # Test dengan input None
        result = create_dashboard(None)
        
        sys.stdout = sys.__stdout__
        output = captured_output.getvalue()
        
        # Verifikasi bahwa tetap menampilkan template code
        self.assertIn("Template kode untuk dashboard", output)
        
    def test_create_dashboard_with_empty_dataframe(self):
        """
        Test create_dashboard dengan DataFrame kosong
        """
        empty_df = pd.DataFrame()
        
        captured_output = io.StringIO()
        sys.stdout = captured_output
        
        result = create_dashboard(empty_df)
        
        sys.stdout = sys.__stdout__
        output = captured_output.getvalue()
        
        # Verifikasi output tetap valid
        self.assertIn("Template kode untuk dashboard", output)
        
    def test_dashboard_template_content(self):
        """
        Test konten template dashboard
        """
        captured_output = io.StringIO()
        sys.stdout = captured_output
        
        create_dashboard(self.test_data)
        
        sys.stdout = sys.__stdout__
        output = captured_output.getvalue()
        
        # Verifikasi komponen utama dashboard ada dalam template
        essential_components = [
            "st.title('Dashboard Analisis Cuaca BMKG')",
            "st.sidebar.header('Filter')",
            "st.sidebar.selectbox('Pilih Kota'",
            "st.header(f'Data Cuaca untuk {selected_city}')",
            "st.subheader('Statistik Cuaca')",
            "st.subheader('Tren Suhu')",
            "st.subheader('Distribusi Kondisi Cuaca')",
            "st.checkbox('Tampilkan data mentah')"
        ]
        
        for component in essential_components:
            self.assertIn(component, output, f"Missing component: {component}")
            
    def test_dashboard_imports(self):
        """
        Test apakah semua import yang diperlukan ada dalam template
        """
        captured_output = io.StringIO()
        sys.stdout = captured_output
        
        create_dashboard(self.test_data)
        
        sys.stdout = sys.__stdout__
        output = captured_output.getvalue()
        
        # Verifikasi import statements
        required_imports = [
            "import streamlit as st",
            "import pandas as pd",
            "import matplotlib.pyplot as plt",
            "import seaborn as sns"
        ]
        
        for import_stmt in required_imports:
            self.assertIn(import_stmt, output, f"Missing import: {import_stmt}")
            
    def test_dashboard_data_operations(self):
        """
        Test operasi data dalam template dashboard
        """
        captured_output = io.StringIO()
        sys.stdout = captured_output
        
        create_dashboard(self.test_data)
        
        sys.stdout = sys.__stdout__
        output = captured_output.getvalue()
        
        # Verifikasi operasi data
        data_operations = [
            "pd.read_csv('data/weather_data.csv')",
            "pd.to_datetime(df['date'])",
            "df['city'].unique()",
            "df[df['city'] == selected_city]",
            ".describe()",
            ".value_counts()"
        ]
        
        for operation in data_operations:
            self.assertIn(operation, output, f"Missing data operation: {operation}")
            
    def test_dashboard_visualization_components(self):
        """
        Test komponen visualisasi dalam template
        """
        captured_output = io.StringIO()
        sys.stdout = captured_output
        
        create_dashboard(self.test_data)
        
        sys.stdout = sys.__stdout__
        output = captured_output.getvalue()
        
        # Verifikasi komponen visualisasi
        viz_components = [
            "plt.subplots(figsize=(10, 6))",
            "ax.plot(",
            "ax.set_xlabel('Tanggal')",
            "ax.set_ylabel('Suhu (°C)')",
            "st.pyplot(fig)",
            "st.bar_chart(condition_counts)",
            "st.dataframe(stats)"
        ]
        
        for component in viz_components:
            self.assertIn(component, output, f"Missing visualization component: {component}")


class TestDashboardIntegration(unittest.TestCase):
    """
    Integration tests untuk dashboard template
    """
    
    def setUp(self):
        """
        Setup untuk integration test
        """
        # Buat data yang lebih komprehensif
        np.random.seed(123)
        dates = pd.date_range('2023-01-01', periods=100, freq='D')
        cities = ['Jakarta', 'Bandung', 'Surabaya', 'Medan', 'Makassar', 'Denpasar']
        
        # Buat data dengan pattern yang realistis per kota
        data_list = []
        for city in cities:
            n_days = len(dates)
            if city == 'Jakarta':
                base_temp = 30
                base_humidity = 80
            elif city == 'Bandung':
                base_temp = 24
                base_humidity = 75
            elif city == 'Surabaya':
                base_temp = 32
                base_humidity = 85
            else:
                base_temp = 28
                base_humidity = 78
                
            city_data = pd.DataFrame({
                'date': dates,
                'city': city,
                'temperature': np.random.normal(base_temp, 3, n_days),
                'humidity': np.random.normal(base_humidity, 8, n_days),
                'wind_speed': np.random.exponential(8, n_days),
                'rainfall': np.random.exponential(3, n_days),
                'condition': np.random.choice(['Cerah', 'Berawan', 'Hujan', 'Hujan Ringan'], n_days)
            })
            data_list.append(city_data)
            
        self.integration_data = pd.concat(data_list, ignore_index=True)
        
    def test_dashboard_template_completeness(self):
        """
        Test kelengkapan template dashboard
        """
        captured_output = io.StringIO()
        sys.stdout = captured_output
        
        create_dashboard(self.integration_data)
        
        sys.stdout = sys.__stdout__
        template_code = captured_output.getvalue()
        
        # Verifikasi struktur lengkap dashboard
        self.assertIn("# Dashboard Analisis Cuaca BMKG", template_code)
        self.assertIn("# Implementasi dengan Streamlit", template_code)
        
        # Verifikasi section utama
        sections = [
            "# Load data",
            "# Sidebar untuk filter", 
            "# Filter data berdasarkan kota",
            "# Tampilkan informasi umum",
            "# Tampilkan statistik",
            "# Plot tren suhu",
            "# Plot kondisi cuaca",
            "# Tampilkan data mentah jika diminta"
        ]
        
        for section in sections:
            self.assertIn(section, template_code, f"Missing section: {section}")
            
    def test_dashboard_functionality_simulation(self):
        """
        Test simulasi fungsionalitas dashboard
        """
        # Simulasi operasi yang akan dilakukan dashboard
        
        # 1. Test filtering by city
        cities = self.integration_data['city'].unique()
        self.assertGreater(len(cities), 0)
        
        for city in cities:
            filtered_data = self.integration_data[self.integration_data['city'] == city]
            self.assertGreater(len(filtered_data), 0, f"No data for city: {city}")
            
        # 2. Test statistics calculation
        numeric_cols = ['temperature', 'humidity', 'wind_speed', 'rainfall']
        stats = self.integration_data[numeric_cols].describe()
        self.assertEqual(stats.shape[1], 4)  # 4 kolom numerik
        self.assertEqual(stats.shape[0], 8)  # 8 statistik (count, mean, std, dll)
        
        # 3. Test condition value counts
        for city in cities:
            city_data = self.integration_data[self.integration_data['city'] == city]
            condition_counts = city_data['condition'].value_counts()
            self.assertGreater(len(condition_counts), 0)
            
    def test_data_quality_for_dashboard(self):
        """
        Test kualitas data untuk dashboard
        """
        # Verifikasi data memiliki kolom yang diperlukan
        required_columns = ['date', 'city', 'temperature', 'humidity', 'wind_speed', 'rainfall', 'condition']
        for col in required_columns:
            self.assertIn(col, self.integration_data.columns, f"Missing required column: {col}")
            
        # Verifikasi tipe data
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(self.integration_data['date']))
        self.assertTrue(pd.api.types.is_numeric_dtype(self.integration_data['temperature']))
        self.assertTrue(pd.api.types.is_numeric_dtype(self.integration_data['humidity']))
        
        # Verifikasi tidak ada nilai null pada kolom kritis
        critical_columns = ['city', 'temperature', 'date']
        for col in critical_columns:
            null_count = self.integration_data[col].isnull().sum()
            self.assertEqual(null_count, 0, f"Found null values in critical column: {col}")


class TestDashboardErrorHandling(unittest.TestCase):
    """
    Test error handling untuk dashboard
    """
    
    def test_dashboard_with_missing_columns(self):
        """
        Test dashboard dengan DataFrame yang memiliki kolom hilang
        """
        # DataFrame dengan kolom tidak lengkap
        incomplete_df = pd.DataFrame({
            'temperature': [25, 30, 28],
            'humidity': [80, 75, 85]
        })
        
        # Dashboard seharusnya tetap berjalan karena hanya menampilkan template
        captured_output = io.StringIO()
        sys.stdout = captured_output
        
        try:
            create_dashboard(incomplete_df)
            sys.stdout = sys.__stdout__
            output = captured_output.getvalue()
            
            # Verifikasi template tetap ditampilkan
            self.assertIn("Template kode untuk dashboard", output)
        except Exception as e:
            self.fail(f"Dashboard should handle incomplete data gracefully: {e}")
            
    def test_dashboard_with_invalid_data_types(self):
        """
        Test dashboard dengan tipe data yang salah
        """
        # DataFrame dengan tipe data yang salah
        invalid_df = pd.DataFrame({
            'date': ['not_a_date', 'also_not_date'],
            'city': [123, 456],  # Seharusnya string
            'temperature': ['hot', 'cold'],  # Seharusnya numeric
            'humidity': [True, False]  # Seharusnya numeric
        })
        
        captured_output = io.StringIO()
        sys.stdout = captured_output
        
        # Dashboard seharusnya tetap berjalan
        create_dashboard(invalid_df)
        
        sys.stdout = sys.__stdout__
        output = captured_output.getvalue()
        
        # Verifikasi template tetap ditampilkan
        self.assertIn("Template kode untuk dashboard", output)


if __name__ == '__main__':
    # Setup test runner dengan custom formatting
    class CustomTestResult(unittest.TextTestResult):
        def addSuccess(self, test):
            super().addSuccess(test)
            if self.verbosity > 1:
                self.stream.write(f"✓ {test._testMethodName}\n")
                
        def addError(self, test, err):
            super().addError(test, err)
            self.stream.write(f"✗ {test._testMethodName} - ERROR\n")
            
        def addFailure(self, test, err):
            super().addFailure(test, err)
            self.stream.write(f"✗ {test._testMethodName} - FAILED\n")
    
    class CustomTestRunner(unittest.TextTestRunner):
        resultclass = CustomTestResult
    
    # Load dan run tests
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTests(loader.loadTestsFromTestCase(TestDashboard))
    suite.addTests(loader.loadTestsFromTestCase(TestDashboardIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestDashboardErrorHandling))
    
    # Run tests
    print("🧪 Running Dashboard Tests...")
    print("=" * 60)
    
    runner = CustomTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print detailed summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    print(f"Total Tests Run: {result.testsRun}")
    print(f"✓ Passed: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"✗ Failed: {len(result.failures)}")
    print(f"⚠ Errors: {len(result.errors)}")
    
    if result.failures:
        print(f"\n❌ FAILURES:")
        for test, traceback in result.failures:
            print(f"  - {test}")
            
    if result.errors:
        print(f"\n⚠️ ERRORS:")
        for test, traceback in result.errors:
            print(f"  - {test}")
    
    # Calculate success rate
    success_rate = ((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100) if result.testsRun > 0 else 0
    print(f"\n🎯 Success Rate: {success_rate:.1f}%")
    
    if success_rate == 100:
        print("🎉 All tests passed! Dashboard is ready for deployment.")
    elif success_rate >= 80:
        print("⚡ Most tests passed. Minor issues need attention.")
    else:
        print("🔧 Several issues found. Please review failed tests.")
        
    print("=" * 60)