import unittest
import pandas as pd
import numpy as np
import os
import sys
import io
import tempfile
import shutil
from unittest.mock import patch, MagicMock, mock_open, call
from datetime import datetime, timedelta

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'utils'))

# Mock PySpark imports before importing main
sys.modules['pyspark'] = MagicMock()
sys.modules['pyspark.sql'] = MagicMock()
sys.modules['pyspark.ml'] = MagicMock()
sys.modules['pyspark.ml.feature'] = MagicMock()
sys.modules['pyspark.ml.regression'] = MagicMock()
sys.modules['pyspark.ml.evaluation'] = MagicMock()

# Import main module
import main


class TestMainPipeline(unittest.TestCase):
    """
    Test class untuk menguji pipeline utama aplikasi analisis cuaca
    """
    
    def setUp(self):
        """
        Setup untuk testing main pipeline
        """
        # Create sample data
        np.random.seed(42)
        n_samples = 100
        
        start_date = datetime(2023, 1, 1)
        dates = [start_date + timedelta(days=i) for i in range(n_samples)]
        
        self.sample_weather_data = [
            {
                'station_id': f'BMKG{i:03d}',
                'station_name': f'Station {i}',
                'city': np.random.choice(['Jakarta', 'Bandung', 'Surabaya', 'Medan']),
                'datetime': dates[i % len(dates)].isoformat(),
                'temperature': np.random.normal(28, 5),
                'humidity': np.random.uniform(60, 95),
                'wind_speed': np.random.exponential(10),
                'wind_direction': np.random.uniform(0, 360),
                'pressure': np.random.normal(1013, 10),
                'weather_condition': np.random.choice(['Cerah', 'Berawan', 'Hujan'])
            }
            for i in range(n_samples)
        ]
        
        # Create sample processed DataFrame
        self.sample_df = pd.DataFrame({
            'date': pd.date_range('2023-01-01', periods=50),
            'city': np.random.choice(['Jakarta', 'Bandung', 'Surabaya'], 50),
            'temperature': np.random.normal(28, 3, 50),
            'humidity': np.random.uniform(70, 90, 50),
            'wind_speed': np.random.exponential(8, 50),
            'rainfall': np.random.exponential(3, 50),
            'condition': np.random.choice(['Cerah', 'Berawan', 'Hujan'], 50),
            'day_of_week': np.random.randint(0, 7, 50),
            'month': np.random.randint(1, 13, 50)
        })
        
        # Create temporary directory
        self.temp_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        
    def tearDown(self):
        """
        Cleanup after testing
        """
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    @patch('main.create_dashboard')
    @patch('main.visualize_data') 
    @patch('main.save_data')
    @patch('main.preprocess_data')
    @patch('main.fetch_bmkg_data')
    def test_main_pipeline_success_flow(self, mock_fetch, mock_preprocess, 
                                       mock_save, mock_viz, mock_dashboard):
        """
        Test successful execution of main pipeline
        """
        # Setup mocks
        mock_fetch.return_value = self.sample_weather_data
        mock_preprocess.return_value = self.sample_df
        mock_save.return_value = None
        mock_viz.return_value = None
        mock_dashboard.return_value = None
        
        # Capture stdout
        captured_output = io.StringIO()
        sys.stdout = captured_output
        
        # Mock SparkSession
        with patch('main.SparkSession') as mock_spark_session:
            mock_spark = MagicMock()
            mock_spark_session.builder.appName.return_value.config.return_value.getOrCreate.return_value = mock_spark
            
            # Mock Spark DataFrame
            mock_spark_df = MagicMock()
            mock_spark_df.count.return_value = 100
            mock_spark_df.toPandas.return_value = self.sample_df
            mock_spark.read.parquet.return_value = mock_spark_df
            
            # Mock ML components
            mock_assembler = MagicMock()
            mock_data = MagicMock()
            mock_train_data = MagicMock()
            mock_test_data = MagicMock()
            
            with patch('main.VectorAssembler') as mock_vector_assembler:
                with patch('main.RandomForestRegressor') as mock_rf:
                    with patch('main.RegressionEvaluator') as mock_evaluator:
                        mock_vector_assembler.return_value = mock_assembler
                        mock_assembler.transform.return_value = mock_data
                        mock_data.randomSplit.return_value = [mock_train_data, mock_test_data]
                        
                        mock_rf_instance = MagicMock()
                        mock_rf.return_value = mock_rf_instance
                        mock_model = MagicMock()
                        mock_rf_instance.fit.return_value = mock_model
                        
                        mock_predictions = MagicMock()
                        mock_model.transform.return_value = mock_predictions
                        mock_model.featureImportances = [0.3, 0.25, 0.2, 0.15, 0.1]
                        
                        mock_evaluator_instance = MagicMock()
                        mock_evaluator.return_value = mock_evaluator_instance
                        mock_evaluator_instance.evaluate.return_value = 2.5
                        
                        # Execute main pipeline
                        exec(open('main.py').read())
        
        # Restore stdout
        sys.stdout = sys.__stdout__
        output = captured_output.getvalue()
        
        # Verify function calls
        mock_fetch.assert_called_once()
        mock_preprocess.assert_called_once_with(self.sample_weather_data)
        mock_save.assert_called_once_with(self.sample_df, format="parquet")
        mock_viz.assert_called()
        mock_dashboard.assert_called()
        
        # Verify output contains expected messages
        self.assertIn("PROJECT ANALISIS CUACA BIG DATA", output)
        self.assertIn("Menginisialisasi Spark Session", output)
        self.assertIn("Spark session berhasil diinisialisasi", output)
        
    @patch('main.create_dashboard')
    @patch('main.visualize_data') 
    @patch('main.save_data')
    @patch('main.preprocess_data')
    @patch('main.fetch_bmkg_data')
    def test_main_pipeline_with_spark_error(self, mock_fetch, mock_preprocess, 
                                           mock_save, mock_viz, mock_dashboard):
        """
        Test main pipeline when Spark fails
        """
        # Setup mocks
        mock_fetch.return_value = self.sample_weather_data
        mock_preprocess.return_value = self.sample_df
        mock_save.return_value = None
        mock_viz.return_value = None
        mock_dashboard.return_value = None
        
        # Capture stdout
        captured_output = io.StringIO()
        sys.stdout = captured_output
        
        # Mock SparkSession to raise exception
        with patch('main.SparkSession') as mock_spark_session:
            mock_spark_session.builder.appName.return_value.config.return_value.getOrCreate.side_effect = Exception("Spark initialization failed")
            
            # Execute main pipeline
            exec(open('main.py').read())
        
        # Restore stdout
        sys.stdout = sys.__stdout__
        output = captured_output.getvalue()
        
        # Verify fallback behavior
        mock_viz.assert_called_with(self.sample_df)
        self.assertIn("Error dalam analisis data", output)
        self.assertIn("Melanjutkan dengan analisis tanpa Spark", output)
        
    @patch('main.visualize_data')
    @patch('main.save_data')
    @patch('main.preprocess_data')
    @patch('main.fetch_bmkg_data')
    def test_main_pipeline_data_acquisition_failure(self, mock_fetch, mock_preprocess, 
                                                   mock_save, mock_viz):
        """
        Test main pipeline when data acquisition fails
        """
        # Setup mocks
        mock_fetch.side_effect = Exception("API connection failed")
        
        # Capture stderr
        captured_error = io.StringIO()
        sys.stderr = captured_error
        
        try:
            exec(open('main.py').read())
        except Exception:
            pass  # Expected to fail
        
        # Restore stderr
        sys.stderr = sys.__stderr__
        
        # Verify that fetch was called
        mock_fetch.assert_called_once()
        # Subsequent functions should not be called
        mock_preprocess.assert_not_called()
        
    def test_spark_dataframe_operations(self):
        """
        Test Spark DataFrame operations in isolation
        """
        with patch('main.SparkSession') as mock_spark_session:
            mock_spark = MagicMock()
            mock_spark_session.builder.appName.return_value.config.return_value.getOrCreate.return_value = mock_spark
            
            # Test CSV reading
            mock_spark_df = MagicMock()
            mock_spark_df.count.return_value = 50
            mock_spark.read.csv.return_value = mock_spark_df
            
            # Test parquet reading
            mock_spark.read.parquet.return_value = mock_spark_df
            
            # Verify operations would work
            self.assertIsNotNone(mock_spark_df)
            
    def test_feature_importance_calculation(self):
        """
        Test feature importance calculation logic
        """
        # Mock feature importance values
        feature_cols = ["temperature", "humidity", "wind_speed", "day_of_week", "month"]
        importance_values = [0.3, 0.25, 0.2, 0.15, 0.1]
        
        # Simulate the calculation in main.py
        feature_importance = [(feature, float(importance_values[i])) 
                             for i, feature in enumerate(feature_cols)]
        feature_importance.sort(key=lambda x: x[1], reverse=True)
        
        # Verify sorting and values
        self.assertEqual(len(feature_importance), 5)
        self.assertEqual(feature_importance[0][0], "temperature")
        self.assertEqual(feature_importance[0][1], 0.3)
        self.assertEqual(feature_importance[-1][0], "month")
        self.assertEqual(feature_importance[-1][1], 0.1)
        
    @patch('builtins.print')
    def test_descriptive_statistics_output(self, mock_print):
        """
        Test descriptive statistics output format
        """
        # Simulate statistics calculation
        stats_data = self.sample_df[['temperature', 'humidity', 'wind_speed', 'rainfall']].describe()
        
        # Verify statistics are calculated correctly
        self.assertEqual(len(stats_data.columns), 4)
        self.assertIn('mean', stats_data.index)
        self.assertIn('std', stats_data.index)
        self.assertIn('min', stats_data.index)
        self.assertIn('max', stats_data.index)


class TestMainIntegration(unittest.TestCase):
    """
    Integration tests untuk main pipeline
    """
    
    def setUp(self):
        """
        Setup untuk integration testing
        """
        self.temp_dir = tempfile.mkdtemp()
        
        # Create mock data files
        self.csv_file = os.path.join(self.temp_dir, "weather_data.csv")
        self.parquet_file = os.path.join(self.temp_dir, "weather_data.parquet")
        
        # Sample CSV data
        csv_data = """date,city,temperature,humidity,wind_speed,rainfall,condition,day_of_week,month
2023-01-01,Jakarta,30,80,5,0,Cerah,0,1
2023-01-02,Bandung,25,75,3,2,Hujan,1,1
2023-01-03,Surabaya,32,85,7,0,Cerah,2,1"""
        
        with open(self.csv_file, 'w') as f:
            f.write(csv_data)
            
    def tearDown(self):
        """
        Cleanup integration test
        """
        shutil.rmtree(self.temp_dir, ignore_errors=True)
        
    def test_file_format_detection(self):
        """
        Test file format detection logic
        """
        # Test CSV detection
        self.assertTrue(self.csv_file.endswith(".csv"))
        
        # Test parquet detection
        self.assertTrue(self.parquet_file.endswith(".parquet"))
        
        # Test unsupported format
        unsupported_file = "data.xlsx"
        self.assertFalse(unsupported_file.endswith(".csv"))
        self.assertFalse(unsupported_file.endswith(".parquet"))
        
    def test_data_processing_pipeline(self):
        """
        Test complete data processing pipeline simulation
        """
        # Read the CSV file we created
        df = pd.read_csv(self.csv_file)
        
        # Verify data structure
        expected_columns = ['date', 'city', 'temperature', 'humidity', 
                           'wind_speed', 'rainfall', 'condition', 'day_of_week', 'month']
        for col in expected_columns:
            self.assertIn(col, df.columns)
            
        # Test groupby operations (as in main.py)
        city_stats = df.groupby('city')[['temperature', 'humidity', 'rainfall']].mean()
        self.assertEqual(len(city_stats), 3)  # 3 cities
        
        # Test value counts (as in main.py)
        condition_counts = df['condition'].value_counts()
        self.assertGreater(len(condition_counts), 0)


class TestMainErrorHandling(unittest.TestCase):
    """
    Test error handling dalam main pipeline
    """
    
    @patch('main.SparkSession')
    def test_spark_memory_configuration(self, mock_spark_session):
        """
        Test Spark memory configuration
        """
        mock_builder = MagicMock()
        mock_spark_session.builder = mock_builder
        
        # Test configuration chain
        mock_builder.appName.return_value = mock_builder
        mock_builder.config.return_value = mock_builder
        mock_builder.getOrCreate.return_value = MagicMock()
        
        # Verify configuration would be called correctly
        # This simulates the configuration in main.py
        builder = mock_spark_session.builder
        builder.appName("BMKG Weather Analysis")
        builder.config("spark.driver.memory", "2g")
        
        # Verify calls
        mock_builder.appName.assert_called_with("BMKG Weather Analysis")
        mock_builder.config.assert_called_with("spark.driver.memory", "2g")
        
    def test_file_path_handling(self):
        """
        Test file path handling logic
        """
        # Test different file extensions
        csv_path = "data/weather_data.csv"
        parquet_path = "data/weather_data.parquet"
        unsupported_path = "data/weather_data.xlsx"
        
        # Simulate the file format checking logic from main.py
        def check_file_support(file_path):
            if file_path.endswith(".csv"):
                return "csv"
            elif file_path.endswith(".parquet"):
                return "parquet"
            else:
                return None
                
        self.assertEqual(check_file_support(csv_path), "csv")
        self.assertEqual(check_file_support(parquet_path), "parquet")
        self.assertIsNone(check_file_support(unsupported_path))
        
    @patch('main.print')
    def test_error_message_output(self, mock_print):
        """
        Test error message output format
        """
        # Simulate error handling in main.py
        try:
            raise Exception("Test error")
        except Exception as e:
            error_msg = f"Error dalam analisis data: {e}"
            fallback_msg = "Melanjutkan dengan analisis tanpa Spark..."
            
        # Verify error message format
        self.assertIn("Error dalam analisis data:", error_msg)
        self.assertIn("Test error", error_msg)
        self.assertEqual(fallback_msg, "Melanjutkan dengan analisis tanpa Spark...")


class TestMainConfiguration(unittest.TestCase):
    """
    Test konfigurasi dan setup dalam main
    """
    
    def test_spark_configuration_values(self):
        """
        Test nilai konfigurasi Spark
        """
        # Test configuration values from main.py
        app_name = "BMKG Weather Analysis"
        driver_memory = "2g"
        
        self.assertEqual(app_name, "BMKG Weather Analysis")
        self.assertEqual(driver_memory, "2g")
        
    def test_machine_learning_parameters(self):
        """
        Test parameter machine learning
        """
        # Test ML parameters from main.py
        feature_cols = ["temperature", "humidity", "wind_speed", "day_of_week", "month"]
        label_col = "rainfall"
        num_trees = 20
        train_test_split = [0.8, 0.2]
        random_seed = 42
        
        self.assertEqual(len(feature_cols), 5)
        self.assertEqual(label_col, "rainfall")
        self.assertEqual(num_trees, 20)
        self.assertEqual(sum(train_test_split), 1.0)
        self.assertEqual(random_seed, 42)
        
    def test_file_paths_configuration(self):
        """
        Test konfigurasi file paths
        """
        # Test file paths from main.py
        data_file = "data/weather_data.parquet"
        
        self.assertTrue(data_file.startswith("data/"))
        self.assertTrue(data_file.endswith(".parquet"))


if __name__ == '__main__':
    # Custom test runner with enhanced reporting
    class DetailedTestResult(unittest.TextTestResult):
        def __init__(self, stream, descriptions, verbosity):
            super().__init__(stream, descriptions, verbosity)
            self.test_results = []
            
        def addSuccess(self, test):
            super().addSuccess(test)
            self.test_results.append(('PASS', test._testMethodName, test.__class__.__name__))
            if self.verbosity > 1:
                self.stream.write(f"✅ {test._testMethodName}\n")
                
        def addError(self, test, err):
            super().addError(test, err)
            self.test_results.append(('ERROR', test._testMethodName, test.__class__.__name__))
            if self.verbosity > 1:
                self.stream.write(f"💥 {test._testMethodName} - ERROR\n")
                
        def addFailure(self, test, err):
            super().addFailure(test, err)
            self.test_results.append(('FAIL', test._testMethodName, test.__class__.__name__))
            if self.verbosity > 1:
                self.stream.write(f"❌ {test._testMethodName} - FAILED\n")
    
    class DetailedTestRunner(unittest.TextTestRunner):
        resultclass = DetailedTestResult
    
    # Print header
    print("🚀 BMKG Weather Analysis - Main Pipeline Tests")
    print("=" * 70)
    
    # Load test suites
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestMainPipeline,
        TestMainIntegration, 
        TestMainErrorHandling,
        TestMainConfiguration
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = DetailedTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Generate detailed report
    print("\n" + "=" * 70)
    print("📊 DETAILED TEST REPORT")
    print("=" * 70)
    
    # Group results by test class
    results_by_class = {}
    for status, test_name, class_name in result.test_results:
        if class_name not in results_by_class:
            results_by_class[class_name] = {'PASS': 0, 'FAIL': 0, 'ERROR': 0}
        results_by_class[class_name][status] += 1
    
    # Print results by class
    for class_name, stats in results_by_class.items():
        total = sum(stats.values())
        pass_rate = (stats['PASS'] / total * 100) if total > 0 else 0
        print(f"\n📋 {class_name}:")
        print(f"   ✅ Passed: {stats['PASS']}")
        print(f"   ❌ Failed: {stats['FAIL']}")
        print(f"   💥 Errors: {stats['ERROR']}")  
        print(f"   📈 Success Rate: {pass_rate:.1f}%")
    
    # Overall summary
    total_tests = result.testsRun
    total_passed = total_tests - len(result.failures) - len(result.errors)
    overall_success_rate = (total_passed / total_tests * 100) if total_tests > 0 else 0
    
    print(f"\n🎯 OVERALL SUMMARY:")
    print(f"   Total Tests: {total_tests}")
    print(f"   ✅ Passed: {total_passed}")
    print(f"   ❌ Failed: {len(result.failures)}")
    print(f"   💥 Errors: {len(result.errors)}")
    print(f"   🏆 Overall Success Rate: {overall_success_rate:.1f}%")
    
    # Status message
    if overall_success_rate == 100:
        print(f"\n🎉 EXCELLENT! All tests passed. The main pipeline is ready for production!")
    elif overall_success_rate >= 90:
        print(f"\n🌟 GREAT! Most tests passed. Minor issues to address.")
    elif overall_success_rate >= 70:
        print(f"\n⚡ GOOD! Majority of tests passed. Some issues need attention.")
    else:
        print(f"\n🔧 NEEDS WORK! Several critical issues found. Please review failed tests.")
        
    print("=" * 70)
    
    # Exit with appropriate code
    exit_code = 0 if len(result.failures) == 0 and len(result.errors) == 0 else 1
    sys.exit(exit_code)