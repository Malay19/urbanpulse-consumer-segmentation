"""
Comprehensive Test Suite for Consumer Segmentation Analysis
Includes unit tests, integration tests, data quality tests, and performance benchmarks
"""

import pytest
import pandas as pd
import numpy as np
import geopandas as gpd
from datetime import datetime, timedelta
import tempfile
import shutil
from pathlib import Path
import json
import time
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any
import warnings

# Import all modules for testing
from src.data_loader import DataLoader
from src.data_validator import DataValidator
from src.spatial_processor import SpatialProcessor
from src.feature_engineering import FeatureEngineer
from src.clustering_engine import ClusteringEngine
from src.persona_generator import PersonaGenerator, PersonaType, ConsumerPersona, BusinessOpportunity
from src.dashboard_generator import DashboardGenerator
from pipeline_manager import PipelineManager, PipelineConfig, PipelineStage, CacheManager
from config import DATA_CONFIG, MODEL_CONFIG


class TestDataQuality:
    """Test data quality and validation"""
    
    def test_data_completeness(self):
        """Test that all required data is present"""
        loader = DataLoader()
        
        # Test trip data completeness
        trips_df = loader.download_divvy_data(2023, 6)
        required_trip_columns = [
            'trip_id', 'start_time', 'end_time', 'start_station_id',
            'start_lat', 'start_lng', 'end_lat', 'end_lng', 'member_type'
        ]
        
        for col in required_trip_columns:
            assert col in trips_df.columns, f"Missing required column: {col}"
            assert not trips_df[col].isnull().all(), f"Column {col} is entirely null"
        
        # Test spending data completeness
        spending_df = loader.download_spending_data(['17031'])
        required_spending_columns = ['county_fips', 'month', 'year', 'category', 'spending_amount']
        
        for col in required_spending_columns:
            assert col in spending_df.columns, f"Missing required column: {col}"
            assert not spending_df[col].isnull().all(), f"Column {col} is entirely null"
    
    def test_data_consistency(self):
        """Test data consistency across datasets"""
        loader = DataLoader()
        
        # Load data
        trips_df = loader.download_divvy_data(2023, 6)
        spending_df = loader.download_spending_data(['17031', '36061'])
        boundaries = loader.load_county_boundaries(['17031', '36061'])
        
        # Test coordinate consistency
        assert trips_df['start_lat'].between(-90, 90).all(), "Invalid start latitudes"
        assert trips_df['start_lng'].between(-180, 180).all(), "Invalid start longitudes"
        assert trips_df['end_lat'].between(-90, 90).all(), "Invalid end latitudes"
        assert trips_df['end_lng'].between(-180, 180).all(), "Invalid end longitudes"
        
        # Test temporal consistency
        assert (trips_df['end_time'] >= trips_df['start_time']).all(), "End time before start time"
        
        # Test spending data consistency
        assert (spending_df['spending_amount'] >= 0).all(), "Negative spending amounts"
        assert spending_df['month'].between(1, 12).all(), "Invalid months"
        
        # Test boundary data consistency
        assert boundaries.geometry.is_valid.all(), "Invalid geometries in boundaries"
        assert not boundaries.geometry.is_empty.any(), "Empty geometries in boundaries"
    
    def test_data_ranges(self):
        """Test that data values are within expected ranges"""
        loader = DataLoader()
        trips_df = loader.download_divvy_data(2023, 6)
        
        # Test trip duration ranges
        durations = (trips_df['end_time'] - trips_df['start_time']).dt.total_seconds() / 60
        assert durations.min() >= 0, "Negative trip durations"
        assert durations.max() <= 1440, "Trip durations over 24 hours"
        
        # Test reasonable coordinate ranges (Chicago area)
        assert trips_df['start_lat'].between(41.6, 42.1).all(), "Start latitudes outside Chicago area"
        assert trips_df['start_lng'].between(-88.0, -87.5).all(), "Start longitudes outside Chicago area"
        
        # Test member type values
        valid_member_types = {'member', 'casual'}
        assert set(trips_df['member_type'].unique()).issubset(valid_member_types), "Invalid member types"


class TestSpatialProcessing:
    """Test spatial data processing functionality"""
    
    @pytest.fixture
    def sample_spatial_data(self):
        """Create sample spatial data for testing"""
        # Create sample trips
        trips_data = {
            'trip_id': ['trip_001', 'trip_002', 'trip_003'],
            'start_time': pd.to_datetime(['2023-06-01 08:00:00', '2023-06-01 09:00:00', '2023-06-01 10:00:00']),
            'end_time': pd.to_datetime(['2023-06-01 08:15:00', '2023-06-01 09:20:00', '2023-06-01 10:10:00']),
            'start_station_id': [1, 2, 1],
            'end_station_id': [2, 3, 3],
            'start_lat': [41.88, 41.89, 41.88],
            'start_lng': [-87.63, -87.64, -87.63],
            'end_lat': [41.89, 41.90, 41.90],
            'end_lng': [-87.64, -87.65, -87.65],
            'member_type': ['member', 'casual', 'member']
        }
        trips_df = pd.DataFrame(trips_data)
        
        # Create sample boundaries
        from shapely.geometry import Polygon
        county1 = Polygon([(-87.70, 41.85), (-87.60, 41.85), (-87.60, 41.95), (-87.70, 41.95)])
        boundaries_data = {
            'county_fips': ['17031'],
            'county_name': ['Cook County, IL'],
            'geometry': [county1]
        }
        boundaries = gpd.GeoDataFrame(boundaries_data, crs='EPSG:4326')
        
        return trips_df, boundaries
    
    def test_station_extraction(self, sample_spatial_data):
        """Test station extraction from trip data"""
        trips_df, _ = sample_spatial_data
        processor = SpatialProcessor()
        
        stations_gdf = processor.extract_stations_from_trips(trips_df)
        
        # Check that stations were extracted
        assert len(stations_gdf) == 3  # Unique stations: 1, 2, 3
        assert 'station_id' in stations_gdf.columns
        assert 'lat' in stations_gdf.columns
        assert 'lng' in stations_gdf.columns
        assert 'geometry' in stations_gdf.columns
        assert 'trip_count' in stations_gdf.columns
        
        # Check trip counts
        station_1_trips = stations_gdf[stations_gdf['station_id'] == 1]['trip_count'].iloc[0]
        assert station_1_trips == 3  # 2 starts + 1 end
    
    def test_county_assignment(self, sample_spatial_data):
        """Test station-to-county assignment"""
        trips_df, boundaries = sample_spatial_data
        processor = SpatialProcessor()
        
        stations_gdf = processor.extract_stations_from_trips(trips_df)
        stations_with_counties = processor.assign_stations_to_counties(stations_gdf, boundaries)
        
        # Check that county assignments were made
        assert 'county_fips' in stations_with_counties.columns
        assert not stations_with_counties['county_fips'].isnull().any()
        
        # Check that all assignments are valid
        valid_fips = set(boundaries['county_fips'])
        assigned_fips = set(stations_with_counties['county_fips'])
        assert assigned_fips.issubset(valid_fips)
    
    def test_trip_aggregation(self, sample_spatial_data):
        """Test trip aggregation to county level"""
        trips_df, boundaries = sample_spatial_data
        processor = SpatialProcessor()
        processor.county_boundaries = boundaries
        
        county_aggregations = processor.aggregate_trips_to_county_level(trips_df)
        
        # Check aggregation results
        assert len(county_aggregations) > 0
        assert 'county_fips' in county_aggregations.columns
        assert 'total_trips' in county_aggregations.columns
        assert 'avg_trip_duration_minutes' in county_aggregations.columns
        
        # Check that trip counts match
        assert county_aggregations['total_trips'].sum() == len(trips_df)


class TestFeatureEngineering:
    """Test feature engineering functionality"""
    
    @pytest.fixture
    def sample_feature_data(self):
        """Create sample data for feature engineering"""
        mobility_data = pd.DataFrame({
            'county_fips': ['17031', '17043'],
            'total_trips': [10000, 5000],
            'avg_trip_duration_minutes': [15.5, 12.3],
            'member_trips': [7500, 3750],
            'casual_trips': [2500, 1250]
        })
        
        spending_data = pd.DataFrame({
            'county_fips': ['17031', '17031', '17043', '17043'],
            'month': [6, 6, 6, 6],
            'year': [2023, 2023, 2023, 2023],
            'category': ['restaurants', 'retail', 'restaurants', 'retail'],
            'spending_amount': [100000, 150000, 50000, 75000]
        })
        
        return mobility_data, spending_data
    
    def test_mobility_feature_creation(self, sample_feature_data):
        """Test mobility feature creation"""
        mobility_data, _ = sample_feature_data
        engineer = FeatureEngineer()
        
        mobility_features = engineer.create_mobility_features(mobility_data)
        
        # Check that features were added
        assert mobility_features.shape[1] > mobility_data.shape[1]
        assert 'member_ratio' in mobility_features.columns
        
        # Check feature values
        assert mobility_features['member_ratio'].between(0, 1).all()
    
    def test_spending_feature_creation(self, sample_feature_data):
        """Test spending feature creation"""
        _, spending_data = sample_feature_data
        engineer = FeatureEngineer()
        
        spending_features = engineer.create_spending_features(spending_data)
        
        # Check that spending categories were pivoted
        assert 'spending_restaurants' in spending_features.columns
        assert 'spending_retail' in spending_features.columns
        
        # Check proportion columns
        prop_cols = [col for col in spending_features.columns if 'spending_pct_' in col]
        assert len(prop_cols) > 0
        
        # Check that proportions sum to approximately 1
        prop_sums = spending_features[prop_cols].sum(axis=1)
        assert np.allclose(prop_sums, 1.0, atol=0.01)
    
    def test_feature_standardization(self, sample_feature_data):
        """Test feature standardization"""
        mobility_data, _ = sample_feature_data
        engineer = FeatureEngineer()
        
        standardized_data = engineer.standardize_features(mobility_data)
        
        # Check that numeric features were standardized
        numeric_cols = ['total_trips', 'avg_trip_duration_minutes']
        for col in numeric_cols:
            assert abs(standardized_data[col].mean()) < 0.1  # Close to 0
            assert abs(standardized_data[col].std() - 1.0) < 0.1  # Close to 1


class TestClusteringEngine:
    """Test clustering functionality"""
    
    @pytest.fixture
    def sample_clustering_data(self):
        """Create sample data for clustering"""
        np.random.seed(42)
        n_samples = 50
        
        # Create data with clear cluster structure
        data = {
            'county_fips': [f'county_{i:03d}' for i in range(n_samples)],
            'total_trips': np.concatenate([
                np.random.normal(10000, 1000, 15),  # High trips
                np.random.normal(5000, 500, 20),    # Medium trips
                np.random.normal(2000, 300, 15)     # Low trips
            ]),
            'spending_restaurants': np.concatenate([
                np.random.normal(50000, 5000, 15),   # Low spending
                np.random.normal(100000, 10000, 20), # Medium spending
                np.random.normal(150000, 15000, 15)  # High spending
            ]),
            'member_ratio': np.concatenate([
                np.random.normal(0.8, 0.1, 15),     # High member ratio
                np.random.normal(0.5, 0.1, 20),     # Medium member ratio
                np.random.normal(0.3, 0.1, 15)      # Low member ratio
            ])
        }
        
        return pd.DataFrame(data)
    
    def test_clustering_data_preparation(self, sample_clustering_data):
        """Test clustering data preparation"""
        engine = ClusteringEngine()
        
        feature_matrix, feature_cols, clustering_data = engine.prepare_clustering_data(sample_clustering_data)
        
        # Check output types and dimensions
        assert isinstance(feature_matrix, np.ndarray)
        assert isinstance(feature_cols, list)
        assert isinstance(clustering_data, pd.DataFrame)
        
        assert feature_matrix.shape[0] == len(sample_clustering_data)
        assert feature_matrix.shape[1] == len(feature_cols)
        assert 'county_fips' not in feature_cols
    
    def test_hdbscan_clustering(self, sample_clustering_data):
        """Test HDBSCAN clustering"""
        engine = ClusteringEngine()
        
        feature_matrix, feature_cols, clustering_data = engine.prepare_clustering_data(sample_clustering_data)
        
        clusterer = engine.fit_hdbscan(feature_matrix, min_cluster_size=5, min_samples=3)
        
        # Check that clustering was performed
        assert 'hdbscan' in engine.clusterers
        assert 'hdbscan' in engine.cluster_results
        
        results = engine.cluster_results['hdbscan']
        assert 'labels' in results
        assert 'n_clusters' in results
        assert len(results['labels']) == feature_matrix.shape[0]
    
    def test_kmeans_clustering(self, sample_clustering_data):
        """Test K-Means clustering"""
        engine = ClusteringEngine()
        
        feature_matrix, feature_cols, clustering_data = engine.prepare_clustering_data(sample_clustering_data)
        
        clusterer = engine.fit_kmeans_optimal(feature_matrix, k_range=(2, 5))
        
        # Check that clustering was performed
        assert 'kmeans' in engine.clusterers
        assert 'kmeans' in engine.cluster_results
        
        results = engine.cluster_results['kmeans']
        assert 'optimal_k' in results
        assert 2 <= results['optimal_k'] <= 5
    
    def test_clustering_validation(self, sample_clustering_data):
        """Test clustering validation metrics"""
        engine = ClusteringEngine()
        
        feature_matrix, feature_cols, clustering_data = engine.prepare_clustering_data(sample_clustering_data)
        engine.fit_hdbscan(feature_matrix, min_cluster_size=5, min_samples=3)
        
        cluster_labels = engine.cluster_results['hdbscan']['labels']
        metrics = engine.calculate_internal_metrics(feature_matrix, cluster_labels, 'hdbscan')
        
        # Check that metrics were calculated
        expected_metrics = ['silhouette_score', 'calinski_harabasz_score', 'davies_bouldin_score']
        for metric in expected_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], (int, float))


class TestPersonaGeneration:
    """Test persona generation functionality"""
    
    @pytest.fixture
    def sample_persona_data(self):
        """Create sample data for persona generation"""
        cluster_profiles = {
            'cluster_profiles': {
                'cluster_0': {
                    'cluster_id': 0,
                    'size': 5000,
                    'counties': ['17031'],
                    'feature_statistics': {
                        'total_trips': {'mean': 10000, 'std': 2000},
                        'member_ratio': {'mean': 0.8, 'std': 0.1},
                        'spending_restaurants': {'mean': 100000, 'std': 20000}
                    },
                    'distinguishing_features': {
                        'member_ratio': 2.5,
                        'total_trips': 2.0
                    }
                }
            }
        }
        
        features_df = pd.DataFrame({
            'county_fips': ['17031'],
            'total_trips': [10000],
            'member_ratio': [0.8]
        })
        
        return cluster_profiles, features_df
    
    def test_persona_generation(self, sample_persona_data):
        """Test persona generation from cluster data"""
        cluster_profiles, features_df = sample_persona_data
        generator = PersonaGenerator()
        
        # Load demographics
        generator.load_census_demographics(['17031'])
        
        # Analyze clusters
        cluster_analysis = generator.analyze_cluster_characteristics(cluster_profiles, features_df)
        
        # Generate personas
        personas = generator.generate_persona_narratives(cluster_analysis)
        
        # Check persona generation
        assert isinstance(personas, dict)
        assert len(personas) > 0
        
        for persona_id, persona in personas.items():
            assert isinstance(persona, ConsumerPersona)
            assert persona.persona_name is not None
            assert isinstance(persona.persona_type, PersonaType)
            assert persona.estimated_population > 0
            assert persona.market_value > 0
    
    def test_business_opportunities(self, sample_persona_data):
        """Test business opportunity generation"""
        cluster_profiles, features_df = sample_persona_data
        generator = PersonaGenerator()
        
        # Create sample personas
        sample_personas = {
            'persona_1': ConsumerPersona(
                persona_id='persona_1',
                persona_name='Test Persona',
                persona_type=PersonaType.URBAN_COMMUTER,
                cluster_ids=[0],
                estimated_population=5000,
                median_income=70000,
                age_distribution={},
                education_level={},
                mobility_profile={},
                spending_profile={},
                temporal_patterns={},
                market_value=100000,
                targeting_effectiveness=0.8,
                seasonal_trends={},
                description='Test description',
                key_motivations=['Test'],
                preferred_channels=['Test'],
                pain_points=['Test'],
                marketing_strategies=['Test'],
                product_opportunities=['Test'],
                infrastructure_needs=['Test']
            )
        }
        
        opportunities = generator.generate_business_opportunities(sample_personas)
        
        # Check opportunity generation
        assert isinstance(opportunities, list)
        assert len(opportunities) > 0
        
        for opportunity in opportunities:
            assert isinstance(opportunity, BusinessOpportunity)
            assert opportunity.opportunity_type is not None
            assert opportunity.estimated_market_size > 0


class TestPipelineIntegration:
    """Test complete pipeline integration"""
    
    def test_pipeline_configuration(self):
        """Test pipeline configuration"""
        config = PipelineConfig(
            year=2023,
            month=6,
            counties=['17031'],
            enable_caching=True,
            clustering_algorithms=['kmeans']
        )
        
        assert config.year == 2023
        assert config.month == 6
        assert config.counties == ['17031']
        assert config.enable_caching == True
        assert config.clustering_algorithms == ['kmeans']
    
    def test_cache_manager(self):
        """Test cache management functionality"""
        with tempfile.TemporaryDirectory() as temp_dir:
            cache_manager = CacheManager(Path(temp_dir))
            config = PipelineConfig()
            
            # Test caching
            test_data = {'test': 'data'}
            cache_manager.cache_result(PipelineStage.DATA_LOADING, config, test_data)
            
            # Test retrieval
            cached_data = cache_manager.get_cached_result(PipelineStage.DATA_LOADING, config)
            assert cached_data == test_data
            
            # Test cache clearing
            cache_manager.clear_cache(PipelineStage.DATA_LOADING)
            cached_data = cache_manager.get_cached_result(PipelineStage.DATA_LOADING, config)
            assert cached_data is None
    
    def test_pipeline_stage_execution(self):
        """Test individual pipeline stage execution"""
        config = PipelineConfig(
            counties=['17031'],
            clustering_algorithms=['kmeans'],
            include_visualizations=False
        )
        
        pipeline = PipelineManager(config)
        
        # Test data loading stage
        data_result = pipeline.run_stage_only(PipelineStage.DATA_LOADING)
        
        assert 'trips_df' in data_result
        assert 'spending_df' in data_result
        assert 'boundaries' in data_result
        assert isinstance(data_result['trips_df'], pd.DataFrame)
        assert len(data_result['trips_df']) > 0


class TestPerformanceBenchmarks:
    """Performance benchmarking tests"""
    
    def test_data_loading_performance(self):
        """Test data loading performance"""
        loader = DataLoader()
        
        start_time = time.time()
        trips_df = loader.download_divvy_data(2023, 6)
        loading_time = time.time() - start_time
        
        # Should load data in reasonable time (< 10 seconds for sample data)
        assert loading_time < 10, f"Data loading took too long: {loading_time:.2f} seconds"
        assert len(trips_df) > 0, "No data loaded"
    
    def test_clustering_performance(self):
        """Test clustering performance"""
        # Create sample data
        np.random.seed(42)
        n_samples = 1000
        n_features = 10
        
        data = pd.DataFrame({
            'county_fips': [f'county_{i:03d}' for i in range(n_samples)],
            **{f'feature_{j}': np.random.normal(0, 1, n_samples) for j in range(n_features)}
        })
        
        engine = ClusteringEngine()
        
        start_time = time.time()
        feature_matrix, feature_cols, clustering_data = engine.prepare_clustering_data(data)
        engine.fit_kmeans_optimal(feature_matrix, k_range=(2, 5))
        clustering_time = time.time() - start_time
        
        # Should complete clustering in reasonable time (< 30 seconds)
        assert clustering_time < 30, f"Clustering took too long: {clustering_time:.2f} seconds"
    
    def test_memory_usage(self):
        """Test memory usage during processing"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Run data loading
        loader = DataLoader()
        trips_df = loader.download_divvy_data(2023, 6)
        spending_df = loader.download_spending_data(['17031'])
        
        peak_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = peak_memory - initial_memory
        
        # Should not use excessive memory (< 500 MB increase for sample data)
        assert memory_increase < 500, f"Memory usage too high: {memory_increase:.2f} MB"


class TestErrorHandling:
    """Test error handling and edge cases"""
    
    def test_empty_data_handling(self):
        """Test handling of empty datasets"""
        empty_df = pd.DataFrame()
        
        validator = DataValidator()
        
        # Should handle empty data gracefully
        with pytest.raises((ValueError, AssertionError)):
            validator.validate_divvy_data(empty_df)
    
    def test_invalid_coordinates(self):
        """Test handling of invalid coordinates"""
        invalid_trips = pd.DataFrame({
            'trip_id': ['trip_001'],
            'start_time': [datetime.now()],
            'end_time': [datetime.now()],
            'start_station_id': [1],
            'end_station_id': [2],
            'start_lat': [91.0],  # Invalid latitude
            'start_lng': [-87.63],
            'end_lat': [41.89],
            'end_lng': [-87.64],
            'member_type': ['member']
        })
        
        validator = DataValidator()
        result = validator.validate_divvy_data(invalid_trips)
        
        # Should detect invalid coordinates
        assert not result['passed'] or len(result['warnings']) > 0
    
    def test_missing_required_columns(self):
        """Test handling of missing required columns"""
        incomplete_df = pd.DataFrame({
            'trip_id': ['trip_001'],
            'start_time': [datetime.now()]
            # Missing other required columns
        })
        
        validator = DataValidator()
        result = validator.validate_divvy_data(incomplete_df)
        
        # Should detect missing columns
        assert not result['passed']
        assert len(result['issues']) > 0


def run_comprehensive_test_suite():
    """Run the complete test suite with reporting"""
    print("🧪 Running Comprehensive Test Suite")
    print("=" * 50)
    
    # Configure pytest
    pytest_args = [
        __file__,
        "-v",  # Verbose output
        "--tb=short",  # Short traceback format
        "--durations=10",  # Show 10 slowest tests
        "--cov=src",  # Coverage for src directory
        "--cov-report=html",  # HTML coverage report
        "--cov-report=term-missing",  # Terminal coverage report
    ]
    
    # Run tests
    exit_code = pytest.main(pytest_args)
    
    if exit_code == 0:
        print("\n✅ All tests passed!")
    else:
        print(f"\n❌ Tests failed with exit code: {exit_code}")
    
    return exit_code


def run_performance_benchmarks():
    """Run performance benchmarks separately"""
    print("⚡ Running Performance Benchmarks")
    print("=" * 50)
    
    # Run only performance tests
    pytest_args = [
        __file__ + "::TestPerformanceBenchmarks",
        "-v",
        "--tb=short"
    ]
    
    exit_code = pytest.main(pytest_args)
    return exit_code


def run_data_quality_tests():
    """Run data quality tests separately"""
    print("📊 Running Data Quality Tests")
    print("=" * 50)
    
    # Run only data quality tests
    pytest_args = [
        __file__ + "::TestDataQuality",
        "-v",
        "--tb=short"
    ]
    
    exit_code = pytest.main(pytest_args)
    return exit_code


if __name__ == "__main__":
    # Run comprehensive test suite
    exit_code = run_comprehensive_test_suite()
    
    if exit_code == 0:
        print("\n🎯 Running additional test suites...")
        
        # Run performance benchmarks
        perf_code = run_performance_benchmarks()
        
        # Run data quality tests
        quality_code = run_data_quality_tests()
        
        if perf_code == 0 and quality_code == 0:
            print("\n🎉 All test suites completed successfully!")
        else:
            print(f"\n⚠️  Some test suites failed (Performance: {perf_code}, Quality: {quality_code})")
    
    exit(exit_code)