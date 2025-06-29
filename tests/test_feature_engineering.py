"""
Tests for feature engineering functionality
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from feature_engineering import FeatureEngineer


class TestFeatureEngineer:
    
    @pytest.fixture
    def sample_mobility_data(self):
        """Create sample mobility data for testing"""
        data = {
            'county_fips': ['17031', '17043', '36061'],
            'total_trips': [10000, 5000, 15000],
            'avg_trip_duration_minutes': [15.5, 12.3, 18.7],
            'member_trips': [7500, 3750, 11250],
            'casual_trips': [2500, 1250, 3750],
            'inter_county_trips': [500, 250, 750],
            'area_sq_km': [2445, 899, 60]
        }
        return pd.DataFrame(data)
    
    @pytest.fixture
    def sample_spending_data(self):
        """Create sample spending data for testing"""
        data = []
        counties = ['17031', '17043', '36061']
        categories = ['restaurants', 'retail', 'grocery', 'entertainment']
        
        for county in counties:
            for category in categories:
                for month in range(1, 13):
                    data.append({
                        'county_fips': county,
                        'month': month,
                        'year': 2023,
                        'category': category,
                        'spending_amount': np.random.uniform(50000, 200000)
                    })
        
        return pd.DataFrame(data)
    
    @pytest.fixture
    def feature_engineer(self):
        """Create a FeatureEngineer instance for testing"""
        return FeatureEngineer()
    
    def test_population_data_generation(self, feature_engineer):
        """Test population data generation"""
        counties = ['17031', '17043']
        pop_data = feature_engineer.load_population_data(counties)
        
        # Check data structure
        assert isinstance(pop_data, pd.DataFrame)
        assert len(pop_data) > 0
        
        # Check required columns
        required_columns = ['county_fips', 'year', 'population']
        for col in required_columns:
            assert col in pop_data.columns
        
        # Check county coverage
        assert set(pop_data['county_fips'].unique()) == set(counties)
        
        # Check population values are reasonable
        assert (pop_data['population'] > 0).all()
        assert (pop_data['population'] < 20000000).all()  # Reasonable upper bound
    
    def test_mobility_features_creation(self, feature_engineer, sample_mobility_data):
        """Test mobility feature creation"""
        mobility_features = feature_engineer.create_mobility_features(sample_mobility_data)
        
        # Check that original columns are preserved
        for col in sample_mobility_data.columns:
            assert col in mobility_features.columns
        
        # Check that new features were added
        original_cols = set(sample_mobility_data.columns)
        new_cols = set(mobility_features.columns) - original_cols
        assert len(new_cols) > 0
        
        # Check specific features
        expected_features = [
            'peak_hour_ratio', 'weekend_ratio', 'night_trips_ratio',
            'member_ratio', 'trip_volume_category'
        ]
        
        for feature in expected_features:
            if feature not in sample_mobility_data.columns:  # Only check if not already present
                assert feature in mobility_features.columns
        
        # Check value ranges
        if 'member_ratio' in mobility_features.columns:
            assert mobility_features['member_ratio'].between(0, 1).all()
        
        if 'peak_hour_ratio' in mobility_features.columns:
            assert mobility_features['peak_hour_ratio'].between(0, 1).all()
    
    def test_spending_features_creation(self, feature_engineer, sample_spending_data):
        """Test spending feature creation"""
        spending_features = feature_engineer.create_spending_features(sample_spending_data)
        
        # Check data structure
        assert isinstance(spending_features, pd.DataFrame)
        assert len(spending_features) > 0
        
        # Check that spending categories were pivoted
        expected_spending_cols = ['spending_restaurants', 'spending_retail', 'spending_grocery', 'spending_entertainment']
        for col in expected_spending_cols:
            assert col in spending_features.columns
        
        # Check proportion columns
        expected_prop_cols = ['spending_pct_restaurants', 'spending_pct_retail']
        for col in expected_prop_cols:
            assert col in spending_features.columns
        
        # Check that proportions sum to approximately 1
        prop_cols = [col for col in spending_features.columns if 'spending_pct_' in col]
        prop_sums = spending_features[prop_cols].sum(axis=1)
        assert np.allclose(prop_sums, 1.0, atol=0.01)
        
        # Check total spending
        if 'total_spending' in spending_features.columns:
            assert (spending_features['total_spending'] > 0).all()
    
    def test_skewed_distribution_handling(self, feature_engineer):
        """Test skewed distribution transformation"""
        # Create data with skewed distribution
        skewed_data = pd.DataFrame({
            'county_fips': ['17031', '17043', '36061'],
            'normal_feature': [1, 2, 3],
            'skewed_feature': [1, 10, 100]  # Highly skewed
        })
        
        # Apply transformation
        transformed_data, transformers = feature_engineer.handle_skewed_distributions(
            skewed_data, columns=['skewed_feature']
        )
        
        # Check that transformation was applied
        assert 'skewed_feature' in transformers
        
        # Check that skewness was reduced
        original_skew = abs(skewed_data['skewed_feature'].skew())
        transformed_skew = abs(transformed_data['skewed_feature'].skew())
        assert transformed_skew < original_skew
    
    def test_correlation_removal(self, feature_engineer):
        """Test highly correlated feature removal"""
        # Create data with highly correlated features
        corr_data = pd.DataFrame({
            'county_fips': ['17031', '17043', '36061'],
            'feature_1': [1, 2, 3],
            'feature_2': [1.1, 2.1, 3.1],  # Highly correlated with feature_1
            'feature_3': [10, 20, 30]  # Not correlated
        })
        
        # Remove correlated features
        reduced_data, removed_features = feature_engineer.remove_correlated_features(
            corr_data, threshold=0.9
        )
        
        # Check that one of the correlated features was removed
        assert len(removed_features) > 0
        assert 'feature_2' in removed_features or 'feature_1' in removed_features
        
        # Check that uncorrelated feature was kept
        assert 'feature_3' in reduced_data.columns
    
    def test_feature_standardization(self, feature_engineer):
        """Test feature standardization"""
        # Create data with different scales
        scale_data = pd.DataFrame({
            'county_fips': ['17031', '17043', '36061'],
            'small_feature': [1, 2, 3],
            'large_feature': [1000, 2000, 3000]
        })
        
        # Standardize features
        standardized_data = feature_engineer.standardize_features(scale_data)
        
        # Check that numeric features were standardized (mean ~0, std ~1)
        numeric_cols = ['small_feature', 'large_feature']
        for col in numeric_cols:
            assert abs(standardized_data[col].mean()) < 0.1  # Close to 0
            assert abs(standardized_data[col].std() - 1.0) < 0.1  # Close to 1
        
        # Check that non-numeric columns were preserved
        assert 'county_fips' in standardized_data.columns
        assert standardized_data['county_fips'].equals(scale_data['county_fips'])
    
    def test_composite_indices_creation(self, feature_engineer):
        """Test PCA composite indices creation"""
        # Create data with multiple related features
        pca_data = pd.DataFrame({
            'county_fips': ['17031', '17043', '36061'],
            'mobility_feature_1': [1, 2, 3],
            'mobility_feature_2': [1.5, 2.5, 3.5],
            'spending_feature_1': [10, 20, 30],
            'spending_feature_2': [15, 25, 35]
        })
        
        # Define feature groups
        feature_groups = {
            'mobility_index': ['mobility_feature_1', 'mobility_feature_2'],
            'spending_index': ['spending_feature_1', 'spending_feature_2']
        }
        
        # Create composite indices
        indexed_data, pca_results = feature_engineer.create_composite_indices(
            pca_data, feature_groups, n_components=1
        )
        
        # Check that PCA components were added
        assert 'mobility_index_pc1' in indexed_data.columns
        assert 'spending_index_pc1' in indexed_data.columns
        
        # Check PCA results
        assert 'mobility_index' in pca_results
        assert 'spending_index' in pca_results
        
        # Check that variance explained is reasonable
        for index_name, results in pca_results.items():
            assert results['total_variance_explained'] > 0.5  # At least 50% variance explained
    
    def test_complete_pipeline(self, feature_engineer, sample_mobility_data, sample_spending_data):
        """Test complete feature engineering pipeline"""
        # Run complete pipeline
        engineered_features, pipeline_results = feature_engineer.create_feature_pipeline(
            sample_mobility_data, sample_spending_data
        )
        
        # Check output structure
        assert isinstance(engineered_features, pd.DataFrame)
        assert isinstance(pipeline_results, dict)
        
        # Check that pipeline steps were recorded
        assert 'processing_steps' in pipeline_results
        assert len(pipeline_results['processing_steps']) > 0
        
        # Check that feature counts were tracked
        assert 'feature_counts' in pipeline_results
        assert 'final_features' in pipeline_results['feature_counts']
        
        # Check that final dataset has reasonable number of features
        assert engineered_features.shape[1] > len(sample_mobility_data.columns)
        
        # Check that county_fips is preserved
        assert 'county_fips' in engineered_features.columns
        
        # Check that all counties are present
        expected_counties = set(sample_mobility_data['county_fips'])
        actual_counties = set(engineered_features['county_fips'])
        assert expected_counties.issubset(actual_counties)
    
    def test_per_capita_features(self, feature_engineer, sample_mobility_data):
        """Test per-capita feature calculation"""
        # Load population data first
        feature_engineer.load_population_data(['17031', '17043', '36061'])
        
        # Add per-capita features
        mobility_with_per_capita = feature_engineer._add_per_capita_mobility_features(sample_mobility_data)
        
        # Check that per-capita columns were added
        expected_per_capita_cols = ['trips_per_capita', 'mobility_penetration_rate']
        for col in expected_per_capita_cols:
            assert col in mobility_with_per_capita.columns
        
        # Check that per-capita values are reasonable
        assert (mobility_with_per_capita['trips_per_capita'] > 0).all()
        assert mobility_with_per_capita['mobility_penetration_rate'].between(0, 1).all()


if __name__ == "__main__":
    pytest.main([__file__])