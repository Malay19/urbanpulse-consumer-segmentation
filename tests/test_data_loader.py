"""
Tests for data loading functionality
"""

import pytest
import pandas as pd
import geopandas as gpd
from datetime import datetime
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from data_loader import DataLoader
from config import DATA_CONFIG


class TestDataLoader:
    
    @pytest.fixture
    def data_loader(self):
        """Create a DataLoader instance for testing"""
        return DataLoader()
    
    def test_divvy_data_generation(self, data_loader):
        """Test Divvy data generation"""
        df = data_loader.download_divvy_data(2023, 6)
        
        # Check data structure
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        
        # Check required columns
        required_columns = [
            'trip_id', 'start_time', 'end_time', 'start_station_id',
            'start_lat', 'start_lng', 'end_lat', 'end_lng', 'member_type'
        ]
        for col in required_columns:
            assert col in df.columns
        
        # Check data types
        assert pd.api.types.is_datetime64_any_dtype(df['start_time'])
        assert pd.api.types.is_datetime64_any_dtype(df['end_time'])
        assert pd.api.types.is_numeric_dtype(df['start_lat'])
        assert pd.api.types.is_numeric_dtype(df['start_lng'])
        
        # Check coordinate bounds (Chicago area)
        assert df['start_lat'].between(41.6, 42.1).all()
        assert df['start_lng'].between(-88.0, -87.5).all()
        
        # Check member types
        assert df['member_type'].isin(['member', 'casual']).all()
    
    def test_spending_data_generation(self, data_loader):
        """Test spending data generation"""
        counties = ["17031", "36061"]  # Cook County, NY County
        df = data_loader.download_spending_data(counties)
        
        # Check data structure
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        
        # Check required columns
        required_columns = ['county_fips', 'month', 'year', 'category', 'spending_amount']
        for col in required_columns:
            assert col in df.columns
        
        # Check county coverage
        assert set(df['county_fips'].unique()) == set(counties)
        
        # Check date ranges
        assert df['month'].between(1, 12).all()
        assert df['year'].between(2022, 2024).all()
        
        # Check spending amounts
        assert (df['spending_amount'] >= 0).all()
        
        # Check categories
        expected_categories = {
            'restaurants', 'retail', 'grocery', 'entertainment',
            'transportation', 'healthcare', 'education', 'services'
        }
        assert set(df['category'].unique()).issubset(expected_categories)
    
    def test_boundary_data_generation(self, data_loader):
        """Test boundary data generation"""
        counties = ["17031", "36061"]
        gdf = data_loader.load_county_boundaries(counties)
        
        # Check data structure
        assert isinstance(gdf, gpd.GeoDataFrame)
        assert len(gdf) > 0
        
        # Check required columns
        assert 'county_fips' in gdf.columns
        assert 'geometry' in gdf.columns
        
        # Check county coverage
        assert set(gdf['county_fips'].unique()) == set(counties)
        
        # Check geometries
        assert gdf.geometry.is_valid.all()
        assert not gdf.geometry.is_empty.any()
        
        # Check CRS
        assert gdf.crs is not None
    
    def test_data_save_load(self, data_loader, tmp_path):
        """Test data saving and loading"""
        # Generate sample data
        df = data_loader.download_divvy_data(2023, 6)
        
        # Temporarily change data directory
        original_dir = data_loader.raw_data_dir
        data_loader.raw_data_dir = tmp_path
        
        try:
            # Save data
            filename = "test_divvy.csv"
            data_loader.save_data(df, filename)
            
            # Check file exists
            assert (tmp_path / filename).exists()
            
            # Load data back
            loaded_df = data_loader.load_saved_data(filename)
            
            # Compare (excluding datetime precision issues)
            assert len(loaded_df) == len(df)
            assert list(loaded_df.columns) == list(df.columns)
            
        finally:
            # Restore original directory
            data_loader.raw_data_dir = original_dir
    
    def test_coordinate_offset_calculation(self, data_loader):
        """Test coordinate offset calculations"""
        lat, lng = 41.8781, -87.6298  # Chicago
        distance_km = 5.0
        bearing = 90.0  # East
        
        new_lat, new_lng = data_loader._offset_coordinates(lat, lng, distance_km, bearing)
        
        # New longitude should be greater (going east)
        assert new_lng > lng
        # Latitude should be approximately the same
        assert abs(new_lat - lat) < 0.1
        
        # Test distance is approximately correct
        # (simplified check - actual distance calculation would be more complex)
        lng_diff = abs(new_lng - lng)
        assert 0.03 < lng_diff < 0.1  # Rough check for ~5km at Chicago latitude
    
    def test_hourly_weights(self, data_loader):
        """Test hourly distribution weights"""
        weights = data_loader._get_hourly_weights()
        
        # Check array properties
        assert len(weights) == 24
        assert np.isclose(weights.sum(), 1.0)
        assert (weights >= 0).all()
        
        # Check peak hours have higher weights
        morning_peak = weights[7:9].sum()  # 7-8 AM
        evening_peak = weights[17:19].sum()  # 5-6 PM
        night_hours = weights[0:5].sum()    # Midnight to 5 AM
        
        assert morning_peak > night_hours
        assert evening_peak > night_hours


if __name__ == "__main__":
    pytest.main([__file__])