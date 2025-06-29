"""
Tests for spatial processing functionality
"""

import pytest
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon
import numpy as np
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from spatial_processor import SpatialProcessor
from data_loader import DataLoader


class TestSpatialProcessor:
    
    @pytest.fixture
    def sample_trips_data(self):
        """Create sample trip data for testing"""
        data = {
            'trip_id': ['trip_001', 'trip_002', 'trip_003', 'trip_004'],
            'start_time': pd.to_datetime(['2023-06-01 08:00:00', '2023-06-01 09:00:00', 
                                        '2023-06-01 10:00:00', '2023-06-01 11:00:00']),
            'end_time': pd.to_datetime(['2023-06-01 08:15:00', '2023-06-01 09:20:00', 
                                      '2023-06-01 10:10:00', '2023-06-01 11:25:00']),
            'start_station_id': [1, 2, 1, 3],
            'end_station_id': [2, 3, 3, 1],
            'start_lat': [41.88, 41.89, 41.88, 41.90],
            'start_lng': [-87.63, -87.64, -87.63, -87.65],
            'end_lat': [41.89, 41.90, 41.90, 41.88],
            'end_lng': [-87.64, -87.65, -87.65, -87.63],
            'member_type': ['member', 'casual', 'member', 'member']
        }
        return pd.DataFrame(data)
    
    @pytest.fixture
    def sample_boundaries(self):
        """Create sample county boundaries for testing"""
        # Create simple rectangular polygons
        county1 = Polygon([(-87.70, 41.85), (-87.60, 41.85), (-87.60, 41.95), (-87.70, 41.95)])
        county2 = Polygon([(-87.60, 41.85), (-87.50, 41.85), (-87.50, 41.95), (-87.60, 41.95)])
        
        data = {
            'county_fips': ['17031', '17043'],
            'county_name': ['Cook County, IL', 'DuPage County, IL'],
            'geometry': [county1, county2]
        }
        return gpd.GeoDataFrame(data, crs='EPSG:4326')
    
    @pytest.fixture
    def spatial_processor(self):
        """Create a SpatialProcessor instance for testing"""
        return SpatialProcessor()
    
    def test_extract_stations_from_trips(self, spatial_processor, sample_trips_data):
        """Test station extraction from trip data"""
        stations_gdf = spatial_processor.extract_stations_from_trips(sample_trips_data)
        
        # Check data structure
        assert isinstance(stations_gdf, gpd.GeoDataFrame)
        assert len(stations_gdf) == 3  # Unique stations: 1, 2, 3
        
        # Check required columns
        required_columns = ['station_id', 'lat', 'lng', 'geometry', 'trip_count']
        for col in required_columns:
            assert col in stations_gdf.columns
        
        # Check geometries are Points
        assert all(isinstance(geom, Point) for geom in stations_gdf.geometry)
        
        # Check trip counts
        assert stations_gdf[stations_gdf['station_id'] == 1]['trip_count'].iloc[0] == 3  # 2 starts + 1 end
        assert stations_gdf[stations_gdf['station_id'] == 2]['trip_count'].iloc[0] == 2  # 1 start + 1 end
        assert stations_gdf[stations_gdf['station_id'] == 3]['trip_count'].iloc[0] == 3  # 1 start + 2 ends
    
    def test_assign_stations_to_counties(self, spatial_processor, sample_trips_data, sample_boundaries):
        """Test station-to-county assignment"""
        # Extract stations
        stations_gdf = spatial_processor.extract_stations_from_trips(sample_trips_data)
        
        # Assign to counties
        stations_with_counties = spatial_processor.assign_stations_to_counties(
            stations_gdf, sample_boundaries
        )
        
        # Check that county assignments were made
        assert 'county_fips' in stations_with_counties.columns
        assert 'county_name' in stations_with_counties.columns
        
        # Check that all stations have county assignments
        assert not stations_with_counties['county_fips'].isna().any()
        
        # Check that assignments are valid county FIPS codes
        valid_fips = set(sample_boundaries['county_fips'])
        assigned_fips = set(stations_with_counties['county_fips'])
        assert assigned_fips.issubset(valid_fips)
    
    def test_aggregate_trips_to_county_level(self, spatial_processor, sample_trips_data, sample_boundaries):
        """Test trip aggregation to county level"""
        # Set up processor with boundaries
        spatial_processor.county_boundaries = sample_boundaries
        
        # Aggregate trips
        county_aggregations = spatial_processor.aggregate_trips_to_county_level(sample_trips_data)
        
        # Check data structure
        assert isinstance(county_aggregations, pd.DataFrame)
        assert len(county_aggregations) > 0
        
        # Check required columns
        expected_columns = [
            'county_fips', 'total_trips', 'avg_trip_duration_minutes',
            'avg_trip_distance_km', 'member_trips', 'casual_trips'
        ]
        for col in expected_columns:
            assert col in county_aggregations.columns
        
        # Check that trip counts are reasonable
        assert county_aggregations['total_trips'].sum() == len(sample_trips_data)
        assert (county_aggregations['total_trips'] > 0).all()
        
        # Check member/casual split
        total_member_trips = county_aggregations['member_trips'].sum()
        total_casual_trips = county_aggregations['casual_trips'].sum()
        assert total_member_trips + total_casual_trips == len(sample_trips_data)
    
    def test_join_mobility_spending_data(self, spatial_processor):
        """Test joining mobility and spending data"""
        # Create sample mobility data
        mobility_data = pd.DataFrame({
            'county_fips': ['17031', '17043'],
            'total_trips': [1000, 500],
            'avg_trip_duration_minutes': [15.5, 12.3]
        })
        
        # Create sample spending data
        spending_data = pd.DataFrame({
            'county_fips': ['17031', '17031', '17043', '17043'],
            'category': ['restaurants', 'retail', 'restaurants', 'retail'],
            'spending_amount': [100000, 150000, 50000, 75000]
        })
        
        # Join data
        combined_data = spatial_processor.join_mobility_spending_data(mobility_data, spending_data)
        
        # Check structure
        assert isinstance(combined_data, pd.DataFrame)
        assert len(combined_data) == 2  # Two counties
        
        # Check that both mobility and spending columns are present
        assert 'total_trips' in combined_data.columns
        assert 'spending_restaurants' in combined_data.columns
        assert 'spending_retail' in combined_data.columns
        
        # Check values
        cook_county = combined_data[combined_data['county_fips'] == '17031']
        assert cook_county['spending_restaurants'].iloc[0] == 100000
        assert cook_county['spending_retail'].iloc[0] == 150000
    
    def test_validate_spatial_data(self, spatial_processor, sample_trips_data, sample_boundaries):
        """Test spatial data validation"""
        # Extract and assign stations
        stations_gdf = spatial_processor.extract_stations_from_trips(sample_trips_data)
        stations_with_counties = spatial_processor.assign_stations_to_counties(
            stations_gdf, sample_boundaries
        )
        
        # Validate
        validation_results = spatial_processor.validate_spatial_data(
            stations_with_counties, sample_boundaries
        )
        
        # Check validation structure
        assert isinstance(validation_results, dict)
        assert 'passed' in validation_results
        assert 'issues' in validation_results
        assert 'warnings' in validation_results
        assert 'stations_total' in validation_results
        
        # With good sample data, validation should pass
        assert validation_results['passed'] == True
        assert len(validation_results['issues']) == 0
    
    def test_coordinate_validation(self, spatial_processor):
        """Test validation of invalid coordinates"""
        # Create data with invalid coordinates
        invalid_stations = gpd.GeoDataFrame({
            'station_id': [1, 2, 3],
            'lat': [91.0, 41.88, -91.0],  # Invalid latitudes
            'lng': [-87.63, 181.0, -87.65],  # Invalid longitude
            'geometry': [Point(-87.63, 91.0), Point(181.0, 41.88), Point(-87.65, -91.0)]
        }, crs='EPSG:4326')
        
        validation_results = spatial_processor.validate_spatial_data(invalid_stations)
        
        # Should detect coordinate issues
        assert validation_results['passed'] == False
        assert len(validation_results['issues']) > 0
        assert any('Invalid coordinates' in issue for issue in validation_results['issues'])
    
    def test_boundary_preparation(self, spatial_processor):
        """Test boundary data preparation"""
        # Create boundaries without CRS
        boundaries_no_crs = gpd.GeoDataFrame({
            'county_fips': ['17031'],
            'county_name': ['Cook County, IL'],
            'geometry': [Polygon([(-87.70, 41.85), (-87.60, 41.85), (-87.60, 41.95), (-87.70, 41.95)])]
        })
        
        # Prepare boundaries
        prepared = spatial_processor._prepare_boundaries(boundaries_no_crs)
        
        # Check that CRS was set
        assert prepared.crs is not None
        assert prepared.crs.to_string() == 'EPSG:4326'
        
        # Check that area was calculated
        assert 'area_sq_km' in prepared.columns
        assert prepared['area_sq_km'].iloc[0] > 0
        
        # Check that centroids were added
        assert 'centroid_lat' in prepared.columns
        assert 'centroid_lng' in prepared.columns


if __name__ == "__main__":
    pytest.main([__file__])