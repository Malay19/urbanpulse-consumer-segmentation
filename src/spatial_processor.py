"""
Spatial data processing module for Multimodal Consumer Segmentation Project
Handles geographic alignment, aggregation, and validation
"""

import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.geometry import Point, Polygon
from typing import Dict, List, Any, Tuple, Optional
from loguru import logger
import warnings
from pathlib import Path
import time

from config import DATA_CONFIG, VIZ_CONFIG
from src.utils import haversine_distance, calculate_data_summary


class SpatialProcessor:
    """Main class for spatial data processing and alignment"""
    
    def __init__(self):
        self.county_boundaries = None
        self.station_assignments = {}
        self.processing_stats = {}
    
    def load_county_boundaries(self, counties: List[str] = None) -> gpd.GeoDataFrame:
        """
        Load and prepare county boundary data
        """
        if counties is None:
            counties = DATA_CONFIG.SAMPLE_COUNTIES
        
        logger.info(f"Loading county boundaries for {len(counties)} counties")
        
        try:
            # Load boundaries using data_loader
            from src.data_loader import DataLoader
            loader = DataLoader()
            boundaries = loader.load_county_boundaries(counties)
            
            # Validate and prepare boundaries
            boundaries = self._prepare_boundaries(boundaries)
            self.county_boundaries = boundaries
            
            logger.info(f"Successfully loaded {len(boundaries)} county boundaries")
            return boundaries
            
        except Exception as e:
            logger.error(f"Error loading county boundaries: {e}")
            raise
    
    def _prepare_boundaries(self, boundaries: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Prepare and validate boundary data
        """
        boundaries = boundaries.copy()
        
        # Ensure proper CRS
        if boundaries.crs is None:
            logger.warning("No CRS found, assuming EPSG:4326")
            boundaries = boundaries.set_crs('EPSG:4326')
        elif boundaries.crs.to_string() != 'EPSG:4326':
            logger.info(f"Converting CRS from {boundaries.crs} to EPSG:4326")
            boundaries = boundaries.to_crs('EPSG:4326')
        
        # Validate geometries
        invalid_geoms = ~boundaries.geometry.is_valid
        if invalid_geoms.any():
            logger.warning(f"Found {invalid_geoms.sum()} invalid geometries, attempting to fix")
            boundaries.loc[invalid_geoms, 'geometry'] = boundaries.loc[invalid_geoms, 'geometry'].buffer(0)
        
        # Calculate area for density calculations
        # Convert to equal-area projection for accurate area calculation
        boundaries_area = boundaries.to_crs('EPSG:3857')  # Web Mercator
        boundaries['area_sq_km'] = boundaries_area.geometry.area / 1e6  # Convert to km²
        
        # Add centroid coordinates
        centroids = boundaries.geometry.centroid
        boundaries['centroid_lat'] = centroids.y
        boundaries['centroid_lng'] = centroids.x
        
        return boundaries
    
    def extract_stations_from_trips(self, trips_df: pd.DataFrame) -> gpd.GeoDataFrame:
        """
        Extract unique station locations from trip data
        """
        logger.info("Extracting station locations from trip data")
        
        # Extract start stations
        start_stations = trips_df[['start_station_id', 'start_lat', 'start_lng']].drop_duplicates()
        start_stations.columns = ['station_id', 'lat', 'lng']
        start_stations['station_type'] = 'start'
        
        # Extract end stations
        end_stations = trips_df[['end_station_id', 'end_lat', 'end_lng']].drop_duplicates()
        end_stations.columns = ['station_id', 'lat', 'lng']
        end_stations['station_type'] = 'end'
        
        # Combine and deduplicate
        all_stations = pd.concat([start_stations, end_stations], ignore_index=True)
        
        # Group by station_id and take mean coordinates (handles slight variations)
        stations = all_stations.groupby('station_id').agg({
            'lat': 'mean',
            'lng': 'mean'
        }).reset_index()
        
        # Create Point geometries
        stations['geometry'] = stations.apply(
            lambda row: Point(row['lng'], row['lat']), axis=1
        )
        
        # Convert to GeoDataFrame
        stations_gdf = gpd.GeoDataFrame(stations, crs='EPSG:4326')
        
        # Add trip counts per station
        start_counts = trips_df['start_station_id'].value_counts()
        end_counts = trips_df['end_station_id'].value_counts()
        total_counts = start_counts.add(end_counts, fill_value=0)
        
        stations_gdf['trip_count'] = stations_gdf['station_id'].map(total_counts).fillna(0)
        
        logger.info(f"Extracted {len(stations_gdf)} unique stations")
        return stations_gdf
    
    def assign_stations_to_counties(self, stations_gdf: gpd.GeoDataFrame, 
                                  boundaries: gpd.GeoDataFrame = None) -> gpd.GeoDataFrame:
        """
        Assign each bike station to its county using spatial join
        """
        if boundaries is None:
            if self.county_boundaries is None:
                raise ValueError("County boundaries not loaded. Call load_county_boundaries() first.")
            boundaries = self.county_boundaries
        
        logger.info(f"Assigning {len(stations_gdf)} stations to counties")
        
        # Perform spatial join
        stations_with_counties = gpd.sjoin(
            stations_gdf, 
            boundaries[['county_fips', 'county_name', 'geometry']], 
            how='left', 
            predicate='within'
        )
        
        # Handle stations not assigned to any county (edge cases)
        unassigned = stations_with_counties['county_fips'].isna()
        if unassigned.any():
            logger.warning(f"Found {unassigned.sum()} stations not assigned to counties")
            
            # For unassigned stations, find nearest county
            stations_with_counties = self._assign_nearest_county(
                stations_with_counties, boundaries, unassigned
            )
        
        # Clean up columns
        stations_with_counties = stations_with_counties.drop(columns=['index_right'], errors='ignore')
        
        # Store assignments for later use
        self.station_assignments = dict(
            zip(stations_with_counties['station_id'], stations_with_counties['county_fips'])
        )
        
        logger.info(f"Successfully assigned all stations to counties")
        return stations_with_counties
    
    def _assign_nearest_county(self, stations_gdf: gpd.GeoDataFrame, 
                              boundaries: gpd.GeoDataFrame, 
                              unassigned_mask: pd.Series) -> gpd.GeoDataFrame:
        """
        Assign unassigned stations to nearest county
        """
        logger.info(f"Assigning {unassigned_mask.sum()} stations to nearest counties")
        
        unassigned_stations = stations_gdf[unassigned_mask].copy()
        
        for idx, station in unassigned_stations.iterrows():
            station_point = station.geometry
            
            # Calculate distance to all county centroids
            distances = boundaries.geometry.centroid.distance(station_point)
            nearest_county_idx = distances.idxmin()
            
            # Assign to nearest county
            stations_gdf.loc[idx, 'county_fips'] = boundaries.loc[nearest_county_idx, 'county_fips']
            stations_gdf.loc[idx, 'county_name'] = boundaries.loc[nearest_county_idx, 'county_name']
        
        return stations_gdf
    
    def aggregate_trips_to_county_level(self, trips_df: pd.DataFrame, 
                                      stations_gdf: gpd.GeoDataFrame = None) -> pd.DataFrame:
        """
        Aggregate trip data from station-level to county-level
        """
        logger.info("Aggregating trip data to county level")
        
        # If stations not provided, extract and assign them
        if stations_gdf is None:
            stations_gdf = self.extract_stations_from_trips(trips_df)
            stations_gdf = self.assign_stations_to_counties(stations_gdf)
        
        # Add county assignments to trips
        trips_with_counties = self._add_county_info_to_trips(trips_df, stations_gdf)
        
        # Calculate trip-level metrics
        trips_with_counties = self._calculate_trip_metrics(trips_with_counties)
        
        # Aggregate to county level
        county_aggregations = self._perform_county_aggregation(trips_with_counties)
        
        # Add county metadata
        if self.county_boundaries is not None:
            county_aggregations = county_aggregations.merge(
                self.county_boundaries[['county_fips', 'county_name', 'area_sq_km', 'centroid_lat', 'centroid_lng']],
                on='county_fips',
                how='left'
            )
            
            # Calculate density metrics
            county_aggregations['trips_per_sq_km'] = (
                county_aggregations['total_trips'] / county_aggregations['area_sq_km']
            )
        
        logger.info(f"Aggregated data for {len(county_aggregations)} counties")
        return county_aggregations
    
    def _add_county_info_to_trips(self, trips_df: pd.DataFrame, 
                                 stations_gdf: gpd.GeoDataFrame) -> pd.DataFrame:
        """
        Add county information to trip records
        """
        trips_with_counties = trips_df.copy()
        
        # Create station-to-county mapping
        station_county_map = dict(zip(stations_gdf['station_id'], stations_gdf['county_fips']))
        
        # Add county info for start and end stations
        trips_with_counties['start_county_fips'] = trips_with_counties['start_station_id'].map(station_county_map)
        trips_with_counties['end_county_fips'] = trips_with_counties['end_station_id'].map(station_county_map)
        
        # For analysis, use start county as primary county
        trips_with_counties['county_fips'] = trips_with_counties['start_county_fips']
        
        # Flag inter-county trips
        trips_with_counties['is_inter_county'] = (
            trips_with_counties['start_county_fips'] != trips_with_counties['end_county_fips']
        )
        
        return trips_with_counties
    
    def _calculate_trip_metrics(self, trips_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate trip-level metrics
        """
        trips_df = trips_df.copy()
        
        # Trip duration in minutes
        trips_df['trip_duration_minutes'] = (
            pd.to_datetime(trips_df['end_time']) - pd.to_datetime(trips_df['start_time'])
        ).dt.total_seconds() / 60
        
        # Trip distance using haversine formula
        trips_df['trip_distance_km'] = trips_df.apply(
            lambda row: haversine_distance(
                row['start_lat'], row['start_lng'],
                row['end_lat'], row['end_lng']
            ), axis=1
        )
        
        # Trip speed (km/h)
        trips_df['trip_speed_kmh'] = (
            trips_df['trip_distance_km'] / (trips_df['trip_duration_minutes'] / 60)
        ).replace([np.inf, -np.inf], np.nan)
        
        return trips_df
    
    def _perform_county_aggregation(self, trips_df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform county-level aggregation of trip data
        """
        # Define aggregation functions
        agg_functions = {
            'trip_id': 'count',  # Total trips
            'trip_duration_minutes': ['mean', 'median', 'std'],
            'trip_distance_km': ['mean', 'median', 'std'],
            'trip_speed_kmh': ['mean', 'median'],
            'is_inter_county': 'sum',  # Count of inter-county trips
            'member_type': lambda x: (x == 'member').sum()  # Count of member trips
        }
        
        # Perform aggregation
        county_agg = trips_df.groupby('county_fips').agg(agg_functions).reset_index()
        
        # Flatten column names
        county_agg.columns = [
            f"{col[0]}_{col[1]}" if col[1] else col[0] 
            for col in county_agg.columns
        ]
        
        # Rename for clarity
        county_agg = county_agg.rename(columns={
            'trip_id_count': 'total_trips',
            'trip_duration_minutes_mean': 'avg_trip_duration_minutes',
            'trip_duration_minutes_median': 'median_trip_duration_minutes',
            'trip_duration_minutes_std': 'std_trip_duration_minutes',
            'trip_distance_km_mean': 'avg_trip_distance_km',
            'trip_distance_km_median': 'median_trip_distance_km',
            'trip_distance_km_std': 'std_trip_distance_km',
            'trip_speed_kmh_mean': 'avg_trip_speed_kmh',
            'trip_speed_kmh_median': 'median_trip_speed_kmh',
            'is_inter_county_sum': 'inter_county_trips',
            'member_type_<lambda>': 'member_trips'
        })
        
        # Calculate additional metrics
        county_agg['casual_trips'] = county_agg['total_trips'] - county_agg['member_trips']
        county_agg['member_ratio'] = county_agg['member_trips'] / county_agg['total_trips']
        county_agg['inter_county_ratio'] = county_agg['inter_county_trips'] / county_agg['total_trips']
        
        return county_agg
    
    def join_mobility_spending_data(self, mobility_df: pd.DataFrame, 
                                   spending_df: pd.DataFrame) -> pd.DataFrame:
        """
        Join county-level mobility data with spending data
        """
        logger.info("Joining mobility and spending data")
        
        # Aggregate spending data to county level if needed
        if 'month' in spending_df.columns and 'year' in spending_df.columns:
            # Aggregate spending by county and category
            spending_agg = spending_df.groupby(['county_fips', 'category']).agg({
                'spending_amount': ['sum', 'mean', 'std']
            }).reset_index()
            
            # Flatten columns
            spending_agg.columns = [
                f"{col[0]}_{col[1]}" if col[1] else col[0] 
                for col in spending_agg.columns
            ]
            
            # Pivot to get categories as columns
            spending_pivot = spending_agg.pivot(
                index='county_fips', 
                columns='category', 
                values='spending_amount_sum'
            ).reset_index()
            
            # Add prefix to spending columns
            spending_cols = [col for col in spending_pivot.columns if col != 'county_fips']
            spending_pivot = spending_pivot.rename(columns={
                col: f"spending_{col}" for col in spending_cols
            })
            
        else:
            spending_pivot = spending_df
        
        # Join datasets
        combined_data = mobility_df.merge(
            spending_pivot,
            on='county_fips',
            how='inner'
        )
        
        logger.info(f"Combined dataset has {len(combined_data)} counties with {combined_data.shape[1]} features")
        return combined_data
    
    def validate_spatial_data(self, stations_gdf: gpd.GeoDataFrame, 
                            boundaries: gpd.GeoDataFrame = None) -> Dict[str, Any]:
        """
        Validate spatial data quality and assignments
        """
        logger.info("Validating spatial data")
        
        if boundaries is None:
            boundaries = self.county_boundaries
        
        validation_results = {
            'validation_timestamp': pd.Timestamp.now(),
            'stations_total': len(stations_gdf),
            'counties_total': len(boundaries) if boundaries is not None else 0,
            'issues': [],
            'warnings': [],
            'passed': True
        }
        
        # Check coordinate validity
        invalid_coords = (
            (stations_gdf['lat'] < -90) | (stations_gdf['lat'] > 90) |
            (stations_gdf['lng'] < -180) | (stations_gdf['lng'] > 180)
        ).sum()
        
        if invalid_coords > 0:
            validation_results['issues'].append(f"Invalid coordinates: {invalid_coords} stations")
            validation_results['passed'] = False
        
        # Check for missing county assignments
        if 'county_fips' in stations_gdf.columns:
            unassigned = stations_gdf['county_fips'].isna().sum()
            if unassigned > 0:
                validation_results['warnings'].append(f"Unassigned stations: {unassigned}")
        
        # Check geometry validity
        if 'geometry' in stations_gdf.columns:
            invalid_geoms = ~stations_gdf.geometry.is_valid
            if invalid_geoms.any():
                validation_results['issues'].append(f"Invalid geometries: {invalid_geoms.sum()}")
        
        # Check for duplicate stations
        if 'station_id' in stations_gdf.columns:
            duplicates = stations_gdf['station_id'].duplicated().sum()
            if duplicates > 0:
                validation_results['warnings'].append(f"Duplicate station IDs: {duplicates}")
        
        # Spatial distribution check
        if boundaries is not None:
            stations_per_county = stations_gdf.groupby('county_fips').size()
            empty_counties = len(boundaries) - len(stations_per_county)
            if empty_counties > 0:
                validation_results['warnings'].append(f"Counties with no stations: {empty_counties}")
        
        validation_results['stations_per_county_stats'] = {
            'mean': stations_per_county.mean() if 'stations_per_county' in locals() else 0,
            'min': stations_per_county.min() if 'stations_per_county' in locals() else 0,
            'max': stations_per_county.max() if 'stations_per_county' in locals() else 0
        }
        
        logger.info(f"Spatial validation completed. Issues: {len(validation_results['issues'])}")
        return validation_results
    
    def create_spatial_visualization_data(self, stations_gdf: gpd.GeoDataFrame, 
                                        boundaries: gpd.GeoDataFrame = None) -> Dict[str, Any]:
        """
        Prepare data for spatial visualization
        """
        if boundaries is None:
            boundaries = self.county_boundaries
        
        viz_data = {
            'stations': {
                'coordinates': list(zip(stations_gdf['lng'], stations_gdf['lat'])),
                'station_ids': stations_gdf['station_id'].tolist(),
                'county_assignments': stations_gdf.get('county_fips', pd.Series()).tolist(),
                'trip_counts': stations_gdf.get('trip_count', pd.Series()).tolist()
            }
        }
        
        if boundaries is not None:
            viz_data['boundaries'] = {
                'county_fips': boundaries['county_fips'].tolist(),
                'county_names': boundaries.get('county_name', pd.Series()).tolist(),
                'geometries': [geom.__geo_interface__ for geom in boundaries.geometry],
                'centroids': list(zip(boundaries['centroid_lng'], boundaries['centroid_lat']))
            }
        
        return viz_data
    
    def export_processed_data(self, data: pd.DataFrame, filename: str, 
                            metadata: Dict[str, Any] = None) -> str:
        """
        Export processed spatial data to parquet format
        """
        from src.utils import save_processed_data
        
        # Add spatial processing metadata
        if metadata is None:
            metadata = {}
        
        metadata.update({
            'processing_module': 'spatial_processor',
            'counties_processed': len(data['county_fips'].unique()) if 'county_fips' in data.columns else 0,
            'total_records': len(data),
            'processing_stats': self.processing_stats
        })
        
        # Save as parquet for better performance with large datasets
        if not filename.endswith('.parquet'):
            filename = filename.replace('.csv', '.parquet')
        
        filepath = save_processed_data(data, filename, metadata)
        logger.info(f"Exported processed spatial data to {filepath}")
        
        return filepath


def main():
    """Demo function for spatial processing"""
    from src.data_loader import DataLoader
    
    # Initialize components
    loader = DataLoader()
    processor = SpatialProcessor()
    
    logger.info("Starting spatial processing demo...")
    
    # Load sample data
    trips_df = loader.download_divvy_data(2023, 6)
    spending_df = loader.download_spending_data()
    
    # Load county boundaries
    boundaries = processor.load_county_boundaries()
    
    # Extract and assign stations
    stations_gdf = processor.extract_stations_from_trips(trips_df)
    stations_with_counties = processor.assign_stations_to_counties(stations_gdf, boundaries)
    
    # Aggregate to county level
    county_mobility = processor.aggregate_trips_to_county_level(trips_df, stations_with_counties)
    
    # Join with spending data
    combined_data = processor.join_mobility_spending_data(county_mobility, spending_df)
    
    # Validate results
    validation_results = processor.validate_spatial_data(stations_with_counties, boundaries)
    
    # Export processed data
    processor.export_processed_data(combined_data, 'county_mobility_spending.parquet')
    processor.export_processed_data(stations_with_counties, 'stations_with_counties.parquet')
    
    # Print summary
    print(f"\nSpatial Processing Summary:")
    print(f"- Processed {len(stations_gdf)} stations")
    print(f"- Aggregated {len(trips_df)} trips to {len(county_mobility)} counties")
    print(f"- Combined with spending data: {combined_data.shape}")
    print(f"- Validation passed: {validation_results['passed']}")
    
    logger.info("Spatial processing demo completed successfully!")
    
    return combined_data, stations_with_counties, validation_results


if __name__ == "__main__":
    main()