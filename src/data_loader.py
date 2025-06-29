"""
Data loading and ingestion module for Multimodal Consumer Segmentation Project
"""

import pandas as pd
import geopandas as gpd
import requests
import json
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from loguru import logger
import time
import random

from config import DATA_CONFIG, CENSUS_API_KEY


class DataLoader:
    """Main data loading class for all data sources"""
    
    def __init__(self):
        self.raw_data_dir = DATA_CONFIG.RAW_DATA_DIR
        self.processed_data_dir = DATA_CONFIG.PROCESSED_DATA_DIR
        
    def download_divvy_data(self, year: int = 2023, month: int = None) -> pd.DataFrame:
        """
        Download Divvy bike-share data for specified year/month
        For demo purposes, generates realistic sample data
        """
        logger.info(f"Loading Divvy data for year {year}, month {month}")
        
        # In production, this would download from actual Divvy API
        # For now, generate realistic sample data
        return self._generate_sample_divvy_data(year, month)
    
    def _generate_sample_divvy_data(self, year: int, month: Optional[int] = None) -> pd.DataFrame:
        """Generate realistic sample Divvy trip data"""
        
        # Chicago area coordinates (rough bounds)
        chicago_bounds = {
            'lat_min': 41.644, 'lat_max': 42.023,
            'lng_min': -87.940, 'lng_max': -87.524
        }
        
        # Generate sample trips
        n_trips = DATA_CONFIG.SAMPLE_TRIP_COUNT
        
        # Generate realistic temporal distribution
        if month:
            start_date = datetime(year, month, 1)
            end_date = datetime(year, month + 1, 1) - timedelta(days=1)
        else:
            start_date = datetime(year, 1, 1)
            end_date = datetime(year, 12, 31)
        
        # Create trip data
        data = []
        station_ids = list(range(1, 501))  # 500 stations
        member_types = ['member', 'casual']
        
        for i in range(n_trips):
            # Random timestamp with realistic hourly distribution
            random_hours = np.random.choice(24, p=self._get_hourly_weights())
            trip_date = start_date + timedelta(
                days=random.randint(0, (end_date - start_date).days),
                hours=random_hours,
                minutes=random.randint(0, 59)
            )
            
            # Generate start location
            start_lat = random.uniform(chicago_bounds['lat_min'], chicago_bounds['lat_max'])
            start_lng = random.uniform(chicago_bounds['lng_min'], chicago_bounds['lng_max'])
            
            # Generate end location (typically within reasonable distance)
            distance_km = np.random.exponential(2.5)  # Most trips are short
            bearing = random.uniform(0, 360)
            
            end_lat, end_lng = self._offset_coordinates(start_lat, start_lng, distance_km, bearing)
            
            # Trip duration (realistic distribution)
            duration_minutes = np.random.lognormal(2.5, 0.8)  # Log-normal distribution
            duration_minutes = max(1, min(duration_minutes, 180))  # 1 min to 3 hours
            
            data.append({
                'trip_id': f"trip_{i+1:06d}",
                'start_time': trip_date,
                'end_time': trip_date + timedelta(minutes=duration_minutes),
                'start_station_id': random.choice(station_ids),
                'end_station_id': random.choice(station_ids),
                'start_lat': round(start_lat, 6),
                'start_lng': round(start_lng, 6),
                'end_lat': round(end_lat, 6),
                'end_lng': round(end_lng, 6),
                'member_type': np.random.choice(member_types, p=[0.75, 0.25])  # More members
            })
        
        df = pd.DataFrame(data)
        df['start_time'] = pd.to_datetime(df['start_time'])
        df['end_time'] = pd.to_datetime(df['end_time'])
        
        logger.info(f"Generated {len(df)} sample Divvy trips")
        return df
    
    def _get_hourly_weights(self) -> np.ndarray:
        """Get realistic hourly distribution weights for bike trips"""
        # Based on typical urban bike-share patterns
        weights = np.array([
            0.005, 0.002, 0.001, 0.001, 0.002, 0.010,  # 0-5 AM
            0.035, 0.080, 0.120, 0.060, 0.040, 0.045,  # 6-11 AM
            0.055, 0.050, 0.045, 0.055, 0.080, 0.100,  # 12-5 PM
            0.085, 0.060, 0.040, 0.025, 0.015, 0.010   # 6-11 PM
        ])
        return weights / weights.sum()
    
    def _offset_coordinates(self, lat: float, lng: float, distance_km: float, bearing: float) -> Tuple[float, float]:
        """Calculate new coordinates given distance and bearing"""
        # Simplified calculation for demo purposes
        R = 6371  # Earth's radius in km
        
        bearing_rad = np.radians(bearing)
        lat_rad = np.radians(lat)
        lng_rad = np.radians(lng)
        
        new_lat_rad = np.arcsin(
            np.sin(lat_rad) * np.cos(distance_km / R) +
            np.cos(lat_rad) * np.sin(distance_km / R) * np.cos(bearing_rad)
        )
        
        new_lng_rad = lng_rad + np.arctan2(
            np.sin(bearing_rad) * np.sin(distance_km / R) * np.cos(lat_rad),
            np.cos(distance_km / R) - np.sin(lat_rad) * np.sin(new_lat_rad)
        )
        
        return np.degrees(new_lat_rad), np.degrees(new_lng_rad)
    
    def download_spending_data(self, counties: List[str] = None) -> pd.DataFrame:
        """
        Download consumer spending data from Opportunity Insights
        For demo purposes, generates realistic sample data
        """
        if counties is None:
            counties = DATA_CONFIG.SAMPLE_COUNTIES
            
        logger.info(f"Loading spending data for {len(counties)} counties")
        
        # In production, this would fetch from Opportunity Insights GitHub
        return self._generate_sample_spending_data(counties)
    
    def _generate_sample_spending_data(self, counties: List[str]) -> pd.DataFrame:
        """Generate realistic sample consumer spending data"""
        
        categories = [
            'restaurants', 'retail', 'grocery', 'entertainment',
            'transportation', 'healthcare', 'education', 'services'
        ]
        
        # Generate data for the last 24 months
        end_date = datetime.now()
        start_date = end_date - timedelta(days=730)
        
        data = []
        
        for county_fips in counties:
            current_date = start_date
            
            while current_date <= end_date:
                for category in categories:
                    # Base spending amount with seasonal and category variations
                    base_amount = self._get_base_spending(category)
                    seasonal_factor = self._get_seasonal_factor(current_date.month, category)
                    
                    # Add realistic noise and trends
                    noise_factor = np.random.normal(1.0, 0.15)
                    trend_factor = self._get_trend_factor(current_date, category)
                    
                    spending_amount = base_amount * seasonal_factor * noise_factor * trend_factor
                    
                    data.append({
                        'county_fips': county_fips,
                        'year': current_date.year,
                        'month': current_date.month,
                        'category': category,
                        'spending_amount': max(0, spending_amount),
                        'date': current_date
                    })
                
                # Move to next month
                if current_date.month == 12:
                    current_date = current_date.replace(year=current_date.year + 1, month=1)
                else:
                    current_date = current_date.replace(month=current_date.month + 1)
        
        df = pd.DataFrame(data)
        logger.info(f"Generated {len(df)} spending records")
        return df
    
    def _get_base_spending(self, category: str) -> float:
        """Get base spending amount by category (millions of dollars)"""
        base_amounts = {
            'restaurants': 150.0,
            'retail': 200.0,
            'grocery': 120.0,
            'entertainment': 80.0,
            'transportation': 90.0,
            'healthcare': 110.0,
            'education': 60.0,
            'services': 75.0
        }
        return base_amounts.get(category, 100.0)
    
    def _get_seasonal_factor(self, month: int, category: str) -> float:
        """Get seasonal adjustment factor for spending category"""
        # Different categories have different seasonal patterns
        seasonal_patterns = {
            'restaurants': [0.9, 0.85, 0.95, 1.0, 1.05, 1.1, 1.15, 1.1, 1.0, 1.05, 1.2, 1.3],
            'retail': [0.8, 0.9, 1.0, 1.0, 1.05, 1.0, 0.95, 0.95, 1.0, 1.1, 1.3, 1.4],
            'grocery': [1.0, 0.95, 1.0, 1.0, 1.0, 1.05, 1.1, 1.05, 1.0, 1.0, 1.15, 1.2],
            'entertainment': [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.3, 1.0, 1.1, 1.0, 1.1],
            'transportation': [0.9, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.3, 1.1, 1.0, 0.9, 0.8],
            'healthcare': [1.1, 1.0, 1.0, 1.0, 0.95, 0.9, 0.9, 0.9, 1.0, 1.0, 1.0, 1.1],
            'education': [1.2, 1.0, 1.0, 1.0, 1.0, 0.7, 0.6, 1.3, 1.4, 1.1, 1.0, 0.9],
            'services': [1.0, 1.0, 1.0, 1.1, 1.1, 1.1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
        }
        
        pattern = seasonal_patterns.get(category, [1.0] * 12)
        return pattern[month - 1]
    
    def _get_trend_factor(self, date: datetime, category: str) -> float:
        """Get trend adjustment factor based on date and category"""
        # Simulate various trends (recovery, growth, etc.)
        months_from_start = (date.year - 2022) * 12 + date.month
        
        trend_patterns = {
            'restaurants': 1.0 + 0.005 * months_from_start,  # Gradual recovery
            'retail': 1.0 + 0.003 * months_from_start,       # Slow growth
            'entertainment': 1.0 + 0.008 * months_from_start, # Strong recovery
            'transportation': 1.0 + 0.006 * months_from_start, # Mobility recovery
        }
        
        return trend_patterns.get(category, 1.0)
    
    def load_county_boundaries(self, counties: List[str] = None) -> gpd.GeoDataFrame:
        """
        Load US county boundary data
        Uses Census API or generates sample boundaries
        """
        if counties is None:
            counties = DATA_CONFIG.SAMPLE_COUNTIES
            
        logger.info(f"Loading boundary data for {len(counties)} counties")
        
        try:
            # Try to fetch from Census API
            if CENSUS_API_KEY:
                return self._fetch_census_boundaries(counties)
            else:
                logger.warning("No Census API key found, generating sample boundaries")
                return self._generate_sample_boundaries(counties)
        except Exception as e:
            logger.error(f"Error loading county boundaries: {e}")
            return self._generate_sample_boundaries(counties)
    
    def _fetch_census_boundaries(self, counties: List[str]) -> gpd.GeoDataFrame:
        """Fetch county boundaries from Census API"""
        # This would implement actual Census API calls
        # For now, fall back to sample data
        logger.info("Census API integration not yet implemented, using sample data")
        return self._generate_sample_boundaries(counties)
    
    def _generate_sample_boundaries(self, counties: List[str]) -> gpd.GeoDataFrame:
        """Generate sample county boundary polygons"""
        from shapely.geometry import Polygon
        
        # County information (name, center coordinates, approximate size)
        county_info = {
            "17031": {"name": "Cook County, IL", "center": [41.8781, -87.6298], "size": 0.5},
            "36061": {"name": "New York County, NY", "center": [40.7831, -73.9712], "size": 0.2},
            "06037": {"name": "Los Angeles County, CA", "center": [34.0522, -118.2437], "size": 0.8},
            "48201": {"name": "Harris County, TX", "center": [29.7604, -95.3698], "size": 0.6},
            "04013": {"name": "Maricopa County, AZ", "center": [33.4484, -112.0740], "size": 0.7},
        }
        
        data = []
        for county_fips in counties:
            if county_fips in county_info:
                info = county_info[county_fips]
                center_lat, center_lng = info["center"]
                size = info["size"]
                
                # Create a rough rectangular polygon around the center
                polygon = Polygon([
                    [center_lng - size, center_lat - size],
                    [center_lng + size, center_lat - size],
                    [center_lng + size, center_lat + size],
                    [center_lng - size, center_lat + size],
                    [center_lng - size, center_lat - size]
                ])
                
                data.append({
                    'county_fips': county_fips,
                    'county_name': info["name"],
                    'geometry': polygon
                })
        
        gdf = gpd.GeoDataFrame(data, crs='EPSG:4326')
        logger.info(f"Generated {len(gdf)} county boundary polygons")
        return gdf
    
    def save_data(self, data: pd.DataFrame, filename: str, processed: bool = False) -> None:
        """Save data to appropriate directory"""
        directory = self.processed_data_dir if processed else self.raw_data_dir
        filepath = directory / filename
        
        if filename.endswith('.csv'):
            data.to_csv(filepath, index=False)
        elif filename.endswith('.parquet'):
            data.to_parquet(filepath, index=False)
        else:
            raise ValueError(f"Unsupported file format: {filename}")
        
        logger.info(f"Saved data to {filepath}")
    
    def load_saved_data(self, filename: str, processed: bool = False) -> pd.DataFrame:
        """Load previously saved data"""
        directory = self.processed_data_dir if processed else self.raw_data_dir
        filepath = directory / filename
        
        if not filepath.exists():
            raise FileNotFoundError(f"Data file not found: {filepath}")
        
        if filename.endswith('.csv'):
            return pd.read_csv(filepath)
        elif filename.endswith('.parquet'):
            return pd.read_parquet(filepath)
        else:
            raise ValueError(f"Unsupported file format: {filename}")


def main():
    """Demo function to test data loading capabilities"""
    loader = DataLoader()
    
    # Load sample data
    logger.info("Starting data loading demo...")
    
    # Generate and save Divvy data
    divvy_data = loader.download_divvy_data(2023, 6)
    loader.save_data(divvy_data, 'divvy_sample_202306.csv')
    
    # Generate and save spending data
    spending_data = loader.download_spending_data()
    loader.save_data(spending_data, 'spending_sample.csv')
    
    # Load county boundaries
    boundaries = loader.load_county_boundaries()
    logger.info(f"Loaded boundaries for {len(boundaries)} counties")
    
    logger.info("Data loading demo completed successfully!")
    
    return divvy_data, spending_data, boundaries


if __name__ == "__main__":
    main()