"""
Configuration settings for Multimodal Consumer Segmentation Project
"""
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List

# Project root directory
PROJECT_ROOT = Path(__file__).parent.absolute()

@dataclass
class DataConfig:
    """Data source and processing configuration"""
    # Data directories
    RAW_DATA_DIR: Path = PROJECT_ROOT / "data" / "raw"
    PROCESSED_DATA_DIR: Path = PROJECT_ROOT / "data" / "processed"
    
    # External data sources
    DIVVY_DATA_URL: str = "https://divvy-tripdata.s3.amazonaws.com/"
    OPPORTUNITY_INSIGHTS_URL: str = "https://github.com/OpportunityInsights/EconomicTracker"
    CENSUS_API_BASE: str = "https://api.census.gov/data"
    
    # Sample data parameters
    SAMPLE_TRIP_COUNT: int = 50000
    SAMPLE_COUNTIES: List[str] = None
    
    def __post_init__(self):
        if self.SAMPLE_COUNTIES is None:
            # Focus on major metropolitan areas
            self.SAMPLE_COUNTIES = [
                "17031",  # Cook County, IL (Chicago)
                "36061",  # New York County, NY (Manhattan)
                "06037",  # Los Angeles County, CA
                "48201",  # Harris County, TX (Houston)
                "04013",  # Maricopa County, AZ (Phoenix)
            ]

@dataclass
class ModelConfig:
    """Machine learning and segmentation configuration"""
    # Clustering parameters
    MIN_CLUSTER_SIZE: int = 100
    MIN_SAMPLES: int = 50
    CLUSTER_SELECTION_EPSILON: float = 0.1
    
    # Feature engineering
    TEMPORAL_FEATURES: List[str] = None
    SPATIAL_FEATURES: List[str] = None
    SPENDING_CATEGORIES: List[str] = None
    
    def __post_init__(self):
        if self.TEMPORAL_FEATURES is None:
            self.TEMPORAL_FEATURES = [
                'hour_of_day', 'day_of_week', 'month', 'season'
            ]
        
        if self.SPATIAL_FEATURES is None:
            self.SPATIAL_FEATURES = [
                'start_lat', 'start_lng', 'end_lat', 'end_lng',
                'trip_distance', 'trip_bearing'
            ]
        
        if self.SPENDING_CATEGORIES is None:
            self.SPENDING_CATEGORIES = [
                'restaurants', 'retail', 'grocery', 'entertainment',
                'transportation', 'healthcare', 'education'
            ]

@dataclass
class VisualizationConfig:
    """Visualization and dashboard configuration"""
    # Map settings
    DEFAULT_MAP_CENTER: List[float] = None
    DEFAULT_ZOOM: int = 10
    MAP_STYLE: str = "OpenStreetMap"
    
    # Color schemes
    CLUSTER_COLORS: List[str] = None
    SPENDING_COLORS: List[str] = None
    
    def __post_init__(self):
        if self.DEFAULT_MAP_CENTER is None:
            self.DEFAULT_MAP_CENTER = [41.8781, -87.6298]  # Chicago
        
        if self.CLUSTER_COLORS is None:
            self.CLUSTER_COLORS = [
                '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
            ]
        
        if self.SPENDING_COLORS is None:
            self.SPENDING_COLORS = [
                '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57',
                '#FF9FF3', '#54A0FF', '#5F27CD', '#00D2D3', '#FF9F43'
            ]

# Global configuration instances
DATA_CONFIG = DataConfig()
MODEL_CONFIG = ModelConfig()
VIZ_CONFIG = VisualizationConfig()

# Environment-specific settings
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
CACHE_ENABLED = os.getenv("CACHE_ENABLED", "true").lower() == "true"

# API Keys (to be set via environment variables)
CENSUS_API_KEY = os.getenv("CENSUS_API_KEY")
MAPBOX_ACCESS_TOKEN = os.getenv("MAPBOX_ACCESS_TOKEN")

# Create data directories if they don't exist
for directory in [DATA_CONFIG.RAW_DATA_DIR, DATA_CONFIG.PROCESSED_DATA_DIR]:
    directory.mkdir(parents=True, exist_ok=True)