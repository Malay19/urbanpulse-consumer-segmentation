"""
Utility functions for the Multimodal Consumer Segmentation Project
"""

import pandas as pd
import numpy as np
from typing import Union, List, Dict, Any, Tuple
from datetime import datetime, timedelta
import math
from loguru import logger


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    Returns distance in kilometers
    """
    # Convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    
    # Radius of earth in kilometers
    r = 6371
    
    return c * r


def calculate_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the initial bearing from point 1 to point 2
    Returns bearing in degrees (0-360)
    """
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    
    dlon = lon2 - lon1
    
    y = math.sin(dlon) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    
    bearing = math.atan2(y, x)
    bearing = math.degrees(bearing)
    bearing = (bearing + 360) % 360
    
    return bearing


def add_temporal_features(df: pd.DataFrame, datetime_column: str) -> pd.DataFrame:
    """
    Add temporal features from a datetime column
    """
    df = df.copy()
    dt_col = pd.to_datetime(df[datetime_column])
    
    df['hour'] = dt_col.dt.hour
    df['day_of_week'] = dt_col.dt.dayofweek
    df['month'] = dt_col.dt.month
    df['year'] = dt_col.dt.year
    df['day_of_year'] = dt_col.dt.dayofyear
    df['week_of_year'] = dt_col.dt.isocalendar().week
    
    # Business day indicator
    df['is_weekend'] = dt_col.dt.dayofweek.isin([5, 6])
    
    # Season
    def get_season(month):
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:
            return 'Fall'
    
    df['season'] = df['month'].apply(get_season)
    
    # Time of day categories
    def get_time_period(hour):
        if 5 <= hour < 12:
            return 'Morning'
        elif 12 <= hour < 17:
            return 'Afternoon'
        elif 17 <= hour < 21:
            return 'Evening'
        else:
            return 'Night'
    
    df['time_period'] = df['hour'].apply(get_time_period)
    
    logger.info(f"Added temporal features to {len(df)} records")
    return df


def add_spatial_features(df: pd.DataFrame, start_lat_col: str = 'start_lat', 
                        start_lng_col: str = 'start_lng',
                        end_lat_col: str = 'end_lat', 
                        end_lng_col: str = 'end_lng') -> pd.DataFrame:
    """
    Add spatial features based on coordinate columns
    """
    df = df.copy()
    
    # Calculate trip distance
    df['trip_distance_km'] = df.apply(
        lambda row: haversine_distance(
            row[start_lat_col], row[start_lng_col],
            row[end_lat_col], row[end_lng_col]
        ), axis=1
    )
    
    # Calculate trip bearing
    df['trip_bearing'] = df.apply(
        lambda row: calculate_bearing(
            row[start_lat_col], row[start_lng_col],
            row[end_lat_col], row[end_lng_col]
        ), axis=1
    )
    
    # Cardinal direction
    def bearing_to_direction(bearing):
        directions = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']
        idx = round(bearing / 45) % 8
        return directions[idx]
    
    df['trip_direction'] = df['trip_bearing'].apply(bearing_to_direction)
    
    # Distance categories
    def categorize_distance(distance):
        if distance < 1:
            return 'Short'
        elif distance < 5:
            return 'Medium'
        elif distance < 15:
            return 'Long'
        else:
            return 'Very Long'
    
    df['distance_category'] = df['trip_distance_km'].apply(categorize_distance)
    
    logger.info(f"Added spatial features to {len(df)} records")
    return df


def aggregate_spending_data(df: pd.DataFrame, 
                          groupby_cols: List[str],
                          agg_functions: Dict[str, Union[str, List[str]]] = None) -> pd.DataFrame:
    """
    Aggregate spending data with flexible grouping and aggregation
    """
    if agg_functions is None:
        agg_functions = {
            'spending_amount': ['sum', 'mean', 'std', 'count']
        }
    
    aggregated = df.groupby(groupby_cols).agg(agg_functions).reset_index()
    
    # Flatten column names
    aggregated.columns = [
        f"{col[0]}_{col[1]}" if col[1] else col[0] 
        for col in aggregated.columns
    ]
    
    logger.info(f"Aggregated data from {len(df)} to {len(aggregated)} records")
    return aggregated


def normalize_features(df: pd.DataFrame, 
                      columns: List[str],
                      method: str = 'standard') -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Normalize numerical features
    Returns normalized dataframe and scaling parameters
    """
    df_norm = df.copy()
    scaling_params = {}
    
    for col in columns:
        if col not in df.columns:
            logger.warning(f"Column {col} not found in dataframe")
            continue
        
        if method == 'standard':
            mean_val = df[col].mean()
            std_val = df[col].std()
            df_norm[col] = (df[col] - mean_val) / std_val
            scaling_params[col] = {'method': 'standard', 'mean': mean_val, 'std': std_val}
            
        elif method == 'minmax':
            min_val = df[col].min()
            max_val = df[col].max()
            df_norm[col] = (df[col] - min_val) / (max_val - min_val)
            scaling_params[col] = {'method': 'minmax', 'min': min_val, 'max': max_val}
            
        elif method == 'robust':
            median_val = df[col].median()
            mad_val = (df[col] - median_val).abs().median()
            df_norm[col] = (df[col] - median_val) / mad_val
            scaling_params[col] = {'method': 'robust', 'median': median_val, 'mad': mad_val}
    
    logger.info(f"Normalized {len(columns)} features using {method} method")
    return df_norm, scaling_params


def detect_outliers(df: pd.DataFrame, 
                   columns: List[str],
                   method: str = 'iqr',
                   threshold: float = 1.5) -> pd.DataFrame:
    """
    Detect outliers in specified columns
    Returns dataframe with outlier flags
    """
    df_outliers = df.copy()
    
    for col in columns:
        if col not in df.columns:
            continue
        
        if method == 'iqr':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            
            df_outliers[f'{col}_outlier'] = (
                (df[col] < lower_bound) | (df[col] > upper_bound)
            )
            
        elif method == 'zscore':
            z_scores = np.abs((df[col] - df[col].mean()) / df[col].std())
            df_outliers[f'{col}_outlier'] = z_scores > threshold
    
    total_outliers = df_outliers[[col for col in df_outliers.columns if col.endswith('_outlier')]].any(axis=1).sum()
    logger.info(f"Detected {total_outliers} outlier records using {method} method")
    
    return df_outliers


def create_time_windows(df: pd.DataFrame, 
                       datetime_column: str,
                       window_size: str = '1H') -> pd.DataFrame:
    """
    Create time windows for temporal aggregation
    """
    df = df.copy()
    df[datetime_column] = pd.to_datetime(df[datetime_column])
    
    # Create time window
    df['time_window'] = df[datetime_column].dt.floor(window_size)
    
    logger.info(f"Created time windows of size {window_size}")
    return df


def geographic_clustering_prep(df: pd.DataFrame,
                             lat_col: str,
                             lng_col: str,
                             additional_features: List[str] = None) -> np.ndarray:
    """
    Prepare geographic data for clustering
    """
    # Start with coordinates
    features = [lat_col, lng_col]
    
    if additional_features:
        features.extend(additional_features)
    
    # Extract feature matrix
    feature_matrix = df[features].values
    
    # Handle any missing values
    if np.isnan(feature_matrix).any():
        logger.warning("Missing values found in clustering features, filling with column means")
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy='mean')
        feature_matrix = imputer.fit_transform(feature_matrix)
    
    logger.info(f"Prepared {feature_matrix.shape[0]} samples with {feature_matrix.shape[1]} features for clustering")
    return feature_matrix


def calculate_data_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate comprehensive summary statistics for a dataframe
    """
    summary = {
        'shape': df.shape,
        'memory_usage_mb': df.memory_usage(deep=True).sum() / 1024**2,
        'null_counts': df.isnull().sum().to_dict(),
        'dtypes': df.dtypes.astype(str).to_dict(),
        'numeric_summary': {},
        'categorical_summary': {}
    }
    
    # Numeric columns summary
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        summary['numeric_summary'][col] = {
            'mean': df[col].mean(),
            'median': df[col].median(),
            'std': df[col].std(),
            'min': df[col].min(),
            'max': df[col].max(),
            'unique_values': df[col].nunique()
        }
    
    # Categorical columns summary
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    for col in categorical_cols:
        summary['categorical_summary'][col] = {
            'unique_values': df[col].nunique(),
            'most_common': df[col].value_counts().head(5).to_dict(),
            'null_count': df[col].isnull().sum()
        }
    
    return summary


def save_processed_data(df: pd.DataFrame, 
                       filename: str,
                       metadata: Dict[str, Any] = None) -> str:
    """
    Save processed data with metadata
    """
    from config import DATA_CONFIG
    import json
    
    # Save main data
    filepath = DATA_CONFIG.PROCESSED_DATA_DIR / filename
    
    if filename.endswith('.csv'):
        df.to_csv(filepath, index=False)
    elif filename.endswith('.parquet'):
        df.to_parquet(filepath, index=False)
    else:
        raise ValueError(f"Unsupported file format: {filename}")
    
    # Save metadata if provided
    if metadata:
        metadata_file = filepath.with_suffix('.json')
        metadata['processing_timestamp'] = datetime.now().isoformat()
        metadata['record_count'] = len(df)
        metadata['columns'] = list(df.columns)
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        logger.info(f"Saved metadata to {metadata_file}")
    
    logger.info(f"Saved processed data to {filepath}")
    return str(filepath)