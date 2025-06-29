"""
Feature engineering module for Multimodal Consumer Segmentation Project
Transforms raw mobility and spending data into analysis-ready features
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from loguru import logger
from sklearn.preprocessing import StandardScaler, PowerTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression
from scipy import stats
import warnings
from datetime import datetime

from config import DATA_CONFIG, MODEL_CONFIG
from src.utils import calculate_data_summary, save_processed_data


class FeatureEngineer:
    """Main class for feature engineering and transformation"""
    
    def __init__(self):
        self.scalers = {}
        self.transformers = {}
        self.feature_stats = {}
        self.population_data = None
        
    def load_population_data(self, counties: List[str] = None) -> pd.DataFrame:
        """
        Load county population data from Census API or generate sample data
        """
        if counties is None:
            counties = DATA_CONFIG.SAMPLE_COUNTIES
            
        logger.info(f"Loading population data for {len(counties)} counties")
        
        try:
            # In production, this would use Census API
            # For now, generate realistic population data
            population_data = self._generate_sample_population_data(counties)
            self.population_data = population_data
            return population_data
            
        except Exception as e:
            logger.error(f"Error loading population data: {e}")
            return self._generate_sample_population_data(counties)
    
    def _generate_sample_population_data(self, counties: List[str]) -> pd.DataFrame:
        """Generate realistic sample population data"""
        
        # Realistic population estimates for major counties
        population_estimates = {
            "17031": 5150233,  # Cook County, IL (Chicago)
            "36061": 1694251,  # New York County, NY (Manhattan)
            "06037": 9829544,  # Los Angeles County, CA
            "48201": 4713325,  # Harris County, TX (Houston)
            "04013": 4420568,  # Maricopa County, AZ (Phoenix)
        }
        
        data = []
        for county_fips in counties:
            base_pop = population_estimates.get(county_fips, 1000000)
            
            # Add some year-over-year variation
            for year in [2022, 2023]:
                growth_rate = np.random.normal(0.01, 0.005)  # ~1% growth with variation
                population = int(base_pop * (1 + growth_rate) ** (year - 2022))
                
                data.append({
                    'county_fips': county_fips,
                    'year': year,
                    'population': population,
                    'population_density_per_sq_km': population / 1000  # Simplified density
                })
        
        df = pd.DataFrame(data)
        logger.info(f"Generated population data for {len(df)} county-year combinations")
        return df
    
    def create_mobility_features(self, mobility_df: pd.DataFrame, 
                               trips_df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Create comprehensive mobility features from county-level data
        """
        logger.info("Creating mobility features")
        
        features_df = mobility_df.copy()
        
        # Basic volume metrics (already calculated in spatial processor)
        # Ensure we have the base metrics
        required_base_metrics = ['total_trips', 'avg_trip_duration_minutes', 'member_trips', 'casual_trips']
        for metric in required_base_metrics:
            if metric not in features_df.columns:
                logger.warning(f"Missing base metric: {metric}")
        
        # Calculate additional mobility features
        features_df = self._add_temporal_mobility_features(features_df, trips_df)
        features_df = self._add_usage_pattern_features(features_df)
        features_df = self._add_volume_intensity_features(features_df)
        
        # Add population-normalized features if population data available
        if self.population_data is not None:
            features_df = self._add_per_capita_mobility_features(features_df)
        
        logger.info(f"Created mobility features: {features_df.shape[1]} total columns")
        return features_df
    
    def _add_temporal_mobility_features(self, df: pd.DataFrame, 
                                      trips_df: pd.DataFrame = None) -> pd.DataFrame:
        """Add temporal pattern features"""
        
        if trips_df is not None:
            # Calculate temporal patterns from raw trip data
            logger.info("Calculating temporal patterns from trip data")
            
            # Add temporal features to trips
            from src.utils import add_temporal_features
            trips_with_temporal = add_temporal_features(trips_df, 'start_time')
            
            # Aggregate temporal patterns by county
            temporal_agg = self._aggregate_temporal_patterns(trips_with_temporal)
            
            # Merge with main dataframe
            df = df.merge(temporal_agg, on='county_fips', how='left')
        else:
            # Generate realistic temporal patterns based on existing data
            logger.info("Generating estimated temporal patterns")
            df = self._estimate_temporal_patterns(df)
        
        return df
    
    def _aggregate_temporal_patterns(self, trips_df: pd.DataFrame) -> pd.DataFrame:
        """Aggregate temporal patterns from trip data"""
        
        # Add county assignments if not present
        if 'county_fips' not in trips_df.columns:
            # This would need station-to-county mapping from spatial processor
            logger.warning("County assignments not found in trip data")
            return pd.DataFrame()
        
        # Calculate temporal ratios
        temporal_features = []
        
        for county in trips_df['county_fips'].unique():
            county_trips = trips_df[trips_df['county_fips'] == county]
            total_trips = len(county_trips)
            
            if total_trips == 0:
                continue
            
            # Peak hour ratio (7-9 AM and 5-7 PM)
            peak_hours = county_trips['hour'].isin([7, 8, 17, 18])
            peak_hour_ratio = peak_hours.sum() / total_trips
            
            # Weekend ratio
            weekend_ratio = county_trips['is_weekend'].sum() / total_trips
            
            # Night trips ratio (10 PM - 5 AM)
            night_hours = county_trips['hour'].isin([22, 23, 0, 1, 2, 3, 4, 5])
            night_trips_ratio = night_hours.sum() / total_trips
            
            # Seasonal patterns
            summer_trips = county_trips['season'].eq('Summer').sum() / total_trips
            winter_trips = county_trips['season'].eq('Winter').sum() / total_trips
            
            temporal_features.append({
                'county_fips': county,
                'peak_hour_ratio': peak_hour_ratio,
                'weekend_ratio': weekend_ratio,
                'night_trips_ratio': night_trips_ratio,
                'summer_trips_ratio': summer_trips,
                'winter_trips_ratio': winter_trips,
                'seasonal_variation': abs(summer_trips - winter_trips)
            })
        
        return pd.DataFrame(temporal_features)
    
    def _estimate_temporal_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Estimate temporal patterns based on county characteristics"""
        
        # Generate realistic temporal patterns based on trip volume
        df['peak_hour_ratio'] = np.random.beta(2, 3, len(df)) * 0.4 + 0.1  # 10-50%
        df['weekend_ratio'] = np.random.beta(1.5, 3, len(df)) * 0.4 + 0.1  # 10-50%
        df['night_trips_ratio'] = np.random.beta(1, 4, len(df)) * 0.2  # 0-20%
        df['summer_trips_ratio'] = np.random.beta(3, 2, len(df)) * 0.4 + 0.2  # 20-60%
        df['winter_trips_ratio'] = np.random.beta(2, 3, len(df)) * 0.3 + 0.1  # 10-40%
        df['seasonal_variation'] = abs(df['summer_trips_ratio'] - df['winter_trips_ratio'])
        
        return df
    
    def _add_usage_pattern_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add usage pattern features"""
        
        # Member vs casual ratio (already calculated in spatial processor)
        if 'member_ratio' not in df.columns and 'member_trips' in df.columns:
            df['member_ratio'] = df['member_trips'] / df['total_trips']
        
        # Trip duration categories
        if 'avg_trip_duration_minutes' in df.columns:
            df['short_trip_preference'] = (df['avg_trip_duration_minutes'] < 10).astype(int)
            df['long_trip_preference'] = (df['avg_trip_duration_minutes'] > 30).astype(int)
            
            # Duration variability (estimated)
            df['trip_duration_variability'] = df['avg_trip_duration_minutes'] * np.random.uniform(0.3, 0.8, len(df))
        
        # Distance patterns (if available)
        if 'avg_trip_distance_km' in df.columns:
            df['short_distance_preference'] = (df['avg_trip_distance_km'] < 2).astype(int)
            df['long_distance_preference'] = (df['avg_trip_distance_km'] > 8).astype(int)
        
        # Inter-county mobility
        if 'inter_county_ratio' in df.columns:
            df['high_inter_county_mobility'] = (df['inter_county_ratio'] > 0.1).astype(int)
        
        return df
    
    def _add_volume_intensity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add volume and intensity features"""
        
        # Station density (if area available)
        if 'area_sq_km' in df.columns:
            # Estimate station count from trip volume
            df['estimated_stations'] = np.sqrt(df['total_trips'] / 100)  # Rough estimate
            df['station_density_per_sq_km'] = df['estimated_stations'] / df['area_sq_km']
        
        # Trip intensity categories
        trip_volume_percentiles = df['total_trips'].quantile([0.33, 0.67])
        df['trip_volume_category'] = pd.cut(
            df['total_trips'], 
            bins=[-np.inf, trip_volume_percentiles.iloc[0], trip_volume_percentiles.iloc[1], np.inf],
            labels=['Low', 'Medium', 'High']
        )
        
        # Usage efficiency (trips per station estimate)
        if 'estimated_stations' in df.columns:
            df['trips_per_station'] = df['total_trips'] / df['estimated_stations']
        
        return df
    
    def _add_per_capita_mobility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add population-normalized mobility features"""
        
        # Merge with population data
        pop_data = self.population_data.copy()
        
        # Use most recent year available
        latest_year = pop_data['year'].max()
        pop_latest = pop_data[pop_data['year'] == latest_year][['county_fips', 'population']]
        
        df_with_pop = df.merge(pop_latest, on='county_fips', how='left')
        
        # Calculate per-capita metrics
        df_with_pop['trips_per_capita'] = df_with_pop['total_trips'] / df_with_pop['population'] * 1000  # per 1000 people
        df_with_pop['member_trips_per_capita'] = df_with_pop['member_trips'] / df_with_pop['population'] * 1000
        df_with_pop['casual_trips_per_capita'] = df_with_pop['casual_trips'] / df_with_pop['population'] * 1000
        
        # Mobility penetration rate
        df_with_pop['mobility_penetration_rate'] = np.minimum(df_with_pop['trips_per_capita'] / 10, 1.0)  # Cap at 100%
        
        return df_with_pop
    
    def create_spending_features(self, spending_df: pd.DataFrame) -> pd.DataFrame:
        """
        Create comprehensive spending features from county-level data
        """
        logger.info("Creating spending features")
        
        # Pivot spending data to get categories as columns
        spending_pivot = self._pivot_spending_data(spending_df)
        
        # Calculate spending proportions
        spending_features = self._add_spending_proportions(spending_pivot)
        
        # Add spending intensity features
        spending_features = self._add_spending_intensity_features(spending_features)
        
        # Add seasonal and temporal patterns
        spending_features = self._add_spending_temporal_features(spending_features, spending_df)
        
        # Add population-normalized features if available
        if self.population_data is not None:
            spending_features = self._add_per_capita_spending_features(spending_features)
        
        logger.info(f"Created spending features: {spending_features.shape[1]} total columns")
        return spending_features
    
    def _pivot_spending_data(self, spending_df: pd.DataFrame) -> pd.DataFrame:
        """Pivot spending data to get categories as columns"""
        
        # Aggregate by county and category (sum across time periods)
        spending_agg = spending_df.groupby(['county_fips', 'category']).agg({
            'spending_amount': ['sum', 'mean', 'std']
        }).reset_index()
        
        # Flatten column names
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
        
        # Fill missing values with 0
        spending_pivot = spending_pivot.fillna(0)
        
        # Rename columns to add spending_ prefix
        spending_cols = [col for col in spending_pivot.columns if col != 'county_fips']
        spending_pivot = spending_pivot.rename(columns={
            col: f"spending_{col}" for col in spending_cols
        })
        
        return spending_pivot
    
    def _add_spending_proportions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add spending category proportions"""
        
        # Get spending columns
        spending_cols = [col for col in df.columns if col.startswith('spending_')]
        
        # Calculate total spending
        df['total_spending'] = df[spending_cols].sum(axis=1)
        
        # Calculate proportions
        for col in spending_cols:
            prop_col = col.replace('spending_', 'spending_pct_')
            df[prop_col] = df[col] / df['total_spending']
            df[prop_col] = df[prop_col].fillna(0)  # Handle division by zero
        
        # Calculate spending diversity (entropy)
        spending_props = df[[col for col in df.columns if 'spending_pct_' in col]]
        # Add small constant to avoid log(0)
        spending_props_safe = spending_props + 1e-10
        df['spending_diversity'] = -(spending_props_safe * np.log(spending_props_safe)).sum(axis=1)
        
        return df
    
    def _add_spending_intensity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add spending intensity and variability features"""
        
        # Spending intensity categories
        if 'total_spending' in df.columns:
            spending_percentiles = df['total_spending'].quantile([0.25, 0.75])
            df['spending_intensity_category'] = pd.cut(
                df['total_spending'],
                bins=[-np.inf, spending_percentiles.iloc[0], spending_percentiles.iloc[1], np.inf],
                labels=['Low', 'Medium', 'High']
            )
        
        # Dominant spending category
        spending_pct_cols = [col for col in df.columns if 'spending_pct_' in col]
        if spending_pct_cols:
            df['dominant_spending_category'] = df[spending_pct_cols].idxmax(axis=1)
            df['dominant_spending_share'] = df[spending_pct_cols].max(axis=1)
        
        # Essential vs discretionary spending
        essential_categories = ['spending_grocery', 'spending_healthcare', 'spending_transportation']
        discretionary_categories = ['spending_restaurants', 'spending_entertainment', 'spending_retail']
        
        essential_cols = [col for col in essential_categories if col in df.columns]
        discretionary_cols = [col for col in discretionary_categories if col in df.columns]
        
        if essential_cols:
            df['essential_spending'] = df[essential_cols].sum(axis=1)
        if discretionary_cols:
            df['discretionary_spending'] = df[discretionary_cols].sum(axis=1)
        
        if essential_cols and discretionary_cols:
            df['discretionary_ratio'] = df['discretionary_spending'] / (df['essential_spending'] + df['discretionary_spending'])
            df['discretionary_ratio'] = df['discretionary_ratio'].fillna(0)
        
        return df
    
    def _add_spending_temporal_features(self, df: pd.DataFrame, 
                                      original_spending_df: pd.DataFrame) -> pd.DataFrame:
        """Add temporal spending patterns"""
        
        # Calculate month-over-month changes
        temporal_features = []
        
        for county in df['county_fips'].unique():
            county_spending = original_spending_df[original_spending_df['county_fips'] == county]
            
            if len(county_spending) < 2:
                continue
            
            # Sort by date
            if 'date' in county_spending.columns:
                county_spending = county_spending.sort_values('date')
            else:
                county_spending = county_spending.sort_values(['year', 'month'])
            
            # Calculate volatility (coefficient of variation)
            spending_volatility = {}
            for category in county_spending['category'].unique():
                cat_data = county_spending[county_spending['category'] == category]['spending_amount']
                if len(cat_data) > 1 and cat_data.mean() > 0:
                    cv = cat_data.std() / cat_data.mean()
                    spending_volatility[f'spending_volatility_{category}'] = cv
            
            # Calculate trend (simple linear trend)
            total_by_period = county_spending.groupby(['year', 'month'])['spending_amount'].sum()
            if len(total_by_period) > 2:
                x = np.arange(len(total_by_period))
                slope, _, _, _, _ = stats.linregress(x, total_by_period.values)
                spending_volatility['spending_trend'] = slope
            
            spending_volatility['county_fips'] = county
            temporal_features.append(spending_volatility)
        
        if temporal_features:
            temporal_df = pd.DataFrame(temporal_features)
            df = df.merge(temporal_df, on='county_fips', how='left')
        
        return df
    
    def _add_per_capita_spending_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add population-normalized spending features"""
        
        # Merge with population data
        pop_data = self.population_data.copy()
        latest_year = pop_data['year'].max()
        pop_latest = pop_data[pop_data['year'] == latest_year][['county_fips', 'population']]
        
        df_with_pop = df.merge(pop_latest, on='county_fips', how='left')
        
        # Calculate per-capita spending metrics
        spending_cols = [col for col in df.columns if col.startswith('spending_') and not col.endswith('_pct_')]
        
        for col in spending_cols:
            per_capita_col = col.replace('spending_', 'spending_per_capita_')
            df_with_pop[per_capita_col] = df_with_pop[col] / df_with_pop['population']
        
        # Total spending per capita
        if 'total_spending' in df_with_pop.columns:
            df_with_pop['total_spending_per_capita'] = df_with_pop['total_spending'] / df_with_pop['population']
        
        return df_with_pop
    
    def handle_skewed_distributions(self, df: pd.DataFrame, 
                                  columns: List[str] = None,
                                  method: str = 'yeo-johnson') -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Handle skewed distributions using power transformations
        """
        if columns is None:
            # Auto-detect skewed columns
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            columns = []
            for col in numeric_cols:
                if abs(df[col].skew()) > 1.0:  # Threshold for skewness
                    columns.append(col)
        
        logger.info(f"Applying {method} transformation to {len(columns)} skewed columns")
        
        df_transformed = df.copy()
        transformers = {}
        
        for col in columns:
            if col not in df.columns:
                continue
            
            # Handle negative values for log transform
            if method == 'log' and (df[col] <= 0).any():
                # Add constant to make all values positive
                min_val = df[col].min()
                shift = abs(min_val) + 1 if min_val <= 0 else 0
                df_transformed[col] = df[col] + shift
                transformers[col] = {'method': 'log', 'shift': shift}
                df_transformed[col] = np.log1p(df_transformed[col])
            
            elif method in ['box-cox', 'yeo-johnson']:
                transformer = PowerTransformer(method=method, standardize=False)
                # Reshape for sklearn
                values = df[col].values.reshape(-1, 1)
                df_transformed[col] = transformer.fit_transform(values).flatten()
                transformers[col] = {'method': method, 'transformer': transformer}
        
        self.transformers.update(transformers)
        return df_transformed, transformers
    
    def remove_correlated_features(self, df: pd.DataFrame, 
                                 threshold: float = 0.85,
                                 exclude_cols: List[str] = None) -> Tuple[pd.DataFrame, List[str]]:
        """
        Remove highly correlated features
        """
        if exclude_cols is None:
            exclude_cols = ['county_fips']
        
        # Get numeric columns for correlation analysis
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        if len(feature_cols) < 2:
            return df, []
        
        # Calculate correlation matrix
        corr_matrix = df[feature_cols].corr().abs()
        
        # Find highly correlated pairs
        upper_triangle = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        
        # Find features to drop
        to_drop = [column for column in upper_triangle.columns if any(upper_triangle[column] > threshold)]
        
        logger.info(f"Removing {len(to_drop)} highly correlated features (threshold: {threshold})")
        
        # Drop correlated features
        df_reduced = df.drop(columns=to_drop)
        
        return df_reduced, to_drop
    
    def standardize_features(self, df: pd.DataFrame, 
                           exclude_cols: List[str] = None,
                           fit_scaler: bool = True) -> pd.DataFrame:
        """
        Standardize features using StandardScaler
        """
        if exclude_cols is None:
            exclude_cols = ['county_fips']
        
        # Get numeric columns for scaling
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        if len(feature_cols) == 0:
            return df
        
        df_scaled = df.copy()
        
        if fit_scaler:
            # Fit new scaler
            scaler = StandardScaler()
            df_scaled[feature_cols] = scaler.fit_transform(df[feature_cols])
            self.scalers['standard'] = scaler
        else:
            # Use existing scaler
            if 'standard' in self.scalers:
                scaler = self.scalers['standard']
                df_scaled[feature_cols] = scaler.transform(df[feature_cols])
            else:
                logger.warning("No fitted scaler found, fitting new one")
                return self.standardize_features(df, exclude_cols, fit_scaler=True)
        
        logger.info(f"Standardized {len(feature_cols)} features")
        return df_scaled
    
    def create_composite_indices(self, df: pd.DataFrame, 
                               feature_groups: Dict[str, List[str]] = None,
                               n_components: int = 2) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Create composite indices using PCA
        """
        if feature_groups is None:
            # Default feature groups
            feature_groups = {
                'mobility_index': [col for col in df.columns if 'trip' in col.lower() or 'mobility' in col.lower()],
                'spending_index': [col for col in df.columns if 'spending' in col.lower()],
                'temporal_index': [col for col in df.columns if any(x in col.lower() for x in ['hour', 'weekend', 'seasonal'])]
            }
        
        df_with_indices = df.copy()
        pca_results = {}
        
        for index_name, feature_list in feature_groups.items():
            # Filter to existing columns
            available_features = [col for col in feature_list if col in df.columns]
            
            if len(available_features) < 2:
                logger.warning(f"Insufficient features for {index_name}: {len(available_features)}")
                continue
            
            # Apply PCA
            pca = PCA(n_components=min(n_components, len(available_features)))
            
            # Handle missing values
            feature_data = df[available_features].fillna(df[available_features].mean())
            
            # Fit PCA
            pca_components = pca.fit_transform(feature_data)
            
            # Add components to dataframe
            for i in range(pca_components.shape[1]):
                component_name = f"{index_name}_pc{i+1}"
                df_with_indices[component_name] = pca_components[:, i]
            
            # Store PCA results
            pca_results[index_name] = {
                'pca': pca,
                'features': available_features,
                'explained_variance_ratio': pca.explained_variance_ratio_,
                'total_variance_explained': pca.explained_variance_ratio_.sum()
            }
            
            logger.info(f"Created {index_name} with {pca_components.shape[1]} components "
                       f"(variance explained: {pca.explained_variance_ratio_.sum():.3f})")
        
        return df_with_indices, pca_results
    
    def create_feature_pipeline(self, mobility_df: pd.DataFrame, 
                              spending_df: pd.DataFrame,
                              trips_df: pd.DataFrame = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Complete feature engineering pipeline
        """
        logger.info("Starting complete feature engineering pipeline")
        
        pipeline_results = {
            'processing_steps': [],
            'feature_counts': {},
            'transformations': {},
            'removed_features': []
        }
        
        # Step 1: Load population data
        self.load_population_data()
        pipeline_results['processing_steps'].append('population_data_loaded')
        
        # Step 2: Create mobility features
        mobility_features = self.create_mobility_features(mobility_df, trips_df)
        pipeline_results['feature_counts']['mobility_features'] = mobility_features.shape[1]
        pipeline_results['processing_steps'].append('mobility_features_created')
        
        # Step 3: Create spending features
        spending_features = self.create_spending_features(spending_df)
        pipeline_results['feature_counts']['spending_features'] = spending_features.shape[1]
        pipeline_results['processing_steps'].append('spending_features_created')
        
        # Step 4: Combine features
        combined_features = mobility_features.merge(spending_features, on='county_fips', how='inner')
        pipeline_results['feature_counts']['combined_features'] = combined_features.shape[1]
        pipeline_results['processing_steps'].append('features_combined')
        
        # Step 5: Handle skewed distributions
        combined_features, transformers = self.handle_skewed_distributions(combined_features)
        pipeline_results['transformations']['skewness'] = transformers
        pipeline_results['processing_steps'].append('skewness_handled')
        
        # Step 6: Remove correlated features
        combined_features, removed_features = self.remove_correlated_features(combined_features)
        pipeline_results['removed_features'] = removed_features
        pipeline_results['feature_counts']['after_correlation_removal'] = combined_features.shape[1]
        pipeline_results['processing_steps'].append('correlation_removed')
        
        # Step 7: Standardize features
        combined_features = self.standardize_features(combined_features)
        pipeline_results['processing_steps'].append('features_standardized')
        
        # Step 8: Create composite indices
        combined_features, pca_results = self.create_composite_indices(combined_features)
        pipeline_results['transformations']['pca'] = pca_results
        pipeline_results['feature_counts']['final_features'] = combined_features.shape[1]
        pipeline_results['processing_steps'].append('composite_indices_created')
        
        # Calculate feature statistics
        self.feature_stats = calculate_data_summary(combined_features)
        
        logger.info(f"Feature engineering pipeline completed. Final dataset: {combined_features.shape}")
        return combined_features, pipeline_results
    
    def export_engineered_features(self, features_df: pd.DataFrame, 
                                 pipeline_results: Dict[str, Any],
                                 filename: str = 'engineered_features.parquet') -> str:
        """
        Export engineered features with metadata
        """
        metadata = {
            'feature_engineering_pipeline': pipeline_results,
            'feature_statistics': self.feature_stats,
            'scalers_info': {name: 'fitted' for name in self.scalers.keys()},
            'transformers_info': {name: info.get('method', 'unknown') for name, info in self.transformers.items()}
        }
        
        filepath = save_processed_data(features_df, filename, metadata)
        logger.info(f"Exported engineered features to {filepath}")
        
        return filepath


def main():
    """Demo function for feature engineering pipeline"""
    from src.data_loader import DataLoader
    from src.spatial_processor import SpatialProcessor
    
    # Initialize components
    loader = DataLoader()
    processor = SpatialProcessor()
    engineer = FeatureEngineer()
    
    logger.info("Starting feature engineering demo...")
    
    # Load sample data
    trips_df = loader.download_divvy_data(2023, 6)
    spending_df = loader.download_spending_data()
    
    # Process spatial data
    boundaries = processor.load_county_boundaries()
    stations_gdf = processor.extract_stations_from_trips(trips_df)
    stations_with_counties = processor.assign_stations_to_counties(stations_gdf, boundaries)
    county_mobility = processor.aggregate_trips_to_county_level(trips_df, stations_with_counties)
    
    # Run feature engineering pipeline
    engineered_features, pipeline_results = engineer.create_feature_pipeline(
        county_mobility, spending_df, trips_df
    )
    
    # Export results
    filepath = engineer.export_engineered_features(engineered_features, pipeline_results)
    
    # Print summary
    print(f"\nFeature Engineering Summary:")
    print(f"- Final dataset shape: {engineered_features.shape}")
    print(f"- Processing steps completed: {len(pipeline_results['processing_steps'])}")
    print(f"- Features removed due to correlation: {len(pipeline_results['removed_features'])}")
    print(f"- Transformations applied: {len(pipeline_results['transformations'])}")
    
    # Show sample of engineered features
    print(f"\nSample of engineered features:")
    print(engineered_features.head())
    
    logger.info("Feature engineering demo completed successfully!")
    
    return engineered_features, pipeline_results


if __name__ == "__main__":
    main()