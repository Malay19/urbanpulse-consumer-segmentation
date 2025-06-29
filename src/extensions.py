"""
Advanced Analytics Extensions for Consumer Segmentation Analysis
Implements predictive modeling, scalability features, and advanced visualizations
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple
from loguru import logger
import warnings
from datetime import datetime, timedelta
import json
from pathlib import Path

# Machine Learning imports
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, classification_report
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

# Advanced visualization imports
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import networkx as nx

# Privacy and ethics imports
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist, squareform

from config import DATA_CONFIG, MODEL_CONFIG


class PredictiveModeling:
    """Advanced predictive modeling for spending pattern prediction"""
    
    def __init__(self):
        self.models = {}
        self.feature_importance = {}
        self.performance_metrics = {}
        
    def prepare_prediction_data(self, mobility_df: pd.DataFrame, 
                              spending_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Prepare data for predictive modeling with temporal alignment
        """
        logger.info("Preparing data for predictive modeling")
        
        # Aggregate spending by county and time period
        spending_agg = spending_df.groupby(['county_fips', 'year', 'month']).agg({
            'spending_amount': 'sum'
        }).reset_index()
        
        # Create time-based features
        spending_agg['date'] = pd.to_datetime(spending_agg[['year', 'month']].assign(day=1))
        spending_agg['quarter'] = spending_agg['date'].dt.quarter
        spending_agg['season'] = spending_agg['month'].map({
            12: 'Winter', 1: 'Winter', 2: 'Winter',
            3: 'Spring', 4: 'Spring', 5: 'Spring',
            6: 'Summer', 7: 'Summer', 8: 'Summer',
            9: 'Fall', 10: 'Fall', 11: 'Fall'
        })
        
        # Add lagged features for time series prediction
        spending_agg = spending_agg.sort_values(['county_fips', 'date'])
        spending_agg['spending_lag_1'] = spending_agg.groupby('county_fips')['spending_amount'].shift(1)
        spending_agg['spending_lag_3'] = spending_agg.groupby('county_fips')['spending_amount'].shift(3)
        spending_agg['spending_ma_3'] = spending_agg.groupby('county_fips')['spending_amount'].rolling(3).mean().reset_index(0, drop=True)
        
        # Merge with mobility data
        combined_data = mobility_df.merge(spending_agg, on='county_fips', how='inner')
        
        # Create target variables
        targets = combined_data[['county_fips', 'date', 'spending_amount']].copy()
        
        # Feature engineering for predictors
        feature_cols = [col for col in combined_data.columns if col not in [
            'county_fips', 'date', 'spending_amount', 'year', 'month'
        ]]
        
        features = combined_data[['county_fips', 'date'] + feature_cols].copy()
        
        logger.info(f"Prepared {len(features)} samples with {len(feature_cols)} features")
        return features, targets
    
    def train_spending_predictor(self, features_df: pd.DataFrame, 
                               targets_df: pd.DataFrame,
                               model_type: str = 'random_forest') -> Dict[str, Any]:
        """
        Train predictive model for spending patterns
        """
        logger.info(f"Training {model_type} spending predictor")
        
        # Prepare data
        X = features_df.select_dtypes(include=[np.number]).fillna(0)
        y = targets_df['spending_amount']
        
        # Remove any remaining non-numeric or infinite values
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=3)
        
        if model_type == 'random_forest':
            model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            
            # Hyperparameter tuning
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15],
                'min_samples_split': [2, 5, 10]
            }
            
        elif model_type == 'xgboost':
            model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
            
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.05, 0.1, 0.2]
            }
        
        # Grid search with time series CV
        grid_search = GridSearchCV(
            model, param_grid, cv=tscv, 
            scoring='neg_mean_squared_error', n_jobs=-1
        )
        
        grid_search.fit(X, y)
        best_model = grid_search.best_estimator_
        
        # Cross-validation scores
        cv_scores = cross_val_score(best_model, X, y, cv=tscv, scoring='r2')
        
        # Feature importance
        if hasattr(best_model, 'feature_importances_'):
            feature_importance = dict(zip(X.columns, best_model.feature_importances_))
            feature_importance = dict(sorted(feature_importance.items(), 
                                           key=lambda x: x[1], reverse=True))
        else:
            feature_importance = {}
        
        # Performance metrics
        y_pred = best_model.predict(X)
        performance = {
            'r2_score': r2_score(y, y_pred),
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'cv_r2_mean': cv_scores.mean(),
            'cv_r2_std': cv_scores.std(),
            'best_params': grid_search.best_params_
        }
        
        # Store results
        self.models[model_type] = best_model
        self.feature_importance[model_type] = feature_importance
        self.performance_metrics[model_type] = performance
        
        logger.info(f"Model trained. R² score: {performance['r2_score']:.3f}")
        
        return {
            'model': best_model,
            'feature_importance': feature_importance,
            'performance': performance,
            'cv_scores': cv_scores
        }
    
    def predict_future_spending(self, features_df: pd.DataFrame, 
                              model_type: str = 'random_forest',
                              periods: int = 6) -> pd.DataFrame:
        """
        Predict future spending patterns
        """
        if model_type not in self.models:
            raise ValueError(f"Model {model_type} not trained yet")
        
        model = self.models[model_type]
        
        # Prepare features for prediction
        X = features_df.select_dtypes(include=[np.number]).fillna(0)
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        # Make predictions
        predictions = model.predict(X)
        
        # Create prediction dataframe
        pred_df = features_df[['county_fips']].copy()
        pred_df['predicted_spending'] = predictions
        pred_df['prediction_date'] = datetime.now()
        
        return pred_df
    
    def analyze_prediction_drivers(self, model_type: str = 'random_forest') -> Dict[str, Any]:
        """
        Analyze what drives spending predictions
        """
        if model_type not in self.feature_importance:
            raise ValueError(f"No feature importance available for {model_type}")
        
        importance = self.feature_importance[model_type]
        
        # Categorize features
        mobility_features = {k: v for k, v in importance.items() if 'trip' in k.lower() or 'mobility' in k.lower()}
        temporal_features = {k: v for k, v in importance.items() if any(x in k.lower() for x in ['hour', 'day', 'month', 'season'])}
        demographic_features = {k: v for k, v in importance.items() if any(x in k.lower() for x in ['income', 'age', 'education'])}
        
        analysis = {
            'top_features': dict(list(importance.items())[:10]),
            'mobility_importance': sum(mobility_features.values()),
            'temporal_importance': sum(temporal_features.values()),
            'demographic_importance': sum(demographic_features.values()),
            'feature_categories': {
                'mobility': mobility_features,
                'temporal': temporal_features,
                'demographic': demographic_features
            }
        }
        
        return analysis


class ScalabilityFramework:
    """Framework for scaling analysis to multiple cities and systems"""
    
    def __init__(self):
        self.city_configs = {}
        self.data_adapters = {}
        
    def register_city_config(self, city_name: str, config: Dict[str, Any]):
        """
        Register configuration for a new city
        """
        required_keys = ['data_sources', 'geographic_bounds', 'population', 'bike_share_system']
        
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required config key: {key}")
        
        self.city_configs[city_name] = config
        logger.info(f"Registered configuration for {city_name}")
    
    def create_data_adapter(self, city_name: str, system_type: str) -> 'DataAdapter':
        """
        Create data adapter for specific bike-share system
        """
        if city_name not in self.city_configs:
            raise ValueError(f"No configuration found for {city_name}")
        
        config = self.city_configs[city_name]
        
        if system_type == 'divvy':
            adapter = DivvyAdapter(config)
        elif system_type == 'citibike':
            adapter = CitiBikeAdapter(config)
        elif system_type == 'generic':
            adapter = GenericAdapter(config)
        else:
            raise ValueError(f"Unsupported system type: {system_type}")
        
        self.data_adapters[city_name] = adapter
        return adapter
    
    def run_multi_city_analysis(self, cities: List[str]) -> Dict[str, Any]:
        """
        Run analysis across multiple cities
        """
        results = {}
        
        for city in cities:
            if city not in self.data_adapters:
                logger.warning(f"No adapter found for {city}, skipping")
                continue
            
            try:
                adapter = self.data_adapters[city]
                city_results = adapter.run_analysis()
                results[city] = city_results
                logger.info(f"Completed analysis for {city}")
                
            except Exception as e:
                logger.error(f"Analysis failed for {city}: {e}")
                results[city] = {'error': str(e)}
        
        return results
    
    def compare_cities(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare results across cities
        """
        comparison = {
            'city_metrics': {},
            'cluster_comparison': {},
            'best_practices': []
        }
        
        for city, city_results in results.items():
            if 'error' in city_results:
                continue
            
            # Extract key metrics
            if 'personas' in city_results:
                personas = city_results['personas']
                comparison['city_metrics'][city] = {
                    'total_market_value': sum(p.market_value for p in personas.values()),
                    'total_population': sum(p.estimated_population for p in personas.values()),
                    'avg_effectiveness': np.mean([p.targeting_effectiveness for p in personas.values()]),
                    'num_segments': len(personas)
                }
        
        return comparison


class DataAdapter:
    """Base class for city-specific data adapters"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
    def load_trip_data(self) -> pd.DataFrame:
        """Load trip data - to be implemented by subclasses"""
        raise NotImplementedError
        
    def load_spending_data(self) -> pd.DataFrame:
        """Load spending data - to be implemented by subclasses"""
        raise NotImplementedError
        
    def run_analysis(self) -> Dict[str, Any]:
        """Run complete analysis pipeline"""
        from pipeline_manager import PipelineManager, PipelineConfig
        
        # Load data
        trips_df = self.load_trip_data()
        spending_df = self.load_spending_data()
        
        # Configure pipeline
        config = PipelineConfig(
            counties=self.config.get('counties', []),
            clustering_algorithms=['hdbscan', 'kmeans']
        )
        
        # Run pipeline
        pipeline = PipelineManager(config)
        results = pipeline.run_complete_pipeline()
        
        return results


class DivvyAdapter(DataAdapter):
    """Adapter for Divvy bike-share system (Chicago)"""
    
    def load_trip_data(self) -> pd.DataFrame:
        from src.data_loader import DataLoader
        loader = DataLoader()
        return loader.download_divvy_data(2023, 6)
    
    def load_spending_data(self) -> pd.DataFrame:
        from src.data_loader import DataLoader
        loader = DataLoader()
        return loader.download_spending_data(self.config.get('counties', ['17031']))


class CitiBikeAdapter(DataAdapter):
    """Adapter for Citi Bike system (NYC)"""
    
    def load_trip_data(self) -> pd.DataFrame:
        # Implement Citi Bike specific data loading
        # This would connect to Citi Bike's API or data sources
        logger.info("Loading Citi Bike trip data")
        # For demo, return sample data
        return self._generate_sample_citibike_data()
    
    def _generate_sample_citibike_data(self) -> pd.DataFrame:
        """Generate sample Citi Bike data for demonstration"""
        # NYC coordinates
        nyc_bounds = {
            'lat_min': 40.7, 'lat_max': 40.8,
            'lng_min': -74.0, 'lng_max': -73.9
        }
        
        n_trips = 25000
        data = []
        
        for i in range(n_trips):
            data.append({
                'trip_id': f'citibike_{i:06d}',
                'start_time': datetime(2023, 6, 1) + timedelta(
                    days=np.random.randint(0, 30),
                    hours=np.random.randint(0, 24),
                    minutes=np.random.randint(0, 60)
                ),
                'end_time': datetime(2023, 6, 1) + timedelta(
                    days=np.random.randint(0, 30),
                    hours=np.random.randint(0, 24),
                    minutes=np.random.randint(5, 60)
                ),
                'start_station_id': np.random.randint(1, 500),
                'end_station_id': np.random.randint(1, 500),
                'start_lat': np.random.uniform(nyc_bounds['lat_min'], nyc_bounds['lat_max']),
                'start_lng': np.random.uniform(nyc_bounds['lng_min'], nyc_bounds['lng_max']),
                'end_lat': np.random.uniform(nyc_bounds['lat_min'], nyc_bounds['lat_max']),
                'end_lng': np.random.uniform(nyc_bounds['lng_min'], nyc_bounds['lng_max']),
                'member_type': np.random.choice(['member', 'casual'], p=[0.7, 0.3])
            })
        
        return pd.DataFrame(data)


class GenericAdapter(DataAdapter):
    """Generic adapter for any bike-share system"""
    
    def load_trip_data(self) -> pd.DataFrame:
        # Implement generic data loading based on configuration
        logger.info("Loading generic bike-share data")
        return self._generate_generic_data()
    
    def _generate_generic_data(self) -> pd.DataFrame:
        """Generate generic bike-share data"""
        bounds = self.config.get('geographic_bounds', {
            'lat_min': 40.0, 'lat_max': 41.0,
            'lng_min': -88.0, 'lng_max': -87.0
        })
        
        n_trips = self.config.get('sample_size', 10000)
        data = []
        
        for i in range(n_trips):
            data.append({
                'trip_id': f'generic_{i:06d}',
                'start_time': datetime(2023, 6, 1) + timedelta(
                    days=np.random.randint(0, 30),
                    hours=np.random.randint(0, 24)
                ),
                'end_time': datetime(2023, 6, 1) + timedelta(
                    days=np.random.randint(0, 30),
                    hours=np.random.randint(0, 24),
                    minutes=np.random.randint(5, 60)
                ),
                'start_station_id': np.random.randint(1, 200),
                'end_station_id': np.random.randint(1, 200),
                'start_lat': np.random.uniform(bounds['lat_min'], bounds['lat_max']),
                'start_lng': np.random.uniform(bounds['lng_min'], bounds['lng_max']),
                'end_lat': np.random.uniform(bounds['lat_min'], bounds['lat_max']),
                'end_lng': np.random.uniform(bounds['lng_min'], bounds['lng_max']),
                'member_type': np.random.choice(['member', 'casual'])
            })
        
        return pd.DataFrame(data)


class AdvancedVisualizations:
    """Advanced visualization capabilities"""
    
    def __init__(self):
        self.animations = {}
        self.interactive_plots = {}
    
    def create_animated_seasonal_map(self, data_by_season: Dict[str, pd.DataFrame]) -> go.Figure:
        """
        Create animated map showing seasonal changes
        """
        logger.info("Creating animated seasonal map")
        
        fig = go.Figure()
        
        # Create frames for each season
        frames = []
        for season, data in data_by_season.items():
            frame = go.Frame(
                data=[
                    go.Scattermapbox(
                        lat=data['lat'],
                        lon=data['lng'],
                        mode='markers',
                        marker=dict(
                            size=data['value'] / data['value'].max() * 20 + 5,
                            color=data['cluster'],
                            colorscale='Viridis',
                            opacity=0.7
                        ),
                        text=data['hover_text'],
                        hovertemplate='<b>%{text}</b><br>Season: ' + season + '<extra></extra>'
                    )
                ],
                name=season
            )
            frames.append(frame)
        
        # Add initial frame
        if frames:
            fig.add_trace(frames[0].data[0])
        
        # Update layout for animation
        fig.update_layout(
            mapbox=dict(
                style='open-street-map',
                center=dict(lat=41.8781, lon=-87.6298),
                zoom=10
            ),
            title='Seasonal Mobility Patterns',
            updatemenus=[{
                'type': 'buttons',
                'showactive': False,
                'buttons': [
                    {
                        'label': 'Play',
                        'method': 'animate',
                        'args': [None, {
                            'frame': {'duration': 1000, 'redraw': True},
                            'transition': {'duration': 300}
                        }]
                    },
                    {
                        'label': 'Pause',
                        'method': 'animate',
                        'args': [[None], {
                            'frame': {'duration': 0, 'redraw': False},
                            'mode': 'immediate',
                            'transition': {'duration': 0}
                        }]
                    }
                ]
            }],
            sliders=[{
                'steps': [
                    {
                        'args': [[season], {
                            'frame': {'duration': 300, 'redraw': True},
                            'mode': 'immediate',
                            'transition': {'duration': 300}
                        }],
                        'label': season,
                        'method': 'animate'
                    }
                    for season in data_by_season.keys()
                ],
                'active': 0,
                'currentvalue': {'prefix': 'Season: '},
                'len': 0.9,
                'x': 0.1,
                'xanchor': 'left',
                'y': 0,
                'yanchor': 'top'
            }]
        )
        
        fig.frames = frames
        return fig
    
    def create_sankey_mobility_spending(self, mobility_data: pd.DataFrame, 
                                      spending_data: pd.DataFrame) -> go.Figure:
        """
        Create Sankey diagram showing mobility-spending flows
        """
        logger.info("Creating Sankey diagram for mobility-spending flows")
        
        # Prepare data for Sankey
        # Create mobility categories
        mobility_data['mobility_category'] = pd.cut(
            mobility_data['total_trips'],
            bins=3,
            labels=['Low Mobility', 'Medium Mobility', 'High Mobility']
        )
        
        # Create spending categories
        spending_data['spending_category'] = pd.cut(
            spending_data['total_spending'],
            bins=3,
            labels=['Low Spending', 'Medium Spending', 'High Spending']
        )
        
        # Merge data
        combined = mobility_data.merge(spending_data, on='county_fips', how='inner')
        
        # Create flow matrix
        flow_matrix = pd.crosstab(combined['mobility_category'], combined['spending_category'])
        
        # Prepare Sankey data
        mobility_labels = list(flow_matrix.index)
        spending_labels = list(flow_matrix.columns)
        all_labels = mobility_labels + spending_labels
        
        source = []
        target = []
        value = []
        
        for i, mobility_cat in enumerate(mobility_labels):
            for j, spending_cat in enumerate(spending_labels):
                if flow_matrix.loc[mobility_cat, spending_cat] > 0:
                    source.append(i)
                    target.append(len(mobility_labels) + j)
                    value.append(flow_matrix.loc[mobility_cat, spending_cat])
        
        # Create Sankey diagram
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=15,
                thickness=20,
                line=dict(color="black", width=0.5),
                label=all_labels,
                color=["lightblue"] * len(mobility_labels) + ["lightcoral"] * len(spending_labels)
            ),
            link=dict(
                source=source,
                target=target,
                value=value,
                color="rgba(255, 0, 255, 0.4)"
            )
        )])
        
        fig.update_layout(
            title_text="Mobility-Spending Flow Analysis",
            font_size=12,
            height=600
        )
        
        return fig
    
    def create_3d_cluster_plot(self, features_df: pd.DataFrame, 
                             cluster_labels: np.ndarray,
                             feature_names: List[str]) -> go.Figure:
        """
        Create 3D scatter plot for multi-dimensional cluster analysis
        """
        logger.info("Creating 3D cluster visualization")
        
        if len(feature_names) < 3:
            raise ValueError("Need at least 3 features for 3D plot")
        
        # Select top 3 features
        x_feature, y_feature, z_feature = feature_names[:3]
        
        # Create 3D scatter plot
        fig = go.Figure()
        
        unique_clusters = np.unique(cluster_labels)
        colors = px.colors.qualitative.Set1
        
        for i, cluster in enumerate(unique_clusters):
            mask = cluster_labels == cluster
            cluster_name = f'Cluster {cluster}' if cluster != -1 else 'Outliers'
            
            fig.add_trace(go.Scatter3d(
                x=features_df.loc[mask, x_feature],
                y=features_df.loc[mask, y_feature],
                z=features_df.loc[mask, z_feature],
                mode='markers',
                marker=dict(
                    size=5,
                    color=colors[i % len(colors)],
                    opacity=0.7
                ),
                name=cluster_name,
                text=[f'County: {county}' for county in features_df.loc[mask, 'county_fips']],
                hovertemplate='<b>%{text}</b><br>' +
                             f'{x_feature}: %{{x}}<br>' +
                             f'{y_feature}: %{{y}}<br>' +
                             f'{z_feature}: %{{z}}<extra></extra>'
            ))
        
        fig.update_layout(
            title='3D Cluster Analysis',
            scene=dict(
                xaxis_title=x_feature,
                yaxis_title=y_feature,
                zaxis_title=z_feature
            ),
            height=700
        )
        
        return fig


class ExportCapabilities:
    """Advanced export capabilities for reports and data"""
    
    def __init__(self):
        self.export_formats = ['pdf', 'excel', 'html']
    
    def generate_pdf_report(self, personas: Dict, opportunities: List, 
                          insights: Dict, output_path: str) -> str:
        """
        Generate comprehensive PDF report
        """
        logger.info("Generating PDF report")
        
        try:
            import matplotlib.pyplot as plt
            from matplotlib.backends.backend_pdf import PdfPages
            import matplotlib.patches as mpatches
            
            with PdfPages(output_path) as pdf:
                # Title page
                fig, ax = plt.subplots(figsize=(8.5, 11))
                ax.text(0.5, 0.7, 'Consumer Segmentation Analysis', 
                       ha='center', va='center', fontsize=24, fontweight='bold')
                ax.text(0.5, 0.6, 'Advanced Analytics Report', 
                       ha='center', va='center', fontsize=16)
                ax.text(0.5, 0.5, f'Generated: {datetime.now().strftime("%Y-%m-%d")}', 
                       ha='center', va='center', fontsize=12)
                ax.axis('off')
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()
                
                # Executive summary
                fig, ax = plt.subplots(figsize=(8.5, 11))
                ax.text(0.1, 0.9, 'Executive Summary', fontsize=18, fontweight='bold')
                
                summary_text = f"""
Market Overview:
• Total Market Value: ${insights.get('market_overview', {}).get('total_addressable_market', 0):,.0f}
• Total Population: {insights.get('market_overview', {}).get('total_population', 0):,}
• Number of Segments: {len(personas)}
• Average Targeting Effectiveness: {insights.get('market_overview', {}).get('average_targeting_effectiveness', 0):.1%}

Key Insights:
"""
                for insight in insights.get('key_insights', [])[:5]:
                    summary_text += f"• {insight}\n"
                
                ax.text(0.1, 0.8, summary_text, fontsize=10, va='top', wrap=True)
                ax.axis('off')
                pdf.savefig(fig, bbox_inches='tight')
                plt.close()
                
                # Persona pages
                for persona_id, persona in personas.items():
                    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8.5, 11))
                    fig.suptitle(f'{persona.persona_name}', fontsize=16, fontweight='bold')
                    
                    # Persona details
                    details_text = f"""
Type: {persona.persona_type.value}
Population: {persona.estimated_population:,}
Market Value: ${persona.market_value:,.0f}
Targeting Effectiveness: {persona.targeting_effectiveness:.1%}

Description:
{persona.description}
"""
                    ax1.text(0.05, 0.95, details_text, fontsize=9, va='top', transform=ax1.transAxes)
                    ax1.axis('off')
                    ax1.set_title('Persona Overview')
                    
                    # Seasonal trends
                    seasons = list(persona.seasonal_trends.keys())
                    values = list(persona.seasonal_trends.values())
                    ax2.bar(seasons, values, color='skyblue')
                    ax2.set_title('Seasonal Trends')
                    ax2.set_ylabel('Usage Multiplier')
                    
                    # Key motivations
                    motivations_text = "Key Motivations:\n" + "\n".join([f"• {m}" for m in persona.key_motivations[:5]])
                    ax3.text(0.05, 0.95, motivations_text, fontsize=9, va='top', transform=ax3.transAxes)
                    ax3.axis('off')
                    ax3.set_title('Motivations & Channels')
                    
                    # Pain points
                    pain_points_text = "Pain Points:\n" + "\n".join([f"• {p}" for p in persona.pain_points[:5]])
                    ax4.text(0.05, 0.95, pain_points_text, fontsize=9, va='top', transform=ax4.transAxes)
                    ax4.axis('off')
                    ax4.set_title('Pain Points')
                    
                    plt.tight_layout()
                    pdf.savefig(fig, bbox_inches='tight')
                    plt.close()
            
            logger.info(f"PDF report generated: {output_path}")
            return output_path
            
        except ImportError:
            logger.error("matplotlib not available for PDF generation")
            return None
    
    def export_to_excel(self, personas: Dict, opportunities: List, 
                       insights: Dict, output_path: str) -> str:
        """
        Export data to formatted Excel file
        """
        logger.info("Exporting to Excel")
        
        try:
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                # Personas sheet
                personas_data = []
                for persona_id, persona in personas.items():
                    personas_data.append({
                        'Persona ID': persona_id,
                        'Persona Name': persona.persona_name,
                        'Type': persona.persona_type.value,
                        'Population': persona.estimated_population,
                        'Market Value': persona.market_value,
                        'Targeting Effectiveness': persona.targeting_effectiveness,
                        'Description': persona.description
                    })
                
                personas_df = pd.DataFrame(personas_data)
                personas_df.to_excel(writer, sheet_name='Personas', index=False)
                
                # Opportunities sheet
                opportunities_data = []
                for opp in opportunities:
                    opportunities_data.append({
                        'Opportunity Type': opp.opportunity_type,
                        'Description': opp.description,
                        'Target Segments': ', '.join(opp.target_segments),
                        'Market Size': opp.estimated_market_size,
                        'Investment Level': opp.investment_level,
                        'Expected ROI': opp.expected_roi,
                        'Timeline': opp.implementation_timeline
                    })
                
                opportunities_df = pd.DataFrame(opportunities_data)
                opportunities_df.to_excel(writer, sheet_name='Opportunities', index=False)
                
                # Market insights sheet
                insights_data = []
                market_overview = insights.get('market_overview', {})
                for key, value in market_overview.items():
                    insights_data.append({'Metric': key, 'Value': value})
                
                insights_df = pd.DataFrame(insights_data)
                insights_df.to_excel(writer, sheet_name='Market Overview', index=False)
            
            logger.info(f"Excel file exported: {output_path}")
            return output_path
            
        except ImportError:
            logger.error("openpyxl not available for Excel export")
            return None
    
    def export_static_html(self, personas: Dict, opportunities: List, 
                          insights: Dict, output_path: str) -> str:
        """
        Export static HTML report
        """
        logger.info("Exporting static HTML report")
        
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Consumer Segmentation Analysis Report</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; text-align: center; margin-bottom: 30px; }}
        h2 {{ color: #34495e; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
        .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin: 20px 0; }}
        .metric-card {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 8px; text-align: center; }}
        .persona-card {{ border: 1px solid #ddd; border-radius: 8px; padding: 20px; margin: 15px 0; background: #f9f9f9; }}
        .opportunity-card {{ border-left: 4px solid #3498db; padding: 15px; margin: 10px 0; background: #ecf0f1; }}
        .insight-list {{ list-style-type: none; padding: 0; }}
        .insight-list li {{ background: #e8f5e8; margin: 5px 0; padding: 10px; border-radius: 5px; border-left: 4px solid #27ae60; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Consumer Segmentation Analysis Report</h1>
        <p style="text-align: center; color: #7f8c8d;">Generated on {datetime.now().strftime('%B %d, %Y')}</p>
        
        <h2>Market Overview</h2>
        <div class="metric-grid">
            <div class="metric-card">
                <h3>Total Market Value</h3>
                <p style="font-size: 24px; margin: 0;">${insights.get('market_overview', {}).get('total_addressable_market', 0):,.0f}</p>
            </div>
            <div class="metric-card">
                <h3>Total Population</h3>
                <p style="font-size: 24px; margin: 0;">{insights.get('market_overview', {}).get('total_population', 0):,}</p>
            </div>
            <div class="metric-card">
                <h3>Segments Identified</h3>
                <p style="font-size: 24px; margin: 0;">{len(personas)}</p>
            </div>
            <div class="metric-card">
                <h3>Avg. Effectiveness</h3>
                <p style="font-size: 24px; margin: 0;">{insights.get('market_overview', {}).get('average_targeting_effectiveness', 0):.1%}</p>
            </div>
        </div>
        
        <h2>Consumer Personas</h2>
"""
        
        # Add personas
        for persona_id, persona in personas.items():
            html_content += f"""
        <div class="persona-card">
            <h3 style="color: #2c3e50;">{persona.persona_name}</h3>
            <p><strong>Type:</strong> {persona.persona_type.value}</p>
            <p><strong>Population:</strong> {persona.estimated_population:,}</p>
            <p><strong>Market Value:</strong> ${persona.market_value:,.0f}</p>
            <p><strong>Targeting Effectiveness:</strong> {persona.targeting_effectiveness:.1%}</p>
            <p>{persona.description}</p>
            <div style="margin-top: 15px;">
                <strong>Key Motivations:</strong>
                <ul>
                    {''.join([f'<li>{motivation}</li>' for motivation in persona.key_motivations[:3]])}
                </ul>
            </div>
        </div>
"""
        
        # Add opportunities
        html_content += """
        <h2>Business Opportunities</h2>
"""
        
        for opp in opportunities:
            html_content += f"""
        <div class="opportunity-card">
            <h4 style="margin-top: 0; color: #2c3e50;">{opp.opportunity_type}</h4>
            <p>{opp.description}</p>
            <p><strong>Market Size:</strong> ${opp.estimated_market_size:,.0f} | 
               <strong>Expected ROI:</strong> {opp.expected_roi} | 
               <strong>Timeline:</strong> {opp.implementation_timeline}</p>
        </div>
"""
        
        # Add key insights
        html_content += """
        <h2>Key Insights</h2>
        <ul class="insight-list">
"""
        
        for insight in insights.get('key_insights', []):
            html_content += f"<li>{insight}</li>"
        
        html_content += """
        </ul>
    </div>
</body>
</html>
"""
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"Static HTML report exported: {output_path}")
        return output_path


class PrivacyEthicsFramework:
    """Privacy and ethics framework for responsible analytics"""
    
    def __init__(self):
        self.privacy_checks = {}
        self.bias_metrics = {}
        
    def check_data_anonymization(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Check data anonymization and privacy protection
        """
        logger.info("Checking data anonymization")
        
        checks = {
            'has_direct_identifiers': False,
            'has_quasi_identifiers': False,
            'k_anonymity_level': 0,
            'recommendations': []
        }
        
        # Check for direct identifiers
        direct_id_patterns = ['name', 'email', 'phone', 'ssn', 'id']
        for col in data.columns:
            if any(pattern in col.lower() for pattern in direct_id_patterns):
                checks['has_direct_identifiers'] = True
                checks['recommendations'].append(f"Remove or hash column: {col}")
        
        # Check for quasi-identifiers
        quasi_id_patterns = ['zip', 'birth', 'age', 'income']
        quasi_id_cols = []
        for col in data.columns:
            if any(pattern in col.lower() for pattern in quasi_id_patterns):
                checks['has_quasi_identifiers'] = True
                quasi_id_cols.append(col)
        
        # Calculate k-anonymity for geographic data
        if 'county_fips' in data.columns:
            county_counts = data['county_fips'].value_counts()
            checks['k_anonymity_level'] = county_counts.min()
            
            if checks['k_anonymity_level'] < 5:
                checks['recommendations'].append("Consider geographic aggregation for k-anonymity")
        
        # Privacy recommendations
        if not checks['recommendations']:
            checks['recommendations'].append("Data appears to be appropriately anonymized")
        
        self.privacy_checks = checks
        return checks
    
    def detect_clustering_bias(self, features_df: pd.DataFrame, 
                             cluster_labels: np.ndarray,
                             protected_attributes: List[str] = None) -> Dict[str, Any]:
        """
        Detect potential bias in clustering results
        """
        logger.info("Detecting clustering bias")
        
        if protected_attributes is None:
            protected_attributes = ['income', 'age', 'education']
        
        bias_metrics = {
            'demographic_parity': {},
            'cluster_balance': {},
            'feature_correlation': {},
            'recommendations': []
        }
        
        # Check demographic parity across clusters
        for attr in protected_attributes:
            if attr in features_df.columns:
                # Create bins for continuous variables
                if features_df[attr].dtype in ['float64', 'int64']:
                    attr_binned = pd.cut(features_df[attr], bins=3, labels=['Low', 'Medium', 'High'])
                else:
                    attr_binned = features_df[attr]
                
                # Calculate representation across clusters
                cluster_attr_dist = pd.crosstab(cluster_labels, attr_binned, normalize='columns')
                
                # Calculate bias metric (coefficient of variation)
                bias_score = cluster_attr_dist.std(axis=1).mean()
                bias_metrics['demographic_parity'][attr] = bias_score
                
                if bias_score > 0.3:  # Threshold for concern
                    bias_metrics['recommendations'].append(
                        f"High demographic disparity detected for {attr} across clusters"
                    )
        
        # Check cluster size balance
        cluster_sizes = pd.Series(cluster_labels).value_counts()
        size_cv = cluster_sizes.std() / cluster_sizes.mean()
        bias_metrics['cluster_balance']['size_coefficient_variation'] = size_cv
        
        if size_cv > 1.0:
            bias_metrics['recommendations'].append("Highly imbalanced cluster sizes detected")
        
        # Check feature correlation with protected attributes
        for attr in protected_attributes:
            if attr in features_df.columns:
                correlations = features_df.corrwith(features_df[attr]).abs()
                high_corr_features = correlations[correlations > 0.7].index.tolist()
                
                if high_corr_features:
                    bias_metrics['feature_correlation'][attr] = high_corr_features
                    bias_metrics['recommendations'].append(
                        f"High correlation between {attr} and features: {high_corr_features}"
                    )
        
        if not bias_metrics['recommendations']:
            bias_metrics['recommendations'].append("No significant bias detected in clustering")
        
        self.bias_metrics = bias_metrics
        return bias_metrics
    
    def ensure_geographic_privacy(self, data: pd.DataFrame, 
                                min_population: int = 1000) -> pd.DataFrame:
        """
        Ensure geographic privacy through k-anonymity
        """
        logger.info("Ensuring geographic privacy")
        
        if 'county_fips' not in data.columns:
            return data
        
        # Check population per geographic unit
        county_counts = data['county_fips'].value_counts()
        small_counties = county_counts[county_counts < min_population].index
        
        if len(small_counties) > 0:
            logger.warning(f"Found {len(small_counties)} counties with < {min_population} records")
            
            # Aggregate small counties into "Other" category
            data_protected = data.copy()
            data_protected.loc[data_protected['county_fips'].isin(small_counties), 'county_fips'] = 'OTHER'
            
            logger.info(f"Aggregated {len(small_counties)} small counties for privacy protection")
            return data_protected
        
        return data
    
    def generate_ethics_report(self) -> str:
        """
        Generate comprehensive ethics and privacy report
        """
        report = []
        report.append("PRIVACY AND ETHICS ASSESSMENT REPORT")
        report.append("=" * 50)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Privacy assessment
        if self.privacy_checks:
            report.append("PRIVACY ASSESSMENT")
            report.append("-" * 30)
            report.append(f"Direct Identifiers Present: {self.privacy_checks['has_direct_identifiers']}")
            report.append(f"Quasi-Identifiers Present: {self.privacy_checks['has_quasi_identifiers']}")
            report.append(f"K-Anonymity Level: {self.privacy_checks['k_anonymity_level']}")
            report.append("")
            
            report.append("Privacy Recommendations:")
            for rec in self.privacy_checks['recommendations']:
                report.append(f"• {rec}")
            report.append("")
        
        # Bias assessment
        if self.bias_metrics:
            report.append("BIAS ASSESSMENT")
            report.append("-" * 30)
            
            if self.bias_metrics['demographic_parity']:
                report.append("Demographic Parity Scores:")
                for attr, score in self.bias_metrics['demographic_parity'].items():
                    report.append(f"• {attr}: {score:.3f}")
                report.append("")
            
            report.append("Bias Mitigation Recommendations:")
            for rec in self.bias_metrics['recommendations']:
                report.append(f"• {rec}")
            report.append("")
        
        report.append("ETHICAL GUIDELINES")
        report.append("-" * 30)
        report.append("• Ensure informed consent for data usage")
        report.append("• Regularly audit for algorithmic bias")
        report.append("• Implement data minimization principles")
        report.append("• Provide transparency in methodology")
        report.append("• Enable user control over personal data")
        report.append("")
        
        return "\n".join(report)


def main():
    """Demo function for advanced features"""
    logger.info("Starting advanced features demo")
    
    # Initialize components
    predictive_model = PredictiveModeling()
    scalability_framework = ScalabilityFramework()
    advanced_viz = AdvancedVisualizations()
    export_capabilities = ExportCapabilities()
    privacy_framework = PrivacyEthicsFramework()
    
    # Demo predictive modeling
    logger.info("Demo: Predictive modeling")
    # This would use real data in production
    
    # Demo scalability framework
    logger.info("Demo: Multi-city scalability")
    
    # Register sample city configurations
    chicago_config = {
        'data_sources': {'trips': 'divvy_api', 'spending': 'opportunity_insights'},
        'geographic_bounds': {'lat_min': 41.6, 'lat_max': 42.1, 'lng_min': -88.0, 'lng_max': -87.5},
        'population': 2700000,
        'bike_share_system': 'divvy',
        'counties': ['17031']
    }
    
    nyc_config = {
        'data_sources': {'trips': 'citibike_api', 'spending': 'opportunity_insights'},
        'geographic_bounds': {'lat_min': 40.7, 'lat_max': 40.8, 'lng_min': -74.0, 'lng_max': -73.9},
        'population': 8400000,
        'bike_share_system': 'citibike',
        'counties': ['36061']
    }
    
    scalability_framework.register_city_config('chicago', chicago_config)
    scalability_framework.register_city_config('nyc', nyc_config)
    
    # Demo privacy and ethics
    logger.info("Demo: Privacy and ethics framework")
    
    # Create sample data for privacy testing
    sample_data = pd.DataFrame({
        'county_fips': ['17031'] * 100 + ['17043'] * 50,
        'total_trips': np.random.normal(1000, 200, 150),
        'median_income': np.random.normal(60000, 15000, 150)
    })
    
    privacy_checks = privacy_framework.check_data_anonymization(sample_data)
    print("Privacy Assessment:")
    for key, value in privacy_checks.items():
        print(f"  {key}: {value}")
    
    logger.info("Advanced features demo completed!")
    
    return {
        'predictive_model': predictive_model,
        'scalability_framework': scalability_framework,
        'advanced_viz': advanced_viz,
        'export_capabilities': export_capabilities,
        'privacy_framework': privacy_framework
    }


if __name__ == "__main__":
    main()