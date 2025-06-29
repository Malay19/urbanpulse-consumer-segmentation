# API Reference Documentation

## Consumer Segmentation Analytics API

This document provides comprehensive API reference for the Consumer Segmentation Analytics platform.

## Core Modules

### DataLoader

**Purpose**: Data ingestion and preprocessing from multiple sources

#### Methods

##### `download_divvy_data(year: int, month: int) -> pd.DataFrame`
Downloads and processes Divvy bike-share trip data.

**Parameters:**
- `year` (int): Year for data retrieval (2020-2024)
- `month` (int): Month for data retrieval (1-12)

**Returns:**
- `pd.DataFrame`: Trip data with columns:
  - `trip_id`: Unique trip identifier
  - `start_time`: Trip start timestamp
  - `end_time`: Trip end timestamp
  - `start_station_id`: Starting station ID
  - `end_station_id`: Ending station ID
  - `start_lat`, `start_lng`: Starting coordinates
  - `end_lat`, `end_lng`: Ending coordinates
  - `member_type`: User type (member/casual)

**Example:**
```python
loader = DataLoader()
trips = loader.download_divvy_data(2023, 6)
print(f"Loaded {len(trips)} trips")
```

##### `download_spending_data(counties: List[str] = None) -> pd.DataFrame`
Downloads consumer spending data by county and category.

**Parameters:**
- `counties` (List[str], optional): County FIPS codes to include

**Returns:**
- `pd.DataFrame`: Spending data with columns:
  - `county_fips`: County FIPS code
  - `year`: Year
  - `month`: Month
  - `category`: Spending category
  - `spending_amount`: Amount spent

##### `load_county_boundaries(counties: List[str] = None) -> gpd.GeoDataFrame`
Loads county boundary geometries.

**Parameters:**
- `counties` (List[str], optional): County FIPS codes to include

**Returns:**
- `gpd.GeoDataFrame`: Boundary data with geometry column

---

### SpatialProcessor

**Purpose**: Geographic data processing and spatial analysis

#### Methods

##### `extract_stations_from_trips(trips_df: pd.DataFrame) -> gpd.GeoDataFrame`
Extracts unique station locations from trip data.

**Parameters:**
- `trips_df` (pd.DataFrame): Trip data from DataLoader

**Returns:**
- `gpd.GeoDataFrame`: Station locations with trip counts

##### `assign_stations_to_counties(stations_gdf: gpd.GeoDataFrame, boundaries: gpd.GeoDataFrame) -> gpd.GeoDataFrame`
Assigns stations to counties using spatial joins.

**Parameters:**
- `stations_gdf` (gpd.GeoDataFrame): Station locations
- `boundaries` (gpd.GeoDataFrame): County boundaries

**Returns:**
- `gpd.GeoDataFrame`: Stations with county assignments

##### `aggregate_trips_to_county_level(trips_df: pd.DataFrame, stations_gdf: gpd.GeoDataFrame) -> pd.DataFrame`
Aggregates trip data to county level.

**Parameters:**
- `trips_df` (pd.DataFrame): Trip data
- `stations_gdf` (gpd.GeoDataFrame): Stations with county assignments

**Returns:**
- `pd.DataFrame`: County-level mobility metrics

---

### FeatureEngineer

**Purpose**: Feature creation and transformation for analysis

#### Methods

##### `create_feature_pipeline(mobility_df: pd.DataFrame, spending_df: pd.DataFrame, trips_df: pd.DataFrame = None) -> Tuple[pd.DataFrame, Dict[str, Any]]`
Runs complete feature engineering pipeline.

**Parameters:**
- `mobility_df` (pd.DataFrame): County-level mobility data
- `spending_df` (pd.DataFrame): Spending data
- `trips_df` (pd.DataFrame, optional): Raw trip data for temporal features

**Returns:**
- `Tuple[pd.DataFrame, Dict[str, Any]]`: Engineered features and pipeline metadata

##### `create_mobility_features(mobility_df: pd.DataFrame, trips_df: pd.DataFrame = None) -> pd.DataFrame`
Creates mobility-specific features.

**Parameters:**
- `mobility_df` (pd.DataFrame): Base mobility data
- `trips_df` (pd.DataFrame, optional): Raw trip data

**Returns:**
- `pd.DataFrame`: Enhanced mobility features

##### `create_spending_features(spending_df: pd.DataFrame) -> pd.DataFrame`
Creates spending-specific features.

**Parameters:**
- `spending_df` (pd.DataFrame): Raw spending data

**Returns:**
- `pd.DataFrame`: Processed spending features

---

### ClusteringEngine

**Purpose**: Advanced clustering analysis with validation

#### Methods

##### `run_complete_clustering_analysis(features_df: pd.DataFrame, algorithms: List[str] = None) -> Dict[str, Any]`
Runs comprehensive clustering analysis.

**Parameters:**
- `features_df` (pd.DataFrame): Engineered features
- `algorithms` (List[str], optional): Algorithms to use ['hdbscan', 'kmeans']

**Returns:**
- `Dict[str, Any]`: Complete clustering results including:
  - `algorithms`: Results for each algorithm
  - `validation_metrics`: Performance metrics
  - `cluster_profiles`: Detailed cluster characteristics

##### `fit_hdbscan(feature_matrix: np.ndarray, min_cluster_size: int = None, min_samples: int = None) -> hdbscan.HDBSCAN`
Fits HDBSCAN clustering algorithm.

**Parameters:**
- `feature_matrix` (np.ndarray): Feature matrix for clustering
- `min_cluster_size` (int, optional): Minimum cluster size
- `min_samples` (int, optional): Minimum samples parameter

**Returns:**
- `hdbscan.HDBSCAN`: Fitted HDBSCAN model

##### `fit_kmeans_optimal(feature_matrix: np.ndarray, k_range: Tuple[int, int] = (2, 10)) -> KMeans`
Fits K-Means with optimal k selection.

**Parameters:**
- `feature_matrix` (np.ndarray): Feature matrix for clustering
- `k_range` (Tuple[int, int]): Range of k values to test

**Returns:**
- `KMeans`: Fitted K-Means model with optimal k

---

### PersonaGenerator

**Purpose**: Consumer persona generation and business intelligence

#### Methods

##### `run_complete_persona_analysis(clustering_results: Dict[str, Any], features_df: pd.DataFrame) -> Tuple[Dict[str, ConsumerPersona], List[BusinessOpportunity], Dict[str, Any]]`
Runs complete persona generation pipeline.

**Parameters:**
- `clustering_results` (Dict[str, Any]): Results from ClusteringEngine
- `features_df` (pd.DataFrame): Engineered features

**Returns:**
- `Tuple`: Personas, opportunities, and market insights

##### `generate_persona_narratives(cluster_analysis: Dict[str, Any]) -> Dict[str, ConsumerPersona]`
Generates narrative descriptions for personas.

**Parameters:**
- `cluster_analysis` (Dict[str, Any]): Cluster characteristics analysis

**Returns:**
- `Dict[str, ConsumerPersona]`: Generated personas with narratives

##### `generate_business_opportunities(personas: Dict[str, ConsumerPersona]) -> List[BusinessOpportunity]`
Identifies business opportunities from personas.

**Parameters:**
- `personas` (Dict[str, ConsumerPersona]): Generated personas

**Returns:**
- `List[BusinessOpportunity]`: Identified business opportunities

---

### PredictiveModeling (Extensions)

**Purpose**: Advanced predictive analytics for spending patterns

#### Methods

##### `train_spending_predictor(features_df: pd.DataFrame, targets_df: pd.DataFrame, model_type: str = 'random_forest') -> Dict[str, Any]`
Trains predictive model for spending patterns.

**Parameters:**
- `features_df` (pd.DataFrame): Feature matrix
- `targets_df` (pd.DataFrame): Target spending values
- `model_type` (str): Model type ('random_forest' or 'xgboost')

**Returns:**
- `Dict[str, Any]`: Model results including performance metrics and feature importance

##### `predict_future_spending(features_df: pd.DataFrame, model_type: str = 'random_forest', periods: int = 6) -> pd.DataFrame`
Predicts future spending patterns.

**Parameters:**
- `features_df` (pd.DataFrame): Features for prediction
- `model_type` (str): Trained model to use
- `periods` (int): Number of periods to predict

**Returns:**
- `pd.DataFrame`: Spending predictions

---

## Data Structures

### ConsumerPersona

**Purpose**: Represents a consumer segment with detailed characteristics

#### Attributes

- `persona_id` (str): Unique identifier
- `persona_name` (str): Human-readable name
- `persona_type` (PersonaType): Classification type
- `cluster_ids` (List[int]): Associated cluster IDs
- `estimated_population` (int): Estimated segment size
- `median_income` (float): Median income level
- `age_distribution` (Dict[str, float]): Age group percentages
- `education_level` (Dict[str, float]): Education level distribution
- `mobility_profile` (Dict[str, Any]): Mobility characteristics
- `spending_profile` (Dict[str, Any]): Spending characteristics
- `temporal_patterns` (Dict[str, Any]): Time-based patterns
- `market_value` (float): Estimated market value
- `targeting_effectiveness` (float): Targeting effectiveness score (0-1)
- `seasonal_trends` (Dict[str, float]): Seasonal usage multipliers
- `description` (str): Narrative description
- `key_motivations` (List[str]): Primary motivations
- `preferred_channels` (List[str]): Preferred communication channels
- `pain_points` (List[str]): Key challenges and frustrations
- `marketing_strategies` (List[str]): Recommended marketing approaches
- `product_opportunities` (List[str]): Product development opportunities
- `infrastructure_needs` (List[str]): Infrastructure requirements

### BusinessOpportunity

**Purpose**: Represents a business opportunity with implementation details

#### Attributes

- `opportunity_type` (str): Type of opportunity
- `description` (str): Detailed description
- `target_segments` (List[str]): Target persona segments
- `estimated_market_size` (float): Market size estimate
- `investment_level` (str): Required investment level ('Low', 'Medium', 'High')
- `expected_roi` (str): Expected return on investment
- `implementation_timeline` (str): Implementation timeframe
- `key_metrics` (List[str]): Success metrics to track

### PersonaType (Enum)

**Purpose**: Classification types for consumer personas

#### Values

- `URBAN_COMMUTER`: Regular commuters using structured patterns
- `LEISURE_CYCLIST`: Recreation-focused users
- `TOURIST_EXPLORER`: Visitors and tourists
- `FITNESS_ENTHUSIAST`: Health and fitness motivated users
- `TECH_SAVVY`: Technology-forward early adopters
- `BUDGET_CONSCIOUS`: Price-sensitive users
- `OCCASIONAL_USER`: Infrequent, opportunistic users

---

## Configuration

### Environment Variables

#### Required
- `ENVIRONMENT`: Deployment environment (development/staging/production)
- `CENSUS_API_KEY`: US Census API key for demographic data

#### Optional
- `CACHE_ENABLED`: Enable data caching (default: true)
- `LOG_LEVEL`: Logging level (default: INFO)
- `MAX_WORKERS`: Maximum parallel workers (default: 4)
- `MEMORY_LIMIT_MB`: Memory limit in MB (default: 4096)

### Configuration Classes

#### PipelineConfig

**Purpose**: Configuration for analysis pipeline

#### Attributes

- `year` (int): Analysis year
- `month` (int): Analysis month
- `counties` (List[str]): County FIPS codes to analyze
- `sample_size` (int): Sample size for analysis
- `enable_caching` (bool): Enable result caching
- `force_refresh` (bool): Force cache refresh
- `clustering_algorithms` (List[str]): Algorithms to use
- `min_cluster_size` (int): Minimum cluster size
- `export_formats` (List[str]): Export formats

---

## Error Handling

### Common Exceptions

#### DataValidationError
Raised when data validation fails.

```python
try:
    loader.validate_data(df)
except DataValidationError as e:
    print(f"Data validation failed: {e}")
```

#### ClusteringError
Raised when clustering analysis fails.

```python
try:
    results = engine.run_clustering(features)
except ClusteringError as e:
    print(f"Clustering failed: {e}")
```

#### FeatureEngineeringError
Raised when feature engineering fails.

```python
try:
    features = engineer.create_features(data)
except FeatureEngineeringError as e:
    print(f"Feature engineering failed: {e}")
```

---

## Performance Considerations

### Memory Management

- Use chunked processing for large datasets
- Enable caching for repeated operations
- Monitor memory usage with built-in profiling

### Optimization Tips

1. **Data Loading**
   - Use parquet format for faster I/O
   - Implement data sampling for development
   - Cache processed datasets

2. **Feature Engineering**
   - Vectorize operations using pandas/numpy
   - Use sparse matrices for high-dimensional features
   - Implement incremental feature updates

3. **Clustering**
   - Use approximate algorithms for large datasets
   - Implement parallel processing
   - Cache clustering results

---

## Examples

### Complete Analysis Pipeline

```python
from src.pipeline_manager import PipelineManager, PipelineConfig

# Configure analysis
config = PipelineConfig(
    year=2023,
    month=6,
    counties=['17031', '36061'],
    clustering_algorithms=['hdbscan', 'kmeans'],
    enable_caching=True
)

# Run pipeline
pipeline = PipelineManager(config)
results = pipeline.run_complete_pipeline()

# Access results
print(f"Generated {len(results['personas'])} personas")
print(f"Identified {len(results['opportunities'])} opportunities")
```

### Custom Feature Engineering

```python
from src.feature_engineering import FeatureEngineer

engineer = FeatureEngineer()

# Create custom features
mobility_features = engineer.create_mobility_features(mobility_df)
spending_features = engineer.create_spending_features(spending_df)

# Combine and process
combined_features, pipeline_results = engineer.create_feature_pipeline(
    mobility_df, spending_df
)
```

### Advanced Clustering

```python
from src.clustering_engine import ClusteringEngine

engine = ClusteringEngine()

# Prepare data
feature_matrix, feature_cols, clustering_data = engine.prepare_clustering_data(features_df)

# Run HDBSCAN with custom parameters
clusterer = engine.fit_hdbscan(
    feature_matrix,
    min_cluster_size=50,
    min_samples=25,
    cluster_selection_epsilon=0.1
)

# Validate results
metrics = engine.calculate_internal_metrics(
    feature_matrix, 
    clusterer.labels_, 
    'hdbscan'
)
```

This API reference provides comprehensive documentation for all major components and methods in the Consumer Segmentation Analytics platform.