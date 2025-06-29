"""
Configuration Management System for Consumer Segmentation Analysis
Handles environment-specific settings, feature flags, and parameter management
"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union
from dataclasses import dataclass, asdict, field
from enum import Enum
import logging
from datetime import datetime


class Environment(Enum):
    """Deployment environments"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class FeatureFlag(Enum):
    """Feature flags for enabling/disabling functionality"""
    ENABLE_CACHING = "enable_caching"
    ENABLE_PARALLEL_PROCESSING = "enable_parallel_processing"
    ENABLE_ADVANCED_CLUSTERING = "enable_advanced_clustering"
    ENABLE_REAL_TIME_UPDATES = "enable_real_time_updates"
    ENABLE_EXTERNAL_DATA_SOURCES = "enable_external_data_sources"
    ENABLE_MACHINE_LEARNING_FEATURES = "enable_machine_learning_features"
    ENABLE_DASHBOARD_ANALYTICS = "enable_dashboard_analytics"
    ENABLE_DATA_EXPORT = "enable_data_export"


@dataclass
class DatabaseConfig:
    """Database configuration settings"""
    host: str = "localhost"
    port: int = 5432
    database: str = "consumer_segmentation"
    username: str = "postgres"
    password: str = ""
    ssl_mode: str = "prefer"
    connection_pool_size: int = 10
    connection_timeout: int = 30


@dataclass
class CacheConfig:
    """Cache configuration settings"""
    enabled: bool = True
    backend: str = "file"  # file, redis, memory
    ttl_seconds: int = 3600
    max_size_mb: int = 1024
    compression: bool = True
    cache_directory: str = "cache"


@dataclass
class ClusteringConfig:
    """Clustering algorithm configuration"""
    default_algorithm: str = "hdbscan"
    algorithms_enabled: list = field(default_factory=lambda: ["hdbscan", "kmeans"])
    hdbscan_min_cluster_size: int = 100
    hdbscan_min_samples: int = 50
    hdbscan_cluster_epsilon: float = 0.1
    kmeans_max_k: int = 10
    kmeans_random_state: int = 42
    enable_hyperparameter_tuning: bool = True
    validation_metrics: list = field(default_factory=lambda: ["silhouette", "calinski_harabasz"])


@dataclass
class DataProcessingConfig:
    """Data processing configuration"""
    sample_size: int = 50000
    enable_data_validation: bool = True
    validation_strictness: str = "medium"  # low, medium, high
    outlier_detection_method: str = "iqr"
    missing_value_strategy: str = "median"
    feature_selection_method: str = "variance"
    standardization_method: str = "standard"
    enable_feature_engineering: bool = True


@dataclass
class VisualizationConfig:
    """Visualization and dashboard configuration"""
    default_chart_theme: str = "plotly_white"
    color_palette: list = field(default_factory=lambda: [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd'
    ])
    map_provider: str = "openstreetmap"
    enable_interactive_maps: bool = True
    dashboard_refresh_interval: int = 300  # seconds
    export_formats: list = field(default_factory=lambda: ["html", "png", "pdf"])


@dataclass
class SecurityConfig:
    """Security configuration"""
    enable_authentication: bool = False
    session_timeout: int = 3600
    api_rate_limit: int = 100  # requests per minute
    enable_data_encryption: bool = False
    allowed_origins: list = field(default_factory=lambda: ["*"])
    enable_audit_logging: bool = True


@dataclass
class PerformanceConfig:
    """Performance optimization configuration"""
    max_workers: int = 4
    enable_parallel_processing: bool = True
    memory_limit_mb: int = 4096
    processing_timeout: int = 1800  # seconds
    enable_profiling: bool = False
    chunk_size: int = 10000


@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: str = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_path: str = "logs/application.log"
    max_file_size_mb: int = 100
    backup_count: int = 5
    enable_console_logging: bool = True
    enable_file_logging: bool = True


@dataclass
class ApplicationConfig:
    """Main application configuration"""
    environment: Environment = Environment.DEVELOPMENT
    debug: bool = True
    version: str = "1.0.0"
    
    # Sub-configurations
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    data_processing: DataProcessingConfig = field(default_factory=DataProcessingConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    # Feature flags
    feature_flags: Dict[str, bool] = field(default_factory=lambda: {
        FeatureFlag.ENABLE_CACHING.value: True,
        FeatureFlag.ENABLE_PARALLEL_PROCESSING.value: True,
        FeatureFlag.ENABLE_ADVANCED_CLUSTERING.value: True,
        FeatureFlag.ENABLE_REAL_TIME_UPDATES.value: False,
        FeatureFlag.ENABLE_EXTERNAL_DATA_SOURCES.value: True,
        FeatureFlag.ENABLE_MACHINE_LEARNING_FEATURES.value: True,
        FeatureFlag.ENABLE_DASHBOARD_ANALYTICS.value: True,
        FeatureFlag.ENABLE_DATA_EXPORT.value: True
    })


class ConfigManager:
    """Configuration manager for loading and managing application settings"""
    
    def __init__(self, config_dir: Optional[Path] = None):
        self.config_dir = config_dir or Path("config")
        self.config_dir.mkdir(exist_ok=True)
        
        self._config: Optional[ApplicationConfig] = None
        self._environment = self._detect_environment()
        
        # Create default config files if they don't exist
        self._ensure_config_files()
    
    def _detect_environment(self) -> Environment:
        """Detect current environment from environment variables"""
        env_name = os.getenv("ENVIRONMENT", "development").lower()
        
        try:
            return Environment(env_name)
        except ValueError:
            logging.warning(f"Unknown environment '{env_name}', defaulting to development")
            return Environment.DEVELOPMENT
    
    def _ensure_config_files(self):
        """Create default configuration files if they don't exist"""
        # Create base config file
        base_config_file = self.config_dir / "config.yaml"
        if not base_config_file.exists():
            default_config = ApplicationConfig()
            self._save_config_to_file(default_config, base_config_file)
        
        # Create environment-specific config files
        for env in Environment:
            env_config_file = self.config_dir / f"config.{env.value}.yaml"
            if not env_config_file.exists():
                env_config = self._get_environment_defaults(env)
                self._save_config_to_file(env_config, env_config_file)
    
    def _get_environment_defaults(self, environment: Environment) -> ApplicationConfig:
        """Get default configuration for specific environment"""
        config = ApplicationConfig(environment=environment)
        
        if environment == Environment.DEVELOPMENT:
            config.debug = True
            config.logging.level = "DEBUG"
            config.cache.enabled = True
            config.performance.enable_profiling = True
            
        elif environment == Environment.TESTING:
            config.debug = True
            config.logging.level = "DEBUG"
            config.cache.enabled = False
            config.data_processing.sample_size = 1000
            config.performance.max_workers = 1
            
        elif environment == Environment.STAGING:
            config.debug = False
            config.logging.level = "INFO"
            config.cache.enabled = True
            config.security.enable_authentication = True
            config.performance.enable_profiling = False
            
        elif environment == Environment.PRODUCTION:
            config.debug = False
            config.logging.level = "WARNING"
            config.cache.enabled = True
            config.security.enable_authentication = True
            config.security.enable_data_encryption = True
            config.security.enable_audit_logging = True
            config.performance.enable_profiling = False
            config.feature_flags[FeatureFlag.ENABLE_REAL_TIME_UPDATES.value] = True
        
        return config
    
    def _save_config_to_file(self, config: ApplicationConfig, file_path: Path):
        """Save configuration to YAML file"""
        config_dict = asdict(config)
        
        # Convert enums to strings
        config_dict['environment'] = config.environment.value
        
        with open(file_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    def load_config(self, config_file: Optional[Path] = None) -> ApplicationConfig:
        """Load configuration from file"""
        if config_file is None:
            # Load base config first
            base_config_file = self.config_dir / "config.yaml"
            env_config_file = self.config_dir / f"config.{self._environment.value}.yaml"
            
            # Load base configuration
            if base_config_file.exists():
                base_config = self._load_config_from_file(base_config_file)
            else:
                base_config = ApplicationConfig()
            
            # Override with environment-specific configuration
            if env_config_file.exists():
                env_config_dict = self._load_config_dict_from_file(env_config_file)
                config = self._merge_configs(base_config, env_config_dict)
            else:
                config = base_config
        else:
            config = self._load_config_from_file(config_file)
        
        # Override with environment variables
        config = self._apply_environment_overrides(config)
        
        self._config = config
        return config
    
    def _load_config_from_file(self, file_path: Path) -> ApplicationConfig:
        """Load configuration from YAML file"""
        config_dict = self._load_config_dict_from_file(file_path)
        return self._dict_to_config(config_dict)
    
    def _load_config_dict_from_file(self, file_path: Path) -> Dict[str, Any]:
        """Load configuration dictionary from YAML file"""
        with open(file_path, 'r') as f:
            return yaml.safe_load(f) or {}
    
    def _dict_to_config(self, config_dict: Dict[str, Any]) -> ApplicationConfig:
        """Convert dictionary to ApplicationConfig object"""
        # Handle environment enum
        if 'environment' in config_dict:
            config_dict['environment'] = Environment(config_dict['environment'])
        
        # Create nested configuration objects
        if 'database' in config_dict:
            config_dict['database'] = DatabaseConfig(**config_dict['database'])
        
        if 'cache' in config_dict:
            config_dict['cache'] = CacheConfig(**config_dict['cache'])
        
        if 'clustering' in config_dict:
            config_dict['clustering'] = ClusteringConfig(**config_dict['clustering'])
        
        if 'data_processing' in config_dict:
            config_dict['data_processing'] = DataProcessingConfig(**config_dict['data_processing'])
        
        if 'visualization' in config_dict:
            config_dict['visualization'] = VisualizationConfig(**config_dict['visualization'])
        
        if 'security' in config_dict:
            config_dict['security'] = SecurityConfig(**config_dict['security'])
        
        if 'performance' in config_dict:
            config_dict['performance'] = PerformanceConfig(**config_dict['performance'])
        
        if 'logging' in config_dict:
            config_dict['logging'] = LoggingConfig(**config_dict['logging'])
        
        return ApplicationConfig(**config_dict)
    
    def _merge_configs(self, base_config: ApplicationConfig, override_dict: Dict[str, Any]) -> ApplicationConfig:
        """Merge base configuration with override dictionary"""
        base_dict = asdict(base_config)
        
        # Deep merge dictionaries
        merged_dict = self._deep_merge(base_dict, override_dict)
        
        return self._dict_to_config(merged_dict)
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge two dictionaries"""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _apply_environment_overrides(self, config: ApplicationConfig) -> ApplicationConfig:
        """Apply environment variable overrides to configuration"""
        # Database overrides
        if os.getenv("DATABASE_HOST"):
            config.database.host = os.getenv("DATABASE_HOST")
        if os.getenv("DATABASE_PORT"):
            config.database.port = int(os.getenv("DATABASE_PORT"))
        if os.getenv("DATABASE_NAME"):
            config.database.database = os.getenv("DATABASE_NAME")
        if os.getenv("DATABASE_USERNAME"):
            config.database.username = os.getenv("DATABASE_USERNAME")
        if os.getenv("DATABASE_PASSWORD"):
            config.database.password = os.getenv("DATABASE_PASSWORD")
        
        # Cache overrides
        if os.getenv("CACHE_ENABLED"):
            config.cache.enabled = os.getenv("CACHE_ENABLED").lower() == "true"
        if os.getenv("CACHE_TTL"):
            config.cache.ttl_seconds = int(os.getenv("CACHE_TTL"))
        
        # Performance overrides
        if os.getenv("MAX_WORKERS"):
            config.performance.max_workers = int(os.getenv("MAX_WORKERS"))
        if os.getenv("MEMORY_LIMIT_MB"):
            config.performance.memory_limit_mb = int(os.getenv("MEMORY_LIMIT_MB"))
        
        # Logging overrides
        if os.getenv("LOG_LEVEL"):
            config.logging.level = os.getenv("LOG_LEVEL")
        
        # Debug override
        if os.getenv("DEBUG"):
            config.debug = os.getenv("DEBUG").lower() == "true"
        
        return config
    
    def get_config(self) -> ApplicationConfig:
        """Get current configuration"""
        if self._config is None:
            self._config = self.load_config()
        return self._config
    
    def is_feature_enabled(self, feature: FeatureFlag) -> bool:
        """Check if a feature flag is enabled"""
        config = self.get_config()
        return config.feature_flags.get(feature.value, False)
    
    def set_feature_flag(self, feature: FeatureFlag, enabled: bool):
        """Set a feature flag value"""
        config = self.get_config()
        config.feature_flags[feature.value] = enabled
    
    def get_database_url(self) -> str:
        """Get database connection URL"""
        config = self.get_config()
        db = config.database
        
        return f"postgresql://{db.username}:{db.password}@{db.host}:{db.port}/{db.database}"
    
    def export_config(self, file_path: Optional[Path] = None) -> str:
        """Export current configuration to file"""
        config = self.get_config()
        
        if file_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_path = self.config_dir / f"config_export_{timestamp}.yaml"
        
        self._save_config_to_file(config, file_path)
        return str(file_path)
    
    def validate_config(self) -> Dict[str, Any]:
        """Validate current configuration"""
        config = self.get_config()
        validation_results = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Validate database configuration
        if not config.database.host:
            validation_results['errors'].append("Database host is required")
            validation_results['valid'] = False
        
        if config.database.port <= 0 or config.database.port > 65535:
            validation_results['errors'].append("Database port must be between 1 and 65535")
            validation_results['valid'] = False
        
        # Validate performance configuration
        if config.performance.max_workers <= 0:
            validation_results['errors'].append("Max workers must be greater than 0")
            validation_results['valid'] = False
        
        if config.performance.memory_limit_mb < 512:
            validation_results['warnings'].append("Memory limit is very low (< 512 MB)")
        
        # Validate clustering configuration
        if config.clustering.hdbscan_min_cluster_size <= 0:
            validation_results['errors'].append("HDBSCAN min cluster size must be greater than 0")
            validation_results['valid'] = False
        
        # Validate cache configuration
        if config.cache.enabled and config.cache.max_size_mb <= 0:
            validation_results['errors'].append("Cache max size must be greater than 0 when caching is enabled")
            validation_results['valid'] = False
        
        return validation_results


# Global configuration manager instance
config_manager = ConfigManager()


def get_config() -> ApplicationConfig:
    """Get global configuration instance"""
    return config_manager.get_config()


def is_feature_enabled(feature: FeatureFlag) -> bool:
    """Check if a feature flag is enabled"""
    return config_manager.is_feature_enabled(feature)


def get_database_url() -> str:
    """Get database connection URL"""
    return config_manager.get_database_url()


def main():
    """Demo function for configuration management"""
    print("🔧 Configuration Management Demo")
    print("=" * 50)
    
    # Initialize configuration manager
    manager = ConfigManager()
    
    # Load configuration
    config = manager.load_config()
    
    print(f"Environment: {config.environment.value}")
    print(f"Debug mode: {config.debug}")
    print(f"Cache enabled: {config.cache.enabled}")
    print(f"Max workers: {config.performance.max_workers}")
    
    # Test feature flags
    print(f"\nFeature Flags:")
    for flag in FeatureFlag:
        enabled = manager.is_feature_enabled(flag)
        print(f"- {flag.value}: {enabled}")
    
    # Validate configuration
    validation = manager.validate_config()
    print(f"\nConfiguration Validation:")
    print(f"- Valid: {validation['valid']}")
    print(f"- Errors: {len(validation['errors'])}")
    print(f"- Warnings: {len(validation['warnings'])}")
    
    if validation['errors']:
        print("Errors:")
        for error in validation['errors']:
            print(f"  ❌ {error}")
    
    if validation['warnings']:
        print("Warnings:")
        for warning in validation['warnings']:
            print(f"  ⚠️  {warning}")
    
    # Export configuration
    export_path = manager.export_config()
    print(f"\nConfiguration exported to: {export_path}")
    
    print("\n✅ Configuration management demo completed!")


if __name__ == "__main__":
    main()