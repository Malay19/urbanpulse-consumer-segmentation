"""
Pipeline Manager for Consumer Segmentation Analysis
Orchestrates the complete end-to-end workflow with error handling, logging, and monitoring
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from loguru import logger
from datetime import datetime, timedelta
import json
import time
import traceback
from pathlib import Path
import pickle
import hashlib
from dataclasses import dataclass, asdict
from enum import Enum
import threading
import queue

from config import DATA_CONFIG, MODEL_CONFIG, PROJECT_ROOT
from src.data_loader import DataLoader
from src.data_validator import DataValidator
from src.spatial_processor import SpatialProcessor
from src.feature_engineering import FeatureEngineer
from src.clustering_engine import ClusteringEngine
from src.persona_generator import PersonaGenerator
from src.dashboard_generator import DashboardGenerator


class PipelineStage(Enum):
    """Pipeline execution stages"""
    DATA_LOADING = "data_loading"
    DATA_VALIDATION = "data_validation"
    SPATIAL_PROCESSING = "spatial_processing"
    FEATURE_ENGINEERING = "feature_engineering"
    CLUSTERING_ANALYSIS = "clustering_analysis"
    PERSONA_GENERATION = "persona_generation"
    DASHBOARD_CREATION = "dashboard_creation"
    EXPORT_RESULTS = "export_results"


@dataclass
class PipelineConfig:
    """Pipeline configuration settings"""
    # Data parameters
    year: int = 2023
    month: int = 6
    counties: List[str] = None
    sample_size: int = 50000
    
    # Processing parameters
    enable_caching: bool = True
    force_refresh: bool = False
    parallel_processing: bool = True
    max_workers: int = 4
    
    # Clustering parameters
    clustering_algorithms: List[str] = None
    min_cluster_size: int = None
    cluster_epsilon: float = None
    
    # Output parameters
    export_formats: List[str] = None
    include_visualizations: bool = True
    generate_reports: bool = True
    
    def __post_init__(self):
        if self.counties is None:
            self.counties = DATA_CONFIG.SAMPLE_COUNTIES
        if self.clustering_algorithms is None:
            self.clustering_algorithms = ['hdbscan', 'kmeans']
        if self.min_cluster_size is None:
            self.min_cluster_size = MODEL_CONFIG.MIN_CLUSTER_SIZE
        if self.cluster_epsilon is None:
            self.cluster_epsilon = MODEL_CONFIG.CLUSTER_SELECTION_EPSILON
        if self.export_formats is None:
            self.export_formats = ['json', 'csv', 'html']


@dataclass
class StageResult:
    """Result of a pipeline stage execution"""
    stage: PipelineStage
    status: str  # 'success', 'failed', 'skipped'
    start_time: datetime
    end_time: Optional[datetime] = None
    duration: Optional[float] = None
    data: Any = None
    metadata: Dict[str, Any] = None
    error: Optional[str] = None
    
    def __post_init__(self):
        if self.end_time and self.start_time:
            self.duration = (self.end_time - self.start_time).total_seconds()


class PipelineMonitor:
    """Monitor pipeline execution with real-time updates"""
    
    def __init__(self):
        self.stage_results: List[StageResult] = []
        self.current_stage: Optional[PipelineStage] = None
        self.start_time: Optional[datetime] = None
        self.progress_queue = queue.Queue()
        self.callbacks: List[callable] = []
    
    def add_callback(self, callback: callable):
        """Add progress callback function"""
        self.callbacks.append(callback)
    
    def start_pipeline(self):
        """Mark pipeline start"""
        self.start_time = datetime.now()
        self.stage_results = []
        logger.info("Pipeline execution started")
    
    def start_stage(self, stage: PipelineStage):
        """Mark stage start"""
        self.current_stage = stage
        logger.info(f"Starting stage: {stage.value}")
        
        # Notify callbacks
        for callback in self.callbacks:
            try:
                callback('stage_start', stage)
            except Exception as e:
                logger.warning(f"Callback error: {e}")
    
    def complete_stage(self, stage: PipelineStage, status: str, 
                      data: Any = None, metadata: Dict[str, Any] = None, 
                      error: Optional[str] = None):
        """Mark stage completion"""
        # Find the corresponding start result
        start_result = None
        for result in reversed(self.stage_results):
            if result.stage == stage and result.end_time is None:
                start_result = result
                break
        
        if start_result:
            start_result.end_time = datetime.now()
            start_result.status = status
            start_result.data = data
            start_result.metadata = metadata or {}
            start_result.error = error
            start_result.duration = (start_result.end_time - start_result.start_time).total_seconds()
        else:
            # Create new result if start wasn't recorded
            result = StageResult(
                stage=stage,
                status=status,
                start_time=datetime.now(),
                end_time=datetime.now(),
                data=data,
                metadata=metadata or {},
                error=error
            )
            self.stage_results.append(result)
        
        logger.info(f"Completed stage: {stage.value} ({status})")
        
        # Notify callbacks
        for callback in self.callbacks:
            try:
                callback('stage_complete', stage, status)
            except Exception as e:
                logger.warning(f"Callback error: {e}")
    
    def get_progress_summary(self) -> Dict[str, Any]:
        """Get current progress summary"""
        total_stages = len(PipelineStage)
        completed_stages = len([r for r in self.stage_results if r.status in ['success', 'failed']])
        
        return {
            'total_stages': total_stages,
            'completed_stages': completed_stages,
            'current_stage': self.current_stage.value if self.current_stage else None,
            'progress_percentage': (completed_stages / total_stages) * 100,
            'elapsed_time': (datetime.now() - self.start_time).total_seconds() if self.start_time else 0,
            'stage_results': [asdict(r) for r in self.stage_results]
        }


class CacheManager:
    """Manage pipeline data caching for performance optimization"""
    
    def __init__(self, cache_dir: Path = None):
        self.cache_dir = cache_dir or (PROJECT_ROOT / "cache")
        self.cache_dir.mkdir(exist_ok=True)
        self.cache_index = self._load_cache_index()
    
    def _load_cache_index(self) -> Dict[str, Any]:
        """Load cache index from disk"""
        index_file = self.cache_dir / "cache_index.json"
        if index_file.exists():
            try:
                with open(index_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Error loading cache index: {e}")
        return {}
    
    def _save_cache_index(self):
        """Save cache index to disk"""
        index_file = self.cache_dir / "cache_index.json"
        try:
            with open(index_file, 'w') as f:
                json.dump(self.cache_index, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving cache index: {e}")
    
    def _generate_cache_key(self, stage: PipelineStage, config: PipelineConfig) -> str:
        """Generate cache key for stage and configuration"""
        config_dict = asdict(config)
        config_str = json.dumps(config_dict, sort_keys=True)
        cache_key = hashlib.md5(f"{stage.value}_{config_str}".encode()).hexdigest()
        return cache_key
    
    def get_cached_result(self, stage: PipelineStage, config: PipelineConfig) -> Optional[Any]:
        """Get cached result for stage"""
        if not config.enable_caching:
            return None
        
        cache_key = self._generate_cache_key(stage, config)
        
        if cache_key in self.cache_index:
            cache_info = self.cache_index[cache_key]
            cache_file = self.cache_dir / cache_info['filename']
            
            if cache_file.exists():
                try:
                    # Check if cache is still valid (24 hours)
                    cache_time = datetime.fromisoformat(cache_info['timestamp'])
                    if datetime.now() - cache_time < timedelta(hours=24) and not config.force_refresh:
                        with open(cache_file, 'rb') as f:
                            data = pickle.load(f)
                        logger.info(f"Using cached result for {stage.value}")
                        return data
                except Exception as e:
                    logger.warning(f"Error loading cached result: {e}")
        
        return None
    
    def cache_result(self, stage: PipelineStage, config: PipelineConfig, data: Any):
        """Cache result for stage"""
        if not config.enable_caching:
            return
        
        cache_key = self._generate_cache_key(stage, config)
        filename = f"{stage.value}_{cache_key}.pkl"
        cache_file = self.cache_dir / filename
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
            
            self.cache_index[cache_key] = {
                'stage': stage.value,
                'filename': filename,
                'timestamp': datetime.now().isoformat(),
                'size_bytes': cache_file.stat().st_size
            }
            
            self._save_cache_index()
            logger.info(f"Cached result for {stage.value}")
            
        except Exception as e:
            logger.error(f"Error caching result: {e}")
    
    def clear_cache(self, stage: Optional[PipelineStage] = None):
        """Clear cache for specific stage or all stages"""
        if stage:
            # Clear specific stage
            keys_to_remove = [k for k, v in self.cache_index.items() if v['stage'] == stage.value]
            for key in keys_to_remove:
                cache_info = self.cache_index[key]
                cache_file = self.cache_dir / cache_info['filename']
                if cache_file.exists():
                    cache_file.unlink()
                del self.cache_index[key]
        else:
            # Clear all cache
            for cache_info in self.cache_index.values():
                cache_file = self.cache_dir / cache_info['filename']
                if cache_file.exists():
                    cache_file.unlink()
            self.cache_index = {}
        
        self._save_cache_index()
        logger.info(f"Cleared cache for {stage.value if stage else 'all stages'}")


class PipelineManager:
    """Main pipeline manager for orchestrating the complete workflow"""
    
    def __init__(self, config: PipelineConfig = None):
        self.config = config or PipelineConfig()
        self.monitor = PipelineMonitor()
        self.cache_manager = CacheManager()
        
        # Initialize components
        self.data_loader = DataLoader()
        self.data_validator = DataValidator()
        self.spatial_processor = SpatialProcessor()
        self.feature_engineer = FeatureEngineer()
        self.clustering_engine = ClusteringEngine()
        self.persona_generator = PersonaGenerator()
        self.dashboard_generator = DashboardGenerator()
        
        # Pipeline state
        self.pipeline_data = {}
        self.validation_results = []
        
    def add_progress_callback(self, callback: callable):
        """Add progress monitoring callback"""
        self.monitor.add_callback(callback)
    
    def run_complete_pipeline(self) -> Dict[str, Any]:
        """Execute the complete pipeline with error handling and monitoring"""
        logger.info("Starting complete consumer segmentation pipeline")
        self.monitor.start_pipeline()
        
        try:
            # Execute each stage
            self._execute_stage(PipelineStage.DATA_LOADING, self._stage_data_loading)
            self._execute_stage(PipelineStage.DATA_VALIDATION, self._stage_data_validation)
            self._execute_stage(PipelineStage.SPATIAL_PROCESSING, self._stage_spatial_processing)
            self._execute_stage(PipelineStage.FEATURE_ENGINEERING, self._stage_feature_engineering)
            self._execute_stage(PipelineStage.CLUSTERING_ANALYSIS, self._stage_clustering_analysis)
            self._execute_stage(PipelineStage.PERSONA_GENERATION, self._stage_persona_generation)
            self._execute_stage(PipelineStage.DASHBOARD_CREATION, self._stage_dashboard_creation)
            self._execute_stage(PipelineStage.EXPORT_RESULTS, self._stage_export_results)
            
            # Generate final summary
            pipeline_summary = self._generate_pipeline_summary()
            
            logger.info("Pipeline execution completed successfully")
            return pipeline_summary
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {str(e)}")
            logger.error(traceback.format_exc())
            
            # Mark current stage as failed
            if self.monitor.current_stage:
                self.monitor.complete_stage(
                    self.monitor.current_stage, 
                    'failed', 
                    error=str(e)
                )
            
            raise
    
    def _execute_stage(self, stage: PipelineStage, stage_function: callable):
        """Execute a single pipeline stage with error handling and caching"""
        self.monitor.start_stage(stage)
        
        try:
            # Check cache first
            cached_result = self.cache_manager.get_cached_result(stage, self.config)
            if cached_result is not None:
                self.pipeline_data[stage.value] = cached_result
                self.monitor.complete_stage(stage, 'success', cached_result, {'cached': True})
                return cached_result
            
            # Execute stage
            start_time = time.time()
            result = stage_function()
            execution_time = time.time() - start_time
            
            # Cache result
            self.cache_manager.cache_result(stage, self.config, result)
            
            # Store result
            self.pipeline_data[stage.value] = result
            
            # Complete stage
            metadata = {
                'execution_time': execution_time,
                'cached': False,
                'data_size': len(str(result)) if result else 0
            }
            
            self.monitor.complete_stage(stage, 'success', result, metadata)
            return result
            
        except Exception as e:
            error_msg = f"Stage {stage.value} failed: {str(e)}"
            logger.error(error_msg)
            logger.error(traceback.format_exc())
            
            self.monitor.complete_stage(stage, 'failed', error=error_msg)
            raise
    
    def _stage_data_loading(self) -> Dict[str, Any]:
        """Data loading stage"""
        logger.info("Executing data loading stage")
        
        # Load trip data
        trips_df = self.data_loader.download_divvy_data(self.config.year, self.config.month)
        
        # Load spending data
        spending_df = self.data_loader.download_spending_data(self.config.counties)
        
        # Load boundaries
        boundaries = self.data_loader.load_county_boundaries(self.config.counties)
        
        return {
            'trips_df': trips_df,
            'spending_df': spending_df,
            'boundaries': boundaries,
            'metadata': {
                'trips_count': len(trips_df),
                'spending_records': len(spending_df),
                'counties_count': len(boundaries)
            }
        }
    
    def _stage_data_validation(self) -> Dict[str, Any]:
        """Data validation stage"""
        logger.info("Executing data validation stage")
        
        data_loading_result = self.pipeline_data['data_loading']
        
        # Validate each dataset
        trips_validation = self.data_validator.validate_divvy_data(data_loading_result['trips_df'])
        spending_validation = self.data_validator.validate_spending_data(data_loading_result['spending_df'])
        boundaries_validation = self.data_validator.validate_boundary_data(data_loading_result['boundaries'])
        
        validation_results = [trips_validation, spending_validation, boundaries_validation]
        
        # Generate validation report
        validation_report = self.data_validator.generate_data_quality_report(validation_results)
        
        # Check for critical failures
        critical_failures = [r for r in validation_results if not r['passed']]
        if critical_failures:
            raise ValueError(f"Data validation failed for {len(critical_failures)} datasets")
        
        self.validation_results = validation_results
        
        return {
            'validation_results': validation_results,
            'validation_report': validation_report,
            'all_passed': len(critical_failures) == 0,
            'warnings_count': sum(len(r['warnings']) for r in validation_results),
            'issues_count': sum(len(r['issues']) for r in validation_results)
        }
    
    def _stage_spatial_processing(self) -> Dict[str, Any]:
        """Spatial processing stage"""
        logger.info("Executing spatial processing stage")
        
        data_loading_result = self.pipeline_data['data_loading']
        
        # Extract stations
        stations_gdf = self.spatial_processor.extract_stations_from_trips(data_loading_result['trips_df'])
        
        # Assign stations to counties
        stations_with_counties = self.spatial_processor.assign_stations_to_counties(
            stations_gdf, data_loading_result['boundaries']
        )
        
        # Aggregate to county level
        county_mobility = self.spatial_processor.aggregate_trips_to_county_level(
            data_loading_result['trips_df'], stations_with_counties
        )
        
        # Join with spending data
        combined_data = self.spatial_processor.join_mobility_spending_data(
            county_mobility, data_loading_result['spending_df']
        )
        
        # Validate spatial data
        spatial_validation = self.spatial_processor.validate_spatial_data(
            stations_with_counties, data_loading_result['boundaries']
        )
        
        return {
            'stations_gdf': stations_gdf,
            'stations_with_counties': stations_with_counties,
            'county_mobility': county_mobility,
            'combined_data': combined_data,
            'spatial_validation': spatial_validation,
            'metadata': {
                'stations_count': len(stations_gdf),
                'counties_with_data': len(county_mobility),
                'combined_features': combined_data.shape[1] if combined_data is not None else 0
            }
        }
    
    def _stage_feature_engineering(self) -> Dict[str, Any]:
        """Feature engineering stage"""
        logger.info("Executing feature engineering stage")
        
        spatial_result = self.pipeline_data['spatial_processing']
        data_loading_result = self.pipeline_data['data_loading']
        
        # Run feature engineering pipeline
        engineered_features, pipeline_results = self.feature_engineer.create_feature_pipeline(
            spatial_result['county_mobility'],
            data_loading_result['spending_df'],
            data_loading_result['trips_df']
        )
        
        return {
            'engineered_features': engineered_features,
            'pipeline_results': pipeline_results,
            'metadata': {
                'final_features_count': engineered_features.shape[1],
                'final_samples_count': engineered_features.shape[0],
                'processing_steps': len(pipeline_results['processing_steps']),
                'removed_features': len(pipeline_results.get('removed_features', []))
            }
        }
    
    def _stage_clustering_analysis(self) -> Dict[str, Any]:
        """Clustering analysis stage"""
        logger.info("Executing clustering analysis stage")
        
        feature_result = self.pipeline_data['feature_engineering']
        
        # Run clustering analysis
        clustering_results = self.clustering_engine.run_complete_clustering_analysis(
            feature_result['engineered_features'],
            algorithms=self.config.clustering_algorithms
        )
        
        return {
            'clustering_results': clustering_results,
            'metadata': {
                'algorithms_tested': len(self.config.clustering_algorithms),
                'successful_algorithms': len([
                    alg for alg, results in clustering_results['algorithms'].items()
                    if 'error' not in results
                ])
            }
        }
    
    def _stage_persona_generation(self) -> Dict[str, Any]:
        """Persona generation stage"""
        logger.info("Executing persona generation stage")
        
        clustering_result = self.pipeline_data['clustering_analysis']
        feature_result = self.pipeline_data['feature_engineering']
        
        # Run persona analysis
        personas, opportunities, insights = self.persona_generator.run_complete_persona_analysis(
            clustering_result['clustering_results'],
            feature_result['engineered_features']
        )
        
        return {
            'personas': personas,
            'opportunities': opportunities,
            'insights': insights,
            'metadata': {
                'personas_generated': len(personas),
                'opportunities_identified': len(opportunities),
                'total_market_value': sum(p.market_value for p in personas.values()) if personas else 0
            }
        }
    
    def _stage_dashboard_creation(self) -> Dict[str, Any]:
        """Dashboard creation stage"""
        logger.info("Executing dashboard creation stage")
        
        if not self.config.include_visualizations:
            logger.info("Skipping dashboard creation (disabled in config)")
            return {'skipped': True}
        
        persona_result = self.pipeline_data['persona_generation']
        
        # Create dashboards
        dashboard_files = self.dashboard_generator.export_dashboard_bundle(
            persona_result['personas'],
            persona_result['opportunities'],
            persona_result['insights']
        )
        
        return {
            'dashboard_files': dashboard_files,
            'metadata': {
                'files_generated': len(dashboard_files),
                'dashboard_types': list(dashboard_files.keys())
            }
        }
    
    def _stage_export_results(self) -> Dict[str, Any]:
        """Export results stage"""
        logger.info("Executing export results stage")
        
        # Export all results
        exported_files = {}
        
        # Export personas
        if 'persona_generation' in self.pipeline_data:
            persona_files = self.persona_generator.export_business_intelligence()
            exported_files.update(persona_files)
        
        # Export clustering results
        if 'clustering_analysis' in self.pipeline_data:
            clustering_files = self.clustering_engine.export_clustering_results(
                self.pipeline_data['clustering_analysis']['clustering_results'],
                self.pipeline_data['feature_engineering']['engineered_features']
            )
            exported_files.update(clustering_files)
        
        # Export pipeline summary
        pipeline_summary = self._generate_pipeline_summary()
        summary_file = self._export_pipeline_summary(pipeline_summary)
        exported_files['pipeline_summary'] = summary_file
        
        return {
            'exported_files': exported_files,
            'metadata': {
                'total_files': len(exported_files),
                'export_formats': self.config.export_formats
            }
        }
    
    def _generate_pipeline_summary(self) -> Dict[str, Any]:
        """Generate comprehensive pipeline execution summary"""
        progress = self.monitor.get_progress_summary()
        
        summary = {
            'pipeline_execution': {
                'start_time': self.monitor.start_time.isoformat() if self.monitor.start_time else None,
                'end_time': datetime.now().isoformat(),
                'total_duration': progress['elapsed_time'],
                'stages_completed': progress['completed_stages'],
                'total_stages': progress['total_stages'],
                'success_rate': progress['completed_stages'] / progress['total_stages'] * 100
            },
            'configuration': asdict(self.config),
            'stage_results': progress['stage_results'],
            'data_summary': {},
            'validation_summary': {},
            'performance_metrics': {}
        }
        
        # Add data summaries
        for stage_name, stage_data in self.pipeline_data.items():
            if isinstance(stage_data, dict) and 'metadata' in stage_data:
                summary['data_summary'][stage_name] = stage_data['metadata']
        
        # Add validation summary
        if self.validation_results:
            summary['validation_summary'] = {
                'total_datasets': len(self.validation_results),
                'passed_validation': len([r for r in self.validation_results if r['passed']]),
                'total_warnings': sum(len(r['warnings']) for r in self.validation_results),
                'total_issues': sum(len(r['issues']) for r in self.validation_results)
            }
        
        # Add performance metrics
        stage_durations = {
            r['stage']: r['duration'] for r in progress['stage_results'] 
            if r['duration'] is not None
        }
        
        summary['performance_metrics'] = {
            'stage_durations': stage_durations,
            'total_execution_time': sum(stage_durations.values()),
            'average_stage_time': np.mean(list(stage_durations.values())) if stage_durations else 0,
            'slowest_stage': max(stage_durations.items(), key=lambda x: x[1]) if stage_durations else None
        }
        
        return summary
    
    def _export_pipeline_summary(self, summary: Dict[str, Any]) -> str:
        """Export pipeline summary to file"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"pipeline_summary_{timestamp}.json"
        filepath = DATA_CONFIG.PROCESSED_DATA_DIR / filename
        
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"Pipeline summary exported to {filepath}")
        return str(filepath)
    
    def get_progress(self) -> Dict[str, Any]:
        """Get current pipeline progress"""
        return self.monitor.get_progress_summary()
    
    def clear_cache(self, stage: Optional[PipelineStage] = None):
        """Clear pipeline cache"""
        self.cache_manager.clear_cache(stage)
    
    def run_stage_only(self, stage: PipelineStage) -> Any:
        """Run only a specific pipeline stage"""
        logger.info(f"Running single stage: {stage.value}")
        
        stage_functions = {
            PipelineStage.DATA_LOADING: self._stage_data_loading,
            PipelineStage.DATA_VALIDATION: self._stage_data_validation,
            PipelineStage.SPATIAL_PROCESSING: self._stage_spatial_processing,
            PipelineStage.FEATURE_ENGINEERING: self._stage_feature_engineering,
            PipelineStage.CLUSTERING_ANALYSIS: self._stage_clustering_analysis,
            PipelineStage.PERSONA_GENERATION: self._stage_persona_generation,
            PipelineStage.DASHBOARD_CREATION: self._stage_dashboard_creation,
            PipelineStage.EXPORT_RESULTS: self._stage_export_results
        }
        
        if stage not in stage_functions:
            raise ValueError(f"Unknown stage: {stage}")
        
        return self._execute_stage(stage, stage_functions[stage])


def create_pipeline_config(**kwargs) -> PipelineConfig:
    """Create pipeline configuration with custom parameters"""
    return PipelineConfig(**kwargs)


def run_pipeline_with_monitoring(config: PipelineConfig = None, 
                                progress_callback: callable = None) -> Dict[str, Any]:
    """Run pipeline with optional progress monitoring"""
    pipeline = PipelineManager(config)
    
    if progress_callback:
        pipeline.add_progress_callback(progress_callback)
    
    return pipeline.run_complete_pipeline()


def main():
    """Demo function for pipeline manager"""
    logger.info("Starting pipeline manager demo")
    
    # Create configuration
    config = PipelineConfig(
        year=2023,
        month=6,
        counties=['17031', '36061'],  # Cook County, NY County
        enable_caching=True,
        force_refresh=False,
        clustering_algorithms=['hdbscan', 'kmeans'],
        include_visualizations=True,
        generate_reports=True
    )
    
    # Progress callback
    def progress_callback(event_type: str, stage: PipelineStage, status: str = None):
        if event_type == 'stage_start':
            print(f"🚀 Starting: {stage.value}")
        elif event_type == 'stage_complete':
            print(f"✅ Completed: {stage.value} ({status})")
    
    try:
        # Run pipeline
        pipeline_summary = run_pipeline_with_monitoring(config, progress_callback)
        
        # Print summary
        print(f"\n📊 Pipeline Summary:")
        print(f"- Total duration: {pipeline_summary['pipeline_execution']['total_duration']:.2f} seconds")
        print(f"- Stages completed: {pipeline_summary['pipeline_execution']['stages_completed']}")
        print(f"- Success rate: {pipeline_summary['pipeline_execution']['success_rate']:.1f}%")
        
        if 'data_summary' in pipeline_summary:
            print(f"\n📈 Data Summary:")
            for stage, metadata in pipeline_summary['data_summary'].items():
                print(f"- {stage}: {metadata}")
        
        logger.info("Pipeline demo completed successfully!")
        return pipeline_summary
        
    except Exception as e:
        logger.error(f"Pipeline demo failed: {e}")
        raise


if __name__ == "__main__":
    main()