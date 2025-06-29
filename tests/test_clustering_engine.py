"""
Tests for clustering engine functionality
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from clustering_engine import ClusteringEngine


class TestClusteringEngine:
    
    @pytest.fixture
    def sample_features_data(self):
        """Create sample feature data for testing"""
        np.random.seed(42)
        
        # Create data with clear clusters
        n_samples = 50
        data = {
            'county_fips': [f'county_{i:03d}' for i in range(n_samples)]
        }
        
        # Create features with cluster structure
        # Cluster 1: High mobility, low spending
        cluster1_size = 15
        data.update({
            'total_trips': np.concatenate([
                np.random.normal(10000, 1000, cluster1_size),  # High trips
                np.random.normal(5000, 500, 20),               # Medium trips
                np.random.normal(2000, 300, n_samples - cluster1_size - 20)  # Low trips
            ]),
            'spending_restaurants': np.concatenate([
                np.random.normal(50000, 5000, cluster1_size),  # Low spending
                np.random.normal(100000, 10000, 20),           # Medium spending
                np.random.normal(150000, 15000, n_samples - cluster1_size - 20)  # High spending
            ]),
            'member_ratio': np.concatenate([
                np.random.normal(0.8, 0.1, cluster1_size),     # High member ratio
                np.random.normal(0.5, 0.1, 20),               # Medium member ratio
                np.random.normal(0.3, 0.1, n_samples - cluster1_size - 20)  # Low member ratio
            ])
        })
        
        # Add some noise features
        for i in range(5):
            data[f'noise_feature_{i}'] = np.random.normal(0, 1, n_samples)
        
        return pd.DataFrame(data)
    
    @pytest.fixture
    def clustering_engine(self):
        """Create a ClusteringEngine instance for testing"""
        return ClusteringEngine()
    
    def test_data_preparation(self, clustering_engine, sample_features_data):
        """Test clustering data preparation"""
        feature_matrix, feature_cols, clustering_data = clustering_engine.prepare_clustering_data(
            sample_features_data
        )
        
        # Check output types
        assert isinstance(feature_matrix, np.ndarray)
        assert isinstance(feature_cols, list)
        assert isinstance(clustering_data, pd.DataFrame)
        
        # Check dimensions
        assert feature_matrix.shape[0] == len(sample_features_data)
        assert feature_matrix.shape[1] == len(feature_cols)
        
        # Check that county_fips was excluded
        assert 'county_fips' not in feature_cols
        
        # Check that numeric features were included
        assert 'total_trips' in feature_cols
        assert 'spending_restaurants' in feature_cols
        assert 'member_ratio' in feature_cols
    
    def test_hdbscan_clustering(self, clustering_engine, sample_features_data):
        """Test HDBSCAN clustering"""
        feature_matrix, feature_cols, clustering_data = clustering_engine.prepare_clustering_data(
            sample_features_data
        )
        
        # Fit HDBSCAN with small parameters for test data
        clusterer = clustering_engine.fit_hdbscan(
            feature_matrix,
            min_cluster_size=5,
            min_samples=3,
            cluster_selection_epsilon=0.0
        )
        
        # Check that clusterer was stored
        assert 'hdbscan' in clustering_engine.clusterers
        assert 'hdbscan' in clustering_engine.cluster_results
        
        # Check cluster results
        results = clustering_engine.cluster_results['hdbscan']
        assert 'labels' in results
        assert 'n_clusters' in results
        assert 'n_outliers' in results
        
        # Check that some clusters were found
        assert results['n_clusters'] >= 1
        
        # Check label dimensions
        assert len(results['labels']) == feature_matrix.shape[0]
    
    def test_kmeans_clustering(self, clustering_engine, sample_features_data):
        """Test K-Means clustering with optimal k selection"""
        feature_matrix, feature_cols, clustering_data = clustering_engine.prepare_clustering_data(
            sample_features_data
        )
        
        # Fit K-Means
        clusterer = clustering_engine.fit_kmeans_optimal(
            feature_matrix,
            k_range=(2, 5)
        )
        
        # Check that clusterer was stored
        assert 'kmeans' in clustering_engine.clusterers
        assert 'kmeans' in clustering_engine.cluster_results
        
        # Check cluster results
        results = clustering_engine.cluster_results['kmeans']
        assert 'labels' in results
        assert 'n_clusters' in results
        assert 'optimal_k' in results
        assert 'optimal_silhouette' in results
        
        # Check that optimal k is in the specified range
        assert 2 <= results['optimal_k'] <= 5
        
        # Check that silhouette scores were calculated
        assert 'silhouette_scores' in results
        assert len(results['silhouette_scores']) == 4  # k_range 2-5
    
    def test_internal_metrics_calculation(self, clustering_engine, sample_features_data):
        """Test internal validation metrics calculation"""
        feature_matrix, feature_cols, clustering_data = clustering_engine.prepare_clustering_data(
            sample_features_data
        )
        
        # Fit clustering first
        clustering_engine.fit_hdbscan(feature_matrix, min_cluster_size=5, min_samples=3)
        cluster_labels = clustering_engine.cluster_results['hdbscan']['labels']
        
        # Calculate metrics
        metrics = clustering_engine.calculate_internal_metrics(
            feature_matrix, cluster_labels, 'hdbscan'
        )
        
        # Check that metrics were calculated
        expected_metrics = ['silhouette_score', 'calinski_harabasz_score', 'davies_bouldin_score']
        for metric in expected_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], (int, float))
        
        # Silhouette score should be between -1 and 1
        if metrics['silhouette_score'] != -1:  # -1 indicates calculation failed
            assert -1 <= metrics['silhouette_score'] <= 1
    
    def test_stability_testing(self, clustering_engine, sample_features_data):
        """Test clustering stability analysis"""
        feature_matrix, feature_cols, clustering_data = clustering_engine.prepare_clustering_data(
            sample_features_data
        )
        
        # Fit clustering first
        clustering_engine.fit_hdbscan(feature_matrix, min_cluster_size=5, min_samples=3)
        
        # Test stability with small number of bootstrap samples
        stability_results = clustering_engine.stability_testing(
            feature_matrix, 'hdbscan', n_bootstrap=3, sample_ratio=0.8
        )
        
        # Check stability results structure
        expected_keys = [
            'n_clusters_mean', 'n_clusters_std', 'n_clusters_range',
            'cluster_count_stability', 'bootstrap_results'
        ]
        for key in expected_keys:
            assert key in stability_results
        
        # Check that bootstrap results were recorded
        assert len(stability_results['bootstrap_results']) == 3
        
        # Check stability score is between 0 and 1
        assert 0 <= stability_results['cluster_count_stability'] <= 1
    
    def test_cluster_profiles_analysis(self, clustering_engine, sample_features_data):
        """Test cluster profile generation"""
        feature_matrix, feature_cols, clustering_data = clustering_engine.prepare_clustering_data(
            sample_features_data
        )
        
        # Fit clustering first
        clustering_engine.fit_kmeans_optimal(feature_matrix, k_range=(2, 4))
        cluster_labels = clustering_engine.cluster_results['kmeans']['labels']
        
        # Analyze cluster profiles
        profiles = clustering_engine.analyze_cluster_profiles(
            sample_features_data, cluster_labels, feature_cols, 'kmeans'
        )
        
        # Check profiles structure
        assert 'cluster_profiles' in profiles
        assert 'cluster_summary' in profiles
        assert 'algorithm' in profiles
        
        # Check that profiles were created for each cluster
        n_clusters = len(set(cluster_labels))
        assert len(profiles['cluster_profiles']) == n_clusters
        
        # Check profile content
        for cluster_id, profile in profiles['cluster_profiles'].items():
            assert 'cluster_id' in profile
            assert 'size' in profile
            assert 'counties' in profile
            assert 'feature_statistics' in profile
            assert 'distinguishing_features' in profile
            
            # Check that feature statistics were calculated
            assert len(profile['feature_statistics']) > 0
    
    def test_feature_importance_calculation(self, clustering_engine, sample_features_data):
        """Test feature importance analysis"""
        feature_matrix, feature_cols, clustering_data = clustering_engine.prepare_clustering_data(
            sample_features_data
        )
        
        # Fit clustering first
        clustering_engine.fit_kmeans_optimal(feature_matrix, k_range=(2, 4))
        cluster_labels = clustering_engine.cluster_results['kmeans']['labels']
        
        # Calculate feature importance
        importance_results = clustering_engine.calculate_feature_importance(
            feature_matrix, cluster_labels, feature_cols, 'kmeans'
        )
        
        # Check importance results structure
        assert 'method' in importance_results
        assert 'feature_importance' in importance_results
        assert 'top_features' in importance_results
        
        # Check that importance was calculated for all features
        assert len(importance_results['feature_importance']) == len(feature_cols)
        
        # Check that top features were identified
        assert len(importance_results['top_features']) <= 10
        
        # Check that importance values are numeric
        for feature, importance in importance_results['feature_importance'].items():
            assert isinstance(importance, (int, float))
    
    def test_outlier_analysis(self, clustering_engine, sample_features_data):
        """Test outlier analysis"""
        feature_matrix, feature_cols, clustering_data = clustering_engine.prepare_clustering_data(
            sample_features_data
        )
        
        # Fit HDBSCAN (which can detect outliers)
        clustering_engine.fit_hdbscan(feature_matrix, min_cluster_size=5, min_samples=3)
        cluster_labels = clustering_engine.cluster_results['hdbscan']['labels']
        
        # Analyze outliers
        outlier_results = clustering_engine.analyze_outliers(
            sample_features_data, cluster_labels, feature_matrix, feature_cols, 'hdbscan'
        )
        
        # Check outlier results structure
        assert 'n_outliers' in outlier_results
        assert 'outlier_percentage' in outlier_results
        
        if outlier_results['n_outliers'] > 0:
            assert 'outlier_counties' in outlier_results
            assert 'extreme_features' in outlier_results
            assert 'outlier_feature_stats' in outlier_results
            
            # Check that outlier counties were identified
            assert len(outlier_results['outlier_counties']) == outlier_results['n_outliers']
            
            # Check percentage calculation
            expected_pct = outlier_results['n_outliers'] / len(cluster_labels) * 100
            assert abs(outlier_results['outlier_percentage'] - expected_pct) < 0.01
    
    def test_hyperparameter_tuning(self, clustering_engine, sample_features_data):
        """Test HDBSCAN hyperparameter tuning"""
        feature_matrix, feature_cols, clustering_data = clustering_engine.prepare_clustering_data(
            sample_features_data
        )
        
        # Define small parameter grid for testing
        param_grid = {
            'min_cluster_size': [5, 10],
            'min_samples': [3, 5],
            'cluster_selection_epsilon': [0.0, 0.1]
        }
        
        # Run hyperparameter tuning
        tuning_results = clustering_engine.hyperparameter_tuning_hdbscan(
            feature_matrix, param_grid
        )
        
        # Check tuning results structure
        assert 'best_params' in tuning_results
        assert 'best_score' in tuning_results
        assert 'all_results' in tuning_results
        
        # Check that all parameter combinations were tested
        expected_combinations = 2 * 2 * 2  # 8 combinations
        assert len(tuning_results['all_results']) <= expected_combinations
        
        # Check that best parameters were identified
        if tuning_results['best_params']:
            assert 'min_cluster_size' in tuning_results['best_params']
            assert 'min_samples' in tuning_results['best_params']
            assert 'cluster_selection_epsilon' in tuning_results['best_params']
    
    def test_complete_clustering_analysis(self, clustering_engine, sample_features_data):
        """Test complete clustering analysis pipeline"""
        # Run complete analysis
        analysis_results = clustering_engine.run_complete_clustering_analysis(
            sample_features_data, algorithms=['kmeans']  # Use only kmeans for faster testing
        )
        
        # Check top-level structure
        assert 'data_preparation' in analysis_results
        assert 'algorithms' in analysis_results
        assert 'algorithm_comparison' in analysis_results
        
        # Check data preparation results
        data_prep = analysis_results['data_preparation']
        assert 'n_samples' in data_prep
        assert 'n_features' in data_prep
        assert 'feature_columns' in data_prep
        
        # Check algorithm results
        assert 'kmeans' in analysis_results['algorithms']
        kmeans_results = analysis_results['algorithms']['kmeans']
        
        expected_sections = [
            'hyperparameter_tuning', 'cluster_results', 'validation_metrics',
            'cluster_profiles', 'feature_importance', 'outlier_analysis'
        ]
        for section in expected_sections:
            assert section in kmeans_results
        
        # Check validation metrics structure
        validation = kmeans_results['validation_metrics']
        assert 'internal_metrics' in validation
        assert 'stability_metrics' in validation
        assert 'geographic_metrics' in validation
    
    def test_algorithm_comparison(self, clustering_engine, sample_features_data):
        """Test algorithm comparison functionality"""
        # Run analysis with multiple algorithms
        analysis_results = clustering_engine.run_complete_clustering_analysis(
            sample_features_data, algorithms=['kmeans', 'hdbscan']
        )
        
        # Check comparison results
        comparison = analysis_results['algorithm_comparison']
        assert 'summary' in comparison
        
        if isinstance(comparison['summary'], dict):
            # Check that both algorithms were compared
            for algorithm in ['kmeans', 'hdbscan']:
                if algorithm in analysis_results['algorithms'] and 'error' not in analysis_results['algorithms'][algorithm]:
                    assert algorithm in comparison['summary']
                    
                    # Check comparison metrics
                    alg_summary = comparison['summary'][algorithm]
                    assert 'n_clusters' in alg_summary
                    assert 'silhouette_score' in alg_summary


if __name__ == "__main__":
    pytest.main([__file__])