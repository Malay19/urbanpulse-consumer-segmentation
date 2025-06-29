"""
Clustering engine for Multimodal Consumer Segmentation Project
Implements HDBSCAN and K-Means clustering with comprehensive validation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from loguru import logger
import warnings
from datetime import datetime
import json

# Clustering algorithms
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import hdbscan

# Validation metrics
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.model_selection import cross_val_score
from scipy.spatial.distance import pdist, squareform
from scipy.stats import mode

# Spatial analysis
try:
    from pysal.lib import weights
    from pysal.explore import esda
    SPATIAL_AVAILABLE = True
except ImportError:
    SPATIAL_AVAILABLE = False
    logger.warning("PySAL not available, spatial validation will be limited")

# Feature importance
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    logger.warning("SHAP not available, feature importance analysis will be limited")

from config import MODEL_CONFIG
from src.utils import calculate_data_summary, save_processed_data


class ClusteringEngine:
    """Main clustering engine with multiple algorithms and validation"""
    
    def __init__(self):
        self.clusterers = {}
        self.cluster_results = {}
        self.validation_metrics = {}
        self.feature_importance = {}
        self.outlier_analysis = {}
        
    def prepare_clustering_data(self, features_df: pd.DataFrame,
                              exclude_cols: List[str] = None) -> Tuple[np.ndarray, List[str], pd.DataFrame]:
        """
        Prepare data for clustering by selecting and preprocessing features
        """
        if exclude_cols is None:
            exclude_cols = ['county_fips']
        
        logger.info("Preparing data for clustering")
        
        # Select numeric features for clustering
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns
        feature_cols = [col for col in numeric_cols if col not in exclude_cols]
        
        # Handle missing values
        clustering_data = features_df[feature_cols].copy()
        
        # Fill missing values with median
        for col in clustering_data.columns:
            if clustering_data[col].isnull().any():
                median_val = clustering_data[col].median()
                clustering_data[col].fillna(median_val, inplace=True)
                logger.warning(f"Filled {clustering_data[col].isnull().sum()} missing values in {col}")
        
        # Remove features with zero variance
        zero_var_cols = clustering_data.columns[clustering_data.var() == 0]
        if len(zero_var_cols) > 0:
            clustering_data = clustering_data.drop(columns=zero_var_cols)
            feature_cols = [col for col in feature_cols if col not in zero_var_cols]
            logger.warning(f"Removed {len(zero_var_cols)} zero-variance features")
        
        # Convert to numpy array
        feature_matrix = clustering_data.values
        
        logger.info(f"Prepared clustering data: {feature_matrix.shape[0]} samples, {feature_matrix.shape[1]} features")
        return feature_matrix, feature_cols, clustering_data
    
    def fit_hdbscan(self, feature_matrix: np.ndarray,
                   min_cluster_size: int = None,
                   min_samples: int = None,
                   cluster_selection_epsilon: float = None) -> hdbscan.HDBSCAN:
        """
        Fit HDBSCAN clustering algorithm
        """
        if min_cluster_size is None:
            min_cluster_size = MODEL_CONFIG.MIN_CLUSTER_SIZE
        if min_samples is None:
            min_samples = MODEL_CONFIG.MIN_SAMPLES
        if cluster_selection_epsilon is None:
            cluster_selection_epsilon = MODEL_CONFIG.CLUSTER_SELECTION_EPSILON
        
        logger.info(f"Fitting HDBSCAN with min_cluster_size={min_cluster_size}, "
                   f"min_samples={min_samples}, cluster_selection_epsilon={cluster_selection_epsilon}")
        
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_epsilon=cluster_selection_epsilon,
            metric='euclidean',
            cluster_selection_method='eom'
        )
        
        cluster_labels = clusterer.fit_predict(feature_matrix)
        
        # Store results
        self.clusterers['hdbscan'] = clusterer
        self.cluster_results['hdbscan'] = {
            'labels': cluster_labels,
            'n_clusters': len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0),
            'n_outliers': np.sum(cluster_labels == -1),
            'probabilities': clusterer.probabilities_,
            'outlier_scores': clusterer.outlier_scores_
        }
        
        logger.info(f"HDBSCAN found {self.cluster_results['hdbscan']['n_clusters']} clusters "
                   f"with {self.cluster_results['hdbscan']['n_outliers']} outliers")
        
        return clusterer
    
    def fit_kmeans_optimal(self, feature_matrix: np.ndarray,
                          k_range: Tuple[int, int] = (2, 10),
                          random_state: int = 42) -> KMeans:
        """
        Fit K-Means with optimal k selection using silhouette score
        """
        logger.info(f"Finding optimal K-Means clusters in range {k_range}")
        
        k_min, k_max = k_range
        silhouette_scores = []
        inertias = []
        
        best_k = k_min
        best_score = -1
        
        for k in range(k_min, k_max + 1):
            kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
            cluster_labels = kmeans.fit_predict(feature_matrix)
            
            # Calculate silhouette score
            if len(set(cluster_labels)) > 1:  # Need at least 2 clusters
                score = silhouette_score(feature_matrix, cluster_labels)
                silhouette_scores.append(score)
                inertias.append(kmeans.inertia_)
                
                if score > best_score:
                    best_score = score
                    best_k = k
            else:
                silhouette_scores.append(-1)
                inertias.append(kmeans.inertia_)
        
        # Fit final model with optimal k
        logger.info(f"Optimal K-Means: k={best_k} with silhouette score={best_score:.3f}")
        
        final_kmeans = KMeans(n_clusters=best_k, random_state=random_state, n_init=10)
        cluster_labels = final_kmeans.fit_predict(feature_matrix)
        
        # Store results
        self.clusterers['kmeans'] = final_kmeans
        self.cluster_results['kmeans'] = {
            'labels': cluster_labels,
            'n_clusters': best_k,
            'n_outliers': 0,  # K-means doesn't identify outliers
            'silhouette_scores': silhouette_scores,
            'inertias': inertias,
            'k_range': list(range(k_min, k_max + 1)),
            'optimal_k': best_k,
            'optimal_silhouette': best_score
        }
        
        return final_kmeans
    
    def hyperparameter_tuning_hdbscan(self, feature_matrix: np.ndarray,
                                    param_grid: Dict[str, List] = None) -> Dict[str, Any]:
        """
        Hyperparameter tuning for HDBSCAN using grid search
        """
        if param_grid is None:
            param_grid = {
                'min_cluster_size': [50, 100, 150],
                'min_samples': [25, 50, 75],
                'cluster_selection_epsilon': [0.0, 0.1, 0.2]
            }
        
        logger.info("Performing HDBSCAN hyperparameter tuning")
        
        best_params = None
        best_score = -1
        best_n_clusters = 0
        tuning_results = []
        
        for min_cluster_size in param_grid['min_cluster_size']:
            for min_samples in param_grid['min_samples']:
                for epsilon in param_grid['cluster_selection_epsilon']:
                    try:
                        clusterer = hdbscan.HDBSCAN(
                            min_cluster_size=min_cluster_size,
                            min_samples=min_samples,
                            cluster_selection_epsilon=epsilon,
                            metric='euclidean'
                        )
                        
                        labels = clusterer.fit_predict(feature_matrix)
                        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                        n_outliers = np.sum(labels == -1)
                        
                        # Calculate validation score (silhouette if enough clusters)
                        if n_clusters > 1 and n_clusters < len(feature_matrix) - 1:
                            # Only use non-outlier points for silhouette score
                            non_outlier_mask = labels != -1
                            if np.sum(non_outlier_mask) > 1:
                                score = silhouette_score(
                                    feature_matrix[non_outlier_mask], 
                                    labels[non_outlier_mask]
                                )
                            else:
                                score = -1
                        else:
                            score = -1
                        
                        result = {
                            'min_cluster_size': min_cluster_size,
                            'min_samples': min_samples,
                            'cluster_selection_epsilon': epsilon,
                            'n_clusters': n_clusters,
                            'n_outliers': n_outliers,
                            'silhouette_score': score
                        }
                        tuning_results.append(result)
                        
                        # Update best parameters
                        if score > best_score and n_clusters >= 2:
                            best_score = score
                            best_params = {
                                'min_cluster_size': min_cluster_size,
                                'min_samples': min_samples,
                                'cluster_selection_epsilon': epsilon
                            }
                            best_n_clusters = n_clusters
                        
                    except Exception as e:
                        logger.warning(f"HDBSCAN failed with params {min_cluster_size}, {min_samples}, {epsilon}: {e}")
        
        tuning_summary = {
            'best_params': best_params,
            'best_score': best_score,
            'best_n_clusters': best_n_clusters,
            'all_results': tuning_results
        }
        
        logger.info(f"Best HDBSCAN params: {best_params} with score {best_score:.3f}")
        return tuning_summary
    
    def calculate_internal_metrics(self, feature_matrix: np.ndarray,
                                 cluster_labels: np.ndarray,
                                 algorithm_name: str) -> Dict[str, float]:
        """
        Calculate internal clustering validation metrics
        """
        metrics = {}
        
        # Remove outliers for metric calculation (if any)
        non_outlier_mask = cluster_labels != -1
        if np.sum(non_outlier_mask) < 2:
            logger.warning(f"Too few non-outlier points for {algorithm_name} metrics")
            return {'silhouette_score': -1, 'calinski_harabasz_score': -1, 'davies_bouldin_score': -1}
        
        clean_features = feature_matrix[non_outlier_mask]
        clean_labels = cluster_labels[non_outlier_mask]
        
        n_clusters = len(set(clean_labels))
        
        try:
            # Silhouette Score
            if n_clusters > 1 and n_clusters < len(clean_features):
                metrics['silhouette_score'] = silhouette_score(clean_features, clean_labels)
            else:
                metrics['silhouette_score'] = -1
            
            # Calinski-Harabasz Index
            if n_clusters > 1:
                metrics['calinski_harabasz_score'] = calinski_harabasz_score(clean_features, clean_labels)
            else:
                metrics['calinski_harabasz_score'] = -1
            
            # Davies-Bouldin Index
            if n_clusters > 1:
                metrics['davies_bouldin_score'] = davies_bouldin_score(clean_features, clean_labels)
            else:
                metrics['davies_bouldin_score'] = -1
            
        except Exception as e:
            logger.error(f"Error calculating metrics for {algorithm_name}: {e}")
            metrics = {'silhouette_score': -1, 'calinski_harabasz_score': -1, 'davies_bouldin_score': -1}
        
        return metrics
    
    def stability_testing(self, feature_matrix: np.ndarray,
                         algorithm: str = 'hdbscan',
                         n_bootstrap: int = 10,
                         sample_ratio: float = 0.8) -> Dict[str, Any]:
        """
        Test clustering stability using bootstrap sampling
        """
        logger.info(f"Performing stability testing for {algorithm} with {n_bootstrap} bootstrap samples")
        
        n_samples = feature_matrix.shape[0]
        bootstrap_size = int(n_samples * sample_ratio)
        
        bootstrap_results = []
        all_labels = []
        
        for i in range(n_bootstrap):
            # Bootstrap sample
            bootstrap_indices = np.random.choice(n_samples, bootstrap_size, replace=True)
            bootstrap_features = feature_matrix[bootstrap_indices]
            
            # Fit clustering
            if algorithm == 'hdbscan':
                clusterer = self.clusterers.get('hdbscan')
                if clusterer is None:
                    logger.error("HDBSCAN not fitted yet")
                    return {}
                
                # Create new clusterer with same parameters
                new_clusterer = hdbscan.HDBSCAN(
                    min_cluster_size=clusterer.min_cluster_size,
                    min_samples=clusterer.min_samples,
                    cluster_selection_epsilon=clusterer.cluster_selection_epsilon,
                    metric=clusterer.metric
                )
                labels = new_clusterer.fit_predict(bootstrap_features)
                
            elif algorithm == 'kmeans':
                clusterer = self.clusterers.get('kmeans')
                if clusterer is None:
                    logger.error("K-Means not fitted yet")
                    return {}
                
                new_clusterer = KMeans(
                    n_clusters=clusterer.n_clusters,
                    random_state=42 + i,
                    n_init=10
                )
                labels = new_clusterer.fit_predict(bootstrap_features)
            
            # Store results
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            n_outliers = np.sum(labels == -1) if -1 in labels else 0
            
            bootstrap_results.append({
                'iteration': i,
                'n_clusters': n_clusters,
                'n_outliers': n_outliers,
                'labels': labels
            })
            
            all_labels.append(labels)
        
        # Calculate stability metrics
        n_clusters_list = [result['n_clusters'] for result in bootstrap_results]
        n_outliers_list = [result['n_outliers'] for result in bootstrap_results]
        
        stability_metrics = {
            'n_clusters_mean': np.mean(n_clusters_list),
            'n_clusters_std': np.std(n_clusters_list),
            'n_clusters_range': (min(n_clusters_list), max(n_clusters_list)),
            'n_outliers_mean': np.mean(n_outliers_list),
            'n_outliers_std': np.std(n_outliers_list),
            'cluster_count_stability': 1.0 - (np.std(n_clusters_list) / max(np.mean(n_clusters_list), 1)),
            'bootstrap_results': bootstrap_results
        }
        
        logger.info(f"Stability testing completed. Cluster count stability: {stability_metrics['cluster_count_stability']:.3f}")
        return stability_metrics
    
    def geographic_validation(self, features_df: pd.DataFrame,
                            cluster_labels: np.ndarray,
                            algorithm_name: str) -> Dict[str, Any]:
        """
        Validate clustering results using geographic coherence
        """
        if not SPATIAL_AVAILABLE:
            logger.warning("Spatial validation not available - PySAL not installed")
            return {'spatial_validation': 'not_available'}
        
        logger.info(f"Performing geographic validation for {algorithm_name}")
        
        try:
            # Get coordinates if available
            coord_cols = []
            for col in ['centroid_lat', 'centroid_lng', 'lat', 'lng']:
                if col in features_df.columns:
                    coord_cols.append(col)
            
            if len(coord_cols) < 2:
                logger.warning("Insufficient coordinate data for spatial validation")
                return {'spatial_validation': 'insufficient_coordinates'}
            
            # Create spatial weights matrix
            coords = features_df[coord_cols[:2]].values
            
            # Use KNN weights (k=3 for small datasets)
            k = min(3, len(coords) - 1)
            if k < 1:
                return {'spatial_validation': 'insufficient_data'}
            
            w = weights.KNN.from_array(coords, k=k)
            w.transform = 'r'  # Row standardization
            
            # Calculate Moran's I for cluster assignments
            # Convert cluster labels to numeric (handle outliers)
            numeric_labels = cluster_labels.copy()
            if -1 in numeric_labels:
                # Assign outliers to a separate category
                max_label = max([l for l in numeric_labels if l != -1])
                numeric_labels[numeric_labels == -1] = max_label + 1
            
            moran = esda.Moran(numeric_labels, w)
            
            spatial_metrics = {
                'morans_i': moran.I,
                'morans_i_pvalue': moran.p_norm,
                'morans_i_zscore': moran.z_norm,
                'spatial_autocorrelation': 'positive' if moran.I > 0 else 'negative' if moran.I < 0 else 'none',
                'spatial_significance': 'significant' if moran.p_norm < 0.05 else 'not_significant'
            }
            
            logger.info(f"Moran's I: {moran.I:.3f} (p-value: {moran.p_norm:.3f})")
            return spatial_metrics
            
        except Exception as e:
            logger.error(f"Error in geographic validation: {e}")
            return {'spatial_validation': 'error', 'error_message': str(e)}
    
    def analyze_cluster_profiles(self, features_df: pd.DataFrame,
                               cluster_labels: np.ndarray,
                               feature_cols: List[str],
                               algorithm_name: str) -> Dict[str, Any]:
        """
        Generate detailed cluster profiles with statistics
        """
        logger.info(f"Analyzing cluster profiles for {algorithm_name}")
        
        # Create dataframe with cluster assignments
        profile_df = features_df[['county_fips'] + feature_cols].copy()
        profile_df['cluster'] = cluster_labels
        
        # Calculate cluster profiles
        cluster_profiles = {}
        unique_clusters = sorted(set(cluster_labels))
        
        for cluster_id in unique_clusters:
            cluster_data = profile_df[profile_df['cluster'] == cluster_id]
            
            if len(cluster_data) == 0:
                continue
            
            # Calculate statistics for each feature
            feature_stats = {}
            for feature in feature_cols:
                if feature in cluster_data.columns:
                    feature_stats[feature] = {
                        'mean': float(cluster_data[feature].mean()),
                        'median': float(cluster_data[feature].median()),
                        'std': float(cluster_data[feature].std()),
                        'min': float(cluster_data[feature].min()),
                        'max': float(cluster_data[feature].max()),
                        'count': int(cluster_data[feature].count())
                    }
            
            # Calculate relative importance (z-scores relative to overall mean)
            overall_means = profile_df[feature_cols].mean()
            overall_stds = profile_df[feature_cols].std()
            
            cluster_means = cluster_data[feature_cols].mean()
            z_scores = (cluster_means - overall_means) / overall_stds
            
            # Identify distinguishing features (high absolute z-scores)
            distinguishing_features = z_scores.abs().sort_values(ascending=False).head(5)
            
            cluster_profiles[f'cluster_{cluster_id}'] = {
                'cluster_id': int(cluster_id),
                'size': len(cluster_data),
                'counties': cluster_data['county_fips'].tolist(),
                'feature_statistics': feature_stats,
                'distinguishing_features': {
                    feature: float(z_score) for feature, z_score in distinguishing_features.items()
                },
                'cluster_centroid': cluster_means.to_dict()
            }
        
        # Calculate overall cluster summary
        cluster_summary = {
            'total_clusters': len(unique_clusters),
            'cluster_sizes': {f'cluster_{cid}': len(profile_df[profile_df['cluster'] == cid]) 
                            for cid in unique_clusters},
            'outliers': int(np.sum(cluster_labels == -1)) if -1 in cluster_labels else 0
        }
        
        return {
            'cluster_profiles': cluster_profiles,
            'cluster_summary': cluster_summary,
            'algorithm': algorithm_name
        }
    
    def calculate_feature_importance(self, feature_matrix: np.ndarray,
                                   cluster_labels: np.ndarray,
                                   feature_cols: List[str],
                                   algorithm_name: str) -> Dict[str, Any]:
        """
        Calculate feature importance using SHAP values (if available)
        """
        if not SHAP_AVAILABLE:
            logger.warning("SHAP not available, using alternative feature importance method")
            return self._alternative_feature_importance(feature_matrix, cluster_labels, feature_cols)
        
        logger.info(f"Calculating SHAP feature importance for {algorithm_name}")
        
        try:
            # For clustering, we'll use a classifier to predict cluster assignments
            # and then calculate SHAP values for that classifier
            from sklearn.ensemble import RandomForestClassifier
            
            # Remove outliers for training
            non_outlier_mask = cluster_labels != -1
            if np.sum(non_outlier_mask) < 10:
                logger.warning("Too few non-outlier points for SHAP analysis")
                return self._alternative_feature_importance(feature_matrix, cluster_labels, feature_cols)
            
            clean_features = feature_matrix[non_outlier_mask]
            clean_labels = cluster_labels[non_outlier_mask]
            
            # Train classifier to predict clusters
            classifier = RandomForestClassifier(n_estimators=100, random_state=42)
            classifier.fit(clean_features, clean_labels)
            
            # Calculate SHAP values
            explainer = shap.TreeExplainer(classifier)
            shap_values = explainer.shap_values(clean_features)
            
            # If multi-class, average across classes
            if isinstance(shap_values, list):
                shap_values_mean = np.mean([np.abs(sv).mean(axis=0) for sv in shap_values], axis=0)
            else:
                shap_values_mean = np.abs(shap_values).mean(axis=0)
            
            # Create feature importance ranking
            feature_importance = dict(zip(feature_cols, shap_values_mean))
            feature_importance_sorted = dict(sorted(feature_importance.items(), 
                                                  key=lambda x: x[1], reverse=True))
            
            return {
                'method': 'shap',
                'feature_importance': feature_importance_sorted,
                'top_features': list(feature_importance_sorted.keys())[:10],
                'classifier_accuracy': classifier.score(clean_features, clean_labels)
            }
            
        except Exception as e:
            logger.error(f"Error calculating SHAP importance: {e}")
            return self._alternative_feature_importance(feature_matrix, cluster_labels, feature_cols)
    
    def _alternative_feature_importance(self, feature_matrix: np.ndarray,
                                      cluster_labels: np.ndarray,
                                      feature_cols: List[str]) -> Dict[str, Any]:
        """
        Alternative feature importance using variance ratio
        """
        logger.info("Calculating alternative feature importance using variance ratio")
        
        # Remove outliers
        non_outlier_mask = cluster_labels != -1
        if np.sum(non_outlier_mask) < 2:
            return {'method': 'none', 'feature_importance': {}}
        
        clean_features = feature_matrix[non_outlier_mask]
        clean_labels = cluster_labels[non_outlier_mask]
        
        feature_importance = {}
        
        for i, feature_name in enumerate(feature_cols):
            feature_values = clean_features[:, i]
            
            # Calculate between-cluster variance vs within-cluster variance
            overall_var = np.var(feature_values)
            
            if overall_var == 0:
                feature_importance[feature_name] = 0
                continue
            
            # Calculate within-cluster variance
            within_cluster_var = 0
            total_points = 0
            
            for cluster_id in set(clean_labels):
                cluster_mask = clean_labels == cluster_id
                cluster_values = feature_values[cluster_mask]
                
                if len(cluster_values) > 1:
                    within_cluster_var += np.var(cluster_values) * len(cluster_values)
                    total_points += len(cluster_values)
            
            if total_points > 0:
                within_cluster_var /= total_points
                # Feature importance as ratio of between-cluster to within-cluster variance
                between_cluster_var = overall_var - within_cluster_var
                importance = between_cluster_var / (within_cluster_var + 1e-10)
                feature_importance[feature_name] = importance
            else:
                feature_importance[feature_name] = 0
        
        # Sort by importance
        feature_importance_sorted = dict(sorted(feature_importance.items(), 
                                              key=lambda x: x[1], reverse=True))
        
        return {
            'method': 'variance_ratio',
            'feature_importance': feature_importance_sorted,
            'top_features': list(feature_importance_sorted.keys())[:10]
        }
    
    def analyze_outliers(self, features_df: pd.DataFrame,
                        cluster_labels: np.ndarray,
                        feature_matrix: np.ndarray,
                        feature_cols: List[str],
                        algorithm_name: str) -> Dict[str, Any]:
        """
        Analyze outlier counties and their characteristics
        """
        outlier_mask = cluster_labels == -1
        n_outliers = np.sum(outlier_mask)
        
        if n_outliers == 0:
            return {
                'n_outliers': 0,
                'outlier_analysis': 'no_outliers_detected'
            }
        
        logger.info(f"Analyzing {n_outliers} outlier counties for {algorithm_name}")
        
        outlier_counties = features_df.loc[outlier_mask, 'county_fips'].tolist()
        outlier_features = feature_matrix[outlier_mask]
        
        # Calculate outlier statistics
        outlier_stats = {}
        overall_means = np.mean(feature_matrix, axis=0)
        overall_stds = np.std(feature_matrix, axis=0)
        
        for i, feature_name in enumerate(feature_cols):
            outlier_values = outlier_features[:, i]
            
            outlier_stats[feature_name] = {
                'outlier_mean': float(np.mean(outlier_values)),
                'overall_mean': float(overall_means[i]),
                'z_score': float((np.mean(outlier_values) - overall_means[i]) / (overall_stds[i] + 1e-10)),
                'outlier_std': float(np.std(outlier_values))
            }
        
        # Identify most extreme features for outliers
        z_scores = [(name, abs(stats['z_score'])) for name, stats in outlier_stats.items()]
        extreme_features = sorted(z_scores, key=lambda x: x[1], reverse=True)[:5]
        
        # Get outlier scores if available (from HDBSCAN)
        outlier_scores = None
        if algorithm_name == 'hdbscan' and 'hdbscan' in self.cluster_results:
            outlier_scores = self.cluster_results['hdbscan'].get('outlier_scores')
            if outlier_scores is not None:
                outlier_scores = outlier_scores[outlier_mask].tolist()
        
        return {
            'n_outliers': n_outliers,
            'outlier_counties': outlier_counties,
            'outlier_percentage': float(n_outliers / len(cluster_labels) * 100),
            'extreme_features': [{'feature': name, 'abs_z_score': score} for name, score in extreme_features],
            'outlier_feature_stats': outlier_stats,
            'outlier_scores': outlier_scores
        }
    
    def run_complete_clustering_analysis(self, features_df: pd.DataFrame,
                                       algorithms: List[str] = None) -> Dict[str, Any]:
        """
        Run complete clustering analysis with all algorithms and validation
        """
        if algorithms is None:
            algorithms = ['hdbscan', 'kmeans']
        
        logger.info(f"Running complete clustering analysis with algorithms: {algorithms}")
        
        # Prepare data
        feature_matrix, feature_cols, clustering_data = self.prepare_clustering_data(features_df)
        
        analysis_results = {
            'data_preparation': {
                'n_samples': feature_matrix.shape[0],
                'n_features': feature_matrix.shape[1],
                'feature_columns': feature_cols
            },
            'algorithms': {}
        }
        
        # Run each algorithm
        for algorithm in algorithms:
            logger.info(f"Running {algorithm} clustering")
            
            try:
                if algorithm == 'hdbscan':
                    # Hyperparameter tuning
                    tuning_results = self.hyperparameter_tuning_hdbscan(feature_matrix)
                    
                    # Fit with best parameters
                    if tuning_results['best_params']:
                        clusterer = self.fit_hdbscan(
                            feature_matrix,
                            **tuning_results['best_params']
                        )
                    else:
                        clusterer = self.fit_hdbscan(feature_matrix)
                    
                    cluster_labels = self.cluster_results['hdbscan']['labels']
                    
                elif algorithm == 'kmeans':
                    clusterer = self.fit_kmeans_optimal(feature_matrix)
                    cluster_labels = self.cluster_results['kmeans']['labels']
                    tuning_results = {'optimization': 'silhouette_score'}
                
                # Validation metrics
                internal_metrics = self.calculate_internal_metrics(
                    feature_matrix, cluster_labels, algorithm
                )
                
                # Stability testing
                stability_metrics = self.stability_testing(
                    feature_matrix, algorithm
                )
                
                # Geographic validation
                geographic_metrics = self.geographic_validation(
                    features_df, cluster_labels, algorithm
                )
                
                # Cluster profiles
                cluster_profiles = self.analyze_cluster_profiles(
                    features_df, cluster_labels, feature_cols, algorithm
                )
                
                # Feature importance
                feature_importance = self.calculate_feature_importance(
                    feature_matrix, cluster_labels, feature_cols, algorithm
                )
                
                # Outlier analysis
                outlier_analysis = self.analyze_outliers(
                    features_df, cluster_labels, feature_matrix, feature_cols, algorithm
                )
                
                # Store all results
                analysis_results['algorithms'][algorithm] = {
                    'hyperparameter_tuning': tuning_results,
                    'cluster_results': self.cluster_results[algorithm],
                    'validation_metrics': {
                        'internal_metrics': internal_metrics,
                        'stability_metrics': stability_metrics,
                        'geographic_metrics': geographic_metrics
                    },
                    'cluster_profiles': cluster_profiles,
                    'feature_importance': feature_importance,
                    'outlier_analysis': outlier_analysis
                }
                
                logger.info(f"{algorithm} analysis completed successfully")
                
            except Exception as e:
                logger.error(f"Error running {algorithm}: {e}")
                analysis_results['algorithms'][algorithm] = {
                    'error': str(e),
                    'status': 'failed'
                }
        
        # Generate summary comparison
        analysis_results['algorithm_comparison'] = self._compare_algorithms(analysis_results)
        
        return analysis_results
    
    def _compare_algorithms(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare performance of different clustering algorithms
        """
        comparison = {
            'summary': {},
            'recommendations': []
        }
        
        successful_algorithms = [
            alg for alg, results in analysis_results['algorithms'].items()
            if 'error' not in results
        ]
        
        if not successful_algorithms:
            return {'summary': 'no_successful_algorithms'}
        
        # Compare key metrics
        for algorithm in successful_algorithms:
            results = analysis_results['algorithms'][algorithm]
            
            comparison['summary'][algorithm] = {
                'n_clusters': results['cluster_results']['n_clusters'],
                'n_outliers': results['cluster_results']['n_outliers'],
                'silhouette_score': results['validation_metrics']['internal_metrics'].get('silhouette_score', -1),
                'stability_score': results['validation_metrics']['stability_metrics'].get('cluster_count_stability', -1)
            }
        
        # Generate recommendations
        if len(successful_algorithms) > 1:
            # Compare silhouette scores
            silhouette_scores = {
                alg: comparison['summary'][alg]['silhouette_score']
                for alg in successful_algorithms
            }
            best_silhouette = max(silhouette_scores.items(), key=lambda x: x[1])
            
            comparison['recommendations'].append(
                f"Best silhouette score: {best_silhouette[0]} ({best_silhouette[1]:.3f})"
            )
            
            # Compare stability
            stability_scores = {
                alg: comparison['summary'][alg]['stability_score']
                for alg in successful_algorithms
                if comparison['summary'][alg]['stability_score'] > 0
            }
            
            if stability_scores:
                best_stability = max(stability_scores.items(), key=lambda x: x[1])
                comparison['recommendations'].append(
                    f"Most stable clustering: {best_stability[0]} ({best_stability[1]:.3f})"
                )
        
        return comparison
    
    def export_clustering_results(self, analysis_results: Dict[str, Any],
                                features_df: pd.DataFrame,
                                filename_prefix: str = 'clustering_results') -> Dict[str, str]:
        """
        Export clustering results to multiple formats
        """
        logger.info("Exporting clustering results")
        
        exported_files = {}
        
        # Export cluster assignments
        for algorithm in analysis_results['algorithms'].keys():
            if 'cluster_results' in analysis_results['algorithms'][algorithm]:
                cluster_labels = analysis_results['algorithms'][algorithm]['cluster_results']['labels']
                
                # Create assignment dataframe
                assignments_df = features_df[['county_fips']].copy()
                assignments_df[f'{algorithm}_cluster'] = cluster_labels
                
                # Add confidence scores if available
                if algorithm == 'hdbscan' and 'probabilities' in analysis_results['algorithms'][algorithm]['cluster_results']:
                    probabilities = analysis_results['algorithms'][algorithm]['cluster_results']['probabilities']
                    assignments_df[f'{algorithm}_confidence'] = probabilities
                
                # Export assignments
                assignment_file = f"{filename_prefix}_{algorithm}_assignments.csv"
                filepath = save_processed_data(assignments_df, assignment_file)
                exported_files[f'{algorithm}_assignments'] = filepath
        
        # Export cluster profiles as JSON
        profiles_data = {}
        for algorithm in analysis_results['algorithms'].keys():
            if 'cluster_profiles' in analysis_results['algorithms'][algorithm]:
                profiles_data[algorithm] = analysis_results['algorithms'][algorithm]['cluster_profiles']
        
        if profiles_data:
            profiles_file = f"{filename_prefix}_profiles.json"
            profiles_path = save_processed_data(
                pd.DataFrame([profiles_data]), profiles_file.replace('.json', '.csv')
            )
            
            # Also save as JSON
            from config import DATA_CONFIG
            json_path = DATA_CONFIG.PROCESSED_DATA_DIR / profiles_file
            with open(json_path, 'w') as f:
                json.dump(profiles_data, f, indent=2, default=str)
            exported_files['cluster_profiles'] = str(json_path)
        
        # Export validation report
        validation_report = self._generate_validation_report(analysis_results)
        report_file = f"{filename_prefix}_validation_report.txt"
        
        from config import DATA_CONFIG
        report_path = DATA_CONFIG.PROCESSED_DATA_DIR / report_file
        with open(report_path, 'w') as f:
            f.write(validation_report)
        exported_files['validation_report'] = str(report_path)
        
        logger.info(f"Exported {len(exported_files)} clustering result files")
        return exported_files
    
    def _generate_validation_report(self, analysis_results: Dict[str, Any]) -> str:
        """
        Generate comprehensive validation report
        """
        report = []
        report.append("=" * 80)
        report.append("CLUSTERING VALIDATION REPORT")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Data summary
        data_prep = analysis_results.get('data_preparation', {})
        report.append("DATA PREPARATION")
        report.append("-" * 40)
        report.append(f"Samples: {data_prep.get('n_samples', 'N/A')}")
        report.append(f"Features: {data_prep.get('n_features', 'N/A')}")
        report.append("")
        
        # Algorithm results
        for algorithm, results in analysis_results.get('algorithms', {}).items():
            if 'error' in results:
                report.append(f"{algorithm.upper()} - FAILED")
                report.append(f"Error: {results['error']}")
                report.append("")
                continue
            
            report.append(f"{algorithm.upper()} RESULTS")
            report.append("-" * 40)
            
            # Basic results
            cluster_results = results.get('cluster_results', {})
            report.append(f"Clusters Found: {cluster_results.get('n_clusters', 'N/A')}")
            report.append(f"Outliers: {cluster_results.get('n_outliers', 'N/A')}")
            
            # Validation metrics
            validation = results.get('validation_metrics', {})
            internal = validation.get('internal_metrics', {})
            
            report.append(f"Silhouette Score: {internal.get('silhouette_score', 'N/A'):.3f}")
            report.append(f"Calinski-Harabasz Score: {internal.get('calinski_harabasz_score', 'N/A'):.3f}")
            report.append(f"Davies-Bouldin Score: {internal.get('davies_bouldin_score', 'N/A'):.3f}")
            
            # Stability
            stability = validation.get('stability_metrics', {})
            if 'cluster_count_stability' in stability:
                report.append(f"Stability Score: {stability['cluster_count_stability']:.3f}")
            
            # Geographic validation
            geographic = validation.get('geographic_metrics', {})
            if 'morans_i' in geographic:
                report.append(f"Moran's I: {geographic['morans_i']:.3f} (p={geographic.get('morans_i_pvalue', 'N/A'):.3f})")
            
            # Top features
            feature_imp = results.get('feature_importance', {})
            if 'top_features' in feature_imp:
                report.append(f"Top Features: {', '.join(feature_imp['top_features'][:5])}")
            
            report.append("")
        
        # Algorithm comparison
        comparison = analysis_results.get('algorithm_comparison', {})
        if 'recommendations' in comparison:
            report.append("RECOMMENDATIONS")
            report.append("-" * 40)
            for rec in comparison['recommendations']:
                report.append(f"• {rec}")
            report.append("")
        
        report.append("=" * 80)
        return "\n".join(report)


def main():
    """Demo function for clustering engine"""
    from src.data_loader import DataLoader
    from src.spatial_processor import SpatialProcessor
    from src.feature_engineering import FeatureEngineer
    
    # Initialize components
    loader = DataLoader()
    processor = SpatialProcessor()
    engineer = FeatureEngineer()
    clustering_engine = ClusteringEngine()
    
    logger.info("Starting clustering engine demo...")
    
    # Load and process data
    trips_df = loader.download_divvy_data(2023, 6)
    spending_df = loader.download_spending_data()
    
    # Spatial processing
    boundaries = processor.load_county_boundaries()
    stations_gdf = processor.extract_stations_from_trips(trips_df)
    stations_with_counties = processor.assign_stations_to_counties(stations_gdf, boundaries)
    county_mobility = processor.aggregate_trips_to_county_level(trips_df, stations_with_counties)
    
    # Feature engineering
    engineered_features, pipeline_results = engineer.create_feature_pipeline(
        county_mobility, spending_df, trips_df
    )
    
    # Run clustering analysis
    clustering_results = clustering_engine.run_complete_clustering_analysis(
        engineered_features, algorithms=['hdbscan', 'kmeans']
    )
    
    # Export results
    exported_files = clustering_engine.export_clustering_results(
        clustering_results, engineered_features
    )
    
    # Print summary
    print(f"\nClustering Analysis Summary:")
    print(f"- Algorithms tested: {list(clustering_results['algorithms'].keys())}")
    
    for algorithm, results in clustering_results['algorithms'].items():
        if 'error' not in results:
            cluster_results = results['cluster_results']
            validation = results['validation_metrics']['internal_metrics']
            print(f"- {algorithm}: {cluster_results['n_clusters']} clusters, "
                  f"silhouette={validation.get('silhouette_score', -1):.3f}")
    
    print(f"- Exported files: {list(exported_files.keys())}")
    
    logger.info("Clustering engine demo completed successfully!")
    
    return clustering_results, exported_files


if __name__ == "__main__":
    main()