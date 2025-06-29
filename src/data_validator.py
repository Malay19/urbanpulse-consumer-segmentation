"""
Data validation and quality assessment module
"""

import pandas as pd
import geopandas as gpd
import numpy as np
from typing import Dict, List, Any, Tuple
from loguru import logger
from datetime import datetime, timedelta
import warnings


class DataValidator:
    """Comprehensive data validation for all data sources"""
    
    def __init__(self):
        self.validation_results = {}
    
    def validate_divvy_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate Divvy bike-share trip data"""
        logger.info("Validating Divvy trip data...")
        
        results = {
            'data_type': 'divvy_trips',
            'total_records': len(df),
            'validation_timestamp': datetime.now(),
            'issues': [],
            'warnings': [],
            'passed': True
        }
        
        # Required columns check
        required_columns = [
            'trip_id', 'start_time', 'end_time', 'start_station_id',
            'start_lat', 'start_lng', 'end_lat', 'end_lng', 'member_type'
        ]
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            results['issues'].append(f"Missing required columns: {missing_columns}")
            results['passed'] = False
        
        if not results['passed']:
            return results
        
        # Data type validation
        try:
            df['start_time'] = pd.to_datetime(df['start_time'])
            df['end_time'] = pd.to_datetime(df['end_time'])
        except Exception as e:
            results['issues'].append(f"Date parsing error: {str(e)}")
            results['passed'] = False
        
        # Null value checks
        null_counts = df.isnull().sum()
        critical_nulls = null_counts[null_counts > 0]
        if not critical_nulls.empty:
            results['warnings'].append(f"Null values found: {critical_nulls.to_dict()}")
        
        # Coordinate validation
        lat_bounds = (-90, 90)
        lng_bounds = (-180, 180)
        
        invalid_coords = (
            (df['start_lat'] < lat_bounds[0]) | (df['start_lat'] > lat_bounds[1]) |
            (df['start_lng'] < lng_bounds[0]) | (df['start_lng'] > lng_bounds[1]) |
            (df['end_lat'] < lat_bounds[0]) | (df['end_lat'] > lat_bounds[1]) |
            (df['end_lng'] < lng_bounds[0]) | (df['end_lng'] > lng_bounds[1])
        ).sum()
        
        if invalid_coords > 0:
            results['warnings'].append(f"Invalid coordinates found: {invalid_coords} records")
        
        # Trip duration validation
        if 'start_time' in df.columns and 'end_time' in df.columns:
            try:
                durations = (pd.to_datetime(df['end_time']) - pd.to_datetime(df['start_time'])).dt.total_seconds() / 60
                
                # Check for negative durations
                negative_durations = (durations < 0).sum()
                if negative_durations > 0:
                    results['issues'].append(f"Negative trip durations: {negative_durations} records")
                
                # Check for extremely long trips (> 24 hours)
                long_trips = (durations > 1440).sum()
                if long_trips > 0:
                    results['warnings'].append(f"Very long trips (>24h): {long_trips} records")
                
                # Check for very short trips (< 1 minute)
                short_trips = (durations < 1).sum()
                if short_trips > 0:
                    results['warnings'].append(f"Very short trips (<1min): {short_trips} records")
                
                results['duration_stats'] = {
                    'mean_minutes': durations.mean(),
                    'median_minutes': durations.median(),
                    'std_minutes': durations.std(),
                    'min_minutes': durations.min(),
                    'max_minutes': durations.max()
                }
            except Exception as e:
                results['warnings'].append(f"Duration calculation error: {str(e)}")
        
        # Member type validation
        if 'member_type' in df.columns:
            valid_member_types = {'member', 'casual'}
            invalid_member_types = set(df['member_type'].unique()) - valid_member_types
            if invalid_member_types:
                results['warnings'].append(f"Invalid member types: {invalid_member_types}")
        
        # Duplicate trip IDs
        duplicate_ids = df['trip_id'].duplicated().sum()
        if duplicate_ids > 0:
            results['issues'].append(f"Duplicate trip IDs: {duplicate_ids} records")
        
        logger.info(f"Divvy validation completed. Issues: {len(results['issues'])}, Warnings: {len(results['warnings'])}")
        return results
    
    def validate_spending_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate consumer spending data"""
        logger.info("Validating consumer spending data...")
        
        results = {
            'data_type': 'spending_data',
            'total_records': len(df),
            'validation_timestamp': datetime.now(),
            'issues': [],
            'warnings': [],
            'passed': True
        }
        
        # Required columns check
        required_columns = ['county_fips', 'month', 'year', 'category', 'spending_amount']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            results['issues'].append(f"Missing required columns: {missing_columns}")
            results['passed'] = False
        
        if not results['passed']:
            return results
        
        # FIPS code validation
        if 'county_fips' in df.columns:
            # County FIPS should be 5-digit strings
            invalid_fips = df[~df['county_fips'].astype(str).str.match(r'^\d{5}$')]
            if not invalid_fips.empty:
                results['warnings'].append(f"Invalid FIPS codes: {len(invalid_fips)} records")
        
        # Date validation
        if 'year' in df.columns and 'month' in df.columns:
            invalid_years = df[(df['year'] < 2020) | (df['year'] > 2030)]
            if not invalid_years.empty:
                results['warnings'].append(f"Unusual years: {len(invalid_years)} records")
            
            invalid_months = df[(df['month'] < 1) | (df['month'] > 12)]
            if not invalid_months.empty:
                results['issues'].append(f"Invalid months: {len(invalid_months)} records")
        
        # Spending amount validation
        if 'spending_amount' in df.columns:
            negative_spending = (df['spending_amount'] < 0).sum()
            if negative_spending > 0:
                results['warnings'].append(f"Negative spending amounts: {negative_spending} records")
            
            zero_spending = (df['spending_amount'] == 0).sum()
            if zero_spending > len(df) * 0.1:  # More than 10% zeros is suspicious
                results['warnings'].append(f"High number of zero spending records: {zero_spending}")
            
            # Extreme values detection
            Q1 = df['spending_amount'].quantile(0.25)
            Q3 = df['spending_amount'].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 3 * IQR
            upper_bound = Q3 + 3 * IQR
            
            outliers = df[(df['spending_amount'] < lower_bound) | (df['spending_amount'] > upper_bound)]
            if not outliers.empty:
                results['warnings'].append(f"Potential outliers: {len(outliers)} records")
            
            results['spending_stats'] = {
                'mean': df['spending_amount'].mean(),
                'median': df['spending_amount'].median(),
                'std': df['spending_amount'].std(),
                'min': df['spending_amount'].min(),
                'max': df['spending_amount'].max()
            }
        
        # Category validation
        if 'category' in df.columns:
            expected_categories = {
                'restaurants', 'retail', 'grocery', 'entertainment',
                'transportation', 'healthcare', 'education', 'services'
            }
            actual_categories = set(df['category'].unique())
            unexpected_categories = actual_categories - expected_categories
            if unexpected_categories:
                results['warnings'].append(f"Unexpected categories: {unexpected_categories}")
        
        # Completeness check
        null_counts = df.isnull().sum()
        critical_nulls = null_counts[null_counts > 0]
        if not critical_nulls.empty:
            results['warnings'].append(f"Null values found: {critical_nulls.to_dict()}")
        
        logger.info(f"Spending validation completed. Issues: {len(results['issues'])}, Warnings: {len(results['warnings'])}")
        return results
    
    def validate_boundary_data(self, gdf: gpd.GeoDataFrame) -> Dict[str, Any]:
        """Validate geographic boundary data"""
        logger.info("Validating boundary data...")
        
        results = {
            'data_type': 'boundary_data',
            'total_records': len(gdf),
            'validation_timestamp': datetime.now(),
            'issues': [],
            'warnings': [],
            'passed': True
        }
        
        # Required columns
        required_columns = ['county_fips', 'geometry']
        missing_columns = [col for col in required_columns if col not in gdf.columns]
        if missing_columns:
            results['issues'].append(f"Missing required columns: {missing_columns}")
            results['passed'] = False
        
        if not results['passed']:
            return results
        
        # Geometry validation
        if 'geometry' in gdf.columns:
            invalid_geometries = gdf[~gdf.geometry.is_valid]
            if not invalid_geometries.empty:
                results['issues'].append(f"Invalid geometries: {len(invalid_geometries)} records")
            
            empty_geometries = gdf[gdf.geometry.is_empty]
            if not empty_geometries.empty:
                results['warnings'].append(f"Empty geometries: {len(empty_geometries)} records")
        
        # CRS validation
        if gdf.crs is None:
            results['warnings'].append("No CRS defined for boundary data")
        elif gdf.crs.to_string() != 'EPSG:4326':
            results['warnings'].append(f"Unexpected CRS: {gdf.crs}")
        
        # FIPS code validation
        if 'county_fips' in gdf.columns:
            invalid_fips = gdf[~gdf['county_fips'].astype(str).str.match(r'^\d{5}$')]
            if not invalid_fips.empty:
                results['warnings'].append(f"Invalid FIPS codes: {len(invalid_fips)} records")
        
        logger.info(f"Boundary validation completed. Issues: {len(results['issues'])}, Warnings: {len(results['warnings'])}")
        return results
    
    def generate_data_quality_report(self, validation_results: List[Dict[str, Any]]) -> str:
        """Generate a comprehensive data quality report"""
        
        report = []
        report.append("=" * 60)
        report.append("DATA QUALITY ASSESSMENT REPORT")
        report.append("=" * 60)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        total_issues = 0
        total_warnings = 0
        
        for result in validation_results:
            report.append(f"Dataset: {result['data_type'].upper()}")
            report.append("-" * 40)
            report.append(f"Total Records: {result['total_records']:,}")
            report.append(f"Validation Status: {'PASSED' if result['passed'] else 'FAILED'}")
            
            if result['issues']:
                report.append(f"\nISSUES ({len(result['issues'])}):")
                for issue in result['issues']:
                    report.append(f"  ❌ {issue}")
                total_issues += len(result['issues'])
            
            if result['warnings']:
                report.append(f"\nWARNINGS ({len(result['warnings'])}):")
                for warning in result['warnings']:
                    report.append(f"  ⚠️  {warning}")
                total_warnings += len(result['warnings'])
            
            # Add statistics if available
            if 'duration_stats' in result:
                stats = result['duration_stats']
                report.append(f"\nTrip Duration Statistics:")
                report.append(f"  Mean: {stats['mean_minutes']:.1f} minutes")
                report.append(f"  Median: {stats['median_minutes']:.1f} minutes")
                report.append(f"  Range: {stats['min_minutes']:.1f} - {stats['max_minutes']:.1f} minutes")
            
            if 'spending_stats' in result:
                stats = result['spending_stats']
                report.append(f"\nSpending Statistics:")
                report.append(f"  Mean: ${stats['mean']:,.2f}")
                report.append(f"  Median: ${stats['median']:,.2f}")
                report.append(f"  Range: ${stats['min']:,.2f} - ${stats['max']:,.2f}")
            
            report.append("")
        
        # Summary
        report.append("SUMMARY")
        report.append("-" * 40)
        report.append(f"Total Datasets Validated: {len(validation_results)}")
        report.append(f"Total Issues: {total_issues}")
        report.append(f"Total Warnings: {total_warnings}")
        
        overall_status = "PASSED" if total_issues == 0 else "FAILED"
        report.append(f"Overall Status: {overall_status}")
        report.append("")
        
        return "\n".join(report)
    
    def save_validation_report(self, validation_results: List[Dict[str, Any]], filename: str = None) -> str:
        """Save validation report to file"""
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"data_quality_report_{timestamp}.txt"
        
        report = self.generate_data_quality_report(validation_results)
        
        from config import DATA_CONFIG
        filepath = DATA_CONFIG.PROCESSED_DATA_DIR / filename
        
        with open(filepath, 'w') as f:
            f.write(report)
        
        logger.info(f"Data quality report saved to {filepath}")
        return str(filepath)


def main():
    """Demo function for data validation"""
    from data_loader import DataLoader
    
    # Load sample data
    loader = DataLoader()
    validator = DataValidator()
    
    # Generate sample data
    divvy_data = loader.download_divvy_data(2023, 6)
    spending_data = loader.download_spending_data()
    boundary_data = loader.load_county_boundaries()
    
    # Validate all datasets
    validation_results = []
    
    validation_results.append(validator.validate_divvy_data(divvy_data))
    validation_results.append(validator.validate_spending_data(spending_data))
    validation_results.append(validator.validate_boundary_data(boundary_data))
    
    # Generate and save report
    report_path = validator.save_validation_report(validation_results)
    
    # Print summary
    print(validator.generate_data_quality_report(validation_results))
    
    return validation_results


if __name__ == "__main__":
    main()