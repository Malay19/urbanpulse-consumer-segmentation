"""
Visualization utilities for spatial data and station assignments
"""

import pandas as pd
import geopandas as gpd
import folium
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple
from loguru import logger

from config import VIZ_CONFIG


class SpatialVisualizer:
    """Create visualizations for spatial data processing results"""
    
    def __init__(self):
        self.default_center = VIZ_CONFIG.DEFAULT_MAP_CENTER
        self.default_zoom = VIZ_CONFIG.DEFAULT_ZOOM
        self.colors = VIZ_CONFIG.CLUSTER_COLORS
    
    def create_station_assignment_map(self, stations_gdf: gpd.GeoDataFrame, 
                                    boundaries: gpd.GeoDataFrame = None,
                                    save_path: str = None) -> folium.Map:
        """
        Create an interactive map showing station-to-county assignments
        """
        logger.info("Creating station assignment map")
        
        # Initialize map
        m = folium.Map(
            location=self.default_center,
            zoom_start=self.default_zoom,
            tiles='OpenStreetMap'
        )
        
        # Add county boundaries if provided
        if boundaries is not None:
            folium.GeoJson(
                boundaries,
                style_function=lambda feature: {
                    'fillColor': 'lightblue',
                    'color': 'blue',
                    'weight': 2,
                    'fillOpacity': 0.1,
                },
                popup=folium.GeoJsonPopup(fields=['county_name', 'county_fips']),
                tooltip=folium.GeoJsonTooltip(fields=['county_name'])
            ).add_to(m)
        
        # Add stations with color coding by county
        if 'county_fips' in stations_gdf.columns:
            unique_counties = stations_gdf['county_fips'].unique()
            color_map = {county: self.colors[i % len(self.colors)] 
                        for i, county in enumerate(unique_counties)}
            
            for idx, station in stations_gdf.iterrows():
                color = color_map.get(station.get('county_fips'), 'gray')
                
                folium.CircleMarker(
                    location=[station['lat'], station['lng']],
                    radius=5,
                    popup=f"Station {station['station_id']}<br>County: {station.get('county_fips', 'Unknown')}",
                    color=color,
                    fill=True,
                    fillColor=color,
                    fillOpacity=0.7
                ).add_to(m)
        else:
            # Add stations without county color coding
            for idx, station in stations_gdf.iterrows():
                folium.CircleMarker(
                    location=[station['lat'], station['lng']],
                    radius=5,
                    popup=f"Station {station['station_id']}",
                    color='red',
                    fill=True,
                    fillColor='red',
                    fillOpacity=0.7
                ).add_to(m)
        
        # Add legend for counties
        if 'county_fips' in stations_gdf.columns and boundaries is not None:
            legend_html = self._create_county_legend(stations_gdf, boundaries, color_map)
            m.get_root().html.add_child(folium.Element(legend_html))
        
        if save_path:
            m.save(save_path)
            logger.info(f"Station assignment map saved to {save_path}")
        
        return m
    
    def _create_county_legend(self, stations_gdf: gpd.GeoDataFrame, 
                            boundaries: gpd.GeoDataFrame,
                            color_map: Dict[str, str]) -> str:
        """Create HTML legend for county colors"""
        
        legend_items = []
        for county_fips, color in color_map.items():
            county_name = boundaries[boundaries['county_fips'] == county_fips]['county_name'].iloc[0] \
                         if not boundaries[boundaries['county_fips'] == county_fips].empty else county_fips
            station_count = len(stations_gdf[stations_gdf['county_fips'] == county_fips])
            
            legend_items.append(
                f'<li><span style="color:{color};">●</span> {county_name} ({station_count} stations)</li>'
            )
        
        legend_html = f'''
        <div style="position: fixed; 
                    top: 10px; right: 10px; width: 300px; height: auto; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:14px; padding: 10px">
        <h4>Counties</h4>
        <ul style="list-style-type: none; padding: 0;">
        {''.join(legend_items)}
        </ul>
        </div>
        '''
        return legend_html
    
    def plot_county_mobility_metrics(self, county_data: pd.DataFrame, 
                                   save_path: str = None) -> go.Figure:
        """
        Create dashboard of county-level mobility metrics
        """
        logger.info("Creating county mobility metrics dashboard")
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Total Trips by County',
                'Average Trip Duration',
                'Trips per Square Kilometer',
                'Member vs Casual Ratio'
            ],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Sort by total trips for better visualization
        county_data_sorted = county_data.sort_values('total_trips', ascending=True)
        
        # Plot 1: Total trips
        fig.add_trace(
            go.Bar(
                x=county_data_sorted['total_trips'],
                y=county_data_sorted['county_name'] if 'county_name' in county_data_sorted.columns 
                  else county_data_sorted['county_fips'],
                orientation='h',
                name='Total Trips',
                marker_color='lightblue'
            ),
            row=1, col=1
        )
        
        # Plot 2: Average trip duration
        fig.add_trace(
            go.Bar(
                x=county_data_sorted['avg_trip_duration_minutes'],
                y=county_data_sorted['county_name'] if 'county_name' in county_data_sorted.columns 
                  else county_data_sorted['county_fips'],
                orientation='h',
                name='Avg Duration (min)',
                marker_color='lightgreen'
            ),
            row=1, col=2
        )
        
        # Plot 3: Trip density
        if 'trips_per_sq_km' in county_data_sorted.columns:
            fig.add_trace(
                go.Bar(
                    x=county_data_sorted['trips_per_sq_km'],
                    y=county_data_sorted['county_name'] if 'county_name' in county_data_sorted.columns 
                      else county_data_sorted['county_fips'],
                    orientation='h',
                    name='Trips per km²',
                    marker_color='orange'
                ),
                row=2, col=1
            )
        
        # Plot 4: Member ratio
        if 'member_ratio' in county_data_sorted.columns:
            fig.add_trace(
                go.Bar(
                    x=county_data_sorted['member_ratio'],
                    y=county_data_sorted['county_name'] if 'county_name' in county_data_sorted.columns 
                      else county_data_sorted['county_fips'],
                    orientation='h',
                    name='Member Ratio',
                    marker_color='purple'
                ),
                row=2, col=2
            )
        
        # Update layout
        fig.update_layout(
            height=800,
            title_text="County-Level Mobility Metrics Dashboard",
            showlegend=False
        )
        
        # Update x-axis labels
        fig.update_xaxes(title_text="Number of Trips", row=1, col=1)
        fig.update_xaxes(title_text="Minutes", row=1, col=2)
        fig.update_xaxes(title_text="Trips per km²", row=2, col=1)
        fig.update_xaxes(title_text="Ratio (0-1)", row=2, col=2)
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Mobility metrics dashboard saved to {save_path}")
        
        return fig
    
    def plot_mobility_spending_correlation(self, combined_data: pd.DataFrame,
                                         save_path: str = None) -> go.Figure:
        """
        Create correlation plots between mobility and spending metrics
        """
        logger.info("Creating mobility-spending correlation plots")
        
        # Identify spending columns
        spending_cols = [col for col in combined_data.columns if col.startswith('spending_')]
        mobility_cols = ['total_trips', 'avg_trip_duration_minutes', 'trips_per_sq_km']
        
        if not spending_cols:
            logger.warning("No spending columns found for correlation analysis")
            return None
        
        # Create correlation matrix
        correlation_cols = mobility_cols + spending_cols
        available_cols = [col for col in correlation_cols if col in combined_data.columns]
        
        if len(available_cols) < 2:
            logger.warning("Insufficient columns for correlation analysis")
            return None
        
        corr_matrix = combined_data[available_cols].corr()
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=corr_matrix.round(2).values,
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title="Mobility-Spending Correlation Matrix",
            xaxis_title="Variables",
            yaxis_title="Variables",
            height=600,
            width=800
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Correlation plot saved to {save_path}")
        
        return fig
    
    def create_spatial_summary_report(self, stations_gdf: gpd.GeoDataFrame,
                                    county_data: pd.DataFrame,
                                    validation_results: Dict[str, Any]) -> str:
        """
        Create a text summary report of spatial processing results
        """
        report = []
        report.append("=" * 60)
        report.append("SPATIAL PROCESSING SUMMARY REPORT")
        report.append("=" * 60)
        report.append(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Station summary
        report.append("STATION ANALYSIS")
        report.append("-" * 30)
        report.append(f"Total Stations: {len(stations_gdf):,}")
        
        if 'county_fips' in stations_gdf.columns:
            stations_per_county = stations_gdf.groupby('county_fips').size()
            report.append(f"Counties with Stations: {len(stations_per_county)}")
            report.append(f"Stations per County - Mean: {stations_per_county.mean():.1f}")
            report.append(f"Stations per County - Range: {stations_per_county.min()} - {stations_per_county.max()}")
        
        if 'trip_count' in stations_gdf.columns:
            report.append(f"Total Trips Across All Stations: {stations_gdf['trip_count'].sum():,}")
            report.append(f"Average Trips per Station: {stations_gdf['trip_count'].mean():.1f}")
        
        report.append("")
        
        # County mobility summary
        report.append("COUNTY MOBILITY METRICS")
        report.append("-" * 30)
        report.append(f"Counties Analyzed: {len(county_data)}")
        
        if 'total_trips' in county_data.columns:
            report.append(f"Total Trips: {county_data['total_trips'].sum():,}")
            report.append(f"Average Trips per County: {county_data['total_trips'].mean():.0f}")
        
        if 'avg_trip_duration_minutes' in county_data.columns:
            report.append(f"Average Trip Duration: {county_data['avg_trip_duration_minutes'].mean():.1f} minutes")
        
        if 'member_ratio' in county_data.columns:
            report.append(f"Overall Member Ratio: {county_data['member_ratio'].mean():.2f}")
        
        report.append("")
        
        # Validation results
        report.append("DATA VALIDATION")
        report.append("-" * 30)
        report.append(f"Validation Status: {'PASSED' if validation_results['passed'] else 'FAILED'}")
        report.append(f"Issues Found: {len(validation_results['issues'])}")
        report.append(f"Warnings: {len(validation_results['warnings'])}")
        
        if validation_results['issues']:
            report.append("\nIssues:")
            for issue in validation_results['issues']:
                report.append(f"  ❌ {issue}")
        
        if validation_results['warnings']:
            report.append("\nWarnings:")
            for warning in validation_results['warnings']:
                report.append(f"  ⚠️  {warning}")
        
        report.append("")
        report.append("=" * 60)
        
        return "\n".join(report)


def main():
    """Demo function for visualization utilities"""
    from src.data_loader import DataLoader
    from src.spatial_processor import SpatialProcessor
    
    # Load and process data
    loader = DataLoader()
    processor = SpatialProcessor()
    visualizer = SpatialVisualizer()
    
    # Generate sample data
    trips_df = loader.download_divvy_data(2023, 6)
    boundaries = processor.load_county_boundaries()
    
    # Process spatial data
    stations_gdf = processor.extract_stations_from_trips(trips_df)
    stations_with_counties = processor.assign_stations_to_counties(stations_gdf, boundaries)
    county_mobility = processor.aggregate_trips_to_county_level(trips_df, stations_with_counties)
    
    # Create visualizations
    station_map = visualizer.create_station_assignment_map(
        stations_with_counties, boundaries, 
        save_path="data/processed/station_assignments_map.html"
    )
    
    mobility_dashboard = visualizer.plot_county_mobility_metrics(
        county_mobility,
        save_path="data/processed/mobility_metrics_dashboard.html"
    )
    
    # Generate summary report
    validation_results = processor.validate_spatial_data(stations_with_counties, boundaries)
    summary_report = visualizer.create_spatial_summary_report(
        stations_with_counties, county_mobility, validation_results
    )
    
    # Save report
    with open("data/processed/spatial_processing_report.txt", "w") as f:
        f.write(summary_report)
    
    print("Visualization demo completed!")
    print(summary_report)
    
    return station_map, mobility_dashboard


if __name__ == "__main__":
    main()