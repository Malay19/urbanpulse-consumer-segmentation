"""
Interactive Streamlit Dashboard for Consumer Segmentation Analysis
Comprehensive visualization and exploration platform for clustering results and business intelligence
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import folium
from streamlit_folium import st_folium
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import base64
from io import BytesIO
import zipfile

# Import our modules
from src.data_loader import DataLoader
from src.spatial_processor import SpatialProcessor
from src.feature_engineering import FeatureEngineer
from src.clustering_engine import ClusteringEngine
from src.persona_generator import PersonaGenerator
from src.dashboard_generator import DashboardGenerator
from config import VIZ_CONFIG, MODEL_CONFIG


class ConsumerSegmentationDashboard:
    """Main dashboard class for consumer segmentation analysis"""
    
    def __init__(self):
        self.setup_page_config()
        self.initialize_session_state()
        self.load_cached_data()
    
    def setup_page_config(self):
        """Configure Streamlit page settings"""
        st.set_page_config(
            page_title="Consumer Segmentation Dashboard",
            page_icon="🎯",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS for better styling
        st.markdown("""
        <style>
        .main-header {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            padding: 1rem;
            border-radius: 10px;
            margin-bottom: 2rem;
        }
        .main-header h1 {
            color: white;
            text-align: center;
            margin: 0;
        }
        .metric-card {
            background: white;
            padding: 1rem;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            border-left: 4px solid #667eea;
        }
        .persona-card {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 1.5rem;
            border-radius: 10px;
            margin: 1rem 0;
            border: 1px solid #e1e5e9;
        }
        .sidebar-section {
            background: #f8f9fa;
            padding: 1rem;
            border-radius: 8px;
            margin-bottom: 1rem;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def initialize_session_state(self):
        """Initialize Streamlit session state variables"""
        if 'data_loaded' not in st.session_state:
            st.session_state.data_loaded = False
        if 'clustering_results' not in st.session_state:
            st.session_state.clustering_results = None
        if 'personas' not in st.session_state:
            st.session_state.personas = None
        if 'selected_clusters' not in st.session_state:
            st.session_state.selected_clusters = []
        if 'analysis_parameters' not in st.session_state:
            st.session_state.analysis_parameters = {}
    
    def load_cached_data(self):
        """Load or generate cached analysis data"""
        try:
            # Try to load existing results
            if not st.session_state.data_loaded:
                with st.spinner("Loading analysis data..."):
                    self.run_complete_analysis()
                    st.session_state.data_loaded = True
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            st.info("Using demo data for visualization")
            self.load_demo_data()
    
    def run_complete_analysis(self):
        """Run the complete consumer segmentation analysis pipeline"""
        # Initialize components
        loader = DataLoader()
        processor = SpatialProcessor()
        engineer = FeatureEngineer()
        clustering_engine = ClusteringEngine()
        persona_generator = PersonaGenerator()
        
        # Load and process data
        trips_df = loader.download_divvy_data(2023, 6)
        spending_df = loader.download_spending_data()
        boundaries = processor.load_county_boundaries()
        
        # Spatial processing
        stations_gdf = processor.extract_stations_from_trips(trips_df)
        stations_with_counties = processor.assign_stations_to_counties(stations_gdf, boundaries)
        county_mobility = processor.aggregate_trips_to_county_level(trips_df, stations_with_counties)
        
        # Feature engineering
        engineered_features, pipeline_results = engineer.create_feature_pipeline(
            county_mobility, spending_df, trips_df
        )
        
        # Clustering analysis
        clustering_results = clustering_engine.run_complete_clustering_analysis(
            engineered_features, algorithms=['hdbscan', 'kmeans']
        )
        
        # Persona generation
        personas, opportunities, insights = persona_generator.run_complete_persona_analysis(
            clustering_results, engineered_features
        )
        
        # Store in session state
        st.session_state.clustering_results = clustering_results
        st.session_state.personas = personas
        st.session_state.opportunities = opportunities
        st.session_state.insights = insights
        st.session_state.engineered_features = engineered_features
        st.session_state.county_mobility = county_mobility
        st.session_state.boundaries = boundaries
    
    def load_demo_data(self):
        """Load demo data for visualization when real analysis fails"""
        from src.persona_generator import PersonaType, ConsumerPersona, BusinessOpportunity
        
        # Create demo personas
        demo_personas = {
            'urban_commuter': ConsumerPersona(
                persona_id='urban_commuter',
                persona_name='Urban Commuter Pro',
                persona_type=PersonaType.URBAN_COMMUTER,
                cluster_ids=[0],
                estimated_population=15000,
                median_income=75000,
                age_distribution={'18-34': 0.4, '35-54': 0.4, '55+': 0.2},
                education_level={'bachelor_plus': 0.6, 'high_school': 0.4},
                mobility_profile={'usage_intensity': 'high', 'member_ratio': 0.85},
                spending_profile={'spending_level': 'high', 'category_preferences': {'restaurants': 0.3, 'retail': 0.25}},
                temporal_patterns={'schedule_type': 'structured'},
                market_value=250000,
                targeting_effectiveness=0.85,
                seasonal_trends={'spring': 1.0, 'summer': 1.2, 'fall': 0.9, 'winter': 0.7},
                description='Highly structured professionals who rely on bike-sharing for daily commuting to work.',
                key_motivations=['Reliable transportation', 'Time efficiency', 'Cost savings'],
                preferred_channels=['Mobile app', 'Email', 'LinkedIn'],
                pain_points=['Rush hour bike availability', 'Weather dependency'],
                marketing_strategies=['Corporate partnerships', 'Commuter packages'],
                product_opportunities=['Reserved bikes', 'Express lanes'],
                infrastructure_needs=['High-capacity stations', 'Transit integration']
            ),
            'leisure_cyclist': ConsumerPersona(
                persona_id='leisure_cyclist',
                persona_name='Weekend Explorer',
                persona_type=PersonaType.LEISURE_CYCLIST,
                cluster_ids=[1],
                estimated_population=8000,
                median_income=65000,
                age_distribution={'18-34': 0.3, '35-54': 0.5, '55+': 0.2},
                education_level={'bachelor_plus': 0.5, 'high_school': 0.5},
                mobility_profile={'usage_intensity': 'medium', 'member_ratio': 0.6},
                spending_profile={'spending_level': 'medium', 'category_preferences': {'entertainment': 0.35, 'restaurants': 0.3}},
                temporal_patterns={'schedule_type': 'flexible'},
                market_value=120000,
                targeting_effectiveness=0.75,
                seasonal_trends={'spring': 1.1, 'summer': 1.4, 'fall': 1.0, 'winter': 0.5},
                description='Recreation-focused users who enjoy cycling for leisure and exploration on weekends.',
                key_motivations=['Recreation', 'Fitness', 'Exploration'],
                preferred_channels=['Social media', 'Outdoor magazines', 'Community events'],
                pain_points=['Limited weekend availability', 'Route planning'],
                marketing_strategies=['Weekend promotions', 'Fitness partnerships'],
                product_opportunities=['Scenic routes', 'Fitness tracking'],
                infrastructure_needs=['Recreational paths', 'Tourist-friendly stations']
            )
        }
        
        # Create demo opportunities
        demo_opportunities = [
            BusinessOpportunity(
                opportunity_type='Premium Commuter Services',
                description='Develop premium service tier for high-frequency commuters',
                target_segments=['Urban Commuter Pro'],
                estimated_market_size=300000,
                investment_level='Medium',
                expected_roi='25-35%',
                implementation_timeline='6-9 months',
                key_metrics=['Customer lifetime value', 'Premium conversion rate']
            ),
            BusinessOpportunity(
                opportunity_type='Weekend Recreation Packages',
                description='Create weekend-focused packages for leisure cyclists',
                target_segments=['Weekend Explorer'],
                estimated_market_size=150000,
                investment_level='Low',
                expected_roi='15-25%',
                implementation_timeline='3-6 months',
                key_metrics=['Weekend usage growth', 'Package adoption rate']
            )
        ]
        
        # Create demo insights
        demo_insights = {
            'market_overview': {
                'total_addressable_market': 450000,
                'total_population': 23000,
                'average_targeting_effectiveness': 0.80,
                'number_of_segments': 2
            },
            'key_insights': [
                'Urban commuters represent 67% of total market value despite being 65% of users',
                'Summer season shows 40% increase in leisure cycling activity',
                'Premium services could capture additional $300K in annual revenue',
                'Weekend promotions have highest conversion potential for leisure segment'
            ]
        }
        
        # Store demo data
        st.session_state.personas = demo_personas
        st.session_state.opportunities = demo_opportunities
        st.session_state.insights = demo_insights
        st.session_state.data_loaded = True
    
    def render_sidebar(self):
        """Render the sidebar with filters and controls"""
        st.sidebar.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.sidebar.title("🎯 Dashboard Controls")
        st.sidebar.markdown('</div>', unsafe_allow_html=True)
        
        # Data refresh section
        st.sidebar.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.sidebar.subheader("📊 Data Management")
        
        if st.sidebar.button("🔄 Refresh Analysis", help="Re-run the complete analysis pipeline"):
            st.session_state.data_loaded = False
            st.experimental_rerun()
        
        if st.sidebar.button("📥 Download Results", help="Download all analysis results"):
            self.download_results()
        
        st.sidebar.markdown('</div>', unsafe_allow_html=True)
        
        # Persona filters
        if st.session_state.personas:
            st.sidebar.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
            st.sidebar.subheader("👥 Persona Filters")
            
            persona_names = list(st.session_state.personas.keys())
            selected_personas = st.sidebar.multiselect(
                "Select Personas to Display",
                options=persona_names,
                default=persona_names,
                help="Choose which personas to include in visualizations"
            )
            
            st.session_state.selected_personas = selected_personas
            st.sidebar.markdown('</div>', unsafe_allow_html=True)
        
        # Analysis parameters
        st.sidebar.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.sidebar.subheader("⚙️ Analysis Parameters")
        
        # Clustering parameters
        st.sidebar.write("**Clustering Settings**")
        min_cluster_size = st.sidebar.slider(
            "Min Cluster Size",
            min_value=10, max_value=200, value=MODEL_CONFIG.MIN_CLUSTER_SIZE,
            help="Minimum number of points in a cluster"
        )
        
        cluster_epsilon = st.sidebar.slider(
            "Cluster Selection Epsilon",
            min_value=0.0, max_value=0.5, value=MODEL_CONFIG.CLUSTER_SELECTION_EPSILON,
            step=0.05,
            help="Distance threshold for cluster selection"
        )
        
        # Visualization parameters
        st.sidebar.write("**Visualization Settings**")
        show_outliers = st.sidebar.checkbox("Show Outliers", value=True)
        color_scheme = st.sidebar.selectbox(
            "Color Scheme",
            options=["Default", "Viridis", "Plasma", "Inferno"],
            help="Color scheme for visualizations"
        )
        
        st.session_state.analysis_parameters = {
            'min_cluster_size': min_cluster_size,
            'cluster_epsilon': cluster_epsilon,
            'show_outliers': show_outliers,
            'color_scheme': color_scheme
        }
        
        st.sidebar.markdown('</div>', unsafe_allow_html=True)
        
        # About section
        st.sidebar.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.sidebar.subheader("ℹ️ About")
        st.sidebar.info(
            "This dashboard provides comprehensive analysis of consumer segmentation "
            "based on mobility patterns and spending behavior. Use the controls above "
            "to customize the analysis and explore different aspects of the data."
        )
        st.sidebar.markdown('</div>', unsafe_allow_html=True)
    
    def render_main_header(self):
        """Render the main dashboard header"""
        st.markdown("""
        <div class="main-header">
            <h1>🎯 Consumer Segmentation Dashboard</h1>
            <p style="text-align: center; color: white; margin: 0; opacity: 0.9;">
                Advanced Analytics for Mobility & Spending Patterns
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    def render_overview_metrics(self):
        """Render key overview metrics"""
        if not st.session_state.insights:
            st.warning("No insights data available")
            return
        
        insights = st.session_state.insights
        market_overview = insights.get('market_overview', {})
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h3 style="margin: 0; color: #667eea;">Total Market Value</h3>
                <h2 style="margin: 10px 0 0 0; color: #333;">${:,.0f}</h2>
            </div>
            """.format(market_overview.get('total_addressable_market', 0)), unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <h3 style="margin: 0; color: #667eea;">Total Users</h3>
                <h2 style="margin: 10px 0 0 0; color: #333;">{:,}</h2>
            </div>
            """.format(market_overview.get('total_population', 0)), unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-card">
                <h3 style="margin: 0; color: #667eea;">Avg. Effectiveness</h3>
                <h2 style="margin: 10px 0 0 0; color: #333;">{:.1%}</h2>
            </div>
            """.format(market_overview.get('average_targeting_effectiveness', 0)), unsafe_allow_html=True)
        
        with col4:
            st.markdown("""
            <div class="metric-card">
                <h3 style="margin: 0; color: #667eea;">Segments</h3>
                <h2 style="margin: 10px 0 0 0; color: #333;">{}</h2>
            </div>
            """.format(market_overview.get('number_of_segments', 0)), unsafe_allow_html=True)
    
    def render_interactive_map(self):
        """Render interactive geographic map"""
        st.subheader("🗺️ Geographic Distribution")
        
        if not st.session_state.personas:
            st.warning("No persona data available for mapping")
            return
        
        # Create base map
        m = folium.Map(
            location=VIZ_CONFIG.DEFAULT_MAP_CENTER,
            zoom_start=VIZ_CONFIG.DEFAULT_ZOOM,
            tiles='OpenStreetMap'
        )
        
        # Add persona markers
        colors = VIZ_CONFIG.CLUSTER_COLORS
        for i, (persona_id, persona) in enumerate(st.session_state.personas.items()):
            if persona_id not in st.session_state.get('selected_personas', [persona_id]):
                continue
            
            color = colors[i % len(colors)]
            
            # Add marker for persona (demo coordinates)
            lat_offset = np.random.uniform(-0.1, 0.1)
            lng_offset = np.random.uniform(-0.1, 0.1)
            
            folium.CircleMarker(
                location=[
                    VIZ_CONFIG.DEFAULT_MAP_CENTER[0] + lat_offset,
                    VIZ_CONFIG.DEFAULT_MAP_CENTER[1] + lng_offset
                ],
                radius=max(10, persona.estimated_population / 1000),
                popup=folium.Popup(
                    f"""
                    <div style="width: 200px;">
                        <h4 style="color: {color};">{persona.persona_name}</h4>
                        <p><strong>Type:</strong> {persona.persona_type.value}</p>
                        <p><strong>Population:</strong> {persona.estimated_population:,}</p>
                        <p><strong>Market Value:</strong> ${persona.market_value:,.0f}</p>
                        <p><strong>Effectiveness:</strong> {persona.targeting_effectiveness:.1%}</p>
                    </div>
                    """,
                    max_width=250
                ),
                color=color,
                fill=True,
                fillColor=color,
                fillOpacity=0.6,
                weight=2
            ).add_to(m)
        
        # Display map
        map_data = st_folium(m, width=700, height=500)
        
        # Show clicked persona details
        if map_data['last_object_clicked_popup']:
            st.info("💡 Click on map markers to see detailed persona information")
    
    def render_cluster_analysis(self):
        """Render cluster analysis visualizations"""
        st.subheader("📊 Cluster Analysis")
        
        if not st.session_state.personas:
            st.warning("No cluster data available")
            return
        
        # Create tabs for different analyses
        tab1, tab2, tab3 = st.tabs(["📈 Market Value", "👥 Population", "🎯 Effectiveness"])
        
        with tab1:
            self.render_market_value_analysis()
        
        with tab2:
            self.render_population_analysis()
        
        with tab3:
            self.render_effectiveness_analysis()
    
    def render_market_value_analysis(self):
        """Render market value analysis"""
        personas = st.session_state.personas
        selected_personas = st.session_state.get('selected_personas', list(personas.keys()))
        
        # Filter personas
        filtered_personas = {k: v for k, v in personas.items() if k in selected_personas}
        
        if not filtered_personas:
            st.warning("No personas selected for analysis")
            return
        
        # Create market value comparison
        persona_names = [p.persona_name for p in filtered_personas.values()]
        market_values = [p.market_value for p in filtered_personas.values()]
        populations = [p.estimated_population for p in filtered_personas.values()]
        
        fig = go.Figure()
        
        # Add bar chart
        fig.add_trace(go.Bar(
            x=persona_names,
            y=market_values,
            text=[f'${v:,.0f}' for v in market_values],
            textposition='auto',
            marker_color=VIZ_CONFIG.CLUSTER_COLORS[:len(filtered_personas)],
            hovertemplate='<b>%{x}</b><br>Market Value: $%{y:,.0f}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Market Value by Persona",
            xaxis_title="Consumer Persona",
            yaxis_title="Market Value ($)",
            template="plotly_white",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Add insights
        total_value = sum(market_values)
        highest_value_persona = max(filtered_personas.values(), key=lambda x: x.market_value)
        
        st.info(f"""
        **Market Value Insights:**
        - Total market value across selected personas: ${total_value:,.0f}
        - Highest value persona: {highest_value_persona.persona_name} (${highest_value_persona.market_value:,.0f})
        - Average value per persona: ${total_value/len(filtered_personas):,.0f}
        """)
    
    def render_population_analysis(self):
        """Render population distribution analysis"""
        personas = st.session_state.personas
        selected_personas = st.session_state.get('selected_personas', list(personas.keys()))
        
        # Filter personas
        filtered_personas = {k: v for k, v in personas.items() if k in selected_personas}
        
        if not filtered_personas:
            st.warning("No personas selected for analysis")
            return
        
        # Create population pie chart
        persona_names = [p.persona_name for p in filtered_personas.values()]
        populations = [p.estimated_population for p in filtered_personas.values()]
        
        fig = go.Figure(data=[go.Pie(
            labels=persona_names,
            values=populations,
            hole=0.3,
            marker_colors=VIZ_CONFIG.CLUSTER_COLORS[:len(filtered_personas)],
            textinfo='label+percent',
            hovertemplate='<b>%{label}</b><br>Population: %{value:,}<br>Percentage: %{percent}<extra></extra>'
        )])
        
        fig.update_layout(
            title="Population Distribution by Persona",
            template="plotly_white",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Population statistics
        total_population = sum(populations)
        largest_segment = max(filtered_personas.values(), key=lambda x: x.estimated_population)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Population", f"{total_population:,}")
            st.metric("Largest Segment", largest_segment.persona_name)
        
        with col2:
            st.metric("Average Segment Size", f"{total_population/len(filtered_personas):,.0f}")
            st.metric("Largest Segment Size", f"{largest_segment.estimated_population:,}")
    
    def render_effectiveness_analysis(self):
        """Render targeting effectiveness analysis"""
        personas = st.session_state.personas
        selected_personas = st.session_state.get('selected_personas', list(personas.keys()))
        
        # Filter personas
        filtered_personas = {k: v for k, v in personas.items() if k in selected_personas}
        
        if not filtered_personas:
            st.warning("No personas selected for analysis")
            return
        
        # Create effectiveness scatter plot
        persona_names = [p.persona_name for p in filtered_personas.values()]
        effectiveness = [p.targeting_effectiveness for p in filtered_personas.values()]
        market_values = [p.market_value for p in filtered_personas.values()]
        populations = [p.estimated_population for p in filtered_personas.values()]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=populations,
            y=effectiveness,
            mode='markers+text',
            text=persona_names,
            textposition='top center',
            marker=dict(
                size=[mv/5000 for mv in market_values],  # Size by market value
                color=VIZ_CONFIG.CLUSTER_COLORS[:len(filtered_personas)],
                sizemode='diameter',
                sizeref=2.*max([mv/5000 for mv in market_values])/(40.**2),
                sizemin=4,
                line=dict(width=2, color='white')
            ),
            hovertemplate='<b>%{text}</b><br>Population: %{x:,}<br>Effectiveness: %{y:.1%}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Targeting Effectiveness vs Population Size",
            xaxis_title="Population Size",
            yaxis_title="Targeting Effectiveness",
            template="plotly_white",
            height=400
        )
        
        # Add effectiveness threshold line
        fig.add_hline(y=0.7, line_dash="dash", line_color="red", 
                     annotation_text="70% Effectiveness Threshold")
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Effectiveness insights
        avg_effectiveness = np.mean(effectiveness)
        high_effectiveness = [p for p in filtered_personas.values() if p.targeting_effectiveness > 0.7]
        
        st.info(f"""
        **Targeting Effectiveness Insights:**
        - Average effectiveness across personas: {avg_effectiveness:.1%}
        - High-effectiveness personas (>70%): {len(high_effectiveness)} out of {len(filtered_personas)}
        - Most targetable: {max(filtered_personas.values(), key=lambda x: x.targeting_effectiveness).persona_name}
        """)
    
    def render_time_series_analysis(self):
        """Render time series and seasonal analysis"""
        st.subheader("📅 Seasonal Trends Analysis")
        
        if not st.session_state.personas:
            st.warning("No seasonal data available")
            return
        
        personas = st.session_state.personas
        selected_personas = st.session_state.get('selected_personas', list(personas.keys()))
        
        # Filter personas
        filtered_personas = {k: v for k, v in personas.items() if k in selected_personas}
        
        if not filtered_personas:
            st.warning("No personas selected for analysis")
            return
        
        # Create seasonal trends chart
        seasons = ['Spring', 'Summer', 'Fall', 'Winter']
        fig = go.Figure()
        
        for i, (persona_id, persona) in enumerate(filtered_personas.items()):
            seasonal_values = [
                persona.seasonal_trends.get(season.lower(), 1.0) 
                for season in seasons
            ]
            
            fig.add_trace(go.Scatter(
                x=seasons,
                y=seasonal_values,
                mode='lines+markers',
                name=persona.persona_name,
                line=dict(
                    color=VIZ_CONFIG.CLUSTER_COLORS[i % len(VIZ_CONFIG.CLUSTER_COLORS)],
                    width=3
                ),
                marker=dict(size=8),
                hovertemplate='<b>%{fullData.name}</b><br>Season: %{x}<br>Usage Multiplier: %{y:.2f}<extra></extra>'
            ))
        
        fig.update_layout(
            title="Seasonal Usage Patterns by Persona",
            xaxis_title="Season",
            yaxis_title="Usage Multiplier",
            template="plotly_white",
            height=400,
            hovermode='x unified'
        )
        
        # Add baseline
        fig.add_hline(y=1.0, line_dash="dash", line_color="gray", 
                     annotation_text="Baseline (1.0)")
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Seasonal insights
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Peak Seasons by Persona:**")
            for persona in filtered_personas.values():
                peak_season = max(persona.seasonal_trends.items(), key=lambda x: x[1])
                st.write(f"• {persona.persona_name}: {peak_season[0].title()} ({peak_season[1]:.1f}x)")
        
        with col2:
            st.write("**Seasonal Recommendations:**")
            st.write("• Summer: Focus on leisure and tourism marketing")
            st.write("• Winter: Promote indoor alternatives and maintenance")
            st.write("• Spring/Fall: Target commuter acquisition")
    
    def render_business_intelligence(self):
        """Render business intelligence and opportunities"""
        st.subheader("💼 Business Intelligence")
        
        if not st.session_state.opportunities:
            st.warning("No business opportunities data available")
            return
        
        opportunities = st.session_state.opportunities
        
        # Create tabs for different BI views
        tab1, tab2, tab3 = st.tabs(["🎯 Opportunities", "💰 ROI Analysis", "📋 Recommendations"])
        
        with tab1:
            self.render_opportunities_overview(opportunities)
        
        with tab2:
            self.render_roi_analysis(opportunities)
        
        with tab3:
            self.render_strategic_recommendations()
    
    def render_opportunities_overview(self, opportunities):
        """Render business opportunities overview"""
        # Opportunities summary
        col1, col2, col3 = st.columns(3)
        
        total_market_size = sum(opp.estimated_market_size for opp in opportunities)
        avg_roi = np.mean([float(opp.expected_roi.split('-')[0].replace('%', '')) for opp in opportunities])
        
        with col1:
            st.metric("Total Opportunity Value", f"${total_market_size:,.0f}")
        with col2:
            st.metric("Number of Opportunities", len(opportunities))
        with col3:
            st.metric("Average Expected ROI", f"{avg_roi:.1f}%")
        
        # Opportunities table
        st.write("**Identified Business Opportunities:**")
        
        opp_data = []
        for opp in opportunities:
            opp_data.append({
                'Opportunity': opp.opportunity_type,
                'Market Size': f"${opp.estimated_market_size:,.0f}",
                'Investment': opp.investment_level,
                'Expected ROI': opp.expected_roi,
                'Timeline': opp.implementation_timeline,
                'Target Segments': ', '.join(opp.target_segments)
            })
        
        opp_df = pd.DataFrame(opp_data)
        st.dataframe(opp_df, use_container_width=True)
        
        # Detailed opportunity cards
        st.write("**Opportunity Details:**")
        for i, opp in enumerate(opportunities):
            with st.expander(f"📈 {opp.opportunity_type}"):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.write(f"**Description:** {opp.description}")
                    st.write(f"**Target Segments:** {', '.join(opp.target_segments)}")
                    st.write(f"**Key Metrics:** {', '.join(opp.key_metrics)}")
                
                with col2:
                    st.metric("Market Size", f"${opp.estimated_market_size:,.0f}")
                    st.metric("Expected ROI", opp.expected_roi)
                    st.metric("Timeline", opp.implementation_timeline)
                    
                    # Investment level indicator
                    investment_color = {'Low': 'green', 'Medium': 'orange', 'High': 'red'}
                    st.markdown(f"**Investment Level:** :{investment_color.get(opp.investment_level, 'blue')}[{opp.investment_level}]")
    
    def render_roi_analysis(self, opportunities):
        """Render ROI analysis visualization"""
        if not opportunities:
            st.warning("No opportunities available for ROI analysis")
            return
        
        # Parse ROI data
        opp_names = [opp.opportunity_type for opp in opportunities]
        market_sizes = [opp.estimated_market_size for opp in opportunities]
        investment_levels = [opp.investment_level for opp in opportunities]
        
        # Parse ROI ranges
        roi_data = []
        for opp in opportunities:
            roi_str = opp.expected_roi.replace('%', '')
            if '-' in roi_str:
                low, high = map(float, roi_str.split('-'))
                roi_data.append((low, high, (low + high) / 2))
            else:
                roi_val = float(roi_str)
                roi_data.append((roi_val, roi_val, roi_val))
        
        # Create ROI vs Market Size scatter plot
        fig = go.Figure()
        
        for i, (name, (low, high, mid), market_size, inv_level) in enumerate(zip(opp_names, roi_data, market_sizes, investment_levels)):
            color = VIZ_CONFIG.SPENDING_COLORS[i % len(VIZ_CONFIG.SPENDING_COLORS)]
            
            fig.add_trace(go.Scatter(
                x=[market_size],
                y=[mid],
                mode='markers+text',
                text=[name],
                textposition='top center',
                marker=dict(
                    size=20,
                    color=color,
                    line=dict(width=2, color='white')
                ),
                error_y=dict(
                    type='data',
                    symmetric=False,
                    array=[high - mid],
                    arrayminus=[mid - low],
                    visible=True
                ),
                name=name,
                hovertemplate=f'<b>{name}</b><br>Market Size: ${market_size:,.0f}<br>ROI Range: {low:.1f}% - {high:.1f}%<br>Investment: {inv_level}<extra></extra>'
            ))
        
        fig.update_layout(
            title="ROI vs Market Size Analysis",
            xaxis_title="Market Size ($)",
            yaxis_title="Expected ROI (%)",
            template="plotly_white",
            height=500,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # ROI recommendations
        best_roi_opp = max(opportunities, key=lambda x: float(x.expected_roi.split('-')[-1].replace('%', '')))
        largest_market_opp = max(opportunities, key=lambda x: x.estimated_market_size)
        
        col1, col2 = st.columns(2)
        with col1:
            st.success(f"**Highest ROI Opportunity:** {best_roi_opp.opportunity_type}")
            st.write(f"Expected ROI: {best_roi_opp.expected_roi}")
        
        with col2:
            st.info(f"**Largest Market Opportunity:** {largest_market_opp.opportunity_type}")
            st.write(f"Market Size: ${largest_market_opp.estimated_market_size:,.0f}")
    
    def render_strategic_recommendations(self):
        """Render strategic recommendations"""
        if not st.session_state.insights:
            st.warning("No insights available for recommendations")
            return
        
        insights = st.session_state.insights
        
        # Key insights
        st.write("**🔍 Key Market Insights:**")
        for insight in insights.get('key_insights', []):
            st.write(f"• {insight}")
        
        st.write("**📈 Strategic Recommendations:**")
        recommendations = insights.get('strategic_recommendations', [
            "Focus on high-value urban commuter segment for premium services",
            "Develop seasonal marketing campaigns aligned with usage patterns",
            "Invest in technology infrastructure to support growing demand",
            "Create partnerships with local businesses for cross-promotion"
        ])
        
        for i, rec in enumerate(recommendations, 1):
            st.write(f"{i}. {rec}")
        
        # Action items
        st.write("**✅ Immediate Action Items:**")
        action_items = [
            "Conduct user surveys to validate persona characteristics",
            "Develop pilot programs for top 2 business opportunities",
            "Establish KPI tracking for persona-specific metrics",
            "Create targeted marketing campaigns for each persona"
        ]
        
        for item in action_items:
            st.checkbox(item, key=f"action_{item[:20]}")
    
    def render_persona_cards(self):
        """Render detailed persona cards"""
        st.subheader("👥 Detailed Persona Profiles")
        
        if not st.session_state.personas:
            st.warning("No persona data available")
            return
        
        personas = st.session_state.personas
        selected_personas = st.session_state.get('selected_personas', list(personas.keys()))
        
        # Filter personas
        filtered_personas = {k: v for k, v in personas.items() if k in selected_personas}
        
        for i, (persona_id, persona) in enumerate(filtered_personas.items()):
            color = VIZ_CONFIG.CLUSTER_COLORS[i % len(VIZ_CONFIG.CLUSTER_COLORS)]
            
            # Create persona card
            st.markdown(f"""
            <div class="persona-card" style="border-left: 5px solid {color};">
                <h3 style="color: {color}; margin-top: 0;">{persona.persona_name}</h3>
                <p style="color: #666; font-style: italic;">{persona.persona_type.value}</p>
                <p style="line-height: 1.6;">{persona.description}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Persona details in columns
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.write("**📊 Key Metrics**")
                st.metric("Population", f"{persona.estimated_population:,}")
                st.metric("Market Value", f"${persona.market_value:,.0f}")
                st.metric("Effectiveness", f"{persona.targeting_effectiveness:.1%}")
            
            with col2:
                st.write("**🎯 Motivations**")
                for motivation in persona.key_motivations[:3]:
                    st.write(f"• {motivation}")
                
                st.write("**📱 Preferred Channels**")
                for channel in persona.preferred_channels[:3]:
                    st.write(f"• {channel}")
            
            with col3:
                st.write("**⚠️ Pain Points**")
                for pain_point in persona.pain_points[:3]:
                    st.write(f"• {pain_point}")
                
                st.write("**💡 Opportunities**")
                for opportunity in persona.product_opportunities[:3]:
                    st.write(f"• {opportunity}")
            
            # Spending profile chart
            if hasattr(persona, 'spending_profile') and persona.spending_profile.get('category_preferences'):
                st.write("**💰 Spending Profile**")
                
                categories = list(persona.spending_profile['category_preferences'].keys())
                values = list(persona.spending_profile['category_preferences'].values())
                
                fig = go.Figure(data=[go.Bar(
                    x=categories,
                    y=values,
                    marker_color=color,
                    text=[f"{v:.1%}" for v in values],
                    textposition='auto'
                )])
                
                fig.update_layout(
                    title=f"Spending Categories - {persona.persona_name}",
                    xaxis_title="Category",
                    yaxis_title="Percentage of Spending",
                    template="plotly_white",
                    height=300
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            st.markdown("---")
    
    def download_results(self):
        """Generate and provide download for all results"""
        try:
            # Create a zip file with all results
            zip_buffer = BytesIO()
            
            with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                # Add personas data
                if st.session_state.personas:
                    personas_json = json.dumps({
                        pid: {
                            'persona_name': p.persona_name,
                            'persona_type': p.persona_type.value,
                            'estimated_population': p.estimated_population,
                            'market_value': p.market_value,
                            'targeting_effectiveness': p.targeting_effectiveness,
                            'description': p.description,
                            'key_motivations': p.key_motivations,
                            'pain_points': p.pain_points,
                            'seasonal_trends': p.seasonal_trends
                        } for pid, p in st.session_state.personas.items()
                    }, indent=2)
                    zip_file.writestr('personas.json', personas_json)
                
                # Add opportunities data
                if st.session_state.opportunities:
                    opportunities_json = json.dumps([{
                        'opportunity_type': opp.opportunity_type,
                        'description': opp.description,
                        'target_segments': opp.target_segments,
                        'estimated_market_size': opp.estimated_market_size,
                        'investment_level': opp.investment_level,
                        'expected_roi': opp.expected_roi,
                        'implementation_timeline': opp.implementation_timeline,
                        'key_metrics': opp.key_metrics
                    } for opp in st.session_state.opportunities], indent=2)
                    zip_file.writestr('opportunities.json', opportunities_json)
                
                # Add insights data
                if st.session_state.insights:
                    insights_json = json.dumps(st.session_state.insights, indent=2, default=str)
                    zip_file.writestr('insights.json', insights_json)
                
                # Add executive summary
                summary = self.generate_executive_summary()
                zip_file.writestr('executive_summary.txt', summary)
            
            zip_buffer.seek(0)
            
            # Provide download
            st.download_button(
                label="📥 Download Complete Analysis",
                data=zip_buffer.getvalue(),
                file_name=f"consumer_segmentation_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.zip",
                mime="application/zip"
            )
            
        except Exception as e:
            st.error(f"Error generating download: {str(e)}")
    
    def generate_executive_summary(self):
        """Generate executive summary text"""
        summary = []
        summary.append("CONSUMER SEGMENTATION ANALYSIS - EXECUTIVE SUMMARY")
        summary.append("=" * 60)
        summary.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        summary.append("")
        
        if st.session_state.insights:
            market_overview = st.session_state.insights.get('market_overview', {})
            summary.append("MARKET OVERVIEW")
            summary.append("-" * 30)
            summary.append(f"Total Addressable Market: ${market_overview.get('total_addressable_market', 0):,.0f}")
            summary.append(f"Total Population: {market_overview.get('total_population', 0):,}")
            summary.append(f"Number of Segments: {market_overview.get('number_of_segments', 0)}")
            summary.append(f"Average Targeting Effectiveness: {market_overview.get('average_targeting_effectiveness', 0):.1%}")
            summary.append("")
        
        if st.session_state.personas:
            summary.append("IDENTIFIED PERSONAS")
            summary.append("-" * 30)
            for persona in st.session_state.personas.values():
                summary.append(f"• {persona.persona_name} ({persona.persona_type.value})")
                summary.append(f"  Population: {persona.estimated_population:,}")
                summary.append(f"  Market Value: ${persona.market_value:,.0f}")
                summary.append(f"  Effectiveness: {persona.targeting_effectiveness:.1%}")
                summary.append("")
        
        if st.session_state.opportunities:
            summary.append("BUSINESS OPPORTUNITIES")
            summary.append("-" * 30)
            for opp in st.session_state.opportunities:
                summary.append(f"• {opp.opportunity_type}")
                summary.append(f"  Market Size: ${opp.estimated_market_size:,.0f}")
                summary.append(f"  Expected ROI: {opp.expected_roi}")
                summary.append(f"  Timeline: {opp.implementation_timeline}")
                summary.append("")
        
        if st.session_state.insights and 'key_insights' in st.session_state.insights:
            summary.append("KEY INSIGHTS")
            summary.append("-" * 30)
            for insight in st.session_state.insights['key_insights']:
                summary.append(f"• {insight}")
            summary.append("")
        
        summary.append("=" * 60)
        return "\n".join(summary)
    
    def run(self):
        """Main dashboard execution"""
        # Render sidebar
        self.render_sidebar()
        
        # Render main content
        self.render_main_header()
        
        if not st.session_state.data_loaded:
            st.warning("⏳ Loading analysis data... This may take a few moments.")
            return
        
        # Overview metrics
        self.render_overview_metrics()
        
        st.markdown("---")
        
        # Main content tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🗺️ Geographic View", 
            "📊 Cluster Analysis", 
            "📅 Seasonal Trends", 
            "💼 Business Intelligence", 
            "👥 Persona Profiles"
        ])
        
        with tab1:
            self.render_interactive_map()
        
        with tab2:
            self.render_cluster_analysis()
        
        with tab3:
            self.render_time_series_analysis()
        
        with tab4:
            self.render_business_intelligence()
        
        with tab5:
            self.render_persona_cards()


def main():
    """Main function to run the dashboard"""
    dashboard = ConsumerSegmentationDashboard()
    dashboard.run()


if __name__ == "__main__":
    main()