"""
Main Streamlit Application for Consumer Segmentation Analysis
Optimized for Netlify deployment with static generation capabilities
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import json
from pathlib import Path
import sys
import os

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import our modules
try:
    from src.data_loader import DataLoader
    from src.spatial_processor import SpatialProcessor
    from src.feature_engineering import FeatureEngineer
    from src.clustering_engine import ClusteringEngine
    from src.persona_generator import PersonaGenerator, PersonaType
    from src.dashboard_generator import DashboardGenerator
    from src.extensions import PredictiveModeling, AdvancedVisualizations, ExportCapabilities
    from config import VIZ_CONFIG, MODEL_CONFIG
except ImportError as e:
    st.error(f"Import error: {e}")
    st.stop()


class StreamlitApp:
    """Main Streamlit application class"""
    
    def __init__(self):
        self.setup_page_config()
        self.initialize_session_state()
        
    def setup_page_config(self):
        """Configure Streamlit page"""
        st.set_page_config(
            page_title="Consumer Segmentation Analytics",
            page_icon="🎯",
            layout="wide",
            initial_sidebar_state="expanded",
            menu_items={
                'Get Help': 'https://github.com/your-org/consumer-segmentation',
                'Report a bug': 'https://github.com/your-org/consumer-segmentation/issues',
                'About': """
                # Consumer Segmentation Analytics
                
                Advanced analytics platform for understanding consumer behavior 
                through mobility and spending patterns.
                
                Built with Streamlit, Plotly, and scikit-learn.
                """
            }
        )
        
        # Custom CSS
        st.markdown("""
        <style>
        .main-header {
            background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
            padding: 2rem;
            border-radius: 10px;
            margin-bottom: 2rem;
            text-align: center;
        }
        .main-header h1 {
            color: white;
            margin: 0;
            font-size: 2.5rem;
        }
        .main-header p {
            color: rgba(255,255,255,0.9);
            margin: 0.5rem 0 0 0;
            font-size: 1.1rem;
        }
        .metric-card {
            background: white;
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            border-left: 4px solid #667eea;
            margin-bottom: 1rem;
        }
        .persona-card {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 1.5rem;
            border-radius: 10px;
            margin: 1rem 0;
            border: 1px solid #e1e5e9;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 2px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            padding-left: 20px;
            padding-right: 20px;
        }
        </style>
        """, unsafe_allow_html=True)
    
    def initialize_session_state(self):
        """Initialize session state variables"""
        if 'data_loaded' not in st.session_state:
            st.session_state.data_loaded = False
        if 'analysis_complete' not in st.session_state:
            st.session_state.analysis_complete = False
        if 'personas' not in st.session_state:
            st.session_state.personas = None
        if 'opportunities' not in st.session_state:
            st.session_state.opportunities = None
        if 'insights' not in st.session_state:
            st.session_state.insights = None
    
    def render_header(self):
        """Render main header"""
        st.markdown("""
        <div class="main-header">
            <h1>🎯 Consumer Segmentation Analytics</h1>
            <p>Advanced Analytics for Mobility & Spending Patterns</p>
        </div>
        """, unsafe_allow_html=True)
    
    def render_sidebar(self):
        """Render sidebar controls"""
        st.sidebar.title("🎛️ Controls")
        
        # Data loading section
        st.sidebar.subheader("📊 Data Configuration")
        
        year = st.sidebar.selectbox("Year", [2023, 2022, 2021], index=0)
        month = st.sidebar.selectbox("Month", list(range(1, 13)), index=5)
        
        counties = st.sidebar.multiselect(
            "Counties",
            ["17031", "36061", "06037", "48201", "04013"],
            default=["17031", "36061"],
            help="Select counties for analysis"
        )
        
        # Analysis parameters
        st.sidebar.subheader("⚙️ Analysis Parameters")
        
        clustering_algorithms = st.sidebar.multiselect(
            "Clustering Algorithms",
            ["hdbscan", "kmeans"],
            default=["hdbscan", "kmeans"]
        )
        
        min_cluster_size = st.sidebar.slider(
            "Min Cluster Size",
            min_value=10, max_value=200, value=100
        )
        
        # Advanced options
        with st.sidebar.expander("🔬 Advanced Options"):
            enable_predictive = st.checkbox("Enable Predictive Modeling", value=False)
            enable_privacy_check = st.checkbox("Privacy & Ethics Check", value=True)
            export_format = st.selectbox("Export Format", ["HTML", "PDF", "Excel"])
        
        # Action buttons
        st.sidebar.subheader("🚀 Actions")
        
        if st.sidebar.button("🔄 Run Analysis", type="primary"):
            self.run_analysis(year, month, counties, clustering_algorithms, min_cluster_size)
        
        if st.session_state.analysis_complete:
            if st.sidebar.button("📥 Export Results"):
                self.export_results(export_format.lower())
        
        # Information
        st.sidebar.subheader("ℹ️ Information")
        st.sidebar.info("""
        This dashboard analyzes consumer behavior patterns using:
        - Bike-share mobility data
        - Consumer spending patterns
        - Geographic demographics
        - Advanced clustering algorithms
        """)
        
        return {
            'year': year, 'month': month, 'counties': counties,
            'clustering_algorithms': clustering_algorithms,
            'min_cluster_size': min_cluster_size,
            'enable_predictive': enable_predictive,
            'enable_privacy_check': enable_privacy_check,
            'export_format': export_format
        }
    
    def run_analysis(self, year, month, counties, algorithms, min_cluster_size):
        """Run the complete analysis pipeline"""
        with st.spinner("Running consumer segmentation analysis..."):
            try:
                # Initialize components
                loader = DataLoader()
                processor = SpatialProcessor()
                engineer = FeatureEngineer()
                clustering_engine = ClusteringEngine()
                persona_generator = PersonaGenerator()
                
                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Step 1: Load data
                status_text.text("Loading data...")
                trips_df = loader.download_divvy_data(year, month)
                spending_df = loader.download_spending_data(counties)
                boundaries = loader.load_county_boundaries(counties)
                progress_bar.progress(20)
                
                # Step 2: Spatial processing
                status_text.text("Processing spatial data...")
                stations_gdf = processor.extract_stations_from_trips(trips_df)
                stations_with_counties = processor.assign_stations_to_counties(stations_gdf, boundaries)
                county_mobility = processor.aggregate_trips_to_county_level(trips_df, stations_with_counties)
                progress_bar.progress(40)
                
                # Step 3: Feature engineering
                status_text.text("Engineering features...")
                engineered_features, pipeline_results = engineer.create_feature_pipeline(
                    county_mobility, spending_df, trips_df
                )
                progress_bar.progress(60)
                
                # Step 4: Clustering
                status_text.text("Running clustering analysis...")
                clustering_results = clustering_engine.run_complete_clustering_analysis(
                    engineered_features, algorithms=algorithms
                )
                progress_bar.progress(80)
                
                # Step 5: Persona generation
                status_text.text("Generating personas...")
                personas, opportunities, insights = persona_generator.run_complete_persona_analysis(
                    clustering_results, engineered_features
                )
                progress_bar.progress(100)
                
                # Store results
                st.session_state.personas = personas
                st.session_state.opportunities = opportunities
                st.session_state.insights = insights
                st.session_state.clustering_results = clustering_results
                st.session_state.engineered_features = engineered_features
                st.session_state.analysis_complete = True
                
                status_text.text("Analysis complete!")
                st.success("✅ Analysis completed successfully!")
                
            except Exception as e:
                st.error(f"❌ Analysis failed: {str(e)}")
                st.exception(e)
    
    def render_overview_tab(self):
        """Render overview tab"""
        if not st.session_state.analysis_complete:
            st.info("👈 Run analysis from the sidebar to see results")
            return
        
        personas = st.session_state.personas
        insights = st.session_state.insights
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        market_overview = insights.get('market_overview', {})
        
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
            """.format(len(personas)), unsafe_allow_html=True)
        
        # Key insights
        st.subheader("🔍 Key Insights")
        for insight in insights.get('key_insights', []):
            st.info(f"💡 {insight}")
        
        # Persona overview
        st.subheader("👥 Consumer Personas")
        
        for i, (persona_id, persona) in enumerate(personas.items()):
            color = VIZ_CONFIG.CLUSTER_COLORS[i % len(VIZ_CONFIG.CLUSTER_COLORS)]
            
            with st.expander(f"📊 {persona.persona_name}", expanded=True):
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    st.write(f"**Type:** {persona.persona_type.value}")
                    st.write(f"**Description:** {persona.description}")
                    
                    st.write("**Key Motivations:**")
                    for motivation in persona.key_motivations[:3]:
                        st.write(f"• {motivation}")
                
                with col2:
                    st.metric("Population", f"{persona.estimated_population:,}")
                    st.metric("Market Value", f"${persona.market_value:,.0f}")
                
                with col3:
                    st.metric("Effectiveness", f"{persona.targeting_effectiveness:.1%}")
                    
                    # Seasonal trends mini chart
                    seasons = list(persona.seasonal_trends.keys())
                    values = list(persona.seasonal_trends.values())
                    
                    fig = go.Figure(data=go.Bar(x=seasons, y=values, marker_color=color))
                    fig.update_layout(
                        title="Seasonal Trends",
                        height=200,
                        margin=dict(l=0, r=0, t=30, b=0)
                    )
                    st.plotly_chart(fig, use_container_width=True)
    
    def render_analytics_tab(self):
        """Render analytics tab"""
        if not st.session_state.analysis_complete:
            st.info("👈 Run analysis from the sidebar to see results")
            return
        
        personas = st.session_state.personas
        clustering_results = st.session_state.clustering_results
        
        # Market value analysis
        st.subheader("💰 Market Value Analysis")
        
        persona_names = [p.persona_name for p in personas.values()]
        market_values = [p.market_value for p in personas.values()]
        populations = [p.estimated_population for p in personas.values()]
        effectiveness = [p.targeting_effectiveness for p in personas.values()]
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = go.Figure(data=[go.Bar(
                x=persona_names,
                y=market_values,
                marker_color=VIZ_CONFIG.CLUSTER_COLORS[:len(personas)],
                text=[f'${v:,.0f}' for v in market_values],
                textposition='auto'
            )])
            fig.update_layout(
                title="Market Value by Persona",
                xaxis_title="Persona",
                yaxis_title="Market Value ($)",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = go.Figure(data=[go.Pie(
                labels=persona_names,
                values=populations,
                hole=0.3,
                marker_colors=VIZ_CONFIG.CLUSTER_COLORS[:len(personas)]
            )])
            fig.update_layout(
                title="Population Distribution",
                height=400
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Effectiveness vs Population scatter
        st.subheader("🎯 Targeting Effectiveness Analysis")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=populations,
            y=effectiveness,
            mode='markers+text',
            text=persona_names,
            textposition='top center',
            marker=dict(
                size=[mv/5000 for mv in market_values],
                color=VIZ_CONFIG.CLUSTER_COLORS[:len(personas)],
                sizemode='diameter',
                sizeref=2.*max([mv/5000 for mv in market_values])/(40.**2),
                sizemin=4
            ),
            hovertemplate='<b>%{text}</b><br>Population: %{x:,}<br>Effectiveness: %{y:.1%}<extra></extra>'
        ))
        
        fig.update_layout(
            title="Targeting Effectiveness vs Population Size",
            xaxis_title="Population Size",
            yaxis_title="Targeting Effectiveness",
            height=500
        )
        
        fig.add_hline(y=0.7, line_dash="dash", line_color="red", 
                     annotation_text="70% Effectiveness Threshold")
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Clustering validation metrics
        st.subheader("📊 Clustering Validation")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Algorithm Performance:**")
            for algorithm, results in clustering_results.get('algorithms', {}).items():
                if 'validation_metrics' in results:
                    metrics = results['validation_metrics']['internal_metrics']
                    st.write(f"**{algorithm.upper()}:**")
                    st.write(f"• Silhouette Score: {metrics.get('silhouette_score', 'N/A'):.3f}")
                    st.write(f"• Clusters Found: {results['cluster_results']['n_clusters']}")
                    st.write(f"• Outliers: {results['cluster_results']['n_outliers']}")
        
        with col2:
            # Feature importance
            if clustering_results.get('algorithms', {}).get('hdbscan', {}).get('feature_importance'):
                importance = clustering_results['algorithms']['hdbscan']['feature_importance']
                top_features = list(importance['top_features'])[:10]
                importance_values = [importance['feature_importance'][f] for f in top_features]
                
                fig = go.Figure(data=[go.Bar(
                    x=importance_values,
                    y=top_features,
                    orientation='h',
                    marker_color='lightblue'
                )])
                fig.update_layout(
                    title="Top Feature Importance",
                    xaxis_title="Importance Score",
                    height=400
                )
                st.plotly_chart(fig, use_container_width=True)
    
    def render_opportunities_tab(self):
        """Render business opportunities tab"""
        if not st.session_state.analysis_complete:
            st.info("👈 Run analysis from the sidebar to see results")
            return
        
        opportunities = st.session_state.opportunities
        
        if not opportunities:
            st.warning("No business opportunities identified")
            return
        
        # Opportunities overview
        st.subheader("💼 Business Opportunities Overview")
        
        col1, col2, col3 = st.columns(3)
        
        total_market_size = sum(opp.estimated_market_size for opp in opportunities)
        avg_roi = np.mean([float(opp.expected_roi.split('-')[0].replace('%', '')) for opp in opportunities])
        
        with col1:
            st.metric("Total Opportunity Value", f"${total_market_size:,.0f}")
        with col2:
            st.metric("Number of Opportunities", len(opportunities))
        with col3:
            st.metric("Average Expected ROI", f"{avg_roi:.1f}%")
        
        # Opportunities details
        st.subheader("📋 Opportunity Details")
        
        for i, opp in enumerate(opportunities):
            with st.expander(f"🎯 {opp.opportunity_type}", expanded=True):
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
                    investment_colors = {'Low': '🟢', 'Medium': '🟡', 'High': '🔴'}
                    st.write(f"**Investment Level:** {investment_colors.get(opp.investment_level, '⚪')} {opp.investment_level}")
        
        # ROI Analysis
        st.subheader("📈 ROI Analysis")
        
        opp_names = [opp.opportunity_type for opp in opportunities]
        market_sizes = [opp.estimated_market_size for opp in opportunities]
        investment_levels = [opp.investment_level for opp in opportunities]
        
        # Parse ROI ranges
        roi_data = []
        for opp in opportunities:
            roi_str = opp.expected_roi.replace('%', '')
            if '-' in roi_str:
                low, high = map(float, roi_str.split('-'))
                roi_data.append((low + high) / 2)
            else:
                roi_data.append(float(roi_str))
        
        fig = go.Figure()
        
        for i, (name, market_size, roi, inv_level) in enumerate(zip(opp_names, market_sizes, roi_data, investment_levels)):
            color = VIZ_CONFIG.SPENDING_COLORS[i % len(VIZ_CONFIG.SPENDING_COLORS)]
            
            fig.add_trace(go.Scatter(
                x=[market_size],
                y=[roi],
                mode='markers+text',
                text=[name],
                textposition='top center',
                marker=dict(
                    size=20,
                    color=color,
                    line=dict(width=2, color='white')
                ),
                name=name,
                hovertemplate=f'<b>{name}</b><br>Market Size: ${market_size:,.0f}<br>Expected ROI: {roi:.1f}%<br>Investment: {inv_level}<extra></extra>'
            ))
        
        fig.update_layout(
            title="ROI vs Market Size Analysis",
            xaxis_title="Market Size ($)",
            yaxis_title="Expected ROI (%)",
            height=500,
            showlegend=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_advanced_tab(self):
        """Render advanced analytics tab"""
        if not st.session_state.analysis_complete:
            st.info("👈 Run analysis from the sidebar to see results")
            return
        
        st.subheader("🔬 Advanced Analytics")
        
        # Predictive modeling section
        st.write("### 🤖 Predictive Modeling")
        
        if st.button("Train Spending Predictor"):
            with st.spinner("Training predictive model..."):
                try:
                    predictive_model = PredictiveModeling()
                    
                    # Prepare sample data for demonstration
                    mobility_data = pd.DataFrame({
                        'county_fips': ['17031', '17043', '36061'],
                        'total_trips': [10000, 5000, 15000],
                        'member_ratio': [0.8, 0.6, 0.7]
                    })
                    
                    spending_data = pd.DataFrame({
                        'county_fips': ['17031', '17031', '17043', '17043', '36061', '36061'],
                        'year': [2023, 2023, 2023, 2023, 2023, 2023],
                        'month': [6, 7, 6, 7, 6, 7],
                        'category': ['restaurants', 'restaurants', 'restaurants', 'restaurants', 'restaurants', 'restaurants'],
                        'spending_amount': [100000, 110000, 50000, 55000, 150000, 160000]
                    })
                    
                    features, targets = predictive_model.prepare_prediction_data(mobility_data, spending_data)
                    
                    if len(features) > 0:
                        results = predictive_model.train_spending_predictor(features, targets, 'random_forest')
                        
                        st.success("✅ Model trained successfully!")
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric("R² Score", f"{results['performance']['r2_score']:.3f}")
                            st.metric("RMSE", f"{results['performance']['rmse']:,.0f}")
                        
                        with col2:
                            st.metric("CV R² Mean", f"{results['performance']['cv_r2_mean']:.3f}")
                            st.metric("CV R² Std", f"{results['performance']['cv_r2_std']:.3f}")
                        
                        # Feature importance
                        if results['feature_importance']:
                            st.write("**Top Feature Importance:**")
                            importance_df = pd.DataFrame(
                                list(results['feature_importance'].items())[:5],
                                columns=['Feature', 'Importance']
                            )
                            st.dataframe(importance_df)
                    else:
                        st.warning("Insufficient data for model training")
                        
                except Exception as e:
                    st.error(f"Model training failed: {str(e)}")
        
        # Advanced visualizations
        st.write("### 📊 Advanced Visualizations")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Generate 3D Cluster Plot"):
                try:
                    advanced_viz = AdvancedVisualizations()
                    
                    # Sample data for 3D plot
                    features_df = pd.DataFrame({
                        'county_fips': ['17031', '17043', '36061'],
                        'total_trips': [10000, 5000, 15000],
                        'spending_restaurants': [100000, 50000, 150000],
                        'member_ratio': [0.8, 0.6, 0.7]
                    })
                    cluster_labels = np.array([0, 1, 0])
                    feature_names = ['total_trips', 'spending_restaurants', 'member_ratio']
                    
                    fig = advanced_viz.create_3d_cluster_plot(features_df, cluster_labels, feature_names)
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"3D plot generation failed: {str(e)}")
        
        with col2:
            if st.button("Generate Sankey Diagram"):
                try:
                    advanced_viz = AdvancedVisualizations()
                    
                    # Sample data for Sankey
                    mobility_data = pd.DataFrame({
                        'county_fips': ['17031', '17043', '36061'],
                        'total_trips': [10000, 5000, 15000]
                    })
                    spending_data = pd.DataFrame({
                        'county_fips': ['17031', '17043', '36061'],
                        'total_spending': [250000, 125000, 375000]
                    })
                    
                    fig = advanced_viz.create_sankey_mobility_spending(mobility_data, spending_data)
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Sankey diagram generation failed: {str(e)}")
        
        # Privacy and ethics
        st.write("### 🔒 Privacy & Ethics")
        
        if st.button("Run Privacy Assessment"):
            try:
                from src.extensions import PrivacyEthicsFramework
                
                privacy_framework = PrivacyEthicsFramework()
                
                # Sample data for privacy check
                sample_data = pd.DataFrame({
                    'county_fips': ['17031'] * 100 + ['17043'] * 50,
                    'total_trips': np.random.normal(1000, 200, 150),
                    'median_income': np.random.normal(60000, 15000, 150)
                })
                
                privacy_checks = privacy_framework.check_data_anonymization(sample_data)
                
                st.write("**Privacy Assessment Results:**")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Direct Identifiers:** {'❌' if privacy_checks['has_direct_identifiers'] else '✅'}")
                    st.write(f"**Quasi-Identifiers:** {'❌' if privacy_checks['has_quasi_identifiers'] else '✅'}")
                
                with col2:
                    st.write(f"**K-Anonymity Level:** {privacy_checks['k_anonymity_level']}")
                    st.write(f"**Privacy Status:** {'⚠️ Needs Review' if privacy_checks['has_direct_identifiers'] else '✅ Good'}")
                
                st.write("**Recommendations:**")
                for rec in privacy_checks['recommendations']:
                    st.write(f"• {rec}")
                    
            except Exception as e:
                st.error(f"Privacy assessment failed: {str(e)}")
    
    def export_results(self, format_type):
        """Export analysis results"""
        try:
            export_capabilities = ExportCapabilities()
            
            personas = st.session_state.personas
            opportunities = st.session_state.opportunities
            insights = st.session_state.insights
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            if format_type == 'html':
                output_path = f"consumer_segmentation_report_{timestamp}.html"
                exported_file = export_capabilities.export_static_html(
                    personas, opportunities, insights, output_path
                )
                
            elif format_type == 'pdf':
                output_path = f"consumer_segmentation_report_{timestamp}.pdf"
                exported_file = export_capabilities.generate_pdf_report(
                    personas, opportunities, insights, output_path
                )
                
            elif format_type == 'excel':
                output_path = f"consumer_segmentation_report_{timestamp}.xlsx"
                exported_file = export_capabilities.export_to_excel(
                    personas, opportunities, insights, output_path
                )
            
            if exported_file:
                st.success(f"✅ Results exported to {exported_file}")
                
                # Provide download link
                with open(exported_file, 'rb') as f:
                    st.download_button(
                        label=f"📥 Download {format_type.upper()} Report",
                        data=f.read(),
                        file_name=output_path,
                        mime=f"application/{format_type}"
                    )
            else:
                st.error("❌ Export failed")
                
        except Exception as e:
            st.error(f"Export failed: {str(e)}")
    
    def run(self):
        """Run the main application"""
        self.render_header()
        
        # Sidebar
        config = self.render_sidebar()
        
        # Main content tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Overview", 
            "📈 Analytics", 
            "💼 Opportunities", 
            "🔬 Advanced"
        ])
        
        with tab1:
            self.render_overview_tab()
        
        with tab2:
            self.render_analytics_tab()
        
        with tab3:
            self.render_opportunities_tab()
        
        with tab4:
            self.render_advanced_tab()
        
        # Footer
        st.markdown("---")
        st.markdown("""
        <div style="text-align: center; color: #666; padding: 20px;">
            <p>Consumer Segmentation Analytics Platform | Built with Streamlit & Advanced ML</p>
            <p>🎯 Transforming mobility and spending data into actionable business insights</p>
        </div>
        """, unsafe_allow_html=True)


def main():
    """Main application entry point"""
    app = StreamlitApp()
    app.run()


if __name__ == "__main__":
    main()