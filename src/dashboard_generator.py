"""
Interactive Dashboard Generator for Consumer Segmentation Business Intelligence
Creates comprehensive web-based dashboards for persona insights and business recommendations
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import folium
from folium import plugins
import json
from typing import Dict, List, Any, Optional
from loguru import logger
from datetime import datetime
import base64
from io import BytesIO

from config import VIZ_CONFIG
from src.persona_generator import PersonaGenerator, ConsumerPersona, BusinessOpportunity


class DashboardGenerator:
    """Generate interactive dashboards for business intelligence insights"""
    
    def __init__(self):
        self.color_palette = VIZ_CONFIG.CLUSTER_COLORS
        self.spending_colors = VIZ_CONFIG.SPENDING_COLORS
        self.dashboard_components = {}
        
    def create_persona_overview_dashboard(self, personas: Dict[str, ConsumerPersona]) -> go.Figure:
        """
        Create comprehensive persona overview dashboard
        """
        logger.info("Creating persona overview dashboard")
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=[
                'Market Value by Persona',
                'Population Distribution',
                'Targeting Effectiveness',
                'Seasonal Trends',
                'Persona Types Distribution',
                'Key Metrics Summary'
            ],
            specs=[
                [{"type": "bar"}, {"type": "pie"}, {"type": "scatter"}],
                [{"type": "scatter"}, {"type": "pie"}, {"type": "table"}]
            ]
        )
        
        # Prepare data
        persona_names = [p.persona_name for p in personas.values()]
        market_values = [p.market_value for p in personas.values()]
        populations = [p.estimated_population for p in personas.values()]
        effectiveness = [p.targeting_effectiveness for p in personas.values()]
        persona_types = [p.persona_type.value for p in personas.values()]
        
        # 1. Market Value by Persona (Bar Chart)
        fig.add_trace(
            go.Bar(
                x=persona_names,
                y=market_values,
                name='Market Value',
                marker_color=self.color_palette[:len(personas)],
                text=[f'${v:,.0f}' for v in market_values],
                textposition='auto'
            ),
            row=1, col=1
        )
        
        # 2. Population Distribution (Pie Chart)
        fig.add_trace(
            go.Pie(
                labels=persona_names,
                values=populations,
                name='Population',
                marker_colors=self.color_palette[:len(personas)]
            ),
            row=1, col=2
        )
        
        # 3. Targeting Effectiveness (Scatter Plot)
        fig.add_trace(
            go.Scatter(
                x=populations,
                y=effectiveness,
                mode='markers+text',
                text=persona_names,
                textposition='top center',
                marker=dict(
                    size=[mv/10000 for mv in market_values],  # Size by market value
                    color=self.color_palette[:len(personas)],
                    sizemode='diameter',
                    sizeref=2.*max([mv/10000 for mv in market_values])/(40.**2),
                    sizemin=4
                ),
                name='Effectiveness'
            ),
            row=1, col=3
        )
        
        # 4. Seasonal Trends (Line Plot)
        seasons = ['spring', 'summer', 'fall', 'winter']
        for i, (persona_id, persona) in enumerate(personas.items()):
            seasonal_values = [persona.seasonal_trends.get(season, 1.0) for season in seasons]
            fig.add_trace(
                go.Scatter(
                    x=seasons,
                    y=seasonal_values,
                    mode='lines+markers',
                    name=persona.persona_name,
                    line=dict(color=self.color_palette[i % len(self.color_palette)])
                ),
                row=2, col=1
            )
        
        # 5. Persona Types Distribution (Pie Chart)
        type_counts = {}
        for persona_type in persona_types:
            type_counts[persona_type] = type_counts.get(persona_type, 0) + 1
        
        fig.add_trace(
            go.Pie(
                labels=list(type_counts.keys()),
                values=list(type_counts.values()),
                name='Types',
                marker_colors=self.spending_colors[:len(type_counts)]
            ),
            row=2, col=2
        )
        
        # 6. Key Metrics Table
        table_data = []
        for persona in personas.values():
            table_data.append([
                persona.persona_name,
                f"${persona.market_value:,.0f}",
                f"{persona.estimated_population:,}",
                f"{persona.targeting_effectiveness:.1%}",
                persona.persona_type.value
            ])
        
        fig.add_trace(
            go.Table(
                header=dict(
                    values=['Persona', 'Market Value', 'Population', 'Effectiveness', 'Type'],
                    fill_color='lightblue',
                    align='left'
                ),
                cells=dict(
                    values=list(zip(*table_data)),
                    fill_color='white',
                    align='left'
                )
            ),
            row=2, col=3
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            title_text="Consumer Persona Overview Dashboard",
            showlegend=False
        )
        
        # Update axis labels
        fig.update_xaxes(title_text="Personas", row=1, col=1)
        fig.update_yaxes(title_text="Market Value ($)", row=1, col=1)
        fig.update_xaxes(title_text="Population", row=1, col=3)
        fig.update_yaxes(title_text="Targeting Effectiveness", row=1, col=3)
        fig.update_xaxes(title_text="Season", row=2, col=1)
        fig.update_yaxes(title_text="Usage Multiplier", row=2, col=1)
        
        return fig
    
    def create_business_opportunity_dashboard(self, opportunities: List[BusinessOpportunity]) -> go.Figure:
        """
        Create business opportunity analysis dashboard
        """
        logger.info("Creating business opportunity dashboard")
        
        if not opportunities:
            # Create empty dashboard
            fig = go.Figure()
            fig.add_annotation(
                text="No business opportunities data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False, font=dict(size=20)
            )
            return fig
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Market Size by Opportunity',
                'Investment Level Distribution',
                'ROI vs Timeline Analysis',
                'Implementation Priority Matrix'
            ],
            specs=[
                [{"type": "bar"}, {"type": "pie"}],
                [{"type": "scatter"}, {"type": "scatter"}]
            ]
        )
        
        # Prepare data
        opp_types = [opp.opportunity_type for opp in opportunities]
        market_sizes = [opp.estimated_market_size for opp in opportunities]
        investment_levels = [opp.investment_level for opp in opportunities]
        roi_values = [float(opp.expected_roi.split('-')[0].replace('%', '')) for opp in opportunities]
        timelines = [opp.implementation_timeline for opp in opportunities]
        
        # Convert timelines to numeric (months)
        timeline_months = []
        for timeline in timelines:
            if 'month' in timeline.lower():
                months = int(timeline.split('-')[0])
                timeline_months.append(months)
            else:
                timeline_months.append(12)  # Default to 12 months
        
        # 1. Market Size by Opportunity
        fig.add_trace(
            go.Bar(
                x=opp_types,
                y=market_sizes,
                name='Market Size',
                marker_color=self.color_palette[:len(opportunities)],
                text=[f'${v:,.0f}' for v in market_sizes],
                textposition='auto'
            ),
            row=1, col=1
        )
        
        # 2. Investment Level Distribution
        investment_counts = {}
        for level in investment_levels:
            investment_counts[level] = investment_counts.get(level, 0) + 1
        
        fig.add_trace(
            go.Pie(
                labels=list(investment_counts.keys()),
                values=list(investment_counts.values()),
                name='Investment Levels',
                marker_colors=self.spending_colors[:len(investment_counts)]
            ),
            row=1, col=2
        )
        
        # 3. ROI vs Timeline Analysis
        fig.add_trace(
            go.Scatter(
                x=timeline_months,
                y=roi_values,
                mode='markers+text',
                text=opp_types,
                textposition='top center',
                marker=dict(
                    size=[ms/10000 for ms in market_sizes],
                    color=self.color_palette[:len(opportunities)],
                    sizemode='diameter',
                    sizeref=2.*max([ms/10000 for ms in market_sizes])/(40.**2),
                    sizemin=4
                ),
                name='ROI vs Timeline'
            ),
            row=2, col=1
        )
        
        # 4. Implementation Priority Matrix (ROI vs Investment)
        investment_numeric = {'Low': 1, 'Medium': 2, 'High': 3}
        investment_scores = [investment_numeric.get(level, 2) for level in investment_levels]
        
        fig.add_trace(
            go.Scatter(
                x=investment_scores,
                y=roi_values,
                mode='markers+text',
                text=opp_types,
                textposition='top center',
                marker=dict(
                    size=[ms/10000 for ms in market_sizes],
                    color=self.spending_colors[:len(opportunities)],
                    sizemode='diameter',
                    sizeref=2.*max([ms/10000 for ms in market_sizes])/(40.**2),
                    sizemin=4
                ),
                name='Priority Matrix'
            ),
            row=2, col=2
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            title_text="Business Opportunity Analysis Dashboard",
            showlegend=False
        )
        
        # Update axis labels
        fig.update_xaxes(title_text="Opportunity Type", row=1, col=1)
        fig.update_yaxes(title_text="Market Size ($)", row=1, col=1)
        fig.update_xaxes(title_text="Timeline (Months)", row=2, col=1)
        fig.update_yaxes(title_text="Expected ROI (%)", row=2, col=1)
        fig.update_xaxes(title_text="Investment Level", row=2, col=2, tickvals=[1, 2, 3], ticktext=['Low', 'Medium', 'High'])
        fig.update_yaxes(title_text="Expected ROI (%)", row=2, col=2)
        
        return fig
    
    def create_persona_detail_cards(self, personas: Dict[str, ConsumerPersona]) -> str:
        """
        Create detailed persona cards in HTML format
        """
        logger.info("Creating detailed persona cards")
        
        html_cards = []
        
        for i, (persona_id, persona) in enumerate(personas.items()):
            color = self.color_palette[i % len(self.color_palette)]
            
            # Create spending profile chart
            spending_chart = self._create_mini_spending_chart(persona)
            
            # Create seasonal trend chart
            seasonal_chart = self._create_mini_seasonal_chart(persona)
            
            card_html = f"""
            <div class="persona-card" style="
                border: 2px solid {color};
                border-radius: 10px;
                padding: 20px;
                margin: 20px;
                background: linear-gradient(135deg, {color}20, white);
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                max-width: 600px;
            ">
                <div class="persona-header" style="border-bottom: 1px solid {color}; padding-bottom: 15px; margin-bottom: 15px;">
                    <h2 style="color: {color}; margin: 0; font-size: 24px;">{persona.persona_name}</h2>
                    <p style="color: #666; margin: 5px 0; font-style: italic;">{persona.persona_type.value}</p>
                    <div style="display: flex; justify-content: space-between; margin-top: 10px;">
                        <span style="background: {color}; color: white; padding: 5px 10px; border-radius: 15px; font-size: 12px;">
                            ${persona.market_value:,.0f} Market Value
                        </span>
                        <span style="background: #f0f0f0; color: #333; padding: 5px 10px; border-radius: 15px; font-size: 12px;">
                            {persona.estimated_population:,} Users
                        </span>
                        <span style="background: #e8f5e8; color: #2d5a2d; padding: 5px 10px; border-radius: 15px; font-size: 12px;">
                            {persona.targeting_effectiveness:.1%} Effectiveness
                        </span>
                    </div>
                </div>
                
                <div class="persona-description" style="margin-bottom: 20px;">
                    <p style="line-height: 1.6; color: #444;">{persona.description}</p>
                </div>
                
                <div class="persona-details" style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
                    <div class="motivations">
                        <h4 style="color: {color}; margin-bottom: 10px;">Key Motivations</h4>
                        <ul style="margin: 0; padding-left: 20px;">
                            {''.join([f'<li style="margin-bottom: 5px;">{motivation}</li>' for motivation in persona.key_motivations[:3]])}
                        </ul>
                    </div>
                    
                    <div class="pain-points">
                        <h4 style="color: {color}; margin-bottom: 10px;">Pain Points</h4>
                        <ul style="margin: 0; padding-left: 20px;">
                            {''.join([f'<li style="margin-bottom: 5px;">{pain}</li>' for pain in persona.pain_points[:3]])}
                        </ul>
                    </div>
                </div>
                
                <div class="persona-charts" style="margin-top: 20px;">
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px;">
                        <div>
                            <h5 style="color: {color}; margin-bottom: 10px;">Spending Profile</h5>
                            {spending_chart}
                        </div>
                        <div>
                            <h5 style="color: {color}; margin-bottom: 10px;">Seasonal Trends</h5>
                            {seasonal_chart}
                        </div>
                    </div>
                </div>
                
                <div class="business-recommendations" style="margin-top: 20px; padding-top: 15px; border-top: 1px solid #eee;">
                    <h4 style="color: {color}; margin-bottom: 10px;">Marketing Strategies</h4>
                    <div style="display: flex; flex-wrap: wrap; gap: 5px;">
                        {''.join([f'<span style="background: {color}20; color: {color}; padding: 3px 8px; border-radius: 10px; font-size: 11px;">{strategy}</span>' for strategy in persona.marketing_strategies[:4]])}
                    </div>
                </div>
            </div>
            """
            html_cards.append(card_html)
        
        # Combine all cards
        full_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Consumer Persona Details</title>
            <style>
                body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 20px; background: #f8f9fa; }}
                .container {{ max-width: 1200px; margin: 0 auto; }}
                h1 {{ text-align: center; color: #333; margin-bottom: 30px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Consumer Persona Detailed Analysis</h1>
                {''.join(html_cards)}
            </div>
        </body>
        </html>
        """
        
        return full_html
    
    def _create_mini_spending_chart(self, persona: ConsumerPersona) -> str:
        """Create mini spending profile chart"""
        
        spending_profile = persona.spending_profile
        categories = list(spending_profile.get('category_preferences', {}).keys())
        values = list(spending_profile.get('category_preferences', {}).values())
        
        if not categories:
            return "<p style='color: #999; font-style: italic;'>No spending data available</p>"
        
        # Create simple bar chart using CSS
        max_value = max(values) if values else 1
        bars = []
        
        for i, (category, value) in enumerate(zip(categories, values)):
            width = (value / max_value) * 100
            color = self.spending_colors[i % len(self.spending_colors)]
            bars.append(f"""
                <div style="margin-bottom: 5px;">
                    <div style="font-size: 11px; margin-bottom: 2px;">{category.title()}</div>
                    <div style="background: #f0f0f0; border-radius: 3px; height: 15px;">
                        <div style="background: {color}; height: 100%; width: {width}%; border-radius: 3px; display: flex; align-items: center; padding-left: 5px;">
                            <span style="font-size: 10px; color: white;">{value:.1%}</span>
                        </div>
                    </div>
                </div>
            """)
        
        return f"<div>{''.join(bars)}</div>"
    
    def _create_mini_seasonal_chart(self, persona: ConsumerPersona) -> str:
        """Create mini seasonal trend chart"""
        
        seasonal_trends = persona.seasonal_trends
        seasons = ['Spring', 'Summer', 'Fall', 'Winter']
        values = [seasonal_trends.get(season.lower(), 1.0) for season in seasons]
        
        # Create simple line chart using CSS
        max_value = max(values)
        min_value = min(values)
        range_value = max_value - min_value if max_value != min_value else 1
        
        points = []
        for i, (season, value) in enumerate(zip(seasons, values)):
            x = (i / (len(seasons) - 1)) * 100
            y = 100 - ((value - min_value) / range_value) * 80  # Invert Y and scale to 80% height
            points.append(f"{x},{y}")
        
        polyline = " ".join(points)
        
        chart_html = f"""
        <div style="position: relative; height: 60px; background: #f8f9fa; border-radius: 5px; padding: 10px;">
            <svg width="100%" height="100%" viewBox="0 0 100 100" style="overflow: visible;">
                <polyline points="{polyline}" 
                         fill="none" 
                         stroke="{self.color_palette[0]}" 
                         stroke-width="2"/>
                {' '.join([f'<circle cx="{(i/(len(seasons)-1))*100}" cy="{100-((v-min_value)/range_value)*80}" r="2" fill="{self.color_palette[0]}"/>' for i, v in enumerate(values)])}
            </svg>
            <div style="display: flex; justify-content: space-between; margin-top: 5px; font-size: 9px; color: #666;">
                {''.join([f'<span>{season[:3]}</span>' for season in seasons])}
            </div>
        </div>
        """
        
        return chart_html
    
    def create_geographic_persona_map(self, personas: Dict[str, ConsumerPersona],
                                    county_data: pd.DataFrame = None) -> folium.Map:
        """
        Create geographic map showing persona distribution
        """
        logger.info("Creating geographic persona distribution map")
        
        # Initialize map
        m = folium.Map(
            location=VIZ_CONFIG.DEFAULT_MAP_CENTER,
            zoom_start=VIZ_CONFIG.DEFAULT_ZOOM,
            tiles='OpenStreetMap'
        )
        
        # Create persona color mapping
        persona_colors = {}
        for i, (persona_id, persona) in enumerate(personas.items()):
            persona_colors[persona_id] = self.color_palette[i % len(self.color_palette)]
        
        # Add persona markers/regions
        for persona_id, persona in personas.items():
            color = persona_colors[persona_id]
            
            # Add cluster information to map
            if hasattr(persona, 'cluster_ids') and persona.cluster_ids:
                # Create marker for persona center
                # This would be enhanced with actual geographic data
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
        
        # Add legend
        legend_html = self._create_map_legend(personas, persona_colors)
        m.get_root().html.add_child(folium.Element(legend_html))
        
        return m
    
    def _create_map_legend(self, personas: Dict[str, ConsumerPersona], 
                          persona_colors: Dict[str, str]) -> str:
        """Create HTML legend for persona map"""
        
        legend_items = []
        for persona_id, persona in personas.items():
            color = persona_colors[persona_id]
            legend_items.append(f"""
                <div style="margin-bottom: 8px;">
                    <span style="display: inline-block; width: 15px; height: 15px; background: {color}; border-radius: 50%; margin-right: 8px;"></span>
                    <span style="font-size: 12px; font-weight: bold;">{persona.persona_name}</span>
                    <br>
                    <span style="font-size: 10px; color: #666; margin-left: 23px;">
                        {persona.estimated_population:,} users | ${persona.market_value:,.0f}
                    </span>
                </div>
            """)
        
        legend_html = f"""
        <div style="
            position: fixed; 
            top: 10px; right: 10px; 
            width: 250px; 
            background: white; 
            border: 2px solid #ccc; 
            border-radius: 5px;
            padding: 15px;
            z-index: 9999; 
            font-family: Arial, sans-serif;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        ">
            <h4 style="margin: 0 0 15px 0; color: #333;">Consumer Personas</h4>
            {''.join(legend_items)}
        </div>
        """
        
        return legend_html
    
    def create_roi_analysis_dashboard(self, opportunities: List[BusinessOpportunity]) -> go.Figure:
        """
        Create ROI analysis dashboard for business opportunities
        """
        logger.info("Creating ROI analysis dashboard")
        
        if not opportunities:
            fig = go.Figure()
            fig.add_annotation(
                text="No opportunities data available for ROI analysis",
                xref="paper", yref="paper",
                x=0.5, y=0.5, xanchor='center', yanchor='middle',
                showarrow=False, font=dict(size=16)
            )
            return fig
        
        # Create ROI comparison chart
        fig = go.Figure()
        
        # Prepare data
        opp_names = [opp.opportunity_type for opp in opportunities]
        roi_ranges = []
        market_sizes = [opp.estimated_market_size for opp in opportunities]
        investment_levels = [opp.investment_level for opp in opportunities]
        
        # Parse ROI ranges
        for opp in opportunities:
            roi_str = opp.expected_roi.replace('%', '')
            if '-' in roi_str:
                low, high = map(float, roi_str.split('-'))
                roi_ranges.append((low, high, (low + high) / 2))
            else:
                roi_val = float(roi_str)
                roi_ranges.append((roi_val, roi_val, roi_val))
        
        # Create ROI range chart
        for i, (opp_name, (low, high, mid), market_size, inv_level) in enumerate(zip(opp_names, roi_ranges, market_sizes, investment_levels)):
            color = self.color_palette[i % len(self.color_palette)]
            
            # Add ROI range bar
            fig.add_trace(go.Bar(
                x=[opp_name],
                y=[mid],
                error_y=dict(
                    type='data',
                    symmetric=False,
                    array=[high - mid],
                    arrayminus=[mid - low]
                ),
                name=f'{opp_name} ROI',
                marker_color=color,
                text=f'{mid:.1f}%<br>${market_size:,.0f}<br>{inv_level}',
                textposition='auto',
                hovertemplate=f"""
                <b>{opp_name}</b><br>
                ROI Range: {low:.1f}% - {high:.1f}%<br>
                Market Size: ${market_size:,.0f}<br>
                Investment: {inv_level}<br>
                <extra></extra>
                """
            ))
        
        # Update layout
        fig.update_layout(
            title="ROI Analysis by Business Opportunity",
            xaxis_title="Business Opportunity",
            yaxis_title="Expected ROI (%)",
            showlegend=False,
            height=500,
            template="plotly_white"
        )
        
        return fig
    
    def export_dashboard_bundle(self, personas: Dict[str, ConsumerPersona],
                              opportunities: List[BusinessOpportunity],
                              market_insights: Dict[str, Any],
                              filename_prefix: str = 'dashboard_bundle') -> Dict[str, str]:
        """
        Export complete dashboard bundle with all visualizations
        """
        logger.info("Exporting complete dashboard bundle")
        
        exported_files = {}
        
        # 1. Persona Overview Dashboard
        persona_overview = self.create_persona_overview_dashboard(personas)
        persona_overview_file = f"{filename_prefix}_persona_overview.html"
        from config import DATA_CONFIG
        persona_overview_path = DATA_CONFIG.PROCESSED_DATA_DIR / persona_overview_file
        persona_overview.write_html(str(persona_overview_path))
        exported_files['persona_overview'] = str(persona_overview_path)
        
        # 2. Business Opportunity Dashboard
        opportunity_dashboard = self.create_business_opportunity_dashboard(opportunities)
        opportunity_file = f"{filename_prefix}_opportunities.html"
        opportunity_path = DATA_CONFIG.PROCESSED_DATA_DIR / opportunity_file
        opportunity_dashboard.write_html(str(opportunity_path))
        exported_files['opportunities'] = str(opportunity_path)
        
        # 3. ROI Analysis Dashboard
        roi_dashboard = self.create_roi_analysis_dashboard(opportunities)
        roi_file = f"{filename_prefix}_roi_analysis.html"
        roi_path = DATA_CONFIG.PROCESSED_DATA_DIR / roi_file
        roi_dashboard.write_html(str(roi_path))
        exported_files['roi_analysis'] = str(roi_path)
        
        # 4. Detailed Persona Cards
        persona_cards = self.create_persona_detail_cards(personas)
        cards_file = f"{filename_prefix}_persona_cards.html"
        cards_path = DATA_CONFIG.PROCESSED_DATA_DIR / cards_file
        with open(cards_path, 'w', encoding='utf-8') as f:
            f.write(persona_cards)
        exported_files['persona_cards'] = str(cards_path)
        
        # 5. Geographic Map
        persona_map = self.create_geographic_persona_map(personas)
        map_file = f"{filename_prefix}_geographic_map.html"
        map_path = DATA_CONFIG.PROCESSED_DATA_DIR / map_file
        persona_map.save(str(map_path))
        exported_files['geographic_map'] = str(map_path)
        
        # 6. Executive Dashboard (Combined)
        executive_dashboard = self._create_executive_dashboard(personas, opportunities, market_insights)
        exec_file = f"{filename_prefix}_executive_dashboard.html"
        exec_path = DATA_CONFIG.PROCESSED_DATA_DIR / exec_file
        with open(exec_path, 'w', encoding='utf-8') as f:
            f.write(executive_dashboard)
        exported_files['executive_dashboard'] = str(exec_path)
        
        logger.info(f"Exported {len(exported_files)} dashboard files")
        return exported_files
    
    def _create_executive_dashboard(self, personas: Dict[str, ConsumerPersona],
                                  opportunities: List[BusinessOpportunity],
                                  market_insights: Dict[str, Any]) -> str:
        """
        Create comprehensive executive dashboard HTML
        """
        # Calculate key metrics
        total_market_value = sum(p.market_value for p in personas.values())
        total_population = sum(p.estimated_population for p in personas.values())
        avg_effectiveness = np.mean([p.targeting_effectiveness for p in personas.values()])
        
        # Get top opportunities
        top_opportunities = sorted(opportunities, key=lambda x: x.estimated_market_size, reverse=True)[:3]
        
        # Create summary cards
        summary_cards = f"""
        <div class="summary-cards" style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 20px; margin-bottom: 30px;">
            <div class="card" style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; text-align: center;">
                <h3 style="margin: 0; font-size: 14px; opacity: 0.9;">Total Market Value</h3>
                <p style="margin: 10px 0 0 0; font-size: 28px; font-weight: bold;">${total_market_value:,.0f}</p>
            </div>
            <div class="card" style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); color: white; padding: 20px; border-radius: 10px; text-align: center;">
                <h3 style="margin: 0; font-size: 14px; opacity: 0.9;">Total Users</h3>
                <p style="margin: 10px 0 0 0; font-size: 28px; font-weight: bold;">{total_population:,}</p>
            </div>
            <div class="card" style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); color: white; padding: 20px; border-radius: 10px; text-align: center;">
                <h3 style="margin: 0; font-size: 14px; opacity: 0.9;">Avg. Targeting Effectiveness</h3>
                <p style="margin: 10px 0 0 0; font-size: 28px; font-weight: bold;">{avg_effectiveness:.1%}</p>
            </div>
            <div class="card" style="background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%); color: white; padding: 20px; border-radius: 10px; text-align: center;">
                <h3 style="margin: 0; font-size: 14px; opacity: 0.9;">Identified Segments</h3>
                <p style="margin: 10px 0 0 0; font-size: 28px; font-weight: bold;">{len(personas)}</p>
            </div>
        </div>
        """
        
        # Create top personas section
        top_personas = sorted(personas.values(), key=lambda x: x.market_value, reverse=True)[:3]
        personas_section = f"""
        <div class="top-personas" style="margin-bottom: 30px;">
            <h2 style="color: #333; margin-bottom: 20px;">Top Consumer Personas</h2>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px;">
                {''.join([f'''
                <div style="border: 1px solid #ddd; border-radius: 8px; padding: 15px; background: white;">
                    <h4 style="color: {self.color_palette[i]}; margin: 0 0 10px 0;">{persona.persona_name}</h4>
                    <p style="color: #666; margin: 0 0 10px 0; font-size: 14px;">{persona.persona_type.value}</p>
                    <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                        <span style="font-size: 12px;"><strong>Market Value:</strong> ${persona.market_value:,.0f}</span>
                        <span style="font-size: 12px;"><strong>Users:</strong> {persona.estimated_population:,}</span>
                    </div>
                    <p style="font-size: 12px; line-height: 1.4; margin: 0;">{persona.description[:150]}...</p>
                </div>
                ''' for i, persona in enumerate(top_personas)])}
            </div>
        </div>
        """
        
        # Create opportunities section
        opportunities_section = f"""
        <div class="top-opportunities" style="margin-bottom: 30px;">
            <h2 style="color: #333; margin-bottom: 20px;">Top Business Opportunities</h2>
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px;">
                {''.join([f'''
                <div style="border: 1px solid #ddd; border-radius: 8px; padding: 15px; background: white;">
                    <h4 style="color: {self.spending_colors[i]}; margin: 0 0 10px 0;">{opp.opportunity_type}</h4>
                    <p style="color: #666; margin: 0 0 10px 0; font-size: 14px;">{opp.description}</p>
                    <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px; font-size: 12px;">
                        <span><strong>Market Size:</strong> ${opp.estimated_market_size:,.0f}</span>
                        <span><strong>Expected ROI:</strong> {opp.expected_roi}</span>
                        <span><strong>Investment:</strong> {opp.investment_level}</span>
                        <span><strong>Timeline:</strong> {opp.implementation_timeline}</span>
                    </div>
                </div>
                ''' for i, opp in enumerate(top_opportunities)])}
            </div>
        </div>
        """
        
        # Create insights section
        insights_section = ""
        if market_insights and 'key_insights' in market_insights:
            insights_list = ''.join([f'<li style="margin-bottom: 10px; line-height: 1.5;">{insight}</li>' 
                                   for insight in market_insights['key_insights']])
            insights_section = f"""
            <div class="key-insights" style="margin-bottom: 30px;">
                <h2 style="color: #333; margin-bottom: 20px;">Key Insights</h2>
                <div style="background: #f8f9fa; border-left: 4px solid #007bff; padding: 20px; border-radius: 5px;">
                    <ul style="margin: 0; padding-left: 20px;">
                        {insights_list}
                    </ul>
                </div>
            </div>
            """
        
        # Combine everything
        executive_html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Executive Dashboard - Consumer Segmentation</title>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <style>
                body {{ 
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                    margin: 0; 
                    padding: 20px; 
                    background: #f5f6fa; 
                    line-height: 1.6;
                }}
                .container {{ 
                    max-width: 1200px; 
                    margin: 0 auto; 
                    background: white; 
                    padding: 30px; 
                    border-radius: 10px; 
                    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                }}
                h1 {{ 
                    text-align: center; 
                    color: #2c3e50; 
                    margin-bottom: 30px; 
                    font-size: 32px;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                    background-clip: text;
                }}
                .timestamp {{
                    text-align: center;
                    color: #666;
                    font-size: 14px;
                    margin-bottom: 30px;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Executive Dashboard</h1>
                <div class="timestamp">Consumer Segmentation Business Intelligence | Generated: {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</div>
                
                {summary_cards}
                {personas_section}
                {opportunities_section}
                {insights_section}
                
                <div style="text-align: center; margin-top: 40px; padding-top: 20px; border-top: 1px solid #eee; color: #666; font-size: 12px;">
                    Generated by Consumer Segmentation Business Intelligence System
                </div>
            </div>
        </body>
        </html>
        """
        
        return executive_html


def main():
    """Demo function for dashboard generation"""
    from src.persona_generator import PersonaGenerator, PersonaType, ConsumerPersona, BusinessOpportunity
    
    # Create sample personas for demo
    sample_personas = {
        'persona_1': ConsumerPersona(
            persona_id='persona_1',
            persona_name='Urban Commuter Pro',
            persona_type=PersonaType.URBAN_COMMUTER,
            cluster_ids=[1],
            estimated_population=15000,
            median_income=75000,
            age_distribution={'18-34': 0.4, '35-54': 0.4, '55+': 0.2},
            education_level={'bachelor_plus': 0.6, 'high_school': 0.4},
            mobility_profile={'usage_intensity': 'high', 'member_ratio': 0.85},
            spending_profile={'spending_level': 'high', 'category_preferences': {'restaurants': 0.3, 'retail': 0.25, 'entertainment': 0.15}},
            temporal_patterns={'schedule_type': 'structured'},
            market_value=150000,
            targeting_effectiveness=0.85,
            seasonal_trends={'spring': 1.0, 'summer': 1.2, 'fall': 0.9, 'winter': 0.7},
            description='Highly structured individuals who rely on bike-sharing for daily commuting.',
            key_motivations=['Reliable transportation', 'Time efficiency', 'Cost savings'],
            preferred_channels=['Mobile app', 'Email', 'LinkedIn'],
            pain_points=['Rush hour availability', 'Weather dependency'],
            marketing_strategies=['Commuter packages', 'Corporate partnerships'],
            product_opportunities=['Reserved bikes', 'Express stations'],
            infrastructure_needs=['High-capacity stations', 'Transit integration']
        )
    }
    
    # Create sample opportunities
    sample_opportunities = [
        BusinessOpportunity(
            opportunity_type='Premium Services',
            description='Develop premium service tier for high-value customers',
            target_segments=['Urban Commuter Pro'],
            estimated_market_size=200000,
            investment_level='Medium',
            expected_roi='20-30%',
            implementation_timeline='6-12 months',
            key_metrics=['Customer lifetime value', 'Premium conversion rate']
        )
    ]
    
    # Create sample market insights
    sample_insights = {
        'market_overview': {
            'total_addressable_market': 500000,
            'total_population': 25000,
            'average_targeting_effectiveness': 0.75,
            'number_of_segments': 3
        },
        'key_insights': [
            'Urban commuters represent the highest value opportunity',
            'Summer presents significant growth potential',
            'Technology integration is critical for competitive advantage'
        ]
    }
    
    # Initialize dashboard generator
    dashboard_gen = DashboardGenerator()
    
    # Generate dashboards
    logger.info("Starting dashboard generation demo...")
    
    # Create individual dashboards
    persona_overview = dashboard_gen.create_persona_overview_dashboard(sample_personas)
    opportunity_dashboard = dashboard_gen.create_business_opportunity_dashboard(sample_opportunities)
    persona_cards = dashboard_gen.create_persona_detail_cards(sample_personas)
    
    # Export dashboard bundle
    exported_files = dashboard_gen.export_dashboard_bundle(
        sample_personas, sample_opportunities, sample_insights
    )
    
    print(f"\nDashboard Generation Summary:")
    print(f"- Generated {len(exported_files)} dashboard files")
    print(f"- Files created: {list(exported_files.keys())}")
    
    logger.info("Dashboard generation demo completed successfully!")
    
    return exported_files


if __name__ == "__main__":
    main()