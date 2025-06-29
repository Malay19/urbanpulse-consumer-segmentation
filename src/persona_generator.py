"""
Business Intelligence and Persona Generation Module
Transforms clustering results into actionable business insights and consumer personas
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from loguru import logger
from datetime import datetime
import json
import requests
from dataclasses import dataclass
from enum import Enum

from config import DATA_CONFIG, CENSUS_API_KEY
from src.utils import calculate_data_summary, save_processed_data


class PersonaType(Enum):
    """Enumeration of persona archetypes"""
    URBAN_COMMUTER = "Urban Commuter"
    LEISURE_CYCLIST = "Leisure Cyclist"
    TOURIST_EXPLORER = "Tourist Explorer"
    FITNESS_ENTHUSIAST = "Fitness Enthusiast"
    BUDGET_CONSCIOUS = "Budget Conscious"
    PREMIUM_CONSUMER = "Premium Consumer"
    FAMILY_ORIENTED = "Family Oriented"
    TECH_SAVVY = "Tech Savvy"


@dataclass
class BusinessOpportunity:
    """Structure for business opportunity recommendations"""
    opportunity_type: str
    description: str
    target_segments: List[str]
    estimated_market_size: float
    investment_level: str
    expected_roi: str
    implementation_timeline: str
    key_metrics: List[str]


@dataclass
class ConsumerPersona:
    """Comprehensive consumer persona structure"""
    persona_id: str
    persona_name: str
    persona_type: PersonaType
    cluster_ids: List[int]
    
    # Demographics
    estimated_population: int
    median_income: Optional[float]
    age_distribution: Dict[str, float]
    education_level: Dict[str, float]
    
    # Behavioral Characteristics
    mobility_profile: Dict[str, Any]
    spending_profile: Dict[str, Any]
    temporal_patterns: Dict[str, Any]
    
    # Business Insights
    market_value: float
    targeting_effectiveness: float
    seasonal_trends: Dict[str, float]
    
    # Narrative Description
    description: str
    key_motivations: List[str]
    preferred_channels: List[str]
    pain_points: List[str]
    
    # Business Recommendations
    marketing_strategies: List[str]
    product_opportunities: List[str]
    infrastructure_needs: List[str]


class PersonaGenerator:
    """Main class for generating business intelligence and consumer personas"""
    
    def __init__(self):
        self.census_data = {}
        self.personas = {}
        self.business_opportunities = []
        self.market_insights = {}
        
    def load_census_demographics(self, counties: List[str] = None) -> pd.DataFrame:
        """
        Load demographic data from Census API or generate realistic estimates
        """
        if counties is None:
            counties = DATA_CONFIG.SAMPLE_COUNTIES
            
        logger.info(f"Loading demographic data for {len(counties)} counties")
        
        try:
            if CENSUS_API_KEY:
                return self._fetch_census_data(counties)
            else:
                logger.warning("No Census API key found, generating sample demographic data")
                return self._generate_sample_demographics(counties)
        except Exception as e:
            logger.error(f"Error loading census data: {e}")
            return self._generate_sample_demographics(counties)
    
    def _fetch_census_data(self, counties: List[str]) -> pd.DataFrame:
        """Fetch actual demographic data from Census API"""
        # This would implement actual Census API calls
        # For now, fall back to sample data
        logger.info("Census API integration not yet implemented, using sample data")
        return self._generate_sample_demographics(counties)
    
    def _generate_sample_demographics(self, counties: List[str]) -> pd.DataFrame:
        """Generate realistic demographic data for counties"""
        
        # Realistic demographic profiles for major counties
        demographic_profiles = {
            "17031": {  # Cook County, IL (Chicago)
                "median_income": 65000,
                "age_18_34": 0.28,
                "age_35_54": 0.26,
                "age_55_plus": 0.46,
                "bachelor_plus": 0.42,
                "high_school": 0.35,
                "population_density": 2100
            },
            "36061": {  # New York County, NY (Manhattan)
                "median_income": 85000,
                "age_18_34": 0.35,
                "age_35_54": 0.30,
                "age_55_plus": 0.35,
                "bachelor_plus": 0.65,
                "high_school": 0.25,
                "population_density": 28000
            },
            "06037": {  # Los Angeles County, CA
                "median_income": 70000,
                "age_18_34": 0.30,
                "age_35_54": 0.28,
                "age_55_plus": 0.42,
                "bachelor_plus": 0.38,
                "high_school": 0.40,
                "population_density": 1000
            },
            "48201": {  # Harris County, TX (Houston)
                "median_income": 62000,
                "age_18_34": 0.32,
                "age_35_54": 0.29,
                "age_55_plus": 0.39,
                "bachelor_plus": 0.35,
                "high_school": 0.42,
                "population_density": 800
            },
            "04013": {  # Maricopa County, AZ (Phoenix)
                "median_income": 58000,
                "age_18_34": 0.29,
                "age_35_54": 0.27,
                "age_55_plus": 0.44,
                "bachelor_plus": 0.32,
                "high_school": 0.45,
                "population_density": 500
            }
        }
        
        data = []
        for county_fips in counties:
            profile = demographic_profiles.get(county_fips, demographic_profiles["17031"])
            
            # Add some realistic variation
            variation = np.random.normal(1.0, 0.1)
            
            data.append({
                'county_fips': county_fips,
                'median_income': profile['median_income'] * variation,
                'age_18_34_pct': profile['age_18_34'],
                'age_35_54_pct': profile['age_35_54'],
                'age_55_plus_pct': profile['age_55_plus'],
                'bachelor_plus_pct': profile['bachelor_plus'],
                'high_school_pct': profile['high_school'],
                'population_density': profile['population_density']
            })
        
        df = pd.DataFrame(data)
        self.census_data = df
        logger.info(f"Generated demographic data for {len(df)} counties")
        return df
    
    def analyze_cluster_characteristics(self, cluster_profiles: Dict[str, Any],
                                     features_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze cluster characteristics to identify persona archetypes
        """
        logger.info("Analyzing cluster characteristics for persona identification")
        
        cluster_analysis = {}
        
        for cluster_key, profile in cluster_profiles['cluster_profiles'].items():
            cluster_id = profile['cluster_id']
            
            # Skip outliers
            if cluster_id == -1:
                continue
            
            # Analyze mobility characteristics
            mobility_traits = self._analyze_mobility_traits(profile)
            
            # Analyze spending characteristics
            spending_traits = self._analyze_spending_traits(profile)
            
            # Analyze temporal patterns
            temporal_traits = self._analyze_temporal_traits(profile)
            
            # Determine persona archetype
            persona_type = self._determine_persona_type(mobility_traits, spending_traits, temporal_traits)
            
            cluster_analysis[cluster_key] = {
                'cluster_id': cluster_id,
                'size': profile['size'],
                'counties': profile['counties'],
                'mobility_traits': mobility_traits,
                'spending_traits': spending_traits,
                'temporal_traits': temporal_traits,
                'persona_type': persona_type,
                'distinguishing_features': profile['distinguishing_features']
            }
        
        return cluster_analysis
    
    def _analyze_mobility_traits(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze mobility-related characteristics"""
        
        feature_stats = profile.get('feature_statistics', {})
        
        # Extract mobility metrics
        total_trips = feature_stats.get('total_trips', {}).get('mean', 0)
        trip_duration = feature_stats.get('avg_trip_duration_minutes', {}).get('mean', 0)
        member_ratio = feature_stats.get('member_ratio', {}).get('mean', 0)
        weekend_ratio = feature_stats.get('weekend_ratio', {}).get('mean', 0.3)
        peak_hour_ratio = feature_stats.get('peak_hour_ratio', {}).get('mean', 0.3)
        
        # Categorize mobility patterns
        mobility_traits = {
            'usage_intensity': 'high' if total_trips > 8000 else 'medium' if total_trips > 3000 else 'low',
            'trip_length_preference': 'long' if trip_duration > 20 else 'medium' if trip_duration > 12 else 'short',
            'membership_commitment': 'high' if member_ratio > 0.7 else 'medium' if member_ratio > 0.4 else 'low',
            'weekend_usage': 'high' if weekend_ratio > 0.4 else 'medium' if weekend_ratio > 0.25 else 'low',
            'commuter_pattern': 'strong' if peak_hour_ratio > 0.35 else 'moderate' if peak_hour_ratio > 0.25 else 'weak',
            'total_trips': total_trips,
            'avg_duration': trip_duration,
            'member_ratio': member_ratio
        }
        
        return mobility_traits
    
    def _analyze_spending_traits(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze spending-related characteristics"""
        
        feature_stats = profile.get('feature_statistics', {})
        
        # Extract spending metrics
        total_spending = feature_stats.get('total_spending', {}).get('mean', 0)
        restaurant_pct = feature_stats.get('spending_pct_restaurants', {}).get('mean', 0)
        retail_pct = feature_stats.get('spending_pct_retail', {}).get('mean', 0)
        entertainment_pct = feature_stats.get('spending_pct_entertainment', {}).get('mean', 0)
        discretionary_ratio = feature_stats.get('discretionary_ratio', {}).get('mean', 0.5)
        
        # Categorize spending patterns
        spending_traits = {
            'spending_level': 'high' if total_spending > 800000 else 'medium' if total_spending > 400000 else 'low',
            'dining_preference': 'high' if restaurant_pct > 0.25 else 'medium' if restaurant_pct > 0.15 else 'low',
            'retail_orientation': 'high' if retail_pct > 0.3 else 'medium' if retail_pct > 0.2 else 'low',
            'entertainment_focus': 'high' if entertainment_pct > 0.15 else 'medium' if entertainment_pct > 0.08 else 'low',
            'discretionary_spending': 'high' if discretionary_ratio > 0.6 else 'medium' if discretionary_ratio > 0.4 else 'low',
            'total_spending': total_spending,
            'category_preferences': {
                'restaurants': restaurant_pct,
                'retail': retail_pct,
                'entertainment': entertainment_pct
            }
        }
        
        return spending_traits
    
    def _analyze_temporal_traits(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze temporal usage patterns"""
        
        feature_stats = profile.get('feature_statistics', {})
        
        # Extract temporal metrics
        peak_hour_ratio = feature_stats.get('peak_hour_ratio', {}).get('mean', 0.3)
        weekend_ratio = feature_stats.get('weekend_ratio', {}).get('mean', 0.3)
        night_trips_ratio = feature_stats.get('night_trips_ratio', {}).get('mean', 0.1)
        seasonal_variation = feature_stats.get('seasonal_variation', {}).get('mean', 0.2)
        
        temporal_traits = {
            'schedule_type': 'structured' if peak_hour_ratio > 0.35 else 'flexible',
            'weekend_activity': 'active' if weekend_ratio > 0.35 else 'moderate' if weekend_ratio > 0.25 else 'inactive',
            'night_usage': 'frequent' if night_trips_ratio > 0.15 else 'occasional' if night_trips_ratio > 0.05 else 'rare',
            'seasonal_sensitivity': 'high' if seasonal_variation > 0.3 else 'medium' if seasonal_variation > 0.15 else 'low',
            'peak_hour_ratio': peak_hour_ratio,
            'weekend_ratio': weekend_ratio,
            'seasonal_variation': seasonal_variation
        }
        
        return temporal_traits
    
    def _determine_persona_type(self, mobility_traits: Dict[str, Any],
                               spending_traits: Dict[str, Any],
                               temporal_traits: Dict[str, Any]) -> PersonaType:
        """Determine persona archetype based on behavioral traits"""
        
        # Rule-based persona classification
        
        # Urban Commuter: High peak hour usage, high membership, structured schedule
        if (mobility_traits['commuter_pattern'] == 'strong' and
            mobility_traits['membership_commitment'] == 'high' and
            temporal_traits['schedule_type'] == 'structured'):
            return PersonaType.URBAN_COMMUTER
        
        # Leisure Cyclist: High weekend usage, flexible schedule, moderate spending
        elif (temporal_traits['weekend_activity'] == 'active' and
              temporal_traits['schedule_type'] == 'flexible' and
              mobility_traits['usage_intensity'] in ['medium', 'high']):
            return PersonaType.LEISURE_CYCLIST
        
        # Tourist Explorer: Low membership, short trips, high entertainment spending
        elif (mobility_traits['membership_commitment'] == 'low' and
              mobility_traits['trip_length_preference'] == 'short' and
              spending_traits['entertainment_focus'] == 'high'):
            return PersonaType.TOURIST_EXPLORER
        
        # Fitness Enthusiast: Long trips, high weekend usage, moderate spending
        elif (mobility_traits['trip_length_preference'] == 'long' and
              temporal_traits['weekend_activity'] == 'active' and
              mobility_traits['usage_intensity'] == 'high'):
            return PersonaType.FITNESS_ENTHUSIAST
        
        # Premium Consumer: High spending, high discretionary ratio
        elif (spending_traits['spending_level'] == 'high' and
              spending_traits['discretionary_spending'] == 'high'):
            return PersonaType.PREMIUM_CONSUMER
        
        # Budget Conscious: Low spending, low discretionary ratio
        elif (spending_traits['spending_level'] == 'low' and
              spending_traits['discretionary_spending'] == 'low'):
            return PersonaType.BUDGET_CONSCIOUS
        
        # Tech Savvy: High membership, structured usage, high retail spending
        elif (mobility_traits['membership_commitment'] == 'high' and
              spending_traits['retail_orientation'] == 'high'):
            return PersonaType.TECH_SAVVY
        
        # Default to Family Oriented for remaining segments
        else:
            return PersonaType.FAMILY_ORIENTED
    
    def generate_persona_narratives(self, cluster_analysis: Dict[str, Any]) -> Dict[str, ConsumerPersona]:
        """
        Generate detailed persona narratives and descriptions
        """
        logger.info("Generating persona narratives and descriptions")
        
        personas = {}
        
        for cluster_key, analysis in cluster_analysis.items():
            persona_id = f"persona_{analysis['cluster_id']}"
            persona_type = analysis['persona_type']
            
            # Generate persona name
            persona_name = self._generate_persona_name(persona_type, analysis)
            
            # Create demographic profile
            demographics = self._create_demographic_profile(analysis)
            
            # Generate narrative description
            description = self._generate_persona_description(persona_type, analysis)
            
            # Identify key motivations
            motivations = self._identify_key_motivations(persona_type, analysis)
            
            # Determine preferred channels
            channels = self._determine_preferred_channels(persona_type, analysis)
            
            # Identify pain points
            pain_points = self._identify_pain_points(persona_type, analysis)
            
            # Generate business recommendations
            marketing_strategies = self._generate_marketing_strategies(persona_type, analysis)
            product_opportunities = self._generate_product_opportunities(persona_type, analysis)
            infrastructure_needs = self._generate_infrastructure_needs(persona_type, analysis)
            
            # Calculate business metrics
            market_value = self._calculate_market_value(analysis)
            targeting_effectiveness = self._calculate_targeting_effectiveness(analysis)
            seasonal_trends = self._analyze_seasonal_trends(analysis)
            
            # Create persona object
            persona = ConsumerPersona(
                persona_id=persona_id,
                persona_name=persona_name,
                persona_type=persona_type,
                cluster_ids=[analysis['cluster_id']],
                estimated_population=analysis['size'],
                median_income=demographics.get('median_income'),
                age_distribution=demographics.get('age_distribution', {}),
                education_level=demographics.get('education_level', {}),
                mobility_profile=analysis['mobility_traits'],
                spending_profile=analysis['spending_traits'],
                temporal_patterns=analysis['temporal_traits'],
                market_value=market_value,
                targeting_effectiveness=targeting_effectiveness,
                seasonal_trends=seasonal_trends,
                description=description,
                key_motivations=motivations,
                preferred_channels=channels,
                pain_points=pain_points,
                marketing_strategies=marketing_strategies,
                product_opportunities=product_opportunities,
                infrastructure_needs=infrastructure_needs
            )
            
            personas[persona_id] = persona
        
        self.personas = personas
        return personas
    
    def _generate_persona_name(self, persona_type: PersonaType, analysis: Dict[str, Any]) -> str:
        """Generate descriptive persona names"""
        
        base_names = {
            PersonaType.URBAN_COMMUTER: ["Daily Commuter", "City Navigator", "Rush Hour Rider"],
            PersonaType.LEISURE_CYCLIST: ["Weekend Warrior", "Casual Explorer", "Recreational Rider"],
            PersonaType.TOURIST_EXPLORER: ["City Tourist", "Visitor Explorer", "Sightseeing Cyclist"],
            PersonaType.FITNESS_ENTHUSIAST: ["Fitness Rider", "Active Cyclist", "Health-Focused User"],
            PersonaType.PREMIUM_CONSUMER: ["Premium Spender", "Affluent User", "High-Value Customer"],
            PersonaType.BUDGET_CONSCIOUS: ["Value Seeker", "Budget-Minded User", "Cost-Conscious Rider"],
            PersonaType.FAMILY_ORIENTED: ["Family User", "Community Rider", "Neighborhood Cyclist"],
            PersonaType.TECH_SAVVY: ["Digital Native", "Tech-Forward User", "Connected Cyclist"]
        }
        
        names = base_names.get(persona_type, ["Generic User"])
        
        # Add intensity modifier based on usage
        usage_intensity = analysis['mobility_traits']['usage_intensity']
        if usage_intensity == 'high':
            modifier = "Power"
        elif usage_intensity == 'low':
            modifier = "Casual"
        else:
            modifier = "Regular"
        
        base_name = np.random.choice(names)
        return f"{modifier} {base_name}"
    
    def _create_demographic_profile(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Create demographic profile for persona"""
        
        # Use census data if available
        if hasattr(self, 'census_data') and not self.census_data.empty:
            # Get average demographics for counties in this cluster
            cluster_counties = analysis['counties']
            county_demographics = self.census_data[self.census_data['county_fips'].isin(cluster_counties)]
            
            if not county_demographics.empty:
                return {
                    'median_income': county_demographics['median_income'].mean(),
                    'age_distribution': {
                        '18-34': county_demographics['age_18_34_pct'].mean(),
                        '35-54': county_demographics['age_35_54_pct'].mean(),
                        '55+': county_demographics['age_55_plus_pct'].mean()
                    },
                    'education_level': {
                        'bachelor_plus': county_demographics['bachelor_plus_pct'].mean(),
                        'high_school': county_demographics['high_school_pct'].mean()
                    }
                }
        
        # Default demographic estimates
        return {
            'median_income': 65000,
            'age_distribution': {'18-34': 0.3, '35-54': 0.4, '55+': 0.3},
            'education_level': {'bachelor_plus': 0.4, 'high_school': 0.6}
        }
    
    def _generate_persona_description(self, persona_type: PersonaType, analysis: Dict[str, Any]) -> str:
        """Generate narrative description for persona"""
        
        descriptions = {
            PersonaType.URBAN_COMMUTER: (
                "Highly structured individuals who rely on bike-sharing for daily commuting. "
                "They exhibit strong peak-hour usage patterns and maintain consistent membership. "
                "These users value reliability, convenience, and time efficiency in their transportation choices."
            ),
            PersonaType.LEISURE_CYCLIST: (
                "Recreation-focused users who primarily bike during weekends and leisure time. "
                "They enjoy exploring the city at a relaxed pace and often combine cycling with social activities. "
                "These users value flexibility and scenic routes over speed and efficiency."
            ),
            PersonaType.TOURIST_EXPLORER: (
                "Visitors and occasional users who use bike-sharing to explore the city. "
                "They typically take shorter trips to tourist destinations and entertainment venues. "
                "These users prioritize convenience, ease of use, and access to popular attractions."
            ),
            PersonaType.FITNESS_ENTHUSIAST: (
                "Health-conscious individuals who use cycling as part of their fitness routine. "
                "They tend to take longer trips and are active during both weekdays and weekends. "
                "These users value performance tracking, route variety, and health benefits."
            ),
            PersonaType.PREMIUM_CONSUMER: (
                "High-spending individuals with significant discretionary income. "
                "They are willing to pay for premium services and convenience features. "
                "These users value quality, exclusivity, and personalized experiences."
            ),
            PersonaType.BUDGET_CONSCIOUS: (
                "Cost-sensitive users who prioritize value and affordability. "
                "They carefully consider spending decisions and look for deals and promotions. "
                "These users value transparent pricing, cost-effectiveness, and basic reliability."
            ),
            PersonaType.FAMILY_ORIENTED: (
                "Community-focused individuals who often bike with family members or in groups. "
                "They prioritize safety, family-friendly routes, and community amenities. "
                "These users value safety features, group options, and neighborhood connectivity."
            ),
            PersonaType.TECH_SAVVY: (
                "Technology-forward users who embrace digital solutions and smart features. "
                "They actively use mobile apps, digital payments, and data tracking features. "
                "These users value innovation, connectivity, and seamless digital experiences."
            )
        }
        
        return descriptions.get(persona_type, "General bike-sharing user with varied usage patterns.")
    
    def _identify_key_motivations(self, persona_type: PersonaType, analysis: Dict[str, Any]) -> List[str]:
        """Identify key motivations for each persona"""
        
        motivations = {
            PersonaType.URBAN_COMMUTER: [
                "Reliable daily transportation",
                "Time efficiency",
                "Cost savings vs. car ownership",
                "Environmental consciousness",
                "Avoiding traffic congestion"
            ],
            PersonaType.LEISURE_CYCLIST: [
                "Recreation and enjoyment",
                "Social activities",
                "Exploring the city",
                "Physical activity",
                "Stress relief"
            ],
            PersonaType.TOURIST_EXPLORER: [
                "Convenient city exploration",
                "Access to attractions",
                "Authentic local experience",
                "Flexibility in travel",
                "Photo opportunities"
            ],
            PersonaType.FITNESS_ENTHUSIAST: [
                "Health and fitness goals",
                "Performance tracking",
                "Calorie burning",
                "Endurance building",
                "Active lifestyle"
            ],
            PersonaType.PREMIUM_CONSUMER: [
                "Premium experience",
                "Status and image",
                "Convenience and comfort",
                "Exclusive access",
                "Quality service"
            ],
            PersonaType.BUDGET_CONSCIOUS: [
                "Cost savings",
                "Value for money",
                "Basic transportation needs",
                "Avoiding unnecessary expenses",
                "Practical solutions"
            ],
            PersonaType.FAMILY_ORIENTED: [
                "Family bonding",
                "Safe transportation",
                "Community engagement",
                "Teaching children",
                "Neighborhood exploration"
            ],
            PersonaType.TECH_SAVVY: [
                "Latest technology",
                "Digital convenience",
                "Data insights",
                "Seamless integration",
                "Innovation adoption"
            ]
        }
        
        return motivations.get(persona_type, ["General transportation needs"])
    
    def _determine_preferred_channels(self, persona_type: PersonaType, analysis: Dict[str, Any]) -> List[str]:
        """Determine preferred communication and engagement channels"""
        
        channels = {
            PersonaType.URBAN_COMMUTER: [
                "Mobile app notifications",
                "Email newsletters",
                "Transit station advertising",
                "LinkedIn",
                "Professional networks"
            ],
            PersonaType.LEISURE_CYCLIST: [
                "Social media (Instagram, Facebook)",
                "Community events",
                "Local partnerships",
                "Word of mouth",
                "Outdoor advertising"
            ],
            PersonaType.TOURIST_EXPLORER: [
                "Travel websites",
                "Hotel partnerships",
                "Tourist information centers",
                "Google Maps integration",
                "Travel apps"
            ],
            PersonaType.FITNESS_ENTHUSIAST: [
                "Fitness apps integration",
                "Gym partnerships",
                "Health and wellness blogs",
                "Strava community",
                "Fitness influencers"
            ],
            PersonaType.PREMIUM_CONSUMER: [
                "Premium app features",
                "Concierge services",
                "Exclusive events",
                "High-end partnerships",
                "Personalized communications"
            ],
            PersonaType.BUDGET_CONSCIOUS: [
                "Deal websites",
                "Coupon platforms",
                "Community boards",
                "Student networks",
                "Budget-focused content"
            ],
            PersonaType.FAMILY_ORIENTED: [
                "Family-focused social media",
                "School partnerships",
                "Community centers",
                "Parent networks",
                "Local events"
            ],
            PersonaType.TECH_SAVVY: [
                "Tech blogs",
                "Developer communities",
                "Beta testing programs",
                "Tech conferences",
                "Innovation showcases"
            ]
        }
        
        return channels.get(persona_type, ["General marketing channels"])
    
    def _identify_pain_points(self, persona_type: PersonaType, analysis: Dict[str, Any]) -> List[str]:
        """Identify key pain points for each persona"""
        
        pain_points = {
            PersonaType.URBAN_COMMUTER: [
                "Bike availability during rush hours",
                "Docking station capacity",
                "Weather dependency",
                "Route predictability",
                "Integration with other transit"
            ],
            PersonaType.LEISURE_CYCLIST: [
                "Limited weekend availability",
                "Lack of scenic routes",
                "Safety concerns",
                "Group booking limitations",
                "Weather cancellations"
            ],
            PersonaType.TOURIST_EXPLORER: [
                "Complex registration process",
                "Language barriers",
                "Unfamiliar locations",
                "Tourist area congestion",
                "Limited tourist information"
            ],
            PersonaType.FITNESS_ENTHUSIAST: [
                "Limited performance tracking",
                "Bike quality variations",
                "Route optimization",
                "Fitness app integration",
                "Long-distance limitations"
            ],
            PersonaType.PREMIUM_CONSUMER: [
                "Standard service quality",
                "Lack of premium options",
                "No priority access",
                "Limited customization",
                "Generic experience"
            ],
            PersonaType.BUDGET_CONSCIOUS: [
                "Hidden fees",
                "Price increases",
                "Limited free options",
                "Overage charges",
                "Unclear pricing"
            ],
            PersonaType.FAMILY_ORIENTED: [
                "Safety concerns",
                "Limited family bikes",
                "Child seat availability",
                "Group coordination",
                "Family-friendly routes"
            ],
            PersonaType.TECH_SAVVY: [
                "App limitations",
                "Data privacy concerns",
                "Integration issues",
                "Feature requests",
                "Technical glitches"
            ]
        }
        
        return pain_points.get(persona_type, ["General service issues"])
    
    def _generate_marketing_strategies(self, persona_type: PersonaType, analysis: Dict[str, Any]) -> List[str]:
        """Generate targeted marketing strategies"""
        
        strategies = {
            PersonaType.URBAN_COMMUTER: [
                "Commuter membership packages",
                "Rush hour availability guarantees",
                "Corporate partnership programs",
                "Transit integration promotions",
                "Time-saving messaging"
            ],
            PersonaType.LEISURE_CYCLIST: [
                "Weekend special offers",
                "Group riding events",
                "Scenic route promotions",
                "Social media campaigns",
                "Community partnerships"
            ],
            PersonaType.TOURIST_EXPLORER: [
                "Tourist package deals",
                "Hotel partnership programs",
                "Attraction integration",
                "Multi-language support",
                "Visitor center presence"
            ],
            PersonaType.FITNESS_ENTHUSIAST: [
                "Fitness tracking integration",
                "Health challenge programs",
                "Gym partnerships",
                "Performance-based rewards",
                "Wellness content marketing"
            ],
            PersonaType.PREMIUM_CONSUMER: [
                "Premium membership tiers",
                "Exclusive bike access",
                "Concierge services",
                "VIP events",
                "Luxury partnerships"
            ],
            PersonaType.BUDGET_CONSCIOUS: [
                "Student discounts",
                "Low-income programs",
                "Value messaging",
                "Transparent pricing",
                "Cost comparison tools"
            ],
            PersonaType.FAMILY_ORIENTED: [
                "Family membership plans",
                "Safety campaigns",
                "School partnerships",
                "Community events",
                "Parent testimonials"
            ],
            PersonaType.TECH_SAVVY: [
                "Beta testing programs",
                "API access",
                "Developer partnerships",
                "Innovation showcases",
                "Tech feature highlights"
            ]
        }
        
        return strategies.get(persona_type, ["General marketing approaches"])
    
    def _generate_product_opportunities(self, persona_type: PersonaType, analysis: Dict[str, Any]) -> List[str]:
        """Generate product development opportunities"""
        
        opportunities = {
            PersonaType.URBAN_COMMUTER: [
                "Reserved bike programs",
                "Express docking stations",
                "Weather protection features",
                "Route optimization tools",
                "Multi-modal integration"
            ],
            PersonaType.LEISURE_CYCLIST: [
                "Scenic route recommendations",
                "Group booking features",
                "Social sharing tools",
                "Event calendar integration",
                "Photo spot markers"
            ],
            PersonaType.TOURIST_EXPLORER: [
                "Tourist-specific app features",
                "Audio tour integration",
                "Attraction partnerships",
                "Multi-language support",
                "Simplified registration"
            ],
            PersonaType.FITNESS_ENTHUSIAST: [
                "Performance tracking dashboard",
                "Fitness app integrations",
                "Challenge programs",
                "Health metrics tracking",
                "Training route suggestions"
            ],
            PersonaType.PREMIUM_CONSUMER: [
                "Premium bike fleet",
                "Priority access system",
                "Concierge services",
                "Exclusive stations",
                "Luxury amenities"
            ],
            PersonaType.BUDGET_CONSCIOUS: [
                "Basic membership tiers",
                "Pay-per-use options",
                "Student programs",
                "Referral rewards",
                "Cost tracking tools"
            ],
            PersonaType.FAMILY_ORIENTED: [
                "Family bike options",
                "Child safety features",
                "Group management tools",
                "Family-friendly routes",
                "Safety education programs"
            ],
            PersonaType.TECH_SAVVY: [
                "Advanced app features",
                "API access",
                "IoT integrations",
                "Data analytics tools",
                "Smart bike features"
            ]
        }
        
        return opportunities.get(persona_type, ["General product improvements"])
    
    def _generate_infrastructure_needs(self, persona_type: PersonaType, analysis: Dict[str, Any]) -> List[str]:
        """Generate infrastructure investment recommendations"""
        
        infrastructure = {
            PersonaType.URBAN_COMMUTER: [
                "High-capacity stations at transit hubs",
                "Express lanes for commuters",
                "Weather-protected docking",
                "Real-time availability displays",
                "Integration with transit systems"
            ],
            PersonaType.LEISURE_CYCLIST: [
                "Scenic route development",
                "Park and recreation area stations",
                "Rest areas and amenities",
                "Group gathering spaces",
                "Weekend service expansion"
            ],
            PersonaType.TOURIST_EXPLORER: [
                "Tourist district station density",
                "Attraction-adjacent locations",
                "Multilingual signage",
                "Tourist information integration",
                "Hotel partnership stations"
            ],
            PersonaType.FITNESS_ENTHUSIAST: [
                "Long-distance route connections",
                "Fitness facility partnerships",
                "Performance tracking infrastructure",
                "Health monitoring stations",
                "Training route markers"
            ],
            PersonaType.PREMIUM_CONSUMER: [
                "Premium station locations",
                "Exclusive access areas",
                "Enhanced amenities",
                "Concierge service points",
                "Luxury partnerships"
            ],
            PersonaType.BUDGET_CONSCIOUS: [
                "Basic service area expansion",
                "Cost-effective station designs",
                "Community partnership locations",
                "Student area coverage",
                "Affordable access points"
            ],
            PersonaType.FAMILY_ORIENTED: [
                "Family-safe route development",
                "School and community center stations",
                "Child-friendly amenities",
                "Safety infrastructure",
                "Neighborhood connectivity"
            ],
            PersonaType.TECH_SAVVY: [
                "Smart station technology",
                "IoT sensor networks",
                "Digital integration points",
                "Tech hub partnerships",
                "Innovation showcases"
            ]
        }
        
        return infrastructure.get(persona_type, ["General infrastructure improvements"])
    
    def _calculate_market_value(self, analysis: Dict[str, Any]) -> float:
        """Calculate estimated market value for persona"""
        
        # Base calculation on cluster size and spending patterns
        cluster_size = analysis['size']
        spending_traits = analysis['spending_traits']
        
        # Estimate annual value per user
        spending_level = spending_traits['spending_level']
        if spending_level == 'high':
            value_per_user = 500
        elif spending_level == 'medium':
            value_per_user = 200
        else:
            value_per_user = 100
        
        # Adjust for membership commitment
        mobility_traits = analysis['mobility_traits']
        membership_multiplier = {
            'high': 1.5,
            'medium': 1.2,
            'low': 0.8
        }.get(mobility_traits['membership_commitment'], 1.0)
        
        total_market_value = cluster_size * value_per_user * membership_multiplier
        return round(total_market_value, 2)
    
    def _calculate_targeting_effectiveness(self, analysis: Dict[str, Any]) -> float:
        """Calculate targeting effectiveness score (0-1)"""
        
        # Base score on cluster coherence and distinguishing features
        distinguishing_features = analysis.get('distinguishing_features', {})
        
        if not distinguishing_features:
            return 0.5
        
        # Higher absolute z-scores indicate more distinctive features
        z_scores = list(distinguishing_features.values())
        avg_distinctiveness = np.mean([abs(z) for z in z_scores])
        
        # Normalize to 0-1 scale (z-score of 2+ is highly distinctive)
        effectiveness = min(avg_distinctiveness / 2.0, 1.0)
        
        return round(effectiveness, 3)
    
    def _analyze_seasonal_trends(self, analysis: Dict[str, Any]) -> Dict[str, float]:
        """Analyze seasonal usage trends"""
        
        temporal_traits = analysis['temporal_traits']
        seasonal_variation = temporal_traits.get('seasonal_variation', 0.2)
        
        # Generate realistic seasonal patterns based on persona type
        base_pattern = {
            'spring': 1.0,
            'summer': 1.2,
            'fall': 0.9,
            'winter': 0.7
        }
        
        # Adjust based on seasonal sensitivity
        sensitivity = seasonal_variation
        for season in base_pattern:
            if season in ['summer', 'spring']:
                base_pattern[season] += sensitivity
            else:
                base_pattern[season] -= sensitivity * 0.5
        
        # Normalize to ensure reasonable values
        for season in base_pattern:
            base_pattern[season] = max(0.3, min(1.8, base_pattern[season]))
        
        return base_pattern
    
    def generate_business_opportunities(self, personas: Dict[str, ConsumerPersona]) -> List[BusinessOpportunity]:
        """
        Generate business opportunities based on persona analysis
        """
        logger.info("Generating business opportunities from persona analysis")
        
        opportunities = []
        
        # Market expansion opportunities
        total_market_value = sum(persona.market_value for persona in personas.values())
        high_value_personas = [p for p in personas.values() if p.market_value > total_market_value / len(personas)]
        
        if high_value_personas:
            opportunities.append(BusinessOpportunity(
                opportunity_type="Market Expansion",
                description="Focus on high-value customer segments for premium service offerings",
                target_segments=[p.persona_name for p in high_value_personas],
                estimated_market_size=sum(p.market_value for p in high_value_personas),
                investment_level="Medium",
                expected_roi="15-25%",
                implementation_timeline="6-12 months",
                key_metrics=["Customer lifetime value", "Premium conversion rate", "Revenue per user"]
            ))
        
        # Technology integration opportunities
        tech_personas = [p for p in personas.values() if p.persona_type == PersonaType.TECH_SAVVY]
        if tech_personas:
            opportunities.append(BusinessOpportunity(
                opportunity_type="Technology Integration",
                description="Develop advanced digital features and IoT integrations",
                target_segments=[p.persona_name for p in tech_personas],
                estimated_market_size=sum(p.market_value for p in tech_personas),
                investment_level="High",
                expected_roi="20-35%",
                implementation_timeline="12-18 months",
                key_metrics=["App engagement", "Feature adoption", "Digital conversion"]
            ))
        
        # Infrastructure investment opportunities
        commuter_personas = [p for p in personas.values() if p.persona_type == PersonaType.URBAN_COMMUTER]
        if commuter_personas:
            opportunities.append(BusinessOpportunity(
                opportunity_type="Infrastructure Investment",
                description="Expand station network in high-commuter areas",
                target_segments=[p.persona_name for p in commuter_personas],
                estimated_market_size=sum(p.market_value for p in commuter_personas),
                investment_level="High",
                expected_roi="10-20%",
                implementation_timeline="18-24 months",
                key_metrics=["Station utilization", "Commuter retention", "Peak hour capacity"]
            ))
        
        # Tourism partnerships
        tourist_personas = [p for p in personas.values() if p.persona_type == PersonaType.TOURIST_EXPLORER]
        if tourist_personas:
            opportunities.append(BusinessOpportunity(
                opportunity_type="Tourism Partnerships",
                description="Develop partnerships with hotels and attractions",
                target_segments=[p.persona_name for p in tourist_personas],
                estimated_market_size=sum(p.market_value for p in tourist_personas),
                investment_level="Low",
                expected_roi="25-40%",
                implementation_timeline="3-6 months",
                key_metrics=["Tourist conversion", "Partnership revenue", "Seasonal growth"]
            ))
        
        self.business_opportunities = opportunities
        return opportunities
    
    def generate_market_insights(self, personas: Dict[str, ConsumerPersona],
                               opportunities: List[BusinessOpportunity]) -> Dict[str, Any]:
        """
        Generate comprehensive market insights and recommendations
        """
        logger.info("Generating comprehensive market insights")
        
        # Calculate market metrics
        total_population = sum(persona.estimated_population for persona in personas.values())
        total_market_value = sum(persona.market_value for persona in personas.values())
        avg_targeting_effectiveness = np.mean([persona.targeting_effectiveness for persona in personas.values()])
        
        # Identify dominant persona types
        persona_type_distribution = {}
        for persona in personas.values():
            persona_type = persona.persona_type.value
            if persona_type not in persona_type_distribution:
                persona_type_distribution[persona_type] = 0
            persona_type_distribution[persona_type] += persona.estimated_population
        
        # Calculate seasonal patterns
        seasonal_analysis = {}
        for season in ['spring', 'summer', 'fall', 'winter']:
            seasonal_values = [persona.seasonal_trends.get(season, 1.0) for persona in personas.values()]
            seasonal_analysis[season] = {
                'avg_multiplier': np.mean(seasonal_values),
                'variation': np.std(seasonal_values)
            }
        
        # Generate strategic recommendations
        strategic_recommendations = self._generate_strategic_recommendations(personas, opportunities)
        
        # Risk assessment
        risk_factors = self._assess_market_risks(personas)
        
        insights = {
            'market_overview': {
                'total_addressable_market': total_market_value,
                'total_population': total_population,
                'average_targeting_effectiveness': avg_targeting_effectiveness,
                'number_of_segments': len(personas)
            },
            'persona_distribution': persona_type_distribution,
            'seasonal_patterns': seasonal_analysis,
            'business_opportunities': [
                {
                    'type': opp.opportunity_type,
                    'description': opp.description,
                    'market_size': opp.estimated_market_size,
                    'expected_roi': opp.expected_roi,
                    'timeline': opp.implementation_timeline
                }
                for opp in opportunities
            ],
            'strategic_recommendations': strategic_recommendations,
            'risk_assessment': risk_factors,
            'key_insights': self._generate_key_insights(personas, opportunities)
        }
        
        self.market_insights = insights
        return insights
    
    def _generate_strategic_recommendations(self, personas: Dict[str, ConsumerPersona],
                                         opportunities: List[BusinessOpportunity]) -> List[str]:
        """Generate high-level strategic recommendations"""
        
        recommendations = []
        
        # Analyze persona distribution
        high_value_count = sum(1 for p in personas.values() if p.market_value > 100000)
        total_personas = len(personas)
        
        if high_value_count / total_personas > 0.4:
            recommendations.append(
                "Focus on premium service offerings to capture high-value segments"
            )
        
        # Analyze seasonal patterns
        seasonal_variations = []
        for persona in personas.values():
            summer_winter_diff = persona.seasonal_trends.get('summer', 1.0) - persona.seasonal_trends.get('winter', 1.0)
            seasonal_variations.append(summer_winter_diff)
        
        avg_seasonal_variation = np.mean(seasonal_variations)
        if avg_seasonal_variation > 0.3:
            recommendations.append(
                "Implement dynamic pricing and capacity management for seasonal demand"
            )
        
        # Technology adoption potential
        tech_savvy_personas = [p for p in personas.values() if p.persona_type == PersonaType.TECH_SAVVY]
        if len(tech_savvy_personas) > 0:
            recommendations.append(
                "Invest in digital innovation and smart technology features"
            )
        
        # Market expansion potential
        total_market_value = sum(p.market_value for p in personas.values())
        if total_market_value > 500000:
            recommendations.append(
                "Consider geographic expansion to similar markets"
            )
        
        return recommendations
    
    def _assess_market_risks(self, personas: Dict[str, ConsumerPersona]) -> List[str]:
        """Assess potential market risks"""
        
        risks = []
        
        # Concentration risk
        max_persona_share = max(p.estimated_population for p in personas.values()) / sum(p.estimated_population for p in personas.values())
        if max_persona_share > 0.5:
            risks.append("High concentration in single customer segment")
        
        # Seasonal risk
        winter_dependency = np.mean([p.seasonal_trends.get('winter', 1.0) for p in personas.values()])
        if winter_dependency < 0.6:
            risks.append("High seasonal dependency with winter vulnerability")
        
        # Technology risk
        tech_dependent_personas = [p for p in personas.values() if p.persona_type == PersonaType.TECH_SAVVY]
        if len(tech_dependent_personas) / len(personas) > 0.3:
            risks.append("Technology disruption risk from changing user expectations")
        
        # Economic sensitivity
        budget_conscious_share = sum(p.estimated_population for p in personas.values() if p.persona_type == PersonaType.BUDGET_CONSCIOUS) / sum(p.estimated_population for p in personas.values())
        if budget_conscious_share > 0.3:
            risks.append("Economic downturn sensitivity due to price-conscious segments")
        
        return risks
    
    def _generate_key_insights(self, personas: Dict[str, ConsumerPersona],
                             opportunities: List[BusinessOpportunity]) -> List[str]:
        """Generate key actionable insights"""
        
        insights = []
        
        # Most valuable persona
        highest_value_persona = max(personas.values(), key=lambda p: p.market_value)
        insights.append(
            f"'{highest_value_persona.persona_name}' represents the highest market value opportunity "
            f"(${highest_value_persona.market_value:,.0f})"
        )
        
        # Best targeting opportunity
        best_targeting_persona = max(personas.values(), key=lambda p: p.targeting_effectiveness)
        insights.append(
            f"'{best_targeting_persona.persona_name}' offers the best targeting effectiveness "
            f"({best_targeting_persona.targeting_effectiveness:.1%})"
        )
        
        # Seasonal opportunity
        summer_personas = sorted(personas.values(), 
                               key=lambda p: p.seasonal_trends.get('summer', 1.0), 
                               reverse=True)
        if summer_personas:
            insights.append(
                f"Summer presents the biggest growth opportunity, especially for "
                f"'{summer_personas[0].persona_name}' segment"
            )
        
        # Infrastructure priority
        commuter_personas = [p for p in personas.values() if p.persona_type == PersonaType.URBAN_COMMUTER]
        if commuter_personas:
            total_commuter_value = sum(p.market_value for p in commuter_personas)
            insights.append(
                f"Commuter infrastructure should be prioritized (${total_commuter_value:,.0f} market value)"
            )
        
        return insights
    
    def export_business_intelligence(self, filename_prefix: str = 'business_intelligence') -> Dict[str, str]:
        """
        Export all business intelligence outputs to various formats
        """
        logger.info("Exporting business intelligence outputs")
        
        exported_files = {}
        
        # Export personas as JSON
        personas_data = {}
        for persona_id, persona in self.personas.items():
            personas_data[persona_id] = {
                'persona_name': persona.persona_name,
                'persona_type': persona.persona_type.value,
                'estimated_population': persona.estimated_population,
                'market_value': persona.market_value,
                'targeting_effectiveness': persona.targeting_effectiveness,
                'description': persona.description,
                'key_motivations': persona.key_motivations,
                'preferred_channels': persona.preferred_channels,
                'pain_points': persona.pain_points,
                'marketing_strategies': persona.marketing_strategies,
                'product_opportunities': persona.product_opportunities,
                'infrastructure_needs': persona.infrastructure_needs,
                'mobility_profile': persona.mobility_profile,
                'spending_profile': persona.spending_profile,
                'seasonal_trends': persona.seasonal_trends
            }
        
        # Save personas JSON
        personas_file = f"{filename_prefix}_personas.json"
        from config import DATA_CONFIG
        personas_path = DATA_CONFIG.PROCESSED_DATA_DIR / personas_file
        with open(personas_path, 'w') as f:
            json.dump(personas_data, f, indent=2, default=str)
        exported_files['personas'] = str(personas_path)
        
        # Export business opportunities
        opportunities_data = [
            {
                'opportunity_type': opp.opportunity_type,
                'description': opp.description,
                'target_segments': opp.target_segments,
                'estimated_market_size': opp.estimated_market_size,
                'investment_level': opp.investment_level,
                'expected_roi': opp.expected_roi,
                'implementation_timeline': opp.implementation_timeline,
                'key_metrics': opp.key_metrics
            }
            for opp in self.business_opportunities
        ]
        
        opportunities_file = f"{filename_prefix}_opportunities.json"
        opportunities_path = DATA_CONFIG.PROCESSED_DATA_DIR / opportunities_file
        with open(opportunities_path, 'w') as f:
            json.dump(opportunities_data, f, indent=2)
        exported_files['opportunities'] = str(opportunities_path)
        
        # Export market insights
        insights_file = f"{filename_prefix}_market_insights.json"
        insights_path = DATA_CONFIG.PROCESSED_DATA_DIR / insights_file
        with open(insights_path, 'w') as f:
            json.dump(self.market_insights, f, indent=2, default=str)
        exported_files['insights'] = str(insights_path)
        
        # Generate executive summary report
        summary_report = self._generate_executive_summary()
        summary_file = f"{filename_prefix}_executive_summary.txt"
        summary_path = DATA_CONFIG.PROCESSED_DATA_DIR / summary_file
        with open(summary_path, 'w') as f:
            f.write(summary_report)
        exported_files['executive_summary'] = str(summary_path)
        
        logger.info(f"Exported {len(exported_files)} business intelligence files")
        return exported_files
    
    def _generate_executive_summary(self) -> str:
        """Generate executive summary report"""
        
        report = []
        report.append("=" * 80)
        report.append("EXECUTIVE SUMMARY: CONSUMER SEGMENTATION BUSINESS INTELLIGENCE")
        report.append("=" * 80)
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Market Overview
        if self.market_insights:
            overview = self.market_insights['market_overview']
            report.append("MARKET OVERVIEW")
            report.append("-" * 40)
            report.append(f"Total Addressable Market: ${overview['total_addressable_market']:,.0f}")
            report.append(f"Total Population: {overview['total_population']:,} users")
            report.append(f"Number of Segments: {overview['number_of_segments']}")
            report.append(f"Average Targeting Effectiveness: {overview['average_targeting_effectiveness']:.1%}")
            report.append("")
        
        # Key Personas
        if self.personas:
            report.append("KEY CONSUMER PERSONAS")
            report.append("-" * 40)
            
            # Sort personas by market value
            sorted_personas = sorted(self.personas.values(), key=lambda p: p.market_value, reverse=True)
            
            for i, persona in enumerate(sorted_personas[:3], 1):
                report.append(f"{i}. {persona.persona_name}")
                report.append(f"   Type: {persona.persona_type.value}")
                report.append(f"   Market Value: ${persona.market_value:,.0f}")
                report.append(f"   Population: {persona.estimated_population:,}")
                report.append(f"   Targeting Effectiveness: {persona.targeting_effectiveness:.1%}")
                report.append("")
        
        # Business Opportunities
        if self.business_opportunities:
            report.append("TOP BUSINESS OPPORTUNITIES")
            report.append("-" * 40)
            
            for i, opp in enumerate(self.business_opportunities[:3], 1):
                report.append(f"{i}. {opp.opportunity_type}")
                report.append(f"   Market Size: ${opp.estimated_market_size:,.0f}")
                report.append(f"   Expected ROI: {opp.expected_roi}")
                report.append(f"   Timeline: {opp.implementation_timeline}")
                report.append(f"   Investment: {opp.investment_level}")
                report.append("")
        
        # Strategic Recommendations
        if self.market_insights and 'strategic_recommendations' in self.market_insights:
            report.append("STRATEGIC RECOMMENDATIONS")
            report.append("-" * 40)
            for i, rec in enumerate(self.market_insights['strategic_recommendations'], 1):
                report.append(f"{i}. {rec}")
            report.append("")
        
        # Risk Assessment
        if self.market_insights and 'risk_assessment' in self.market_insights:
            report.append("RISK ASSESSMENT")
            report.append("-" * 40)
            for i, risk in enumerate(self.market_insights['risk_assessment'], 1):
                report.append(f"{i}. {risk}")
            report.append("")
        
        # Key Insights
        if self.market_insights and 'key_insights' in self.market_insights:
            report.append("KEY INSIGHTS")
            report.append("-" * 40)
            for i, insight in enumerate(self.market_insights['key_insights'], 1):
                report.append(f"{i}. {insight}")
            report.append("")
        
        report.append("=" * 80)
        return "\n".join(report)


def main():
    """Demo function for persona generation and business intelligence"""
    from src.data_loader import DataLoader
    from src.spatial_processor import SpatialProcessor
    from src.feature_engineering import FeatureEngineer
    from src.clustering_engine import ClusteringEngine
    
    # Initialize components
    loader = DataLoader()
    processor = SpatialProcessor()
    engineer = FeatureEngineer()
    clustering_engine = ClusteringEngine()
    persona_generator = PersonaGenerator()
    
    logger.info("Starting business intelligence demo...")
    
    # Load and process data (abbreviated for demo)
    trips_df = loader.download_divvy_data(2023, 6)
    spending_df = loader.download_spending_data()
    
    # Process through pipeline
    boundaries = processor.load_county_boundaries()
    stations_gdf = processor.extract_stations_from_trips(trips_df)
    stations_with_counties = processor.assign_stations_to_counties(stations_gdf, boundaries)
    county_mobility = processor.aggregate_trips_to_county_level(trips_df, stations_with_counties)
    
    # Feature engineering
    engineered_features, pipeline_results = engineer.create_feature_pipeline(
        county_mobility, spending_df, trips_df
    )
    
    # Clustering
    clustering_results = clustering_engine.run_complete_clustering_analysis(
        engineered_features, algorithms=['kmeans']
    )
    
    # Extract cluster profiles
    cluster_profiles = clustering_results['algorithms']['kmeans']['cluster_profiles']
    
    # Load demographic data
    persona_generator.load_census_demographics()
    
    # Generate business intelligence
    cluster_analysis = persona_generator.analyze_cluster_characteristics(cluster_profiles, engineered_features)
    personas = persona_generator.generate_persona_narratives(cluster_analysis)
    opportunities = persona_generator.generate_business_opportunities(personas)
    market_insights = persona_generator.generate_market_insights(personas, opportunities)
    
    # Export results
    exported_files = persona_generator.export_business_intelligence()
    
    # Print summary
    print(f"\nBusiness Intelligence Summary:")
    print(f"- Generated {len(personas)} consumer personas")
    print(f"- Identified {len(opportunities)} business opportunities")
    print(f"- Total market value: ${sum(p.market_value for p in personas.values()):,.0f}")
    print(f"- Exported files: {list(exported_files.keys())}")
    
    # Show sample persona
    if personas:
        sample_persona = list(personas.values())[0]
        print(f"\nSample Persona: {sample_persona.persona_name}")
        print(f"Type: {sample_persona.persona_type.value}")
        print(f"Description: {sample_persona.description[:100]}...")
        print(f"Market Value: ${sample_persona.market_value:,.0f}")
    
    logger.info("Business intelligence demo completed successfully!")
    
    return personas, opportunities, market_insights


if __name__ == "__main__":
    main()