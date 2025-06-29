"""
Tests for persona generation and business intelligence functionality
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from persona_generator import PersonaGenerator, PersonaType, ConsumerPersona, BusinessOpportunity


class TestPersonaGenerator:
    
    @pytest.fixture
    def sample_cluster_profiles(self):
        """Create sample cluster profiles for testing"""
        return {
            'cluster_profiles': {
                'cluster_0': {
                    'cluster_id': 0,
                    'size': 5000,
                    'counties': ['17031', '17043'],
                    'feature_statistics': {
                        'total_trips': {'mean': 10000, 'std': 2000},
                        'avg_trip_duration_minutes': {'mean': 15.5, 'std': 3.2},
                        'member_ratio': {'mean': 0.8, 'std': 0.1},
                        'peak_hour_ratio': {'mean': 0.4, 'std': 0.05},
                        'weekend_ratio': {'mean': 0.2, 'std': 0.05},
                        'total_spending': {'mean': 500000, 'std': 100000},
                        'spending_pct_restaurants': {'mean': 0.25, 'std': 0.05},
                        'spending_pct_retail': {'mean': 0.3, 'std': 0.05},
                        'discretionary_ratio': {'mean': 0.6, 'std': 0.1}
                    },
                    'distinguishing_features': {
                        'member_ratio': 2.5,
                        'peak_hour_ratio': 2.0,
                        'total_spending': 1.8
                    }
                },
                'cluster_1': {
                    'cluster_id': 1,
                    'size': 3000,
                    'counties': ['36061'],
                    'feature_statistics': {
                        'total_trips': {'mean': 3000, 'std': 800},
                        'avg_trip_duration_minutes': {'mean': 25.0, 'std': 5.0},
                        'member_ratio': {'mean': 0.3, 'std': 0.1},
                        'peak_hour_ratio': {'mean': 0.15, 'std': 0.05},
                        'weekend_ratio': {'mean': 0.6, 'std': 0.1},
                        'total_spending': {'mean': 200000, 'std': 50000},
                        'spending_pct_entertainment': {'mean': 0.4, 'std': 0.1},
                        'discretionary_ratio': {'mean': 0.8, 'std': 0.1}
                    },
                    'distinguishing_features': {
                        'weekend_ratio': 3.0,
                        'spending_pct_entertainment': 2.8,
                        'member_ratio': -2.0
                    }
                }
            }
        }
    
    @pytest.fixture
    def persona_generator(self):
        """Create a PersonaGenerator instance for testing"""
        return PersonaGenerator()
    
    def test_demographic_data_generation(self, persona_generator):
        """Test demographic data generation"""
        counties = ['17031', '36061']
        demographics = persona_generator.load_census_demographics(counties)
        
        # Check data structure
        assert isinstance(demographics, pd.DataFrame)
        assert len(demographics) > 0
        
        # Check required columns
        required_columns = ['county_fips', 'median_income', 'age_18_34_pct', 'bachelor_plus_pct']
        for col in required_columns:
            assert col in demographics.columns
        
        # Check county coverage
        assert set(demographics['county_fips'].unique()) == set(counties)
        
        # Check reasonable value ranges
        assert (demographics['median_income'] > 30000).all()
        assert (demographics['median_income'] < 150000).all()
        assert (demographics['age_18_34_pct'] >= 0).all()
        assert (demographics['age_18_34_pct'] <= 1).all()
    
    def test_cluster_characteristics_analysis(self, persona_generator, sample_cluster_profiles):
        """Test cluster characteristics analysis"""
        features_df = pd.DataFrame({
            'county_fips': ['17031', '17043', '36061'],
            'total_trips': [10000, 8000, 3000],
            'member_ratio': [0.8, 0.75, 0.3]
        })
        
        cluster_analysis = persona_generator.analyze_cluster_characteristics(
            sample_cluster_profiles, features_df
        )
        
        # Check analysis structure
        assert isinstance(cluster_analysis, dict)
        assert len(cluster_analysis) == 2  # Two clusters (excluding outliers)
        
        # Check cluster analysis content
        for cluster_key, analysis in cluster_analysis.items():
            assert 'cluster_id' in analysis
            assert 'mobility_traits' in analysis
            assert 'spending_traits' in analysis
            assert 'temporal_traits' in analysis
            assert 'persona_type' in analysis
            
            # Check mobility traits
            mobility_traits = analysis['mobility_traits']
            assert 'usage_intensity' in mobility_traits
            assert 'membership_commitment' in mobility_traits
            assert mobility_traits['usage_intensity'] in ['low', 'medium', 'high']
            assert mobility_traits['membership_commitment'] in ['low', 'medium', 'high']
            
            # Check spending traits
            spending_traits = analysis['spending_traits']
            assert 'spending_level' in spending_traits
            assert 'discretionary_spending' in spending_traits
            assert spending_traits['spending_level'] in ['low', 'medium', 'high']
            
            # Check persona type
            assert isinstance(analysis['persona_type'], PersonaType)
    
    def test_persona_type_determination(self, persona_generator):
        """Test persona type determination logic"""
        # Test Urban Commuter classification
        mobility_traits = {
            'commuter_pattern': 'strong',
            'membership_commitment': 'high',
            'usage_intensity': 'high'
        }
        spending_traits = {
            'spending_level': 'medium',
            'discretionary_spending': 'medium'
        }
        temporal_traits = {
            'schedule_type': 'structured',
            'weekend_activity': 'moderate'
        }
        
        persona_type = persona_generator._determine_persona_type(
            mobility_traits, spending_traits, temporal_traits
        )
        assert persona_type == PersonaType.URBAN_COMMUTER
        
        # Test Tourist Explorer classification
        mobility_traits = {
            'membership_commitment': 'low',
            'trip_length_preference': 'short',
            'usage_intensity': 'low'
        }
        spending_traits = {
            'entertainment_focus': 'high',
            'spending_level': 'medium'
        }
        temporal_traits = {
            'schedule_type': 'flexible'
        }
        
        persona_type = persona_generator._determine_persona_type(
            mobility_traits, spending_traits, temporal_traits
        )
        assert persona_type == PersonaType.TOURIST_EXPLORER
    
    def test_persona_narrative_generation(self, persona_generator, sample_cluster_profiles):
        """Test persona narrative generation"""
        # Load demographics first
        persona_generator.load_census_demographics(['17031', '17043', '36061'])
        
        # Analyze clusters
        features_df = pd.DataFrame({
            'county_fips': ['17031', '17043', '36061'],
            'total_trips': [10000, 8000, 3000]
        })
        
        cluster_analysis = persona_generator.analyze_cluster_characteristics(
            sample_cluster_profiles, features_df
        )
        
        # Generate personas
        personas = persona_generator.generate_persona_narratives(cluster_analysis)
        
        # Check personas structure
        assert isinstance(personas, dict)
        assert len(personas) == 2  # Two clusters
        
        # Check persona objects
        for persona_id, persona in personas.items():
            assert isinstance(persona, ConsumerPersona)
            
            # Check required attributes
            assert persona.persona_id == persona_id
            assert persona.persona_name is not None
            assert isinstance(persona.persona_type, PersonaType)
            assert persona.estimated_population > 0
            assert persona.market_value > 0
            assert 0 <= persona.targeting_effectiveness <= 1
            
            # Check narrative elements
            assert len(persona.description) > 50  # Substantial description
            assert len(persona.key_motivations) > 0
            assert len(persona.preferred_channels) > 0
            assert len(persona.pain_points) > 0
            assert len(persona.marketing_strategies) > 0
            
            # Check seasonal trends
            assert 'spring' in persona.seasonal_trends
            assert 'summer' in persona.seasonal_trends
            assert 'fall' in persona.seasonal_trends
            assert 'winter' in persona.seasonal_trends
    
    def test_business_opportunities_generation(self, persona_generator):
        """Test business opportunities generation"""
        # Create sample personas
        sample_personas = {
            'persona_1': ConsumerPersona(
                persona_id='persona_1',
                persona_name='Tech Savvy Commuter',
                persona_type=PersonaType.TECH_SAVVY,
                cluster_ids=[1],
                estimated_population=5000,
                median_income=80000,
                age_distribution={'18-34': 0.5, '35-54': 0.3, '55+': 0.2},
                education_level={'bachelor_plus': 0.7, 'high_school': 0.3},
                mobility_profile={'usage_intensity': 'high'},
                spending_profile={'spending_level': 'high'},
                temporal_patterns={'schedule_type': 'structured'},
                market_value=200000,
                targeting_effectiveness=0.8,
                seasonal_trends={'spring': 1.0, 'summer': 1.2, 'fall': 0.9, 'winter': 0.7},
                description='Tech-forward users who embrace digital solutions',
                key_motivations=['Innovation', 'Convenience'],
                preferred_channels=['Mobile app', 'Tech blogs'],
                pain_points=['App limitations', 'Feature requests'],
                marketing_strategies=['Beta testing', 'Tech partnerships'],
                product_opportunities=['Advanced features', 'API access'],
                infrastructure_needs=['Smart stations', 'IoT integration']
            )
        }
        
        # Generate opportunities
        opportunities = persona_generator.generate_business_opportunities(sample_personas)
        
        # Check opportunities structure
        assert isinstance(opportunities, list)
        assert len(opportunities) > 0
        
        # Check opportunity objects
        for opportunity in opportunities:
            assert isinstance(opportunity, BusinessOpportunity)
            assert opportunity.opportunity_type is not None
            assert opportunity.description is not None
            assert len(opportunity.target_segments) > 0
            assert opportunity.estimated_market_size > 0
            assert opportunity.investment_level in ['Low', 'Medium', 'High']
            assert '%' in opportunity.expected_roi
            assert 'month' in opportunity.implementation_timeline.lower()
            assert len(opportunity.key_metrics) > 0
    
    def test_market_insights_generation(self, persona_generator):
        """Test market insights generation"""
        # Create sample data
        sample_personas = {
            'persona_1': ConsumerPersona(
                persona_id='persona_1',
                persona_name='Sample Persona',
                persona_type=PersonaType.URBAN_COMMUTER,
                cluster_ids=[1],
                estimated_population=5000,
                median_income=70000,
                age_distribution={},
                education_level={},
                mobility_profile={},
                spending_profile={},
                temporal_patterns={},
                market_value=150000,
                targeting_effectiveness=0.75,
                seasonal_trends={'spring': 1.0, 'summer': 1.3, 'fall': 0.8, 'winter': 0.6},
                description='Sample description',
                key_motivations=[],
                preferred_channels=[],
                pain_points=[],
                marketing_strategies=[],
                product_opportunities=[],
                infrastructure_needs=[]
            )
        }
        
        sample_opportunities = [
            BusinessOpportunity(
                opportunity_type='Test Opportunity',
                description='Test description',
                target_segments=['Sample Persona'],
                estimated_market_size=100000,
                investment_level='Medium',
                expected_roi='15-25%',
                implementation_timeline='6-12 months',
                key_metrics=['Test metric']
            )
        ]
        
        # Generate insights
        insights = persona_generator.generate_market_insights(sample_personas, sample_opportunities)
        
        # Check insights structure
        assert isinstance(insights, dict)
        
        # Check required sections
        required_sections = [
            'market_overview', 'persona_distribution', 'seasonal_patterns',
            'business_opportunities', 'strategic_recommendations', 'key_insights'
        ]
        for section in required_sections:
            assert section in insights
        
        # Check market overview
        market_overview = insights['market_overview']
        assert 'total_addressable_market' in market_overview
        assert 'total_population' in market_overview
        assert 'average_targeting_effectiveness' in market_overview
        assert market_overview['total_addressable_market'] > 0
        assert market_overview['total_population'] > 0
        
        # Check seasonal patterns
        seasonal_patterns = insights['seasonal_patterns']
        seasons = ['spring', 'summer', 'fall', 'winter']
        for season in seasons:
            assert season in seasonal_patterns
            assert 'avg_multiplier' in seasonal_patterns[season]
            assert 'variation' in seasonal_patterns[season]
        
        # Check recommendations and insights are lists
        assert isinstance(insights['strategic_recommendations'], list)
        assert isinstance(insights['key_insights'], list)
    
    def test_market_value_calculation(self, persona_generator):
        """Test market value calculation"""
        # Test high spending cluster
        high_spending_analysis = {
            'size': 1000,
            'spending_traits': {'spending_level': 'high'},
            'mobility_traits': {'membership_commitment': 'high'}
        }
        
        market_value = persona_generator._calculate_market_value(high_spending_analysis)
        assert market_value > 500000  # Should be high value
        
        # Test low spending cluster
        low_spending_analysis = {
            'size': 1000,
            'spending_traits': {'spending_level': 'low'},
            'mobility_traits': {'membership_commitment': 'low'}
        }
        
        market_value = persona_generator._calculate_market_value(low_spending_analysis)
        assert market_value < 200000  # Should be lower value
    
    def test_targeting_effectiveness_calculation(self, persona_generator):
        """Test targeting effectiveness calculation"""
        # Test high distinctiveness
        high_distinct_analysis = {
            'distinguishing_features': {
                'feature1': 3.0,  # High z-score
                'feature2': -2.5,
                'feature3': 2.8
            }
        }
        
        effectiveness = persona_generator._calculate_targeting_effectiveness(high_distinct_analysis)
        assert effectiveness > 0.8  # Should be highly effective
        
        # Test low distinctiveness
        low_distinct_analysis = {
            'distinguishing_features': {
                'feature1': 0.5,  # Low z-scores
                'feature2': -0.3,
                'feature3': 0.2
            }
        }
        
        effectiveness = persona_generator._calculate_targeting_effectiveness(low_distinct_analysis)
        assert effectiveness < 0.5  # Should be less effective
    
    def test_export_functionality(self, persona_generator, tmp_path):
        """Test business intelligence export functionality"""
        # Create minimal test data
        sample_personas = {
            'persona_1': ConsumerPersona(
                persona_id='persona_1',
                persona_name='Test Persona',
                persona_type=PersonaType.URBAN_COMMUTER,
                cluster_ids=[1],
                estimated_population=1000,
                median_income=60000,
                age_distribution={},
                education_level={},
                mobility_profile={},
                spending_profile={},
                temporal_patterns={},
                market_value=50000,
                targeting_effectiveness=0.7,
                seasonal_trends={},
                description='Test description',
                key_motivations=['Test motivation'],
                preferred_channels=['Test channel'],
                pain_points=['Test pain point'],
                marketing_strategies=['Test strategy'],
                product_opportunities=['Test opportunity'],
                infrastructure_needs=['Test need']
            )
        }
        
        sample_opportunities = [
            BusinessOpportunity(
                opportunity_type='Test',
                description='Test',
                target_segments=['Test Persona'],
                estimated_market_size=25000,
                investment_level='Low',
                expected_roi='10%',
                implementation_timeline='3 months',
                key_metrics=['Test metric']
            )
        ]
        
        sample_insights = {
            'market_overview': {'total_addressable_market': 50000},
            'key_insights': ['Test insight']
        }
        
        # Set up test data
        persona_generator.personas = sample_personas
        persona_generator.business_opportunities = sample_opportunities
        persona_generator.market_insights = sample_insights
        
        # Temporarily change data directory
        from config import DATA_CONFIG
        original_dir = DATA_CONFIG.PROCESSED_DATA_DIR
        DATA_CONFIG.PROCESSED_DATA_DIR = tmp_path
        
        try:
            # Export business intelligence
            exported_files = persona_generator.export_business_intelligence('test')
            
            # Check that files were created
            assert len(exported_files) > 0
            assert 'personas' in exported_files
            assert 'opportunities' in exported_files
            assert 'insights' in exported_files
            assert 'executive_summary' in exported_files
            
            # Check that files exist
            for file_type, filepath in exported_files.items():
                assert Path(filepath).exists()
                assert Path(filepath).stat().st_size > 0  # File is not empty
            
        finally:
            # Restore original directory
            DATA_CONFIG.PROCESSED_DATA_DIR = original_dir


if __name__ == "__main__":
    pytest.main([__file__])