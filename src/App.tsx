import React, { useState, useEffect } from 'react';
import { MapPin, TrendingUp, Users, DollarSign, Target, BarChart3, Calendar, Settings, Download, RefreshCw } from 'lucide-react';

interface PersonaData {
  id: string;
  name: string;
  type: string;
  population: number;
  marketValue: number;
  effectiveness: number;
  description: string;
  seasonalTrends: { [key: string]: number };
}

interface OpportunityData {
  type: string;
  marketSize: number;
  roi: string;
  timeline: string;
  investment: string;
}

function App() {
  const [activeTab, setActiveTab] = useState('overview');
  const [selectedPersonas, setSelectedPersonas] = useState<string[]>([]);
  const [isLoading, setIsLoading] = useState(true);

  // Demo data
  const personas: PersonaData[] = [
    {
      id: 'urban_commuter',
      name: 'Urban Commuter Pro',
      type: 'Urban Commuter',
      population: 15000,
      marketValue: 250000,
      effectiveness: 0.85,
      description: 'Highly structured professionals who rely on bike-sharing for daily commuting to work.',
      seasonalTrends: { spring: 1.0, summer: 1.2, fall: 0.9, winter: 0.7 }
    },
    {
      id: 'weekend_explorer',
      name: 'Weekend Explorer',
      type: 'Leisure Cyclist',
      population: 8000,
      marketValue: 120000,
      effectiveness: 0.75,
      description: 'Recreation-focused users who enjoy cycling for leisure and exploration on weekends.',
      seasonalTrends: { spring: 1.1, summer: 1.4, fall: 1.0, winter: 0.5 }
    },
    {
      id: 'tech_savvy',
      name: 'Tech Innovator',
      type: 'Tech Savvy',
      population: 5000,
      marketValue: 180000,
      effectiveness: 0.90,
      description: 'Early adopters who embrace technology and seek innovative transportation solutions.',
      seasonalTrends: { spring: 1.0, summer: 1.1, fall: 1.0, winter: 0.8 }
    }
  ];

  const opportunities: OpportunityData[] = [
    {
      type: 'Premium Commuter Services',
      marketSize: 300000,
      roi: '25-35%',
      timeline: '6-9 months',
      investment: 'Medium'
    },
    {
      type: 'Weekend Recreation Packages',
      marketSize: 150000,
      roi: '15-25%',
      timeline: '3-6 months',
      investment: 'Low'
    },
    {
      type: 'Smart Technology Integration',
      marketSize: 200000,
      roi: '30-40%',
      timeline: '9-12 months',
      investment: 'High'
    }
  ];

  useEffect(() => {
    // Simulate loading
    const timer = setTimeout(() => {
      setIsLoading(false);
      setSelectedPersonas(personas.map(p => p.id));
    }, 2000);

    return () => clearTimeout(timer);
  }, []);

  const totalMarketValue = personas
    .filter(p => selectedPersonas.includes(p.id))
    .reduce((sum, p) => sum + p.marketValue, 0);

  const totalPopulation = personas
    .filter(p => selectedPersonas.includes(p.id))
    .reduce((sum, p) => sum + p.population, 0);

  const avgEffectiveness = personas
    .filter(p => selectedPersonas.includes(p.id))
    .reduce((sum, p) => sum + p.effectiveness, 0) / selectedPersonas.length || 0;

  const renderOverview = () => (
    <div className="space-y-6">
      {/* Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <div className="bg-gradient-to-r from-blue-500 to-blue-600 text-white p-6 rounded-lg shadow-lg">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-blue-100 text-sm">Total Market Value</p>
              <p className="text-2xl font-bold">${totalMarketValue.toLocaleString()}</p>
            </div>
            <DollarSign className="h-8 w-8 text-blue-200" />
          </div>
        </div>

        <div className="bg-gradient-to-r from-green-500 to-green-600 text-white p-6 rounded-lg shadow-lg">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-green-100 text-sm">Total Users</p>
              <p className="text-2xl font-bold">{totalPopulation.toLocaleString()}</p>
            </div>
            <Users className="h-8 w-8 text-green-200" />
          </div>
        </div>

        <div className="bg-gradient-to-r from-purple-500 to-purple-600 text-white p-6 rounded-lg shadow-lg">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-purple-100 text-sm">Avg. Effectiveness</p>
              <p className="text-2xl font-bold">{(avgEffectiveness * 100).toFixed(1)}%</p>
            </div>
            <Target className="h-8 w-8 text-purple-200" />
          </div>
        </div>

        <div className="bg-gradient-to-r from-orange-500 to-orange-600 text-white p-6 rounded-lg shadow-lg">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-orange-100 text-sm">Segments</p>
              <p className="text-2xl font-bold">{selectedPersonas.length}</p>
            </div>
            <BarChart3 className="h-8 w-8 text-orange-200" />
          </div>
        </div>
      </div>

      {/* Personas Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6">
        {personas
          .filter(p => selectedPersonas.includes(p.id))
          .map((persona, index) => (
            <div key={persona.id} className="bg-white rounded-lg shadow-lg p-6 border-l-4 border-blue-500">
              <div className="flex items-start justify-between mb-4">
                <div>
                  <h3 className="text-lg font-semibold text-gray-900">{persona.name}</h3>
                  <p className="text-sm text-gray-600">{persona.type}</p>
                </div>
                <div className="text-right">
                  <p className="text-sm text-gray-500">Market Value</p>
                  <p className="text-lg font-bold text-blue-600">${persona.marketValue.toLocaleString()}</p>
                </div>
              </div>

              <p className="text-gray-700 text-sm mb-4 line-clamp-3">{persona.description}</p>

              <div className="grid grid-cols-2 gap-4 text-sm">
                <div>
                  <p className="text-gray-500">Population</p>
                  <p className="font-semibold">{persona.population.toLocaleString()}</p>
                </div>
                <div>
                  <p className="text-gray-500">Effectiveness</p>
                  <p className="font-semibold">{(persona.effectiveness * 100).toFixed(1)}%</p>
                </div>
              </div>

              {/* Seasonal Trends Mini Chart */}
              <div className="mt-4">
                <p className="text-xs text-gray-500 mb-2">Seasonal Trends</p>
                <div className="flex space-x-1">
                  {Object.entries(persona.seasonalTrends).map(([season, value]) => (
                    <div key={season} className="flex-1">
                      <div className="bg-gray-200 rounded h-2">
                        <div 
                          className="bg-blue-500 rounded h-2 transition-all duration-300"
                          style={{ width: `${(value / 1.4) * 100}%` }}
                        />
                      </div>
                      <p className="text-xs text-gray-500 mt-1 capitalize">{season.slice(0, 3)}</p>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          ))}
      </div>
    </div>
  );

  const renderOpportunities = () => (
    <div className="space-y-6">
      <div className="bg-white rounded-lg shadow-lg p-6">
        <h2 className="text-xl font-semibold text-gray-900 mb-6">Business Opportunities</h2>
        
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
          <div className="text-center">
            <p className="text-3xl font-bold text-blue-600">${opportunities.reduce((sum, opp) => sum + opp.marketSize, 0).toLocaleString()}</p>
            <p className="text-gray-600">Total Opportunity Value</p>
          </div>
          <div className="text-center">
            <p className="text-3xl font-bold text-green-600">{opportunities.length}</p>
            <p className="text-gray-600">Identified Opportunities</p>
          </div>
          <div className="text-center">
            <p className="text-3xl font-bold text-purple-600">28%</p>
            <p className="text-gray-600">Average Expected ROI</p>
          </div>
        </div>

        <div className="space-y-4">
          {opportunities.map((opp, index) => (
            <div key={index} className="border border-gray-200 rounded-lg p-6 hover:shadow-md transition-shadow">
              <div className="flex items-start justify-between mb-4">
                <div>
                  <h3 className="text-lg font-semibold text-gray-900">{opp.type}</h3>
                  <div className="flex items-center space-x-4 mt-2 text-sm text-gray-600">
                    <span>Market Size: ${opp.marketSize.toLocaleString()}</span>
                    <span>ROI: {opp.roi}</span>
                    <span>Timeline: {opp.timeline}</span>
                  </div>
                </div>
                <div className="text-right">
                  <span className={`px-3 py-1 rounded-full text-xs font-medium ${
                    opp.investment === 'Low' ? 'bg-green-100 text-green-800' :
                    opp.investment === 'Medium' ? 'bg-yellow-100 text-yellow-800' :
                    'bg-red-100 text-red-800'
                  }`}>
                    {opp.investment} Investment
                  </span>
                </div>
              </div>

              <div className="w-full bg-gray-200 rounded-full h-2">
                <div 
                  className="bg-blue-500 h-2 rounded-full transition-all duration-500"
                  style={{ width: `${(opp.marketSize / 300000) * 100}%` }}
                />
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );

  const renderAnalytics = () => (
    <div className="space-y-6">
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Market Value Chart */}
        <div className="bg-white rounded-lg shadow-lg p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Market Value by Persona</h3>
          <div className="space-y-3">
            {personas
              .filter(p => selectedPersonas.includes(p.id))
              .sort((a, b) => b.marketValue - a.marketValue)
              .map((persona, index) => (
                <div key={persona.id} className="flex items-center justify-between">
                  <span className="text-sm text-gray-700">{persona.name}</span>
                  <div className="flex items-center space-x-2">
                    <div className="w-32 bg-gray-200 rounded-full h-2">
                      <div 
                        className="bg-blue-500 h-2 rounded-full transition-all duration-500"
                        style={{ width: `${(persona.marketValue / 250000) * 100}%` }}
                      />
                    </div>
                    <span className="text-sm font-medium text-gray-900 w-20 text-right">
                      ${persona.marketValue.toLocaleString()}
                    </span>
                  </div>
                </div>
              ))}
          </div>
        </div>

        {/* Effectiveness Chart */}
        <div className="bg-white rounded-lg shadow-lg p-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">Targeting Effectiveness</h3>
          <div className="space-y-3">
            {personas
              .filter(p => selectedPersonas.includes(p.id))
              .sort((a, b) => b.effectiveness - a.effectiveness)
              .map((persona, index) => (
                <div key={persona.id} className="flex items-center justify-between">
                  <span className="text-sm text-gray-700">{persona.name}</span>
                  <div className="flex items-center space-x-2">
                    <div className="w-32 bg-gray-200 rounded-full h-2">
                      <div 
                        className="bg-green-500 h-2 rounded-full transition-all duration-500"
                        style={{ width: `${persona.effectiveness * 100}%` }}
                      />
                    </div>
                    <span className="text-sm font-medium text-gray-900 w-12 text-right">
                      {(persona.effectiveness * 100).toFixed(0)}%
                    </span>
                  </div>
                </div>
              ))}
          </div>
        </div>
      </div>

      {/* Seasonal Trends */}
      <div className="bg-white rounded-lg shadow-lg p-6">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Seasonal Usage Patterns</h3>
        <div className="grid grid-cols-4 gap-4">
          {['spring', 'summer', 'fall', 'winter'].map(season => (
            <div key={season} className="text-center">
              <h4 className="text-sm font-medium text-gray-700 mb-3 capitalize">{season}</h4>
              <div className="space-y-2">
                {personas
                  .filter(p => selectedPersonas.includes(p.id))
                  .map(persona => (
                    <div key={persona.id} className="flex items-center justify-between text-xs">
                      <span className="text-gray-600 truncate">{persona.name.split(' ')[0]}</span>
                      <span className="font-medium">{persona.seasonalTrends[season].toFixed(1)}x</span>
                    </div>
                  ))}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );

  if (isLoading) {
    return (
      <div className="min-h-screen bg-gray-50 flex items-center justify-center">
        <div className="text-center">
          <RefreshCw className="h-8 w-8 animate-spin text-blue-500 mx-auto mb-4" />
          <h2 className="text-xl font-semibold text-gray-900 mb-2">Loading Consumer Segmentation Dashboard</h2>
          <p className="text-gray-600">Analyzing mobility and spending patterns...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex items-center justify-between h-16">
            <div className="flex items-center space-x-3">
              <div className="bg-gradient-to-r from-blue-500 to-purple-600 p-2 rounded-lg">
                <Target className="h-6 w-6 text-white" />
              </div>
              <div>
                <h1 className="text-xl font-bold text-gray-900">Consumer Segmentation Dashboard</h1>
                <p className="text-sm text-gray-600">Advanced Analytics for Mobility & Spending Patterns</p>
              </div>
            </div>
            
            <div className="flex items-center space-x-4">
              <button className="flex items-center space-x-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors">
                <Download className="h-4 w-4" />
                <span>Export</span>
              </button>
              <button 
                onClick={() => setIsLoading(true)}
                className="flex items-center space-x-2 px-4 py-2 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50 transition-colors"
              >
                <RefreshCw className="h-4 w-4" />
                <span>Refresh</span>
              </button>
            </div>
          </div>
        </div>
      </header>

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <div className="flex flex-col lg:flex-row gap-8">
          {/* Sidebar */}
          <div className="lg:w-64 space-y-6">
            {/* Navigation */}
            <div className="bg-white rounded-lg shadow-sm p-4">
              <h3 className="text-sm font-medium text-gray-900 mb-3">Dashboard Sections</h3>
              <nav className="space-y-1">
                {[
                  { id: 'overview', label: 'Overview', icon: BarChart3 },
                  { id: 'opportunities', label: 'Opportunities', icon: TrendingUp },
                  { id: 'analytics', label: 'Analytics', icon: Target },
                ].map(item => (
                  <button
                    key={item.id}
                    onClick={() => setActiveTab(item.id)}
                    className={`w-full flex items-center space-x-3 px-3 py-2 text-sm rounded-lg transition-colors ${
                      activeTab === item.id
                        ? 'bg-blue-50 text-blue-700 border border-blue-200'
                        : 'text-gray-700 hover:bg-gray-50'
                    }`}
                  >
                    <item.icon className="h-4 w-4" />
                    <span>{item.label}</span>
                  </button>
                ))}
              </nav>
            </div>

            {/* Persona Filters */}
            <div className="bg-white rounded-lg shadow-sm p-4">
              <h3 className="text-sm font-medium text-gray-900 mb-3">Persona Filters</h3>
              <div className="space-y-2">
                {personas.map(persona => (
                  <label key={persona.id} className="flex items-center space-x-2">
                    <input
                      type="checkbox"
                      checked={selectedPersonas.includes(persona.id)}
                      onChange={(e) => {
                        if (e.target.checked) {
                          setSelectedPersonas([...selectedPersonas, persona.id]);
                        } else {
                          setSelectedPersonas(selectedPersonas.filter(id => id !== persona.id));
                        }
                      }}
                      className="rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                    />
                    <span className="text-sm text-gray-700">{persona.name}</span>
                  </label>
                ))}
              </div>
            </div>

            {/* Quick Stats */}
            <div className="bg-white rounded-lg shadow-sm p-4">
              <h3 className="text-sm font-medium text-gray-900 mb-3">Quick Stats</h3>
              <div className="space-y-3 text-sm">
                <div className="flex justify-between">
                  <span className="text-gray-600">Active Segments</span>
                  <span className="font-medium">{selectedPersonas.length}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Total Opportunities</span>
                  <span className="font-medium">{opportunities.length}</span>
                </div>
                <div className="flex justify-between">
                  <span className="text-gray-600">Avg ROI</span>
                  <span className="font-medium text-green-600">28%</span>
                </div>
              </div>
            </div>
          </div>

          {/* Main Content */}
          <div className="flex-1">
            {activeTab === 'overview' && renderOverview()}
            {activeTab === 'opportunities' && renderOpportunities()}
            {activeTab === 'analytics' && renderAnalytics()}
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;