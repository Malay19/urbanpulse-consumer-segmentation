#!/bin/bash

# Setup script for Consumer Segmentation Analysis
set -e

echo "🚀 Setting up Consumer Segmentation Analysis Environment"
echo "========================================================="

# Check Python version
python_version=$(python3 --version 2>&1 | grep -o '[0-9]\+\.[0-9]\+')
required_version="3.8"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Python 3.8+ is required. Found: $python_version"
    exit 1
fi

echo "✅ Python version check passed: $python_version"

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📚 Installing dependencies..."
pip install -r requirements.txt

# Install development dependencies if in development mode
if [ "${ENVIRONMENT:-development}" = "development" ]; then
    echo "🛠️  Installing development dependencies..."
    pip install -r requirements-dev.txt
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p data/raw data/processed logs cache config test-results

# Set up pre-commit hooks (development only)
if [ "${ENVIRONMENT:-development}" = "development" ]; then
    echo "🪝 Setting up pre-commit hooks..."
    pre-commit install
fi

# Create default configuration files
echo "⚙️  Creating configuration files..."
python3 -c "
from config_manager import ConfigManager
manager = ConfigManager()
config = manager.load_config()
print('Configuration files created successfully')
"

# Run initial tests
echo "🧪 Running initial tests..."
python3 -m pytest test_suite.py::TestDataQuality::test_data_completeness -v

# Generate sample data
echo "📊 Generating sample data..."
python3 -c "
from src.data_loader import DataLoader
loader = DataLoader()
trips_df = loader.download_divvy_data(2023, 6)
spending_df = loader.download_spending_data()
print(f'Generated {len(trips_df)} trip records and {len(spending_df)} spending records')
"

echo ""
echo "✅ Setup completed successfully!"
echo ""
echo "Next steps:"
echo "1. Activate the virtual environment: source venv/bin/activate"
echo "2. Run the dashboard: python run_dashboard.py"
echo "3. Run the complete pipeline: python pipeline_manager.py"
echo "4. Run tests: python test_suite.py"
echo ""
echo "🎯 Happy analyzing!"