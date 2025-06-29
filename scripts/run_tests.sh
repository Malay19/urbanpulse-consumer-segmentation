#!/bin/bash

# Test runner script for Consumer Segmentation Analysis
set -e

echo "🧪 Running Consumer Segmentation Analysis Test Suite"
echo "===================================================="

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Set testing environment
export ENVIRONMENT=testing
export DEBUG=true
export LOG_LEVEL=DEBUG
export CACHE_ENABLED=false

# Create test results directory
mkdir -p test-results

# Run different test suites based on argument
case "${1:-all}" in
    "unit")
        echo "🔬 Running unit tests..."
        python -m pytest test_suite.py::TestDataQuality test_suite.py::TestSpatialProcessing test_suite.py::TestFeatureEngineering test_suite.py::TestClusteringEngine test_suite.py::TestPersonaGeneration -v --tb=short --junitxml=test-results/unit-tests.xml
        ;;
    
    "integration")
        echo "🔗 Running integration tests..."
        python -m pytest test_suite.py::TestPipelineIntegration -v --tb=short --junitxml=test-results/integration-tests.xml
        ;;
    
    "performance")
        echo "⚡ Running performance tests..."
        python -m pytest test_suite.py::TestPerformanceBenchmarks -v --tb=short --junitxml=test-results/performance-tests.xml
        ;;
    
    "quality")
        echo "📊 Running data quality tests..."
        python -m pytest test_suite.py::TestDataQuality -v --tb=short --junitxml=test-results/quality-tests.xml
        ;;
    
    "error")
        echo "🚨 Running error handling tests..."
        python -m pytest test_suite.py::TestErrorHandling -v --tb=short --junitxml=test-results/error-tests.xml
        ;;
    
    "coverage")
        echo "📈 Running tests with coverage..."
        python -m pytest test_suite.py -v --tb=short --cov=src --cov=pipeline_manager --cov=config_manager --cov-report=html --cov-report=xml --cov-report=term-missing --junitxml=test-results/coverage-tests.xml
        echo "Coverage report generated in htmlcov/index.html"
        ;;
    
    "all")
        echo "🎯 Running complete test suite..."
        python -m pytest test_suite.py -v --tb=short --durations=10 --junitxml=test-results/all-tests.xml
        ;;
    
    "quick")
        echo "⚡ Running quick test suite..."
        python -m pytest test_suite.py -v --tb=short -x --ff --junitxml=test-results/quick-tests.xml
        ;;
    
    *)
        echo "❌ Unknown test suite: $1"
        echo "Available options: unit, integration, performance, quality, error, coverage, all, quick"
        exit 1
        ;;
esac

# Check test results
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ All tests passed!"
    
    # Generate test summary
    echo ""
    echo "📊 Test Summary:"
    echo "================"
    
    if [ -f "test-results/all-tests.xml" ]; then
        python3 -c "
import xml.etree.ElementTree as ET
try:
    tree = ET.parse('test-results/all-tests.xml')
    root = tree.getroot()
    tests = root.get('tests', '0')
    failures = root.get('failures', '0')
    errors = root.get('errors', '0')
    time = root.get('time', '0')
    
    print(f'Total Tests: {tests}')
    print(f'Failures: {failures}')
    print(f'Errors: {errors}')
    print(f'Execution Time: {float(time):.2f} seconds')
    print(f'Success Rate: {((int(tests) - int(failures) - int(errors)) / int(tests) * 100):.1f}%' if int(tests) > 0 else 'N/A')
except Exception as e:
    print('Could not parse test results')
"
    fi
    
else
    echo ""
    echo "❌ Some tests failed!"
    echo "Check the output above for details."
    exit 1
fi