"""
Dashboard launcher script for Consumer Segmentation Analysis
Run this script to start the Streamlit dashboard
"""

import subprocess
import sys
import os
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed"""
    try:
        import streamlit
        import plotly
        import folium
        import pandas
        import numpy
        print("✅ All required dependencies are installed")
        return True
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("Please install requirements: pip install -r requirements.txt")
        return False

def run_dashboard():
    """Launch the Streamlit dashboard"""
    if not check_dependencies():
        return
    
    # Get the directory of this script
    script_dir = Path(__file__).parent
    dashboard_path = script_dir / "dashboard.py"
    
    if not dashboard_path.exists():
        print(f"❌ Dashboard file not found: {dashboard_path}")
        return
    
    print("🚀 Starting Consumer Segmentation Dashboard...")
    print("📊 Dashboard will open in your default web browser")
    print("🔗 URL: http://localhost:8501")
    print("⏹️  Press Ctrl+C to stop the dashboard")
    print("-" * 50)
    
    try:
        # Run streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            str(dashboard_path),
            "--server.port", "8501",
            "--server.address", "localhost",
            "--browser.gatherUsageStats", "false"
        ])
    except KeyboardInterrupt:
        print("\n🛑 Dashboard stopped by user")
    except Exception as e:
        print(f"❌ Error running dashboard: {e}")

if __name__ == "__main__":
    run_dashboard()