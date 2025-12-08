"""
Path configuration for the project.
"""
from pathlib import Path

# Project root (3 levels up from src folder)
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Data paths
RAW_DATA_DIR = PROJECT_ROOT / "data" / "00_raw"
INTERIM_DATA_DIR = PROJECT_ROOT / "data" / "01_interim"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "02_processed"

# Source paths
SRC_DIR = PROJECT_ROOT / "src"
CONFIG_DIR = SRC_DIR / "config"
DATA_DIR = SRC_DIR / "data"
FEATURES_DIR = SRC_DIR / "features"
PIPELINE_DIR = SRC_DIR / "pipeline"
UTILS_DIR = SRC_DIR / "utils"

# Output paths
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

def ensure_directories():
    """Create all necessary directories."""
    dirs = [
        RAW_DATA_DIR,
        INTERIM_DATA_DIR,
        PROCESSED_DATA_DIR,
        REPORTS_DIR,
        FIGURES_DIR
    ]
    
    for directory in dirs:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"✓ Ensured directory exists: {directory}")
    
    return True