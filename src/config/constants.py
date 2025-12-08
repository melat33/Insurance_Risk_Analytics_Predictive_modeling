"""
Project-wide constants.
"""

# Data file constants
RAW_DATA_FILE = "MachineLearningRating_v3.txt"
OUTPUT_FILES = {
    "raw": "raw_data.txt",
    "cleaned": "cleaned_data.txt", 
    "features": "features_data.txt"
}

# Column groups for processing
DATE_COLUMNS = ['transactionmonth', 'vehicleintroductiondate']
FINANCIAL_COLUMNS = ['totalpremium', 'totalclaims', 'suminsured', 'customvalueestimate']
CATEGORICAL_COLUMNS = ['province', 'vehicletype', 'gender', 'make', 'model']

# Data cleaning parameters
CLEANING_CONFIG = {
    "drop_threshold": 50,  # Drop columns with >50% missing
    "outlier_threshold": 1.5,  # IQR multiplier for outliers
    "max_categories": 50,  # Maximum unique values for categorical conversion
}

# DVC configuration
DVC_STAGES = ["load", "clean", "features"]