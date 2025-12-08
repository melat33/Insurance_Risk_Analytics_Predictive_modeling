# test_cleaner.py
import sys
from pathlib import Path

project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.data.loader import InsuranceDataLoader
from src.data.cleaner import InsuranceDataCleaner

# Load data
loader = InsuranceDataLoader()
df_raw = loader.load_raw_data()

# Clean data
cleaner = InsuranceDataCleaner()
df_clean = cleaner.clean_data(df_raw)

print(f"Raw shape: {df_raw.shape}")
print(f"Clean shape: {df_clean.shape}")

# Save cleaning report
cleaner.save_cleaning_report()