#!/usr/bin/env python
"""
Simplified Task 1 pipeline - saves only cleaned_data.txt, features_data.txt, raw_data.txt
"""
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

class SimpleDataPipeline:
    """Simple pipeline that saves only 3 data files."""
    
    def __init__(self):
        self.output_dir = project_root / "data" / "01_interim"
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def run(self):
        """Run the simplified pipeline."""
        print("="*60)
        print("SIMPLIFIED TASK 1 PIPELINE")
        print("="*60)
        
        # Step 1: Load and save raw data
        print("\n1️⃣  LOADING RAW DATA...")
        df_raw = self._load_raw_data()
        if df_raw is None:
            return
        
        raw_path = self.output_dir / "raw_data.txt"
        df_raw.to_csv(raw_path, sep='|', index=False)
        print(f"   ✅ Saved raw data: {raw_path}")
        print(f"   Shape: {df_raw.shape[0]:,} rows × {df_raw.shape[1]} columns")
        
        # Step 2: Clean and save cleaned data
        print("\n2️⃣  CLEANING DATA...")
        df_clean = self._clean_data(df_raw)
        
        clean_path = self.output_dir / "cleaned_data.txt"
        df_clean.to_csv(clean_path, sep='|', index=False)
        print(f"   ✅ Saved cleaned data: {clean_path}")
        print(f"   Shape: {df_clean.shape[0]:,} rows × {df_clean.shape[1]} columns")
        
        # Step 3: Create features and save features data
        print("\n3️⃣  CREATING FEATURES...")
        df_features = self._create_features(df_clean)
        
        features_path = self.output_dir / "features_data.txt"
        df_features.to_csv(features_path, sep='|', index=False)
        print(f"   ✅ Saved features data: {features_path}")
        print(f"   Shape: {df_features.shape[0]:,} rows × {df_features.shape[1]} columns")
        
        print("\n" + "="*60)
        print("✅ PIPELINE COMPLETED - 3 FILES SAVED")
        print("="*60)
        print(f"📂 Output files in: {self.output_dir}")
        print(f"   1. raw_data.txt")
        print(f"   2. cleaned_data.txt")
        print(f"   3. features_data.txt")
        
        return df_features
    
    def _load_raw_data(self):
        """Load the raw pipe-delimited file."""
        data_path = project_root / "data" / "00_raw" / "MachineLearningRating_v3.txt"
        if not data_path.exists():
            print(f"❌ Data file not found: {data_path}")
            return None
        
        print(f"📥 Loading: {data_path}")
        df = pd.read_csv(data_path, delimiter='|', low_memory=False)
        
        # Convert potential date columns
        date_cols = [col for col in df.columns if any(x in col.lower() for x in ['date', 'month', 'year'])]
        for col in date_cols:
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
            except:
                pass
        
        return df
    
    def _clean_data(self, df):
        """Simple cleaning: remove duplicates and handle missing values."""
        df_clean = df.copy()
        
        # Remove duplicates
        dup_count = df_clean.duplicated().sum()
        if dup_count > 0:
            df_clean = df_clean.drop_duplicates()
            print(f"   🧹 Removed {dup_count} duplicate rows")
        
        # Standardize column names
        df_clean.columns = df_clean.columns.str.strip().str.replace(' ', '_').str.lower()
        
        # Fill numeric missing values with median
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df_clean[col].isnull().sum() > 0:
                median_val = df_clean[col].median()
                df_clean[col] = df_clean[col].fillna(median_val)
        
        # Fill categorical missing values with mode
        categorical_cols = df_clean.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df_clean[col].isnull().sum() > 0:
                mode_val = df_clean[col].mode()[0] if not df_clean[col].mode().empty else 'Unknown'
                df_clean[col] = df_clean[col].fillna(mode_val)
        
        return df_clean
    
    def _create_features(self, df):
        """Create basic features."""
        df_features = df.copy()
        
        # Add claim flag if TotalClaims exists
        if 'totalclaims' in df_features.columns:
            df_features['has_claim'] = (df_features['totalclaims'] > 0).astype(int)
        
        # Add claim ratio if both columns exist
        if all(col in df_features.columns for col in ['totalclaims', 'totalpremium']):
            df_features['claim_ratio'] = df_features['totalclaims'] / df_features['totalpremium'].replace(0, 1)
        
        # Add temporal features if TransactionMonth exists
        if 'transactionmonth' in df_features.columns and pd.api.types.is_datetime64_any_dtype(df_features['transactionmonth']):
            df_features['transaction_year'] = df_features['transactionmonth'].dt.year
            df_features['transaction_month'] = df_features['transactionmonth'].dt.month
        
        return df_features

def main():
    """Main entry point."""
    pipeline = SimpleDataPipeline()
    pipeline.run()

if __name__ == "__main__":
    main()