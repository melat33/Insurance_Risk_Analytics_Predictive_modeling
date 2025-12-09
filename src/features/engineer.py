"""
Feature engineering for insurance data.
"""
import pandas as pd
import numpy as np
from ..config import constants
from ..utils.logger import get_logger

logger = get_logger(__name__)

class FeatureEngineer:
    """Feature engineering for insurance analytics."""
    
    def __init__(self):
        self.features_created = []
    
    def create_basic_features(self, df):
        """
        Create basic features from cleaned data.
        
        Args:
            df: Cleaned DataFrame
            
        Returns:
            DataFrame with added features
        """
        if df is None or df.empty:
            logger.error("Cannot create features from empty DataFrame")
            return df
        
        df_features = df.copy()
        original_columns = set(df.columns)
        
        # 1. Claim-related features
        if 'totalclaims' in df_features.columns:
            df_features['has_claim'] = (df_features['totalclaims'] > 0).astype(int)
            df_features['claim_amount'] = df_features['totalclaims'].clip(lower=0)
            self.features_created.append('has_claim')
            self.features_created.append('claim_amount')
        
        # 2. Financial ratio features
        if all(col in df_features.columns for col in ['totalclaims', 'totalpremium']):
            # Avoid division by zero
            premium_no_zero = df_features['totalpremium'].replace(0, np.nan)
            df_features['loss_ratio'] = df_features['totalclaims'] / premium_no_zero
            df_features['loss_ratio'] = df_features['loss_ratio'].fillna(0)
            self.features_created.append('loss_ratio')
        
        # 3. Temporal features
        if 'transactionmonth' in df_features.columns:
            if pd.api.types.is_datetime64_any_dtype(df_features['transactionmonth']):
                df_features['transaction_year'] = df_features['transactionmonth'].dt.year
                df_features['transaction_month'] = df_features['transactionmonth'].dt.month
                df_features['transaction_quarter'] = df_features['transactionmonth'].dt.quarter
                df_features['transaction_day'] = df_features['transactionmonth'].dt.day
                
                self.features_created.extend([
                    'transaction_year', 
                    'transaction_month',
                    'transaction_quarter',
                    'transaction_day'
                ])
        
        # 4. Vehicle age feature
        if all(col in df_features.columns for col in ['registrationyear', 'transactionmonth']):
            if pd.api.types.is_datetime64_any_dtype(df_features['transactionmonth']):
                df_features['vehicle_age'] = (
                    df_features['transactionmonth'].dt.year - df_features['registrationyear']
                )
                df_features['vehicle_age'] = df_features['vehicle_age'].clip(lower=0, upper=50)
                self.features_created.append('vehicle_age')
        
        # 5. Binary flags for categorical columns
        for col in constants.CATEGORICAL_COLUMNS:
            if col in df_features.columns and df_features[col].nunique() == 2:
                # Convert binary categorical to 0/1
                unique_vals = df_features[col].unique()
                if len(unique_vals) == 2:
                    df_features[f'{col}_flag'] = (df_features[col] == unique_vals[0]).astype(int)
                    self.features_created.append(f'{col}_flag')
        
        new_columns = set(df_features.columns) - original_columns
        logger.info(f"Created {len(new_columns)} new features: {list(new_columns)}")
        
        return df_features
    
    def get_feature_summary(self):
        """Get summary of features created."""
        return {
            "total_features_created": len(self.features_created),
            "features": self.features_created
        }