# src/features/financial.py (minimal version)
"""
Fallback financial feature engineering.
"""

class FinancialFeatureEngineer:
    def create_features(self, df):
        return df.copy()
    def get_feature_definitions(self):
        return {}