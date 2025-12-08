"""
Data cleaning and preprocessing module.
Integrated with the new project structure.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional

from ..config import constants, paths
from ..utils.logger import get_logger

logger = get_logger(__name__)

class InsuranceDataCleaner:
    """Cleaner for insurance data with validation and transformation."""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize cleaner with optional configuration.
        
        Args:
            config: Cleaning configuration dictionary
        """
        if config is None:
            config = self._default_config()
        
        self.config = config
        self.cleaning_report = {
            "operations": [],
            "removed_rows": 0,
            "removed_columns": 0,
            "missing_values": {},
            "outliers": {},
            "timestamp": pd.Timestamp.now().isoformat()
        }
        
        logger.info(f"Initialized InsuranceDataCleaner with config: {config}")
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean insurance data with comprehensive preprocessing.
        
        Args:
            df: Raw DataFrame from loader
            
        Returns:
            Cleaned DataFrame
        """
        if df is None or df.empty:
            logger.error("Cannot clean empty DataFrame")
            return df
        
        df_clean = df.copy()
        original_shape = df_clean.shape
        
        logger.info(f"🧹 Starting data cleaning process... Original shape: {original_shape}")
        
        # 1. Remove exact duplicates
        duplicates = df_clean.duplicated().sum()
        if duplicates > 0:
            df_clean = df_clean.drop_duplicates()
            self.cleaning_report["operations"].append(f"Removed {duplicates} duplicate rows")
            self.cleaning_report["removed_rows"] += duplicates
            logger.info(f"Removed {duplicates} duplicate rows")
        
        # 2. Standardize column names
        df_clean.columns = self._standardize_column_names(df_clean.columns)
        self.cleaning_report["operations"].append("Standardized column names")
        logger.info("Standardized column names")
        
        # 3. Handle missing values
        df_clean, missing_report = self._handle_missing_values(df_clean)
        self.cleaning_report["missing_values"] = missing_report
        
        # 4. Fix data types
        df_clean = self._fix_data_types(df_clean)
        self.cleaning_report["operations"].append("Fixed data types")
        logger.info("Fixed data types")
        
        # 5. Handle outliers (if configured)
        if self.config.get("handle_outliers", True):
            df_clean, outlier_report = self._handle_outliers(df_clean)
            self.cleaning_report["outliers"] = outlier_report
        
        # 6. Validate logical consistency
        df_clean = self._validate_logical_consistency(df_clean)
        
        # 7. Create derived features for cleaning
        df_clean = self._create_cleaning_features(df_clean)
        
        final_shape = df_clean.shape
        rows_removed = original_shape[0] - final_shape[0]
        self.cleaning_report["removed_rows"] += rows_removed
        
        logger.info(f"✅ Cleaning complete: {original_shape} → {final_shape}")
        logger.info(f"   Removed {rows_removed} rows total ({duplicates} duplicates)")
        
        return df_clean
    
    def _standardize_column_names(self, columns: pd.Index) -> List[str]:
        """Standardize column names."""
        cleaned = []
        for col in columns:
            # Remove special characters, convert to lowercase with underscores
            col_clean = str(col).strip().lower()
            col_clean = col_clean.replace(' ', '_').replace('-', '_')
            col_clean = col_clean.replace('__', '_').strip('_')
            cleaned.append(col_clean)
        return cleaned
    
    def _handle_missing_values(self, df: pd.DataFrame) -> tuple:
        """Handle missing values based on column type and configuration."""
        missing_report = {}
        
        drop_threshold = self.config.get("drop_threshold", 50)
        
        for column in df.columns:
            missing_count = df[column].isnull().sum()
            if missing_count > 0:
                missing_pct = (missing_count / len(df)) * 100
                missing_report[column] = {
                    "missing_count": int(missing_count),
                    "missing_pct": float(missing_pct),
                    "action": "none"
                }
                
                # Determine imputation strategy
                if missing_pct > drop_threshold:
                    # Drop column if too many missing values
                    df = df.drop(columns=[column])
                    self.cleaning_report["removed_columns"] += 1
                    missing_report[column]["action"] = "dropped_column"
                    logger.debug(f"Dropped column '{column}' ({missing_pct:.1f}% missing)")
                else:
                    # Impute based on column type
                    col_lower = column.lower()
                    
                    # Check if it's a date column
                    if any(pattern in col_lower for pattern in ['date', 'month', 'year']):
                        df[column] = df[column].fillna(pd.Timestamp('2014-01-01'))
                        missing_report[column]["action"] = "imputed_date"
                    
                    # Check if numeric
                    elif df[column].dtype in ['int64', 'float64']:
                        # Check if financial column
                        if any(fin_col in col_lower for fin_col in ['premium', 'claim', 'amount', 'value', 'price']):
                            df[column] = df[column].fillna(0)
                            missing_report[column]["action"] = "imputed_zero"
                        else:
                            df[column] = df[column].fillna(df[column].median())
                            missing_report[column]["action"] = "imputed_median"
                    
                    # Categorical columns
                    else:
                        mode_val = df[column].mode()[0] if not df[column].mode().empty else "Unknown"
                        df[column] = df[column].fillna(mode_val)
                        missing_report[column]["action"] = f"imputed_mode({mode_val})"
        
        total_missing = sum([info["missing_count"] for info in missing_report.values()])
        if total_missing > 0:
            logger.info(f"Handled {total_missing:,} missing values across {len(missing_report)} columns")
        
        return df, missing_report
    
    def _fix_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Fix data types for known column patterns."""
        
        date_patterns = ['date', 'month', 'year', 'intro']
        numeric_patterns = ['premium', 'claim', 'insured', 'value', 'amount', 'price', 'cost', 'sum']
        categorical_patterns = ['type', 'category', 'status', 'gender', 'province', 'make', 'model', 'code']
        
        for col in df.columns:
            col_lower = col.lower()
            
            # Convert to datetime
            if any(pattern in col_lower for pattern in date_patterns):
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    logger.debug(f"Converted '{col}' to datetime")
                except Exception as e:
                    logger.debug(f"Could not convert '{col}' to datetime: {e}")
            
            # Convert to numeric
            elif any(pattern in col_lower for pattern in numeric_patterns):
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                    logger.debug(f"Converted '{col}' to numeric")
                except Exception as e:
                    logger.debug(f"Could not convert '{col}' to numeric: {e}")
            
            # Convert to categorical (if reasonable number of categories)
            elif any(pattern in col_lower for pattern in categorical_patterns):
                if df[col].nunique() < self.config.get("max_categories", 50):
                    df[col] = df[col].astype('category')
                    logger.debug(f"Converted '{col}' to categorical ({df[col].nunique()} categories)")
        
        return df
    
    def _handle_outliers(self, df: pd.DataFrame) -> tuple:
        """Handle outliers in numeric columns using IQR method."""
        outlier_report = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        outlier_threshold = self.config.get("outlier_threshold", 1.5)
        
        for col in numeric_cols:
            col_lower = col.lower()
            
            # Only process financial-like columns for outlier handling
            if any(fin_col in col_lower for fin_col in ['premium', 'claim', 'amount', 'value', 'sum']):
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - outlier_threshold * IQR
                upper_bound = Q3 + outlier_threshold * IQR
                
                # Identify outliers
                outliers_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
                outlier_count = outliers_mask.sum()
                
                if outlier_count > 0:
                    # Cap outliers instead of removing
                    df[col] = df[col].clip(lower_bound, upper_bound)
                    outlier_report[col] = {
                        "outlier_count": int(outlier_count),
                        "method": "capped_iqr",
                        "bounds": [float(lower_bound), float(upper_bound)],
                        "outlier_pct": float((outlier_count / len(df)) * 100)
                    }
                    logger.debug(f"Capped {outlier_count} outliers in '{col}'")
        
        if outlier_report:
            total_outliers = sum([info["outlier_count"] for info in outlier_report.values()])
            logger.info(f"Capped {total_outliers:,} outliers across {len(outlier_report)} columns")
        
        return df, outlier_report
    
    def _validate_logical_consistency(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and fix logical inconsistencies."""
        
        # 1. Age validation (if Age column exists)
        if 'age' in df.columns:
            # Remove negative ages and ages over 100
            mask = (df['age'] < 0) | (df['age'] > 100)
            if mask.sum() > 0:
                df.loc[mask, 'age'] = np.nan
                df['age'] = df['age'].fillna(df['age'].median())
                logger.debug(f"Fixed {mask.sum()} invalid age values")
        
        # 2. Premium vs Claims consistency
        if all(col in df.columns for col in ['totalpremium', 'totalclaims']):
            # Claims can't be higher than premium (with reasonable multiplier)
            mask = df['totalclaims'] > (df['totalpremium'] * 5)  # Allow 5x for extreme cases
            if mask.sum() > 0:
                df.loc[mask, 'totalclaims'] = df['totalpremium'] * 0.5  # Cap at 50% of premium
                logger.debug(f"Fixed {mask.sum()} illogical claim amounts")
        
        # 3. Registration year vs transaction date
        if all(col in df.columns for col in ['registrationyear', 'transactionmonth']):
            if pd.api.types.is_datetime64_any_dtype(df['transactionmonth']):
                # Vehicle can't be registered after transaction
                mask = df['registrationyear'] > df['transactionmonth'].dt.year
                if mask.sum() > 0:
                    df.loc[mask, 'registrationyear'] = df['transactionmonth'].dt.year - 1
                    logger.debug(f"Fixed {mask.sum()} registration year inconsistencies")
        
        # 4. Sum insured consistency
        if all(col in df.columns for col in ['suminsured', 'customvalueestimate']):
            # Sum insured should be related to custom value
            ratio = df['suminsured'] / df['customvalueestimate']
            mask = (ratio > 5) | (ratio < 0.1)  # Remove extreme ratios
            if mask.sum() > 0:
                df.loc[mask, 'suminsured'] = df['customvalueestimate'] * 1.1  # Set to 110% of value
                logger.debug(f"Fixed {mask.sum()} sum insured inconsistencies")
        
        return df
    
    def _create_cleaning_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features that help with data quality assessment."""
        
        # 1. Data quality flags
        df['_has_missing_values'] = df.isnull().any(axis=1).astype(int)
        
        # 2. Outlier flags for financial columns
        financial_cols = [col for col in df.columns if any(x in col for x in ['premium', 'claim', 'value', 'amount'])]
        for col in financial_cols:
            if col in df.columns and df[col].dtype in ['int64', 'float64']:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                outlier_mask = (df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))
                df[f'_{col}_outlier'] = outlier_mask.astype(int)
        
        # 3. Inconsistency flags
        if all(col in df.columns for col in ['totalclaims', 'totalpremium']):
            df['_high_claim_ratio'] = (df['totalclaims'] / df['totalpremium'].replace(0, 1) > 1).astype(int)
        
        return df
    
    def _default_config(self) -> Dict:
        """Default cleaning configuration using project constants."""
        return {
            "drop_threshold": constants.CLEANING_CONFIG["drop_threshold"],
            "outlier_threshold": constants.CLEANING_CONFIG["outlier_threshold"],
            "max_categories": constants.CLEANING_CONFIG["max_categories"],
            "handle_outliers": True,
            "date_columns": constants.DATE_COLUMNS,
            "financial_columns": constants.FINANCIAL_COLUMNS
        }
    
    def get_cleaning_summary(self) -> Dict:
        """Get comprehensive cleaning summary."""
        return self.cleaning_report
    
    def save_cleaning_report(self, path: Optional[str] = None):
        """Save cleaning report to file."""
        import json
        
        if path is None:
            path = paths.REPORTS_DIR / "cleaning_report.json"
        
        # Ensure directory exists
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(self.cleaning_report, f, indent=2, default=str)
        
        logger.info(f"💾 Cleaning report saved to: {path}")
        return path