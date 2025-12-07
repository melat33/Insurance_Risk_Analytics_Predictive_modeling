"""
Data loading utilities.
"""
import pandas as pd
from pathlib import Path
from ..config import paths, constants
from ..utils.logger import get_logger

logger = get_logger(__name__)

class InsuranceDataLoader:
    """Loader for insurance data files."""
    
    def __init__(self):
        self.data_path = paths.RAW_DATA_DIR / constants.RAW_DATA_FILE
    
    def load_raw_data(self, delimiter='|'):
        """
        Load the raw pipe-delimited data file.
        
        Args:
            delimiter: Column delimiter (default '|')
            
        Returns:
            pandas.DataFrame or None if file not found
        """
        if not self.data_path.exists():
            logger.error(f"Data file not found: {self.data_path}")
            return None
        
        logger.info(f"Loading data from: {self.data_path}")
        
        try:
            # Load with optimized settings for large files
            df = pd.read_csv(
                self.data_path,
                delimiter=delimiter,
                low_memory=False,
                on_bad_lines='warn'
            )
            
            logger.info(f"Successfully loaded {len(df):,} rows × {len(df.columns)} columns")
            
            # Convert date columns
            df = self._convert_date_columns(df)
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            return None
    
    def _convert_date_columns(self, df):
        """Convert potential date columns to datetime."""
        date_patterns = ['date', 'month', 'year', 'intro']
        
        for col in df.columns:
            col_lower = str(col).lower()
            if any(pattern in col_lower for pattern in date_patterns):
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    logger.debug(f"Converted column to datetime: {col}")
                except Exception as e:
                    logger.debug(f"Could not convert {col} to datetime: {e}")
        
        return df
    
    def validate_data(self, df):
        """Perform basic data validation."""
        if df is None:
            return {"status": "error", "message": "DataFrame is None"}
        
        validation = {
            "status": "success",
            "row_count": len(df),
            "column_count": len(df.columns),
            "missing_values": df.isnull().sum().sum(),
            "duplicates": df.duplicated().sum(),
            "data_types": dict(df.dtypes)
        }
        
        logger.info(f"Validation: {validation['row_count']:,} rows, "
                   f"{validation['missing_values']:,} missing values")
        
        return validation