"""
Data loading utilities for insurance analytics.
"""
import pandas as pd
from pathlib import Path
import sys

# Add project root to path to handle imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from src.config import paths, constants
    from src.utils.logger import get_logger
except ImportError:
    # Fallback imports if module structure isn't available
    import logging
    
    def get_logger(name):
        logger = logging.getLogger(name)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    
    # Create simple path constants
    class SimplePaths:
        RAW_DATA_DIR = project_root / "data" / "00_raw"
    
    class SimpleConstants:
        RAW_DATA_FILE = "MachineLearningRating_v3.txt"
    
    paths = SimplePaths()
    constants = SimpleConstants()

logger = get_logger(__name__)

class InsuranceDataLoader:
    """Loader for insurance data files."""
    
    def __init__(self, data_path=None):
        """
        Initialize data loader.
        
        Args:
            data_path: Optional custom path to data file
        """
        if data_path is None:
            self.data_path = paths.RAW_DATA_DIR / constants.RAW_DATA_FILE
        else:
            self.data_path = Path(data_path)
    
    def load_raw_data(self, delimiter='|', sample_size=None):
        """
        Load the raw pipe-delimited data file.
        
        Args:
            delimiter: Column delimiter (default '|')
            sample_size: Number of rows to load (None for all)
            
        Returns:
            pandas.DataFrame or None if file not found
        """
        if not self.data_path.exists():
            logger.error(f"Data file not found: {self.data_path}")
            # Try to find alternative locations
            alternative_paths = [
                self.data_path,
                Path("data/00_raw/MachineLearningRating_v3.txt"),
                Path("../data/00_raw/MachineLearningRating_v3.txt"),
                Path("MachineLearningRating_v3.txt")
            ]
            
            for alt_path in alternative_paths:
                if alt_path.exists():
                    self.data_path = alt_path
                    logger.info(f"Found data at: {alt_path}")
                    break
            else:
                logger.error("Could not find data file in any location")
                return None
        
        logger.info(f"Loading data from: {self.data_path}")
        
        try:
            # Load with optimized settings
            read_params = {
                'delimiter': delimiter,
                'low_memory': False,
                'on_bad_lines': 'warn',
                'encoding': 'utf-8'
            }
            
            if sample_size:
                read_params['nrows'] = sample_size
            
            df = pd.read_csv(self.data_path, **read_params)
            
            logger.info(f"✅ Successfully loaded {len(df):,} rows × {len(df.columns)} columns")
            
            # Convert date columns
            df = self._convert_date_columns(df)
            
            # Clean column names
            df.columns = self._clean_column_names(df.columns)
            
            return df
            
        except UnicodeDecodeError:
            # Try different encodings
            logger.warning("UTF-8 failed, trying latin-1 encoding...")
            try:
                df = pd.read_csv(self.data_path, delimiter=delimiter, 
                               low_memory=False, encoding='latin-1')
                logger.info(f"✅ Loaded with latin-1: {len(df):,} rows")
                df = self._convert_date_columns(df)
                df.columns = self._clean_column_names(df.columns)
                return df
            except Exception as e:
                logger.error(f"Failed with latin-1: {e}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to load data: {e}")
            return None
    
    def _clean_column_names(self, columns):
        """Clean and standardize column names."""
        cleaned = []
        for col in columns:
            # Remove special characters, convert to lowercase with underscores
            col_clean = str(col).strip().lower()
            col_clean = col_clean.replace(' ', '_').replace('-', '_')
            col_clean = col_clean.replace('__', '_').strip('_')
            cleaned.append(col_clean)
        return cleaned
    
    def _convert_date_columns(self, df):
        """Convert potential date columns to datetime."""
        date_patterns = ['date', 'month', 'year', 'intro', 'time']
        
        for col in df.columns:
            col_lower = str(col).lower()
            if any(pattern in col_lower for pattern in date_patterns):
                try:
                    before_count = df[col].notna().sum() if hasattr(df[col], 'notna') else 0
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    after_count = df[col].notna().sum() if hasattr(df[col], 'notna') else 0
                    
                    if after_count > before_count:
                        logger.debug(f"Converted {col} to datetime: {after_count:,} valid dates")
                except Exception as e:
                    logger.debug(f"Could not convert {col} to datetime: {e}")
        
        return df
    
    def validate_data(self, df):
        """Perform basic data validation."""
        if df is None or df.empty:
            return {"status": "error", "message": "DataFrame is empty or None"}
        
        validation = {
            "status": "success",
            "row_count": len(df),
            "column_count": len(df.columns),
            "missing_values": int(df.isnull().sum().sum()),
            "duplicates": int(df.duplicated().sum()),
            "memory_usage_mb": float(df.memory_usage(deep=True).sum() / 1024**2),
            "columns": list(df.columns[:20]),  # First 20 columns
            "data_types": {str(k): str(v) for k, v in df.dtypes.to_dict().items()}
        }
        
        logger.info(f"Validation: {validation['row_count']:,} rows, "
                   f"{validation['missing_values']:,} missing values, "
                   f"{validation['memory_usage_mb']:.1f} MB")
        
        return validation
    
    def get_file_info(self):
        """Get information about the data file."""
        if not self.data_path.exists():
            return {"exists": False, "path": str(self.data_path)}
        
        try:
            size_mb = self.data_path.stat().st_size / 1024**2
            return {
                "exists": True,
                "path": str(self.data_path),
                "size_mb": round(size_mb, 2),
                "last_modified": self.data_path.stat().st_mtime
            }
        except Exception as e:
            return {"exists": True, "path": str(self.data_path), "error": str(e)}