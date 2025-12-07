# src/features/transformer.py
"""
Data transformation module for machine learning preparation.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional
import logging
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer

logger = logging.getLogger(__name__)

class InsuranceDataTransformer:
    """Transformer for preparing insurance data for modeling."""
    
    def __init__(self):
        self.transformers = {}
        self.column_categories = {}
        self.feature_names = []
    
    def prepare_for_modeling(self, df: pd.DataFrame, config: Optional[Dict] = None) -> pd.DataFrame:
        """
        Prepare data for machine learning modeling.
        
        Args:
            df: DataFrame with engineered features
            config: Transformation configuration
            
        Returns:
            Transformed DataFrame ready for modeling
        """
        if config is None:
            config = self._default_config()
        
        df_transformed = df.copy()
        
        logger.info("🔄 Preparing data for modeling...")
        
        # 1. Separate features and target
        target = None
        if config.get("target_column") in df_transformed.columns:
            target = df_transformed[config["target_column"]]
            df_transformed = df_transformed.drop(columns=[config["target_column"]])
        
        # 2. Handle missing values
        df_transformed = self._handle_missing_values(df_transformed, config)
        
        # 3. Encode categorical variables
        df_transformed, encoders = self._encode_categorical(df_transformed, config)
        
        # 4. Scale numerical features
        df_transformed, scalers = self._scale_numerical(df_transformed, config)
        
        # 5. Feature selection
        if config.get("feature_selection", False):
            df_transformed = self._select_features(df_transformed, config)
        
        # 6. Add target back if needed
        if target is not None and config.get("include_target", True):
            df_transformed[config["target_column"]] = target
        
        logger.info(f"✅ Data prepared: {df_transformed.shape}")
        
        return df_transformed
    
    def _handle_missing_values(self, df: pd.DataFrame, config: Dict) -> pd.DataFrame:
        """Handle any remaining missing values."""
        
        # Separate numeric and categorical
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        # Impute numeric with median
        if len(numeric_cols) > 0:
            numeric_imputer = SimpleImputer(strategy='median')
            df[numeric_cols] = numeric_imputer.fit_transform(df[numeric_cols])
            self.transformers['numeric_imputer'] = numeric_imputer
        
        # Impute categorical with mode
        if len(categorical_cols) > 0:
            categorical_imputer = SimpleImputer(strategy='most_frequent')
            df[categorical_cols] = categorical_imputer.fit_transform(df[categorical_cols])
            self.transformers['categorical_imputer'] = categorical_imputer
        
        return df
    
    def _encode_categorical(self, df: pd.DataFrame, config: Dict) -> tuple:
        """Encode categorical variables."""
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        if len(categorical_cols) == 0:
            return df, {}
        
        encoders = {}
        
        for col in categorical_cols:
            n_unique = df[col].nunique()
            
            if n_unique <= config.get("max_categories_label_encode", 10):
                # Use label encoding for low cardinality
                encoder = LabelEncoder()
                df[col] = encoder.fit_transform(df[col].astype(str))
                encoders[col] = {'type': 'label', 'encoder': encoder}
                
            elif n_unique <= config.get("max_categories_onehot", 50):
                # Use one-hot encoding for medium cardinality
                encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                encoded = encoder.fit_transform(df[[col]])
                
                # Create column names
                encoded_cols = [f"{col}_{cat}" for cat in encoder.categories_[0]]
                
                # Add to dataframe
                encoded_df = pd.DataFrame(encoded, columns=encoded_cols, index=df.index)
                df = pd.concat([df.drop(columns=[col]), encoded_df], axis=1)
                
                encoders[col] = {'type': 'onehot', 'encoder': encoder, 'columns': encoded_cols}
                
            else:
                # For high cardinality, use frequency encoding
                freq = df[col].value_counts(normalize=True)
                df[col] = df[col].map(freq)
                df[col] = df[col].fillna(0)
                encoders[col] = {'type': 'frequency', 'mapping': freq.to_dict()}
        
        self.transformers['encoders'] = encoders
        return df, encoders
    
    def _scale_numerical(self, df: pd.DataFrame, config: Dict) -> tuple:
        """Scale numerical features."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        if len(numeric_cols) == 0:
            return df, {}
        
        # Remove columns that shouldn't be scaled
        exclude_patterns = ['_outlier', '_flag', 'is_', 'has_']
        cols_to_scale = [
            col for col in numeric_cols 
            if not any(pattern in col for pattern in exclude_patterns)
        ]
        
        if len(cols_to_scale) == 0:
            return df, {}
        
        scaler = StandardScaler()
        df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
        
        self.transformers['scaler'] = scaler
        return df, scaler
    
    def _select_features(self, df: pd.DataFrame, config: Dict) -> pd.DataFrame:
        """Select important features."""
        
        # Correlation-based selection
        if config.get("selection_method") == "correlation":
            df = self._select_by_correlation(df, config)
        
        # Variance-based selection
        elif config.get("selection_method") == "variance":
            df = self._select_by_variance(df, config)
        
        return df
    
    def _select_by_correlation(self, df: pd.DataFrame, config: Dict) -> pd.DataFrame:
        """Select features based on correlation."""
        
        # Calculate correlation matrix for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) < 2:
            return df
        
        corr_matrix = df[numeric_cols].corr().abs()
        
        # Select upper triangle of correlation matrix
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        
        # Find features with correlation greater than threshold
        to_drop = [column for column in upper.columns if any(upper[column] > config.get("correlation_threshold", 0.95))]
        
        if to_drop:
            logger.info(f"Dropping {len(to_drop)} highly correlated features: {to_drop}")
            df = df.drop(columns=to_drop)
        
        return df
    
    def _select_by_variance(self, df: pd.DataFrame, config: Dict) -> pd.DataFrame:
        """Select features based on variance."""
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) == 0:
            return df
        
        # Calculate variance
        variances = df[numeric_cols].var()
        
        # Keep features with variance above threshold
        threshold = config.get("variance_threshold", 0.01)
        low_variance = variances[variances < threshold].index.tolist()
        
        if low_variance:
            logger.info(f"Dropping {len(low_variance)} low-variance features: {low_variance}")
            df = df.drop(columns=low_variance)
        
        return df
    
    def _default_config(self) -> Dict:
        """Default transformation configuration."""
        return {
            "target_column": "has_claim",
            "include_target": True,
            "max_categories_label_encode": 10,
            "max_categories_onehot": 50,
            "feature_selection": True,
            "selection_method": "correlation",
            "correlation_threshold": 0.95,
            "variance_threshold": 0.01
        }
    
    def get_transformers(self) -> Dict:
        """Get all fitted transformers."""
        return self.transformers
    
    def save_transformers(self, path: str):
        """Save transformers to file."""
        import joblib
        joblib.dump(self.transformers, path)
        logger.info(f"💾 Transformers saved to: {path}")
    
    def load_transformers(self, path: str):
        """Load transformers from file."""
        import joblib
        self.transformers = joblib.load(path)
        logger.info(f"📥 Transformers loaded from: {path}")