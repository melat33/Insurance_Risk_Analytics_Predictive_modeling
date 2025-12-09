"""
ML Feature Engineering for Insurance Risk Modeling (Task 4)
Enhanced version specifically for predictive modeling
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class EnhancedFeatureEngineer:
    """Advanced feature engineering for insurance risk modeling."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_stats = {}
        
    def load_and_prepare_data(self, data_path):
        """Load data and prepare for modeling."""
        print("📥 Loading cleaned data...")
        df = pd.read_csv(data_path, sep='|')
        print(f"   Original shape: {df.shape}")
        
        # Clean column names (remove spaces, lowercase)
        df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
        
        # Convert numeric columns
        numeric_columns = ['registrationyear', 'totalclaims', 'totalpremium', 
                          'customvalueestimate', 'suminsured', 'kilowatts', 
                          'cubiccapacity', 'calculatedpremiumperterm']
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    def create_target_variables(self, df):
        """Create target variables for modeling."""
        df_processed = df.copy()
        
        # 1. Claim Severity Target (for claims > 0)
        if 'totalclaims' in df_processed.columns:
            # Create claim flag
            df_processed['has_claim'] = (df_processed['totalclaims'] > 0).astype(int)
            
            # Create log-transformed claim amount (handles skewness)
            df_processed['log_totalclaims'] = np.log1p(df_processed['totalclaims'])
            
            # Claim severity (only for positive claims)
            df_processed['claim_severity'] = df_processed['totalclaims']
            
        # 2. Premium Optimization Target
        if 'totalpremium' in df_processed.columns:
            # Create optimal premium target (premium adjusted by risk)
            df_processed['optimal_premium'] = self._calculate_optimal_premium(df_processed)
            
        # 3. Risk Score Target (for classification)
        df_processed['high_risk'] = self._calculate_risk_score(df_processed)
        
        return df_processed
    
    def _calculate_optimal_premium(self, df):
        """Calculate optimal premium based on risk factors."""
        base_premium = df['totalpremium'].copy()
        
        # Adjust based on risk factors (simplified formula)
        risk_adjustment = 1.0
        
        # Adjust for vehicle age if available
        if 'registrationyear' in df.columns:
            # Convert to numeric and handle errors
            try:
                vehicle_age = 2015 - pd.to_numeric(df['registrationyear'], errors='coerce')
                # Replace NaN with median
                vehicle_age = vehicle_age.fillna(vehicle_age.median())
                age_factor = 1 + (vehicle_age.clip(0, 20) * 0.02)  # 2% increase per year
                risk_adjustment *= age_factor
            except:
                # If conversion fails, use default
                risk_adjustment = risk_adjustment
        
        # Adjust for province risk (if province data exists)
        if 'province' in df.columns:
            # Create simple province risk mapping
            province_risk = {
                'GAUTENG': 1.15, 'KWAZULU-NATAL': 1.10, 'WESTERN CAPE': 1.05,
                'EASTERN CAPE': 1.08, 'MPUMALANGA': 1.07, 'LIMPOPO': 1.06,
                'NORTH WEST': 1.07, 'FREE STATE': 1.05, 'NORTHERN CAPE': 1.04
            }
            
            # Map province to risk factor
            province_factor = df['province'].astype(str).str.upper().map(province_risk).fillna(1.0)
            risk_adjustment *= province_factor
            
        return base_premium * risk_adjustment
    
    def _calculate_risk_score(self, df):
        """Calculate risk score for classification."""
        risk_score = np.zeros(len(df))
        
        # Vehicle age risk
        if 'registrationyear' in df.columns:
            try:
                vehicle_age = 2015 - pd.to_numeric(df['registrationyear'], errors='coerce')
                vehicle_age = vehicle_age.fillna(vehicle_age.median())
                risk_score += np.where(vehicle_age > 10, 2, 
                                      np.where(vehicle_age > 5, 1, 0))
            except:
                pass
        
        # Claim history risk
        if 'totalclaims' in df.columns:
            try:
                median_claims = df['totalclaims'].median()
                risk_score += np.where(df['totalclaims'] > median_claims, 2, 0)
            except:
                pass
            
        # Create binary high-risk flag (top 30%)
        if len(risk_score) > 0:
            threshold = np.percentile(risk_score[~np.isnan(risk_score)], 70)
            return (risk_score >= threshold).astype(int)
        else:
            return np.zeros(len(df))
    
    def create_vehicle_features(self, df):
        """Create vehicle-related features."""
        df_features = df.copy()
        
        # Vehicle age features
        if 'registrationyear' in df_features.columns:
            try:
                df_features['vehicle_age'] = 2015 - pd.to_numeric(df_features['registrationyear'], errors='coerce')
                df_features['vehicle_age'] = df_features['vehicle_age'].fillna(df_features['vehicle_age'].median())
                df_features['vehicle_age_squared'] = df_features['vehicle_age'] ** 2
                df_features['is_vehicle_old'] = (df_features['vehicle_age'] > 10).astype(int)
                df_features['is_vehicle_new'] = (df_features['vehicle_age'] <= 3).astype(int)
            except Exception as e:
                print(f"   Warning: Could not create vehicle age features: {e}")
                df_features['vehicle_age'] = 0
                df_features['vehicle_age_squared'] = 0
                df_features['is_vehicle_old'] = 0
                df_features['is_vehicle_new'] = 0
            
        # Vehicle value features
        if 'customvalueestimate' in df_features.columns:
            df_features['customvalueestimate'] = pd.to_numeric(df_features['customvalueestimate'], errors='coerce')
            df_features['customvalueestimate'] = df_features['customvalueestimate'].fillna(df_features['customvalueestimate'].median())
            df_features['log_vehicle_value'] = np.log1p(df_features['customvalueestimate'])
            if 'vehicle_age' in df_features.columns:
                df_features['value_per_age'] = df_features['customvalueestimate'] / (df_features['vehicle_age'] + 1)
            
        # Engine features
        if 'kilowatts' in df_features.columns:
            df_features['kilowatts'] = pd.to_numeric(df_features['kilowatts'], errors='coerce')
            df_features['kilowatts'] = df_features['kilowatts'].fillna(df_features['kilowatts'].median())
            if 'customvalueestimate' in df_features.columns:
                df_features['power_to_weight'] = df_features['kilowatts'] / df_features['customvalueestimate'].replace(0, 1)
            median_kw = df_features['kilowatts'].median()
            df_features['high_power'] = (df_features['kilowatts'] > median_kw).astype(int)
            
        return df_features
    
    def create_policy_features(self, df):
        """Create policy-related features."""
        df_features = df.copy()
        
        # Premium-related features
        if 'totalpremium' in df_features.columns:
            df_features['totalpremium'] = pd.to_numeric(df_features['totalpremium'], errors='coerce')
            df_features['totalpremium'] = df_features['totalpremium'].fillna(df_features['totalpremium'].median())
            df_features['log_premium'] = np.log1p(df_features['totalpremium'])
            df_features['premium_per_month'] = df_features['totalpremium'] / 12
            
        if 'suminsured' in df_features.columns:
            df_features['suminsured'] = pd.to_numeric(df_features['suminsured'], errors='coerce')
            df_features['suminsured'] = df_features['suminsured'].fillna(df_features['suminsured'].median())
            df_features['premium_to_sum_ratio'] = df_features['totalpremium'] / df_features['suminsured'].replace(0, 1)
            df_features['log_sum_insured'] = np.log1p(df_features['suminsured'])
            
        # Coverage features
        if 'covertype' in df_features.columns:
            cover_keywords = ['comprehensive', 'third party', 'theft', 'fire']
            for keyword in cover_keywords:
                col_name = f'covers_{keyword.replace(" ", "_")}'
                df_features[col_name] = df_features['covertype'].astype(str).str.contains(keyword, case=False, na=False).astype(int)
                
        return df_features
    
    def create_demographic_features(self, df):
        """Create demographic and location features."""
        df_features = df.copy()
        
        # Gender features
        if 'gender' in df_features.columns:
            df_features['gender'] = df_features['gender'].astype(str)
            df_features['is_male'] = df_features['gender'].str.contains('male', case=False, na=False).astype(int)
            df_features['is_female'] = (~df_features['is_male']).astype(int)
            
        # Location risk features
        if 'province' in df_features.columns:
            df_features['province'] = df_features['province'].astype(str)
            # Calculate province risk score based on historical claims
            if 'has_claim' in df_features.columns:
                province_claims = df_features.groupby('province')['has_claim'].mean()
                df_features['province_risk_score'] = df_features['province'].map(province_claims)
                df_features['province_risk_score'] = df_features['province_risk_score'].fillna(df_features['province_risk_score'].median())
            
        if 'postalcode' in df_features.columns:
            df_features['postalcode'] = df_features['postalcode'].astype(str)
            # Create zip code risk clusters
            if 'has_claim' in df_features.columns:
                zip_claims = df_features.groupby('postalcode')['has_claim'].mean()
                df_features['zipcode_risk_score'] = df_features['postalcode'].map(zip_claims)
                df_features['zipcode_risk_score'] = df_features['zipcode_risk_score'].fillna(df_features['zipcode_risk_score'].median())
            
        # Marital status features
        if 'maritalstatus' in df_features.columns:
            df_features['maritalstatus'] = df_features['maritalstatus'].astype(str)
            df_features['is_married'] = df_features['maritalstatus'].str.contains('married', case=False, na=False).astype(int)
            
        return df_features
    
    def create_temporal_features(self, df):
        """Create time-based features."""
        df_features = df.copy()
        
        # Check for transaction month
        date_cols = [col for col in df_features.columns if 'date' in col or 'month' in col]
        
        for col in date_cols:
            try:
                df_features[col] = pd.to_datetime(df_features[col], errors='coerce')
                
                # Extract temporal features
                df_features[f'{col}_year'] = df_features[col].dt.year.fillna(2015)
                df_features[f'{col}_month'] = df_features[col].dt.month.fillna(6)
                df_features[f'{col}_quarter'] = df_features[col].dt.quarter.fillna(2)
                df_features[f'{col}_dayofweek'] = df_features[col].dt.dayofweek.fillna(2)
                
            except Exception as e:
                print(f"   Warning: Could not parse date column {col}: {e}")
                continue
                
        return df_features
    
    def create_interaction_features(self, df):
        """Create interaction features."""
        df_features = df.copy()
        
        # Age-Risk interactions
        if all(col in df_features.columns for col in ['vehicle_age', 'province_risk_score']):
            df_features['age_risk_interaction'] = df_features['vehicle_age'] * df_features['province_risk_score']
            
        # Value-Risk interactions
        if all(col in df_features.columns for col in ['customvalueestimate', 'zipcode_risk_score']):
            df_features['value_risk_interaction'] = df_features['customvalueestimate'] * df_features['zipcode_risk_score']
            
        # Premium-Age interactions
        if all(col in df_features.columns for col in ['totalpremium', 'vehicle_age']):
            df_features['premium_age_ratio'] = df_features['totalpremium'] / (df_features['vehicle_age'] + 1)
            
        return df_features
    
    def prepare_for_modeling(self, df, target_column=None):
        """Final preparation for modeling."""
        df_model = df.copy()
        
        # Handle missing values
        numeric_cols = df_model.select_dtypes(include=[np.number]).columns
        categorical_cols = df_model.select_dtypes(include=['object']).columns
        
        # Fill numeric missing values with median
        for col in numeric_cols:
            if df_model[col].isnull().sum() > 0:
                median_val = df_model[col].median()
                df_model[col] = df_model[col].fillna(median_val)
                
        # Fill categorical missing values with mode
        for col in categorical_cols:
            if df_model[col].isnull().sum() > 0:
                mode_val = df_model[col].mode()[0] if not df_model[col].mode().empty else 'Unknown'
                df_model[col] = df_model[col].fillna(mode_val)
                
        return df_model
    
    def get_feature_sets(self, df):
        """Get organized feature sets for different modeling tasks."""
        
        # Define feature groups
        feature_sets = {
            'vehicle_features': [
                'vehicle_age', 'vehicle_age_squared', 'is_vehicle_old', 'is_vehicle_new',
                'customvalueestimate', 'log_vehicle_value', 'value_per_age',
                'kilowatts', 'power_to_weight', 'high_power'
            ],
            
            'policy_features': [
                'totalpremium', 'log_premium', 'premium_per_month',
                'suminsured', 'log_sum_insured', 'premium_to_sum_ratio',
            ],
            
            'demographic_features': [
                'is_male', 'is_female', 'province_risk_score', 'zipcode_risk_score',
                'is_married'
            ],
            
            'interaction_features': [
                'age_risk_interaction', 'value_risk_interaction', 'premium_age_ratio'
            ],
            
            'coverage_features': [
                col for col in df.columns if col.startswith('covers_')
            ],
            
            'temporal_features': [
                col for col in df.columns if any(x in col for x in ['_year', '_month', '_quarter', '_dayofweek'])
            ]
        }
        
        # Filter to only include columns that exist
        valid_feature_sets = {}
        for set_name, features in feature_sets.items():
            valid_features = [f for f in features if f in df.columns]
            if valid_features:
                valid_feature_sets[set_name] = valid_features
                
        return valid_feature_sets
    
    def run_full_pipeline(self, data_path):
        """Run complete feature engineering pipeline."""
        print("="*60)
        print("ENHANCED FEATURE ENGINEERING PIPELINE")
        print("="*60)
        
        # Step 1: Load data
        df = self.load_and_prepare_data(data_path)
        
        # Step 2: Create targets
        print("\n🎯 Creating target variables...")
        df = self.create_target_variables(df)
        
        # Step 3: Create features
        print("🔧 Creating vehicle features...")
        df = self.create_vehicle_features(df)
        
        print("📋 Creating policy features...")
        df = self.create_policy_features(df)
        
        print("👥 Creating demographic features...")
        df = self.create_demographic_features(df)
        
        print("⏰ Creating temporal features...")
        df = self.create_temporal_features(df)
        
        print("⚡ Creating interaction features...")
        df = self.create_interaction_features(df)
        
        # Step 4: Final preparation
        print("✨ Final preparation for modeling...")
        df = self.prepare_for_modeling(df)
        
        # Get feature sets
        feature_sets = self.get_feature_sets(df)
        
        print("\n" + "="*60)
        print("✅ FEATURE ENGINEERING COMPLETE")
        print("="*60)
        print(f"📊 Final dataset shape: {df.shape}")
        
        # Show available targets
        target_cols = []
        for col in df.columns:
            if any(keyword in col.lower() for keyword in ['claim', 'risk', 'premium', 'target']):
                target_cols.append(col)
        
        if target_cols:
            print(f"🎯 Available targets: {target_cols}")
        
        total_features = sum(len(features) for features in feature_sets.values())
        print(f"🔧 Feature sets created ({total_features} features):")
        for set_name, features in feature_sets.items():
            if features:  # Only show non-empty sets
                print(f"   • {set_name}: {len(features)} features")
            
        return df, feature_sets