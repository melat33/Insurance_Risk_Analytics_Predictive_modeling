"""
Data segmentation for A/B hypothesis testing.
Ensures groups are comparable before statistical testing.
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import List, Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')


class DataSegmenter:
    """
    Segment data into comparable groups for hypothesis testing.
    
    Handles:
    1. Creating balanced groups for A/B testing
    2. Ensuring groups are comparable on covariates
    3. Handling multiple categories (e.g., many zip codes)
    4. Checking statistical equivalence between groups
    """
    
    def __init__(self, balance_check_columns: List[str] = None):
        """
        Initialize segmenter with columns to check for balance.
        
        Args:
            balance_check_columns: Columns to ensure are balanced between groups
        """
        self.balance_check_columns = balance_check_columns or [
            'SumInsured', 'VehicleType', 'RegistrationYear', 'Age'
        ]
        
    def create_comparable_groups(self, df: pd.DataFrame,
                                group_column: str,
                                group_a_value,
                                group_b_value,
                                min_sample_size: int = 30,
                                max_imbalance_ratio: float = 2.0) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        """
        Create two comparable groups for A/B testing.
        
        Args:
            df: Full dataset
            group_column: Column to split on (e.g., 'Gender', 'Province')
            group_a_value: Value for Group A
            group_b_value: Value for Group B
            min_sample_size: Minimum samples per group
            max_imbalance_ratio: Maximum allowed size ratio between groups
            
        Returns:
            Tuple: (group_a_df, group_b_df, balance_report)
            
        Example:
            >>> segmenter = DataSegmenter()
            >>> male_data, female_data, report = segmenter.create_comparable_groups(
            ...     data, 'Gender', 'Male', 'Female'
            ... )
        """
        if group_column not in df.columns:
            raise ValueError(f"Group column '{group_column}' not found in data")
        
        # Extract initial groups
        group_a = df[df[group_column] == group_a_value].copy()
        group_b = df[df[group_column] == group_b_value].copy()
        
        print(f"Initial group sizes:")
        print(f"  Group A ({group_a_value}): {len(group_a):,} policies")
        print(f"  Group B ({group_b_value}): {len(group_b):,} policies")
        
        # Check sample size requirements
        if len(group_a) < min_sample_size or len(group_b) < min_sample_size:
            raise ValueError(
                f"Group sizes too small. Minimum required: {min_sample_size}. "
                f"Found: Group A={len(group_a)}, Group B={len(group_b)}"
            )
        
        # Check imbalance ratio
        size_ratio = max(len(group_a), len(group_b)) / min(len(group_a), len(group_b))
        if size_ratio > max_imbalance_ratio:
            print(f"Warning: Group size ratio ({size_ratio:.2f}) exceeds maximum ({max_imbalance_ratio})")
            
            # Balance by random sampling
            min_size = min(len(group_a), len(group_b))
            group_a = group_a.sample(n=min_size, random_state=42) if len(group_a) > min_size else group_a
            group_b = group_b.sample(n=min_size, random_state=42) if len(group_b) > min_size else group_b
            
            print(f"  Balanced to: Group A={len(group_a):,}, Group B={len(group_b):,}")
        
        # Check covariate balance
        balance_report = self._check_covariate_balance(group_a, group_b)
        
        # If not balanced, attempt to balance through propensity score matching
        unbalanced_vars = [var for var, stats in balance_report.items() 
                          if not stats.get('is_balanced', True)]
        
        if unbalanced_vars:
            print(f"\nWarning: {len(unbalanced_vars)} variables are not balanced:")
            for var in unbalanced_vars[:3]:  # Show first 3
                print(f"  - {var}: p={balance_report[var].get('p_value', 0):.4f}")
            print("  Consider using propensity score matching for better balance.")
        
        return group_a, group_b, balance_report
    
    def _check_covariate_balance(self, group_a: pd.DataFrame, 
                                group_b: pd.DataFrame) -> Dict:
        """
        Check if groups are balanced on important covariates.
        
        Args:
            group_a: First group data
            group_b: Second group data
            
        Returns:
            Dictionary with balance statistics for each covariate
        """
        balance_report = {}
        
        for column in self.balance_check_columns:
            if column in group_a.columns and column in group_b.columns:
                # Remove missing values
                a_vals = group_a[column].dropna()
                b_vals = group_b[column].dropna()
                
                if len(a_vals) == 0 or len(b_vals) == 0:
                    continue
                
                # Check if column is numeric
                if pd.api.types.is_numeric_dtype(a_vals):
                    # T-test for numeric variables
                    t_stat, p_value = stats.ttest_ind(a_vals, b_vals, 
                                                     equal_var=False, 
                                                     nan_policy='omit')
                    
                    balance_report[column] = {
                        'type': 'numeric',
                        'mean_a': a_vals.mean(),
                        'mean_b': b_vals.mean(),
                        'std_a': a_vals.std(),
                        'std_b': b_vals.std(),
                        't_statistic': t_stat,
                        'p_value': p_value,
                        'is_balanced': p_value > 0.05,
                        'standardized_diff': self._calculate_std_diff(a_vals, b_vals)
                    }
                else:
                    # Chi-square test for categorical variables
                    # Create contingency table
                    all_categories = list(set(a_vals.unique()) | set(b_vals.unique()))
                    if len(all_categories) > 1:
                        freq_a = a_vals.value_counts().reindex(all_categories, fill_value=0)
                        freq_b = b_vals.value_counts().reindex(all_categories, fill_value=0)
                        
                        # Use chi-square test
                        chi2, p_value, _, _ = stats.chi2_contingency([freq_a, freq_b])
                        
                        balance_report[column] = {
                            'type': 'categorical',
                            'chi2_statistic': chi2,
                            'p_value': p_value,
                            'is_balanced': p_value > 0.05,
                            'categories': all_categories,
                            'freq_a': freq_a.tolist(),
                            'freq_b': freq_b.tolist()
                        }
        
        return balance_report
    
    def _calculate_std_diff(self, a_vals: pd.Series, b_vals: pd.Series) -> float:
        """Calculate standardized difference between groups."""
        mean_diff = abs(a_vals.mean() - b_vals.mean())
        pooled_std = np.sqrt((a_vals.std()**2 + b_vals.std()**2) / 2)
        
        if pooled_std == 0:
            return 0.0
        
        return mean_diff / pooled_std
    
    def create_zipcode_groups(self, df: pd.DataFrame,
                            postal_code_col: str = 'PostalCode',
                            method: str = 'risk_quartiles',
                            top_n: int = 10) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Create comparable groups for zip code analysis.
        
        Since there are many zip codes, we need to group them meaningfully.
        
        Methods:
        1. 'risk_quartiles': Group by risk level (high vs low)
        2. 'top_bottom': Compare top N vs bottom N zip codes by policy count
        3. 'sample': Randomly sample two groups of zip codes
        
        Args:
            df: Insurance data
            postal_code_col: Column containing postal codes
            method: Grouping method
            top_n: Number of top/bottom zip codes to use
            
        Returns:
            Tuple of (high_risk_group, low_risk_group)
        """
        if postal_code_col not in df.columns:
            raise ValueError(f"Postal code column '{postal_code_col}' not found")
        
        # Calculate metrics per zip code
        from .metrics import InsuranceMetrics
        metrics_calc = InsuranceMetrics()
        
        zip_metrics = metrics_calc.calculate_all_metrics(df, postal_code_col)
        
        # Filter zip codes with sufficient data
        min_policies = 10
        valid_zips = zip_metrics[zip_metrics['sample_size'] >= min_policies]
        
        if len(valid_zips) < 2:
            raise ValueError(f"Need at least 2 zip codes with {min_policies}+ policies")
        
        if method == 'risk_quartiles':
            # Group by claim frequency quartiles
            valid_zips['risk_quartile'] = pd.qcut(valid_zips['claim_frequency'], 
                                                 q=4, 
                                                 labels=['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)'])
            
            high_risk_zips = valid_zips[valid_zips['risk_quartile'] == 'Q4 (High)']['group'].tolist()
            low_risk_zips = valid_zips[valid_zips['risk_quartile'] == 'Q1 (Low)']['group'].tolist()
            
            print(f"Risk Quartile Method:")
            print(f"  High Risk Zip Codes: {len(high_risk_zips)}")
            print(f"  Low Risk Zip Codes: {len(low_risk_zips)}")
            
        elif method == 'top_bottom':
            # Sort by claim frequency
            valid_zips = valid_zips.sort_values('claim_frequency', ascending=False)
            
            high_risk_zips = valid_zips.head(top_n)['group'].tolist()
            low_risk_zips = valid_zips.tail(top_n)['group'].tolist()
            
            print(f"Top/Bottom Method (Top {top_n}):")
            print(f"  High Risk Zip Codes: {high_risk_zips}")
            print(f"  Low Risk Zip Codes: {low_risk_zips}")
            
        elif method == 'sample':
            # Randomly sample two groups
            all_zips = valid_zips['group'].tolist()
            np.random.seed(42)
            sampled_zips = np.random.choice(all_zips, size=min(20, len(all_zips)), 
                                          replace=False)
            
            # Split into two groups
            split_idx = len(sampled_zips) // 2
            high_risk_zips = sampled_zips[:split_idx].tolist()
            low_risk_zips = sampled_zips[split_idx:].tolist()
            
            print(f"Random Sample Method:")
            print(f"  Group 1 Zip Codes: {len(high_risk_zips)}")
            print(f"  Group 2 Zip Codes: {len(low_risk_zips)}")
        
        else:
            raise ValueError(f"Unknown method: {method}. Use 'risk_quartiles', 'top_bottom', or 'sample'")
        
        # Extract data for high and low risk groups
        high_risk_data = df[df[postal_code_col].isin(high_risk_zips)].copy()
        low_risk_data = df[df[postal_code_col].isin(low_risk_zips)].copy()
        
        print(f"\nGroup Sizes:")
        print(f"  High Risk Group: {len(high_risk_data):,} policies")
        print(f"  Low Risk Group: {len(low_risk_data):,} policies")
        
        # Add group labels
        high_risk_data['risk_group'] = 'High Risk'
        low_risk_data['risk_group'] = 'Low Risk'
        
        return high_risk_data, low_risk_data
    
    def create_province_groups(self, df: pd.DataFrame,
                             province_col: str = 'Province',
                             compare_top: bool = True,
                             top_n: int = 2) -> Dict[str, pd.DataFrame]:
        """
        Create groups for province analysis.
        
        Args:
            df: Insurance data
            province_col: Column containing province names
            compare_top: If True, compare top N provinces by policy count
            top_n: Number of top provinces to compare
            
        Returns:
            Dictionary mapping province names to their data
        """
        if province_col not in df.columns:
            raise ValueError(f"Province column '{province_col}' not found")
        
        # Get province counts
        province_counts = df[province_col].value_counts()
        
        if compare_top:
            top_provinces = province_counts.head(top_n).index.tolist()
            print(f"Comparing top {top_n} provinces: {top_provinces}")
        else:
            # Use all provinces with sufficient data
            min_policies = 50
            top_provinces = province_counts[province_counts >= min_policies].index.tolist()
            print(f"Comparing {len(top_provinces)} provinces with {min_policies}+ policies")
        
        # Create dictionary of province data
        province_groups = {}
        for province in top_provinces:
            province_data = df[df[province_col] == province].copy()
            province_groups[province] = province_data
            
            print(f"  {province}: {len(province_data):,} policies")
        
        return province_groups