"""
Module for calculating Key Performance Indicators (KPIs) for insurance risk analysis.
This module computes Claim Frequency, Claim Severity, and Margin for hypothesis testing.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class InsuranceMetrics:
    """
    Class to calculate insurance-specific KPIs for hypothesis testing.
    
    Key Metrics:
    1. Claim Frequency: Proportion of policies with at least one claim
    2. Claim Severity: Average claim amount when a claim occurs
    3. Margin: Profit per policy (Premium - Claims)
    
    These metrics are used to test the 4 hypotheses about risk differences.
    """
    
    def __init__(self, claim_threshold: float = 0):
        """
        Initialize metrics calculator.
        
        Args:
            claim_threshold: Minimum claim amount to count as having a claim
        """
        self.claim_threshold = claim_threshold
        
    def calculate_claim_frequency(self, df: pd.DataFrame, 
                                  claim_col: str = 'TotalClaims') -> float:
        """
        Calculate claim frequency (proportion of policies with claims).
        
        Formula: (# policies with claims > threshold) / (total policies)
        
        Args:
            df: DataFrame containing insurance policies
            claim_col: Column name for claim amounts
            
        Returns:
            float: Claim frequency between 0 and 1
            
        Example:
            >>> metrics = InsuranceMetrics()
            >>> frequency = metrics.calculate_claim_frequency(data)
            >>> print(f"Claim Frequency: {frequency:.2%}")
        """
        if claim_col not in df.columns:
            raise ValueError(f"Column '{claim_col}' not found in DataFrame")
            
        total_policies = len(df)
        if total_policies == 0:
            return 0.0
            
        has_claim = (df[claim_col] > self.claim_threshold).astype(int)
        frequency = has_claim.mean()
        
        return frequency
    
    def calculate_claim_severity(self, df: pd.DataFrame,
                                claim_col: str = 'TotalClaims',
                                return_details: bool = False) -> Dict:
        """
        Calculate claim severity metrics.
        
        Severity = Average claim amount for policies WITH claims.
        Also calculates additional metrics like standard deviation and confidence intervals.
        
        Args:
            df: DataFrame containing insurance policies
            claim_col: Column name for claim amounts
            return_details: If True, return detailed statistics
            
        Returns:
            dict: Dictionary with severity metrics
            
        Example:
            >>> metrics = InsuranceMetrics()
            >>> severity = metrics.calculate_claim_severity(data)
            >>> print(f"Average Claim: R{severity['mean']:.2f}")
        """
        if claim_col not in df.columns:
            raise ValueError(f"Column '{claim_col}' not found in DataFrame")
            
        # Filter to policies with claims
        claims_data = df[df[claim_col] > self.claim_threshold][claim_col]
        
        if len(claims_data) == 0:
            result = {
                'mean': 0.0,
                'std': 0.0,
                'count': 0,
                'min': 0.0,
                'max': 0.0,
                'median': 0.0
            }
            return result if return_details else 0.0
        
        result = {
            'mean': claims_data.mean(),
            'std': claims_data.std(),
            'count': len(claims_data),
            'min': claims_data.min(),
            'max': claims_data.max(),
            'median': claims_data.median(),
            'total_claims': claims_data.sum()
        }
        
        # Calculate 95% confidence interval for mean
        if len(claims_data) > 1:
            from scipy import stats
            se = claims_data.std() / np.sqrt(len(claims_data))
            ci = stats.t.interval(0.95, len(claims_data)-1, 
                                  loc=claims_data.mean(), scale=se)
            result['ci_lower'] = ci[0]
            result['ci_upper'] = ci[1]
        
        return result if return_details else result['mean']
    
    def calculate_margin(self, df: pd.DataFrame,
                        premium_col: str = 'TotalPremium',
                        claim_col: str = 'TotalClaims',
                        per_policy: bool = True) -> float:
        """
        Calculate insurance margin (profit).
        
        Margin = Total Premium - Total Claims
        Can be calculated per policy or in aggregate.
        
        Args:
            df: DataFrame containing insurance policies
            premium_col: Column name for premium amounts
            claim_col: Column name for claim amounts
            per_policy: If True, return average margin per policy
            
        Returns:
            float: Margin amount
            
        Example:
            >>> metrics = InsuranceMetrics()
            >>> margin = metrics.calculate_margin(data)
            >>> print(f"Average Margin per Policy: R{margin:.2f}")
        """
        required_cols = [premium_col, claim_col]
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            raise ValueError(f"Columns not found: {missing_cols}")
        
        # Calculate margin for each policy
        df['_margin'] = df[premium_col] - df[claim_col]
        
        if per_policy:
            margin = df['_margin'].mean()
        else:
            margin = df['_margin'].sum()
        
        # Clean up temporary column
        df.drop('_margin', axis=1, inplace=True, errors='ignore')
        
        return margin
    
    def calculate_all_metrics(self, df: pd.DataFrame,
                             group_col: str = None) -> pd.DataFrame:
        """
        Calculate all KPIs for each group in the data.
        
        Args:
            df: DataFrame containing insurance policies
            group_col: Column to group by (e.g., 'Province', 'Gender')
            
        Returns:
            DataFrame: Metrics for each group
            
        Example:
            >>> metrics = InsuranceMetrics()
            >>> province_metrics = metrics.calculate_all_metrics(data, 'Province')
        """
        if group_col and group_col not in df.columns:
            raise ValueError(f"Group column '{group_col}' not found")
        
        metrics_list = []
        
        if group_col:
            groups = df[group_col].unique()
            for group in groups:
                group_data = df[df[group_col] == group]
                group_metrics = self._calculate_group_metrics(group_data, group, group_col)
                metrics_list.append(group_metrics)
        else:
            # Calculate for entire dataset
            overall_metrics = self._calculate_group_metrics(df, 'Overall', None)
            metrics_list.append(overall_metrics)
        
        metrics_df = pd.DataFrame(metrics_list)
        return metrics_df
    
    def _calculate_group_metrics(self, df: pd.DataFrame, 
                                group_name: str, 
                                group_col: str) -> Dict:
        """Calculate metrics for a specific group."""
        if len(df) == 0:
            return {
                'group': group_name,
                'group_column': group_col,
                'sample_size': 0,
                'claim_frequency': 0.0,
                'claim_severity_mean': 0.0,
                'average_margin': 0.0,
                'total_premium': 0.0,
                'total_claims': 0.0,
                'loss_ratio': 0.0
            }
        
        frequency = self.calculate_claim_frequency(df)
        severity = self.calculate_claim_severity(df, return_details=True)
        margin = self.calculate_margin(df)
        total_premium = df['TotalPremium'].sum()
        total_claims = df['TotalClaims'].sum()
        loss_ratio = total_claims / total_premium if total_premium > 0 else 0
        
        return {
            'group': group_name,
            'group_column': group_col,
            'sample_size': len(df),
            'policies_with_claims': (df['TotalClaims'] > self.claim_threshold).sum(),
            'claim_frequency': frequency,
            'claim_severity_mean': severity['mean'] if isinstance(severity, dict) else severity,
            'claim_severity_std': severity.get('std', 0) if isinstance(severity, dict) else 0,
            'average_margin': margin,
            'total_premium': total_premium,
            'total_claims': total_claims,
            'loss_ratio': loss_ratio,
            'premium_per_policy': df['TotalPremium'].mean()
        }


# Helper function for quick analysis
def quick_metrics_report(df: pd.DataFrame, group_by: str = None) -> None:
    """
    Generate a quick report of insurance metrics.
    
    Args:
        df: Insurance data DataFrame
        group_by: Column to group by for comparison
    """
    metrics = InsuranceMetrics()
    
    print("=" * 80)
    print("INSURANCE METRICS REPORT")
    print("=" * 80)
    
    # Overall metrics
    overall_freq = metrics.calculate_claim_frequency(df)
    overall_severity = metrics.calculate_claim_severity(df)
    overall_margin = metrics.calculate_margin(df)
    loss_ratio = df['TotalClaims'].sum() / df['TotalPremium'].sum()
    
    print(f"\nOVERALL PORTFOLIO:")
    print(f"  Total Policies: {len(df):,}")
    print(f"  Claim Frequency: {overall_freq:.2%}")
    print(f"  Average Claim Severity: R{overall_severity:,.2f}")
    print(f"  Average Margin per Policy: R{overall_margin:,.2f}")
    print(f"  Loss Ratio: {loss_ratio:.2%}")
    
    if group_by and group_by in df.columns:
        print(f"\nMETRICS BY {group_by.upper()}:")
        group_metrics = metrics.calculate_all_metrics(df, group_by)
        
        # Sort by claim frequency (highest risk first)
        group_metrics = group_metrics.sort_values('claim_frequency', ascending=False)
        
        for _, row in group_metrics.iterrows():
            print(f"\n  {row['group']}:")
            print(f"    Policies: {row['sample_size']:,}")
            print(f"    Claim Frequency: {row['claim_frequency']:.2%}")
            print(f"    Average Severity: R{row['claim_severity_mean']:,.2f}")
            print(f"    Loss Ratio: {row['loss_ratio']:.2%}")
    
    print("\n" + "=" * 80)