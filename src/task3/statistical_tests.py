"""
Statistical tests for hypothesis testing in insurance analytics.
Contains tests for proportions (frequency), means (severity), and margins.
"""

import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.stats import proportion, weightstats
from typing import Dict, Tuple, Any
import warnings
warnings.filterwarnings('ignore')


class StatisticalTests:
    """
    Perform statistical tests for insurance hypothesis testing.
    
    Tests included:
    1. Proportion tests (for claim frequency)
    2. Mean comparison tests (for claim severity, margin)
    3. ANOVA (for multiple groups)
    4. Chi-square tests (for categorical comparisons)
    """
    
    def __init__(self, alpha: float = 0.05):
        """
        Initialize statistical tester.
        
        Args:
            alpha: Significance level (default 0.05)
        """
        self.alpha = alpha
        
    def test_claim_frequency(self, group_a_claims: pd.Series,
                           group_b_claims: pd.Series,
                           test_type: str = 'chi_square') -> Dict[str, Any]:
        """
        Test if claim frequencies are significantly different.
        
        Args:
            group_a_claims: Claim amounts for Group A
            group_b_claims: Claim amounts for Group B
            test_type: 'chi_square' or 'z_test'
            
        Returns:
            Dictionary with test results
            
        Example:
            >>> tester = StatisticalTests()
            >>> results = tester.test_claim_frequency(male_claims, female_claims)
            >>> print(f"p-value: {results['p_value']:.4f}")
        """
        # Convert to binary (1 if claim > 0)
        binary_a = (group_a_claims > 0).astype(int)
        binary_b = (group_b_claims > 0).astype(int)
        
        n_a = len(binary_a)
        n_b = len(binary_b)
        claims_a = binary_a.sum()
        claims_b = binary_b.sum()
        
        p_a = claims_a / n_a if n_a > 0 else 0
        p_b = claims_b / n_b if n_b > 0 else 0
        
        # Calculate effect size (risk difference)
        risk_difference = p_a - p_b
        relative_risk = p_a / p_b if p_b > 0 else np.inf
        
        if test_type == 'z_test':
            # Z-test for proportions
            z_stat, p_value = proportion.proportions_ztest(
                count=[claims_a, claims_b],
                nobs=[n_a, n_b],
                alternative='two-sided'
            )
            
            # Calculate confidence interval for risk difference
            se = np.sqrt(p_a*(1-p_a)/n_a + p_b*(1-p_b)/n_b)
            ci_lower = risk_difference - 1.96 * se
            ci_upper = risk_difference + 1.96 * se
            
        elif test_type == 'chi_square':
            # Chi-square test for independence
            contingency_table = pd.DataFrame({
                'Group A': [claims_a, n_a - claims_a],
                'Group B': [claims_b, n_b - claims_b]
            }, index=['Claims', 'No Claims'])
            
            chi2, p_value, dof, expected = stats.chi2_contingency(contingency_table)
            z_stat = np.sqrt(chi2)  # Approximate Z from chi-square
            
            # For chi-square, use normal approximation for CI
            se = np.sqrt(p_a*(1-p_a)/n_a + p_b*(1-p_b)/n_b)
            ci_lower = risk_difference - 1.96 * se
            ci_upper = risk_difference + 1.96 * se
        
        else:
            raise ValueError(f"Unknown test type: {test_type}. Use 'z_test' or 'chi_square'")
        
        # Decision
        reject_null = p_value < self.alpha
        
        return {
            'test_type': 'frequency',
            'method': test_type,
            'statistic': z_stat,
            'p_value': p_value,
            'reject_null': reject_null,
            'group_a_frequency': p_a,
            'group_b_frequency': p_b,
            'risk_difference': risk_difference,
            'relative_risk': relative_risk,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'n_a': n_a,
            'n_b': n_b,
            'claims_a': claims_a,
            'claims_b': claims_b
        }
    
    def test_claim_severity(self, group_a_claims: pd.Series,
                          group_b_claims: pd.Series,
                          test_type: str = 't_test') -> Dict[str, Any]:
        """
        Test if claim severities are significantly different.
        
        Args:
            group_a_claims: Claim amounts for Group A (including zeros)
            group_b_claims: Claim amounts for Group B (including zeros)
            test_type: 't_test' or 'mannwhitney'
            
        Returns:
            Dictionary with test results
        """
        # Filter to only policies with claims
        severity_a = group_a_claims[group_a_claims > 0]
        severity_b = group_b_claims[group_b_claims > 0]
        
        if len(severity_a) == 0 or len(severity_b) == 0:
            return {
                'test_type': 'severity',
                'method': test_type,
                'statistic': np.nan,
                'p_value': 1.0,
                'reject_null': False,
                'message': 'Insufficient claims data for one or both groups'
            }
        
        mean_a = severity_a.mean()
        mean_b = severity_b.mean()
        std_a = severity_a.std()
        std_b = severity_b.std()
        
        if test_type == 't_test':
            # Independent t-test
            t_stat, p_value = stats.ttest_ind(severity_a, severity_b, 
                                            equal_var=False)
            
            # Calculate Cohen's d (effect size)
            n_a = len(severity_a)
            n_b = len(severity_b)
            pooled_std = np.sqrt(((n_a-1)*std_a**2 + (n_b-1)*std_b**2) / (n_a + n_b - 2))
            cohens_d = (mean_a - mean_b) / pooled_std if pooled_std > 0 else 0
            
            # Confidence interval for mean difference
            se_diff = np.sqrt(std_a**2/n_a + std_b**2/n_b)
            mean_diff = mean_a - mean_b
            ci_lower = mean_diff - 1.96 * se_diff
            ci_upper = mean_diff + 1.96 * se_diff
            
        elif test_type == 'mannwhitney':
            # Mann-Whitney U test (non-parametric)
            u_stat, p_value = stats.mannwhitneyu(severity_a, severity_b, 
                                               alternative='two-sided')
            t_stat = u_stat
            
            # For Mann-Whitney, calculate Hodges-Lehmann estimator
            # (pseudo-median difference)
            all_diffs = []
            for x in severity_a:
                for y in severity_b:
                    all_diffs.append(x - y)
            hodges_lehmann = np.median(all_diffs) if all_diffs else 0
            
            cohens_d = hodges_lehmann / np.sqrt(len(all_diffs)) if all_diffs else 0
            ci_lower = ci_upper = np.nan  # Non-parametric CI calculation is complex
            
        else:
            raise ValueError(f"Unknown test type: {test_type}. Use 't_test' or 'mannwhitney'")
        
        reject_null = p_value < self.alpha
        
        return {
            'test_type': 'severity',
            'method': test_type,
            'statistic': t_stat,
            'p_value': p_value,
            'reject_null': reject_null,
            'mean_a': mean_a,
            'mean_b': mean_b,
            'std_a': std_a,
            'std_b': std_b,
            'mean_difference': mean_a - mean_b,
            'cohens_d': cohens_d,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'n_a': len(severity_a),
            'n_b': len(severity_b)
        }
    
    def test_margin_difference(self, group_a_margins: pd.Series,
                             group_b_margins: pd.Series,
                             test_type: str = 't_test') -> Dict[str, Any]:
        """
        Test if margins (profit) are significantly different.
        
        Args:
            group_a_margins: Margin values for Group A
            group_b_margins: Margin values for Group B
            test_type: 't_test' or 'mannwhitney'
            
        Returns:
            Dictionary with test results
        """
        if len(group_a_margins) == 0 or len(group_b_margins) == 0:
            return {
                'test_type': 'margin',
                'method': test_type,
                'statistic': np.nan,
                'p_value': 1.0,
                'reject_null': False,
                'message': 'Insufficient data for one or both groups'
            }
        
        mean_a = group_a_margins.mean()
        mean_b = group_b_margins.mean()
        std_a = group_a_margins.std()
        std_b = group_b_margins.std()
        
        if test_type == 't_test':
            # Independent t-test
            t_stat, p_value = stats.ttest_ind(group_a_margins, group_b_margins,
                                            equal_var=False)
            
            # Calculate effect size
            n_a = len(group_a_margins)
            n_b = len(group_b_margins)
            pooled_std = np.sqrt(((n_a-1)*std_a**2 + (n_b-1)*std_b**2) / (n_a + n_b - 2))
            cohens_d = (mean_a - mean_b) / pooled_std if pooled_std > 0 else 0
            
            # Confidence interval
            se_diff = np.sqrt(std_a**2/n_a + std_b**2/n_b)
            mean_diff = mean_a - mean_b
            ci_lower = mean_diff - 1.96 * se_diff
            ci_upper = mean_diff + 1.96 * se_diff
            
        elif test_type == 'mannwhitney':
            # Mann-Whitney U test
            u_stat, p_value = stats.mannwhitneyu(group_a_margins, group_b_margins,
                                               alternative='two-sided')
            t_stat = u_stat
            cohens_d = ci_lower = ci_upper = np.nan
            
        else:
            raise ValueError(f"Unknown test type: {test_type}")
        
        reject_null = p_value < self.alpha
        
        return {
            'test_type': 'margin',
            'method': test_type,
            'statistic': t_stat,
            'p_value': p_value,
            'reject_null': reject_null,
            'mean_a': mean_a,
            'mean_b': mean_b,
            'std_a': std_a,
            'std_b': std_b,
            'mean_difference': mean_a - mean_b,
            'cohens_d': cohens_d,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'n_a': n_a,
            'n_b': n_b,
            'total_margin_a': group_a_margins.sum(),
            'total_margin_b': group_b_margins.sum()
        }
    
    def test_multiple_groups(self, groups_dict: Dict[str, pd.DataFrame],
                           metric: str = 'frequency',
                           test_type: str = 'anova') -> Dict[str, Any]:
        """
        Test differences across multiple groups (e.g., multiple provinces).
        
        Args:
            groups_dict: Dictionary mapping group names to DataFrames
            metric: 'frequency', 'severity', or 'margin'
            test_type: 'anova' or 'kruskal'
            
        Returns:
            Dictionary with test results
        """
        if len(groups_dict) < 2:
            return {
                'test_type': f'multiple_groups_{metric}',
                'method': test_type,
                'statistic': np.nan,
                'p_value': 1.0,
                'reject_null': False,
                'message': 'Need at least 2 groups for comparison'
            }
        
        # Prepare data based on metric
        data_by_group = {}
        
        for group_name, df in groups_dict.items():
            if metric == 'frequency':
                # Calculate claim frequency for this group
                has_claim = (df['TotalClaims'] > 0).astype(int)
                data_by_group[group_name] = has_claim
                
            elif metric == 'severity':
                # Get claim amounts for policies with claims
                severity = df[df['TotalClaims'] > 0]['TotalClaims']
                data_by_group[group_name] = severity
                
            elif metric == 'margin':
                # Calculate margins
                margins = df['TotalPremium'] - df['TotalClaims']
                data_by_group[group_name] = margins
                
            else:
                raise ValueError(f"Unknown metric: {metric}")
        
        if test_type == 'anova':
            # One-way ANOVA
            anova_result = stats.f_oneway(*list(data_by_group.values()))
            statistic = anova_result.statistic
            p_value = anova_result.pvalue
            
        elif test_type == 'kruskal':
            # Kruskal-Wallis H-test (non-parametric)
            kruskal_result = stats.kruskal(*list(data_by_group.values()))
            statistic = kruskal_result.statistic
            p_value = kruskal_result.pvalue
            
        else:
            raise ValueError(f"Unknown test type: {test_type}")
        
        reject_null = p_value < self.alpha
        
        # Calculate group statistics
        group_stats = {}
        for group_name, data in data_by_group.items():
            if metric == 'frequency':
                group_stats[group_name] = {
                    'mean': data.mean() if len(data) > 0 else 0,
                    'n': len(data),
                    'sum': data.sum() if len(data) > 0 else 0
                }
            else:
                group_stats[group_name] = {
                    'mean': data.mean() if len(data) > 0 else 0,
                    'std': data.std() if len(data) > 0 else 0,
                    'n': len(data)
                }
        
        return {
            'test_type': f'multiple_groups_{metric}',
            'method': test_type,
            'statistic': statistic,
            'p_value': p_value,
            'reject_null': reject_null,
            'num_groups': len(groups_dict),
            'group_stats': group_stats
        }
    
    def run_all_tests_for_groups(self, group_a: pd.DataFrame,
                               group_b: pd.DataFrame,
                               group_a_name: str = 'Group A',
                               group_b_name: str = 'Group B') -> Dict[str, Dict]:
        """
        Run all relevant tests for two groups.
        
        Args:
            group_a: DataFrame for first group
            group_b: DataFrame for second group
            group_a_name: Name for first group
            group_b_name: Name for second group
            
        Returns:
            Dictionary with all test results
        """
        results = {}
        
        # Extract claim data
        claims_a = group_a['TotalClaims']
        claims_b = group_b['TotalClaims']
        
        # Extract margin data
        margins_a = group_a['TotalPremium'] - group_a['TotalClaims']
        margins_b = group_b['TotalPremium'] - group_b['TotalClaims']
        
        # Test claim frequency
        freq_results = self.test_claim_frequency(claims_a, claims_b, test_type='chi_square')
        freq_results['group_a_name'] = group_a_name
        freq_results['group_b_name'] = group_b_name
        results['claim_frequency'] = freq_results
        
        # Test claim severity (only if there are claims in both groups)
        if (claims_a > 0).sum() > 0 and (claims_b > 0).sum() > 0:
            severity_results = self.test_claim_severity(claims_a, claims_b, test_type='t_test')
        else:
            severity_results = {
                'test_type': 'severity',
                'method': 't_test',
                'statistic': np.nan,
                'p_value': 1.0,
                'reject_null': False,
                'message': 'Insufficient claims data for severity test'
            }
        severity_results['group_a_name'] = group_a_name
        severity_results['group_b_name'] = group_b_name
        results['claim_severity'] = severity_results
        
        # Test margin difference
        margin_results = self.test_margin_difference(margins_a, margins_b, test_type='t_test')
        margin_results['group_a_name'] = group_a_name
        margin_results['group_b_name'] = group_b_name
        results['margin'] = margin_results
        
        return results