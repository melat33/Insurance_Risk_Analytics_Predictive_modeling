"""
Main orchestrator for testing all 4 hypotheses in Task 3.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import yaml
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

from .metrics import InsuranceMetrics
from .segmentation import DataSegmenter
from .statistical_tests import StatisticalTests


class HypothesisTester:
    """
    Main class to test all 4 hypotheses for Task 3.
    
    Tests:
    1. Province risk differences (frequency & severity)
    2. Zip code risk differences (frequency)
    3. Zip code margin differences (margin)
    4. Gender risk differences (frequency & severity)
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize hypothesis tester.
        
        Args:
            config_path: Path to configuration YAML file
        """
        # Load configuration
        if config_path:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = {
                'alpha': 0.05,
                'min_samples': 30
            }
        
        self.alpha = self.config.get('alpha', 0.05)
        
        # Initialize components
        self.metrics_calc = InsuranceMetrics()
        self.segmenter = DataSegmenter()
        self.stat_tester = StatisticalTests(alpha=self.alpha)
        
        # Store results
        self.results = {}
        self.detailed_results = {}
        
    def load_data(self, data_path: str) -> pd.DataFrame:
        """
        Load and prepare data for analysis.
        
        Args:
            data_path: Path to CSV file
            
        Returns:
            Cleaned DataFrame
        """
        print(f"Loading data from {data_path}...")
        df = pd.read_csv(data_path)
        
        # Basic cleaning
        df = self._clean_data(df)
        
        print(f"Data loaded: {len(df):,} rows, {len(df.columns)} columns")
        print(f"Date range: {df['TransactionMonth'].min()} to {df['TransactionMonth'].max()}")
        
        return df
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the insurance data.
        """
        # Make copy
        df = df.copy()
        
        # Remove extreme outliers in claims
        if 'TotalClaims' in df.columns:
            claim_threshold = df['TotalClaims'].quantile(0.995)
            df = df[df['TotalClaims'] <= claim_threshold]
            print(f"  Removed claims above R{claim_threshold:,.2f}")
        
        # Remove negative or zero premiums
        if 'TotalPremium' in df.columns:
            initial_count = len(df)
            df = df[df['TotalPremium'] > 0]
            removed = initial_count - len(df)
            if removed > 0:
                print(f"  Removed {removed} policies with non-positive premium")
        
        # Handle missing values in key columns
        for col in ['Gender', 'Province', 'PostalCode']:
            if col in df.columns:
                missing_before = df[col].isna().sum()
                df[col] = df[col].fillna('Unknown')
                if missing_before > 0:
                    print(f"  Filled {missing_before} missing values in {col}")
        
        # Ensure proper data types
        if 'TransactionMonth' in df.columns:
            df['TransactionMonth'] = pd.to_datetime(df['TransactionMonth'])
        
        return df
    
    def test_province_hypothesis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Test H₀: There are no risk differences across provinces
        
        Tests both frequency and severity differences.
        """
        print("\n" + "="*80)
        print("HYPOTHESIS 1: Testing Province Risk Differences")
        print("="*80)
        
        if 'Province' not in df.columns:
            raise ValueError("'Province' column not found in data")
        
        results = {}
        
        # Get top provinces for comparison
        province_counts = df['Province'].value_counts()
        top_provinces = province_counts.head(3).index.tolist()  # Top 3 provinces
        
        print(f"Comparing top 3 provinces: {top_provinces}")
        
        # Create province groups
        province_groups = {}
        for province in top_provinces:
            province_data = df[df['Province'] == province].copy()
            province_groups[province] = province_data
            print(f"  {province}: {len(province_data):,} policies")
        
        # Test 1: Claim frequency across provinces (multiple groups)
        freq_results = self.stat_tester.test_multiple_groups(
            province_groups, metric='frequency', test_type='anova'
        )
        results['frequency_across_provinces'] = freq_results
        
        # Test 2: Claim severity across provinces
        severity_results = self.stat_tester.test_multiple_groups(
            province_groups, metric='severity', test_type='kruskal'
        )
        results['severity_across_provinces'] = severity_results
        
        # Pairwise comparisons between top 2 provinces
        if len(top_provinces) >= 2:
            province_a = top_provinces[0]
            province_b = top_provinces[1]
            
            print(f"\nPairwise comparison: {province_a} vs {province_b}")
            
            pairwise_results = self.stat_tester.run_all_tests_for_groups(
                province_groups[province_a],
                province_groups[province_b],
                group_a_name=province_a,
                group_b_name=province_b
            )
            results['pairwise_comparison'] = pairwise_results
        
        # Calculate metrics for each province
        province_metrics = self.metrics_calc.calculate_all_metrics(df, 'Province')
        results['province_metrics'] = province_metrics.to_dict('records')
        
        # Overall decision for hypothesis
        freq_reject = freq_results['reject_null']
        severity_reject = severity_results['reject_null']
        overall_reject = freq_reject or severity_reject
        
        results['overall_decision'] = {
            'reject_null': overall_reject,
            'reason': 'Reject if either frequency or severity shows significant differences',
            'frequency_significant': freq_reject,
            'severity_significant': severity_reject
        }
        
        print(f"\nFREQUENCY TEST: {'REJECT' if freq_reject else 'FAIL TO REJECT'} H₀ (p={freq_results['p_value']:.4f})")
        print(f"SEVERITY TEST: {'REJECT' if severity_reject else 'FAIL TO REJECT'} H₀ (p={severity_results['p_value']:.4f})")
        print(f"\nOVERALL: {'REJECT' if overall_reject else 'FAIL TO REJECT'} the null hypothesis")
        
        self.results['province_hypothesis'] = results
        return results
    
    def test_zipcode_hypothesis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Test H₀: There are no risk differences between zip codes
        Test H₀: There is no significant margin difference between zip codes
        
        Combines both risk (frequency) and margin tests for zip codes.
        """
        print("\n" + "="*80)
        print("HYPOTHESIS 2 & 3: Testing Zip Code Risk & Margin Differences")
        print("="*80)
        
        if 'PostalCode' not in df.columns:
            raise ValueError("'PostalCode' column not found in data")
        
        results = {}
        
        # Create high and low risk zip code groups
        try:
            high_risk_data, low_risk_data = self.segmenter.create_zipcode_groups(
                df, method='risk_quartiles'
            )
            
            print(f"\nComparing High Risk vs Low Risk Zip Code Groups:")
            print(f"  High Risk Group: {len(high_risk_data):,} policies")
            print(f"  Low Risk Group: {len(low_risk_data):,} policies")
            
            # Run all tests between high and low risk groups
            zipcode_results = self.stat_tester.run_all_tests_for_groups(
                high_risk_data,
                low_risk_data,
                group_a_name='High Risk Zip Codes',
                group_b_name='Low Risk Zip Codes'
            )
            
            results['high_vs_low_risk'] = zipcode_results
            
            # Extract specific test results
            freq_reject = zipcode_results['claim_frequency']['reject_null']
            margin_reject = zipcode_results['margin']['reject_null']
            
            # Hypothesis 2 decision (risk differences)
            hyp2_decision = {
                'reject_null': freq_reject,
                'p_value': zipcode_results['claim_frequency']['p_value'],
                'risk_difference': zipcode_results['claim_frequency']['risk_difference'],
                'relative_risk': zipcode_results['claim_frequency']['relative_risk']
            }
            results['hypothesis_2_decision'] = hyp2_decision
            
            # Hypothesis 3 decision (margin differences)
            hyp3_decision = {
                'reject_null': margin_reject,
                'p_value': zipcode_results['margin']['p_value'],
                'mean_difference': zipcode_results['margin']['mean_difference'],
                'total_margin_high': zipcode_results['margin']['total_margin_a'],
                'total_margin_low': zipcode_results['margin']['total_margin_b']
            }
            results['hypothesis_3_decision'] = hyp3_decision
            
            print(f"\nHYPOTHESIS 2 (Risk Differences):")
            print(f"  {'REJECT' if freq_reject else 'FAIL TO REJECT'} H₀")
            print(f"  p-value: {zipcode_results['claim_frequency']['p_value']:.4f}")
            print(f"  Risk Difference: {zipcode_results['claim_frequency']['risk_difference']:.4f}")
            
            print(f"\nHYPOTHESIS 3 (Margin Differences):")
            print(f"  {'REJECT' if margin_reject else 'FAIL TO REJECT'} H₀")
            print(f"  p-value: {zipcode_results['margin']['p_value']:.4f}")
            print(f"  Mean Margin Difference: R{zipcode_results['margin']['mean_difference']:,.2f}")
            
        except Exception as e:
            print(f"Error in zip code analysis: {str(e)}")
            results['error'] = str(e)
        
        self.results['zipcode_hypotheses'] = results
        return results
    
    def test_gender_hypothesis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Test H₀: There is no significant risk difference between Women and Men
        
        Tests both frequency and severity differences.
        """
        print("\n" + "="*80)
        print("HYPOTHESIS 4: Testing Gender Risk Differences")
        print("="*80)
        
        if 'Gender' not in df.columns:
            raise ValueError("'Gender' column not found in data")
        
        results = {}
        
        # Check available genders
        gender_counts = df['Gender'].value_counts()
        print(f"Gender distribution: {gender_counts.to_dict()}")
        
        # We'll compare Male vs Female
        test_genders = ['Male', 'Female']
        available_genders = [g for g in test_genders if g in gender_counts.index]
        
        if len(available_genders) < 2:
            print(f"Need both Male and Female data. Available: {available_genders}")
            results['error'] = f"Insufficient gender data. Available: {available_genders}"
            return results
        
        # Create comparable groups
        try:
            male_data, female_data, balance_report = self.segmenter.create_comparable_groups(
                df, 'Gender', 'Male', 'Female', min_sample_size=30
            )
            
            print(f"\nGroup Sizes after balancing:")
            print(f"  Male: {len(male_data):,} policies")
            print(f"  Female: {len(female_data):,} policies")
            
            # Check balance
            print("\nBalance Check Results:")
            balanced_vars = [var for var, stats in balance_report.items() 
                           if stats.get('is_balanced', True)]
            unbalanced_vars = [var for var, stats in balance_report.items() 
                             if not stats.get('is_balanced', True)]
            
            print(f"  Balanced variables: {len(balanced_vars)}")
            print(f"  Unbalanced variables: {len(unbalanced_vars)}")
            
            if unbalanced_vars:
                print(f"  Warning: Unbalanced on: {unbalanced_vars[:3]}")
            
            # Run all tests
            gender_results = self.stat_tester.run_all_tests_for_groups(
                male_data, female_data,
                group_a_name='Male', group_b_name='Female'
            )
            
            results['male_vs_female'] = gender_results
            results['balance_report'] = balance_report
            
            # Extract test results
            freq_reject = gender_results['claim_frequency']['reject_null']
            severity_reject = gender_results['claim_severity']['reject_null']
            
            # Overall decision
            overall_reject = freq_reject or severity_reject
            
            results['overall_decision'] = {
                'reject_null': overall_reject,
                'frequency_significant': freq_reject,
                'severity_significant': severity_reject,
                'frequency_p_value': gender_results['claim_frequency']['p_value'],
                'severity_p_value': gender_results['claim_severity']['p_value']
            }
            
            print(f"\nFREQUENCY TEST: {'REJECT' if freq_reject else 'FAIL TO REJECT'} H₀")
            print(f"  p-value: {gender_results['claim_frequency']['p_value']:.4f}")
            print(f"  Male Frequency: {gender_results['claim_frequency']['group_a_frequency']:.2%}")
            print(f"  Female Frequency: {gender_results['claim_frequency']['group_b_frequency']:.2%}")
            print(f"  Risk Difference: {gender_results['claim_frequency']['risk_difference']:.4f}")
            
            print(f"\nSEVERITY TEST: {'REJECT' if severity_reject else 'FAIL TO REJECT'} H₀")
            print(f"  p-value: {gender_results['claim_severity']['p_value']:.4f}")
            print(f"  Male Average Claim: R{gender_results['claim_severity']['mean_a']:,.2f}")
            print(f"  Female Average Claim: R{gender_results['claim_severity']['mean_b']:,.2f}")
            print(f"  Mean Difference: R{gender_results['claim_severity']['mean_difference']:,.2f}")
            
            print(f"\nOVERALL: {'REJECT' if overall_reject else 'FAIL TO REJECT'} the null hypothesis")
            
        except Exception as e:
            print(f"Error in gender analysis: {str(e)}")
            results['error'] = str(e)
        
        self.results['gender_hypothesis'] = results
        return results
    
    def run_all_tests(self, data_path: str) -> Dict[str, Any]:
        """
        Run all hypothesis tests.
        
        Args:
            data_path: Path to cleaned data CSV
            
        Returns:
            Dictionary with all results
        """
        print("="*80)
        print("RUNNING ALL HYPOTHESIS TESTS FOR TASK 3")
        print("="*80)
        
        # Load data
        df = self.load_data(data_path)
        
        # Run all tests
        all_results = {}
        
        # Test 1: Province differences
        all_results['province_tests'] = self.test_province_hypothesis(df)
        
        # Test 2 & 3: Zip code differences
        all_results['zipcode_tests'] = self.test_zipcode_hypothesis(df)
        
        # Test 4: Gender differences
        all_results['gender_tests'] = self.test_gender_hypothesis(df)
        
        # Generate summary
        summary = self._generate_summary(all_results)
        all_results['summary'] = summary
        
        # Save results
        self._save_results(all_results)
        
        return all_results
    
    def _generate_summary(self, results: Dict) -> Dict:
        """Generate a summary of all test results."""
        summary = {
            'test_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'significance_level': self.alpha,
            'hypotheses': {}
        }
        
        # Province hypothesis summary
        if 'province_tests' in results:
            province_result = results['province_tests']
            if 'overall_decision' in province_result:
                decision = province_result['overall_decision']
                summary['hypotheses']['province_risk_differences'] = {
                    'null_hypothesis': 'There are no risk differences across provinces',
                    'decision': 'REJECT' if decision['reject_null'] else 'FAIL TO REJECT',
                    'frequency_significant': decision.get('frequency_significant', False),
                    'severity_significant': decision.get('severity_significant', False),
                    'business_implication': 'Consider regional pricing adjustments if differences exist'
                }
        
        # Zip code hypotheses summary
        if 'zipcode_tests' in results:
            zipcode_result = results['zipcode_tests']
            
            # Hypothesis 2: Risk differences
            if 'hypothesis_2_decision' in zipcode_result:
                hyp2 = zipcode_result['hypothesis_2_decision']
                summary['hypotheses']['zipcode_risk_differences'] = {
                    'null_hypothesis': 'There are no risk differences between zip codes',
                    'decision': 'REJECT' if hyp2['reject_null'] else 'FAIL TO REJECT',
                    'p_value': hyp2.get('p_value', np.nan),
                    'risk_difference': hyp2.get('risk_difference', np.nan),
                    'business_implication': 'Zip codes can be used for risk segmentation'
                }
            
            # Hypothesis 3: Margin differences
            if 'hypothesis_3_decision' in zipcode_result:
                hyp3 = zipcode_result['hypothesis_3_decision']
                summary['hypotheses']['zipcode_margin_differences'] = {
                    'null_hypothesis': 'There is no significant margin difference between zip codes',
                    'decision': 'REJECT' if hyp3['reject_null'] else 'FAIL TO REJECT',
                    'p_value': hyp3.get('p_value', np.nan),
                    'mean_difference': hyp3.get('mean_difference', np.nan),
                    'business_implication': 'Profitability varies by location, adjust marketing strategy'
                }
        
        # Gender hypothesis summary
        if 'gender_tests' in results:
            gender_result = results['gender_tests']
            if 'overall_decision' in gender_result:
                decision = gender_result['overall_decision']
                summary['hypotheses']['gender_risk_differences'] = {
                    'null_hypothesis': 'There is no significant risk difference between Women and Men',
                    'decision': 'REJECT' if decision['reject_null'] else 'FAIL TO REJECT',
                    'frequency_significant': decision.get('frequency_significant', False),
                    'severity_significant': decision.get('severity_significant', False),
                    'business_implication': 'Gender may be a valid risk factor for pricing'
                }
        
        return summary
    
    def _save_results(self, results: Dict):
        """Save results to files."""
        import os
        
        # Create output directory
        output_dir = 'outputs/task3_results'
        os.makedirs(output_dir, exist_ok=True)
        
        # Save detailed results as JSON
        with open(f'{output_dir}/detailed_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save summary as text
        summary = results.get('summary', {})
        with open(f'{output_dir}/summary_report.txt', 'w') as f:
            f.write("="*80 + "\n")
            f.write("HYPOTHESIS TESTING RESULTS - TASK 3\n")
            f.write("="*80 + "\n\n")
            
            f.write(f"Test Date: {summary.get('test_date', 'N/A')}\n")
            f.write(f"Significance Level (α): {summary.get('significance_level', 0.05)}\n\n")
            
            f.write("HYPOTHESIS SUMMARY:\n")
            f.write("-"*40 + "\n")
            
            for hyp_name, hyp_result in summary.get('hypotheses', {}).items():
                f.write(f"\n{hyp_name.replace('_', ' ').title()}:\n")
                f.write(f"  Null Hypothesis: {hyp_result['null_hypothesis']}\n")
                f.write(f"  Decision: {hyp_result['decision']} H₀\n")
                
                if 'p_value' in hyp_result and not pd.isna(hyp_result['p_value']):
                    f.write(f"  p-value: {hyp_result['p_value']:.4f}\n")
                
                if 'business_implication' in hyp_result:
                    f.write(f"  Business Implication: {hyp_result['business_implication']}\n")
            
            f.write("\n" + "="*80 + "\n")
            f.write("END OF REPORT\n")
            f.write("="*80 + "\n")
        
        print(f"\nResults saved to {output_dir}/")
    
    def generate_visualizations(self, df: pd.DataFrame):
        """Generate visualizations for hypothesis testing results."""
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        output_dir = 'outputs/task3_results/figures'
        os.makedirs(output_dir, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
        # Visualization 1: Province risk comparison
        if 'Province' in df.columns:
            province_metrics = self.metrics_calc.calculate_all_metrics(df, 'Province')
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Claim frequency by province
            provinces = province_metrics.sort_values('claim_frequency', ascending=False)
            axes[0, 0].bar(provinces['group'], provinces['claim_frequency'])
            axes[0, 0].set_title('Claim Frequency by Province')
            axes[0, 0].set_ylabel('Claim Frequency')
            axes[0, 0].tick_params(axis='x', rotation=45)
            
            # Average claim severity by province
            axes[0, 1].bar(provinces['group'], provinces['claim_severity_mean'])
            axes[0, 1].set_title('Average Claim Severity by Province')
            axes[0, 1].set_ylabel('Average Claim (R)')
            axes[0, 1].tick_params(axis='x', rotation=45)
            
            # Loss ratio by province
            axes[1, 0].bar(provinces['group'], provinces['loss_ratio'])
            axes[1, 0].axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Break-even')
            axes[1, 0].set_title('Loss Ratio by Province')
            axes[1, 0].set_ylabel('Loss Ratio (Claims/Premium)')
            axes[1, 0].tick_params(axis='x', rotation=45)
            axes[1, 0].legend()
            
            # Policy count by province
            axes[1, 1].bar(provinces['group'], provinces['sample_size'])
            axes[1, 1].set_title('Number of Policies by Province')
            axes[1, 1].set_ylabel('Number of Policies')
            axes[1, 1].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            plt.savefig(f'{output_dir}/province_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # Visualization 2: Gender comparison
        if 'Gender' in df.columns:
            gender_data = df[df['Gender'].isin(['Male', 'Female'])].copy()
            
            if len(gender_data) > 0:
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                
                # Claim frequency by gender
                gender_freq = gender_data.groupby('Gender')['TotalClaims'].apply(
                    lambda x: (x > 0).mean()
                ).reset_index()
                axes[0].bar(gender_freq['Gender'], gender_freq['TotalClaims'])
                axes[0].set_title('Claim Frequency by Gender')
                axes[0].set_ylabel('Claim Frequency')
                
                # Claim severity by gender
                gender_severity = gender_data[gender_data['TotalClaims'] > 0].groupby('Gender')['TotalClaims'].mean()
                if len(gender_severity) > 0:
                    gender_severity.plot(kind='bar', ax=axes[1])
                    axes[1].set_title('Average Claim Severity by Gender')
                    axes[1].set_ylabel('Average Claim (R)')
                
                # Margin by gender
                gender_data['Margin'] = gender_data['TotalPremium'] - gender_data['TotalClaims']
                gender_margin = gender_data.groupby('Gender')['Margin'].mean()
                gender_margin.plot(kind='bar', ax=axes[2])
                axes[2].set_title('Average Margin by Gender')
                axes[2].set_ylabel('Average Margin (R)')
                
                plt.tight_layout()
                plt.savefig(f'{output_dir}/gender_analysis.png', dpi=300, bbox_inches='tight')
                plt.close()