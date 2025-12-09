#!/usr/bin/env python3
"""
Main script to run Task 3: Hypothesis Testing for Insurance Risk Analytics.
This script orchestrates all hypothesis tests for AlphaCare Insurance Solutions.
"""

import sys
import os
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import argparse
warnings.filterwarnings('ignore')

# Add src to path for custom modules
sys.path.append('src')

def check_and_generate_sample_data():
    """Check if sample data exists, generate if not."""
    sample_path = "data/02_interim/sample_data.csv"
    full_data_path = "data/02_interim/cleaned_data.csv"
    
    if os.path.exists(sample_path):
        print(f"✓ Sample data found: {sample_path}")
        return sample_path
    
    print("Sample data not found. Generating from full dataset...")
    
    # Create sample generation script if it doesn't exist
    if not os.path.exists("scripts/regenerate_sample.py"):
        os.makedirs("scripts", exist_ok=True)
        with open("scripts/regenerate_sample.py", "w") as f:
            f.write('''
import pandas as pd
import os
import numpy as np

def generate_sample():
    """Generate sample data from full dataset."""
    print("Generating sample data...")
    
    full_path = "data/02_interim/cleaned_data.csv"
    sample_path = "data/02_interim/sample_data.csv"
    
    if not os.path.exists(full_path):
        print("Full data not found. Creating synthetic sample...")
        # Create synthetic data
        np.random.seed(42)
        n = 10000
        
        data = {
            'totalpremium': np.random.exponential(8000, n) + 2000,
            'totalclaims': np.random.exponential(2000, n) * (np.random.random(n) > 0.85),
            'province': np.random.choice(['Gauteng', 'Western Cape', 'KwaZulu-Natal', 
                                         'Eastern Cape', 'Free State'], n),
            'postalcode': [str(np.random.randint(1000, 9999)) for _ in range(n)],
            'gender': np.random.choice(['Male', 'Female'], n),
            'suminsured': np.random.lognormal(11, 0.8, n),
            'vehicletype': np.random.choice(['Sedan', 'SUV', 'Truck', 'Hatchback'], n),
            'age': np.random.randint(18, 75, n)
        }
        
        df = pd.DataFrame(data)
        os.makedirs(os.path.dirname(sample_path), exist_ok=True)
        df.to_csv(sample_path, index=False)
        print(f"✅ Synthetic sample created: {len(df):,} rows")
        return True
    else:
        try:
            # Read a portion of the full data
            print(f"Reading from {full_path}...")
            df = pd.read_csv(full_path, nrows=200000, low_memory=False)
            
            # Sample 10,000 rows
            sample_size = min(10000, len(df))
            sample = df.sample(n=sample_size, random_state=42)
            
            # Save sample
            os.makedirs(os.path.dirname(sample_path), exist_ok=True)
            sample.to_csv(sample_path, index=False)
            print(f"✅ Sample created: {sample_path} ({len(sample):,} rows)")
            return True
        except Exception as e:
            print(f"Error reading full data: {e}")
            return False

if __name__ == "__main__":
    generate_sample()
''')
    
    # Run the sample generation script
    os.system("python scripts/regenerate_sample.py")
    
    if os.path.exists(sample_path):
        return sample_path
    else:
        print("Failed to generate sample data. Using synthetic fallback...")
        return None

def load_and_validate_data(data_path):
    """Load data and perform basic validation."""
    print(f"\nLoading data from {data_path}...")
    
    if not os.path.exists(data_path):
        print(f"ERROR: Data file not found at {data_path}")
        print("Attempting to generate sample data...")
        new_path = check_and_generate_sample_data()
        if new_path:
            return load_and_validate_data(new_path)
        else:
            return None
    
    try:
        # Try different encodings and separators
        try:
            df = pd.read_csv(data_path, low_memory=False)
        except:
            try:
                df = pd.read_csv(data_path, sep='|', low_memory=False)
            except:
                df = pd.read_csv(data_path, encoding='latin-1', low_memory=False)
        
        print(f"✓ Data loaded successfully: {len(df):,} rows, {len(df.columns)} columns")
        
        # Clean column names (remove whitespace, make lowercase)
        original_columns = list(df.columns)
        df.columns = [str(col).strip().lower() for col in df.columns]
        print(f"  Original columns standardized to lowercase")
        
        # Check for date columns (case insensitive)
        date_columns = [col for col in df.columns if 'date' in col or 'month' in col or 'year' in col]
        if date_columns:
            print(f"  Date-related columns found: {date_columns}")
        
        # Check required columns (case-insensitive)
        required_columns = ['totalpremium', 'totalclaims', 'province', 'postalcode', 'gender']
        available_columns = [col.lower() for col in df.columns]
        
        missing_columns = []
        for req_col in required_columns:
            if req_col not in available_columns:
                # Try to find similar column names
                similar = [col for col in df.columns if req_col in col.lower()]
                if similar:
                    print(f"  Found similar column for '{req_col}': {similar[0]}")
                else:
                    missing_columns.append(req_col)
        
        if missing_columns:
            print(f"WARNING: Missing required columns: {missing_columns}")
            print("Available columns:")
            for col in df.columns[:20]:  # Show first 20 columns
                print(f"  - {col}")
            if len(df.columns) > 20:
                print(f"  ... and {len(df.columns) - 20} more")
        
        # Basic data validation
        print("\nData Validation:")
        
        if 'totalpremium' in df.columns:
            df['totalpremium'] = pd.to_numeric(df['totalpremium'], errors='coerce')
            valid_premium = df['totalpremium'].notna() & (df['totalpremium'] > 0)
            print(f"  Policies with positive premium: {valid_premium.sum():,}")
            print(f"  Average premium: R{df.loc[valid_premium, 'totalpremium'].mean():,.2f}")
        else:
            print("  WARNING: 'totalpremium' column not found or invalid")
        
        if 'totalclaims' in df.columns:
            df['totalclaims'] = pd.to_numeric(df['totalclaims'], errors='coerce')
            df['totalclaims'] = df['totalclaims'].fillna(0)
            has_claims = df['totalclaims'] > 0
            print(f"  Policies with claims: {has_claims.sum():,}")
            if has_claims.any():
                print(f"  Average claim: R{df.loc[has_claims, 'totalclaims'].mean():,.2f}")
            else:
                print(f"  Average claim: R0.00 (no claims)")
        else:
            print("  WARNING: 'totalclaims' column not found")
        
        # Check province distribution
        if 'province' in df.columns:
            province_counts = df['province'].value_counts()
            print(f"\nTop 5 Provinces:")
            for province, count in province_counts.head().items():
                print(f"  {province}: {count:,} policies")
        
        # Check gender distribution
        if 'gender' in df.columns:
            print(f"\nGender Distribution:")
            gender_counts = df['gender'].value_counts()
            for gender, count in gender_counts.items():
                print(f"  {gender}: {count:,} policies")
        
        # Ensure numeric columns are numeric
        numeric_cols = ['totalpremium', 'totalclaims', 'suminsured']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove any rows with missing essential data
        initial_rows = len(df)
        if 'totalpremium' in df.columns:
            df = df[df['totalpremium'].notna() & (df['totalpremium'] > 0)]
        print(f"  Removed {initial_rows - len(df):,} rows with invalid premium data")
        
        return df
        
    except Exception as e:
        print(f"ERROR: Failed to load data: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def generate_quick_insights(df):
    """Generate quick insights before full analysis."""
    print("\n" + "="*80)
    print("QUICK INSIGHTS FROM DATA")
    print("="*80)
    
    # Simple metrics calculation
    if 'totalclaims' in df.columns and 'totalpremium' in df.columns:
        # Claim frequency
        has_claim = (df['totalclaims'] > 0)
        claim_frequency = has_claim.mean()
        
        # Claim severity (for policies with claims)
        if has_claim.any():
            claim_severity = df.loc[has_claim, 'totalclaims'].mean()
        else:
            claim_severity = 0
        
        # Margin
        df['margin'] = df['totalpremium'] - df['totalclaims']
        avg_margin = df['margin'].mean()
        
        # Loss ratio
        total_claims = df['totalclaims'].sum()
        total_premium = df['totalpremium'].sum()
        loss_ratio = total_claims / total_premium if total_premium > 0 else 0
        
        print(f"\nOverall Portfolio:")
        print(f"  Total Policies: {len(df):,}")
        print(f"  Claim Frequency: {claim_frequency:.2%}")
        print(f"  Average Claim Severity: R{claim_severity:,.2f}")
        print(f"  Average Margin per Policy: R{avg_margin:,.2f}")
        print(f"  Loss Ratio: {loss_ratio:.2%}")
        print(f"  Total Premium: R{total_premium:,.2f}")
        print(f"  Total Claims: R{total_claims:,.2f}")
    
    # Province insights
    if 'province' in df.columns and 'totalclaims' in df.columns:
        print(f"\nRisk by Province (Top 5 by claim frequency):")
        
        # Calculate metrics per province
        province_metrics = []
        for province in df['province'].unique():
            province_data = df[df['province'] == province]
            if len(province_data) > 0:
                prov_freq = (province_data['totalclaims'] > 0).mean()
                prov_severity = province_data.loc[province_data['totalclaims'] > 0, 'totalclaims'].mean() if (province_data['totalclaims'] > 0).any() else 0
                prov_premium = province_data['totalpremium'].sum() if 'totalpremium' in province_data.columns else 0
                prov_claims = province_data['totalclaims'].sum()
                prov_loss_ratio = prov_claims / prov_premium if prov_premium > 0 else 0
                
                province_metrics.append({
                    'province': province,
                    'policies': len(province_data),
                    'frequency': prov_freq,
                    'loss_ratio': prov_loss_ratio
                })
        
        # Sort by frequency and show top 5
        province_metrics.sort(key=lambda x: x['frequency'], reverse=True)
        for metric in province_metrics[:5]:
            print(f"  {metric['province']}: Policies={metric['policies']:,}, "
                  f"Frequency={metric['frequency']:.2%}, Loss Ratio={metric['loss_ratio']:.2%}")

def create_output_directory(output_dir):
    """Create output directory structure."""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'figures'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'tables'), exist_ok=True)
    print(f"✓ Output directory created: {output_dir}")

def run_hypothesis_tests(df, output_dir):
    """Run all hypothesis tests using custom modules."""
    print("\n" + "="*80)
    print("RUNNING STATISTICAL HYPOTHESIS TESTS")
    print("="*80)
    
    try:
        # Import custom modules
        from task3.hypothesis_tester import HypothesisTester
        from task3.report_generator import ReportGenerator
        
        # Initialize tester with modified configuration
        tester = HypothesisTester()
        
        # Run all tests
        print("\nStarting hypothesis testing...")
        
        # Save data temporarily for tester (ensure lowercase column names)
        temp_df = df.copy()
        temp_path = os.path.join(output_dir, "temp_data.csv")
        temp_df.to_csv(temp_path, index=False)
        
        # Modify the hypothesis_tester.py to handle lowercase columns
        # We'll patch the load_data method temporarily
        import types
        
        # Create a patched version of load_data
        def patched_load_data(self, data_path):
            print(f"Loading data from {data_path}...")
            df = pd.read_csv(data_path)
            
            # Clean column names
            df.columns = [col.strip().lower() for col in df.columns]
            
            # Basic cleaning
            df = self._clean_data(df)
            
            print(f"Data loaded: {len(df):,} rows, {len(df.columns)} columns")
            
            # Check for date columns
            date_cols = [col for col in df.columns if 'date' in col or 'month' in col or 'year' in col]
            if date_cols:
                for date_col in date_cols[:1]:  # Just show first date column
                    if date_col in df.columns:
                        try:
                            df[date_col] = pd.to_datetime(df[date_col])
                            print(f"Date range ({date_col}): {df[date_col].min()} to {df[date_col].max()}")
                        except:
                            print(f"Could not parse {date_col} as dates")
            
            return df
        
        # Apply the patch
        tester.load_data = types.MethodType(patched_load_data, tester)
        
        all_results = tester.run_all_tests(temp_path)
        
        # Generate comprehensive report
        print("\n" + "="*80)
        print("GENERATING BUSINESS REPORT")
        print("="*80)
        
        report_gen = ReportGenerator(tester.results)
        report_path = os.path.join(output_dir, 'business_report.txt')
        full_report = report_gen.generate_full_report(report_path)
        
        # Display executive summary
        print("\nEXECUTIVE SUMMARY:")
        print("-" * 40)
        
        # Clean up temp file
        if os.path.exists(temp_path):
            os.remove(temp_path)
        
        return tester.results
        
    except ImportError as e:
        print(f"ERROR: Could not import custom modules: {e}")
        print("Please ensure you have copied all Task 3 module files to src/task3/")
        return None
    except Exception as e:
        print(f"ERROR: Hypothesis testing failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def create_simple_hypothesis_tests(df, output_dir):
    """Create simple hypothesis tests if custom modules are not available."""
    print("\nRunning basic hypothesis tests (fallback mode)...")
    
    results = {}
    
    # Test 1: Province differences
    if 'province' in df.columns and 'totalclaims' in df.columns:
        print("\n1. Testing Province Risk Differences:")
        
        # Get top provinces with sufficient data
        province_counts = df['province'].value_counts()
        top_provinces = province_counts[province_counts >= 50].head(3).index.tolist()
        
        if len(top_provinces) >= 2:
            print(f"   Comparing: {top_provinces}")
            
            # Create contingency table
            contingency = []
            valid_provinces = []
            
            for province in top_provinces:
                province_data = df[df['province'] == province]
                if len(province_data) >= 10:  # Minimum sample size
                    has_claim = (province_data['totalclaims'] > 0).sum()
                    no_claim = len(province_data) - has_claim
                    contingency.append([has_claim, no_claim])
                    valid_provinces.append(province)
            
            if len(contingency) >= 2 and len(contingency[0]) == 2:
                try:
                    chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
                    reject_null = p_value < 0.05
                    
                    results['province_frequency'] = {
                        'chi2': chi2,
                        'p_value': p_value,
                        'reject_null': reject_null,
                        'provinces': valid_provinces
                    }
                    
                    print(f"   Chi-square test: χ²={chi2:.2f}, p={p_value:.4f}")
                    print(f"   Decision: {'REJECT' if reject_null else 'FAIL TO REJECT'} H₀")
                    
                    # Show claim frequencies
                    for i, province in enumerate(valid_provinces):
                        total = contingency[i][0] + contingency[i][1]
                        freq = contingency[i][0] / total if total > 0 else 0
                        print(f"   {province}: {contingency[i][0]}/{total} claims ({freq:.2%})")
                        
                except Exception as e:
                    print(f"   Error in chi-square test: {e}")
                    print("   Using Fisher's exact test instead...")
                    
                    # For 2x2 table, use Fisher's exact test
                    if len(contingency) == 2:
                        oddsratio, p_value = stats.fisher_exact(contingency)
                        reject_null = p_value < 0.05
                        
                        results['province_frequency'] = {
                            'method': 'fisher_exact',
                            'oddsratio': oddsratio,
                            'p_value': p_value,
                            'reject_null': reject_null,
                            'provinces': valid_provinces
                        }
                        
                        print(f"   Fisher's exact test: OR={oddsratio:.2f}, p={p_value:.4f}")
                        print(f"   Decision: {'REJECT' if reject_null else 'FAIL TO REJECT'} H₀")
            else:
                print("   Insufficient data for statistical test")
        else:
            print("   Insufficient provinces with enough data for comparison")
    
    # Test 2: Gender differences
    if 'gender' in df.columns and 'totalclaims' in df.columns:
        print("\n2. Testing Gender Risk Differences:")
        
        # Clean gender data
        gender_data = df.copy()
        gender_data['gender'] = gender_data['gender'].astype(str).str.strip().str.title()
        
        # Filter to Male and Female only with sufficient data
        valid_genders = []
        for gender in ['Male', 'Female']:
            gender_mask = gender_data['gender'].str.contains(gender, case=False, na=False)
            if gender_mask.sum() >= 10:  # Minimum sample size
                valid_genders.append(gender)
        
        if len(valid_genders) >= 2:
            print(f"   Comparing: {valid_genders}")
            
            # Create contingency table
            contingency = []
            for gender in valid_genders:
                gender_mask = gender_data['gender'].str.contains(gender, case=False, na=False)
                gender_subset = gender_data[gender_mask]
                has_claim = (gender_subset['totalclaims'] > 0).sum()
                no_claim = len(gender_subset) - has_claim
                contingency.append([has_claim, no_claim])
            
            try:
                chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
                reject_null = p_value < 0.05
                
                results['gender_frequency'] = {
                    'chi2': chi2,
                    'p_value': p_value,
                    'reject_null': reject_null,
                    'genders': valid_genders
                }
                
                print(f"   Chi-square test: χ²={chi2:.2f}, p={p_value:.4f}")
                print(f"   Decision: {'REJECT' if reject_null else 'FAIL TO REJECT'} H₀")
                
                # Show claim frequencies
                for i, gender in enumerate(valid_genders):
                    total = contingency[i][0] + contingency[i][1]
                    freq = contingency[i][0] / total if total > 0 else 0
                    print(f"   {gender}: {contingency[i][0]}/{total} claims ({freq:.2%})")
                    
            except Exception as e:
                print(f"   Error in chi-square test: {e}")
                print("   Using proportion z-test instead...")
                
                # Use two-proportion z-test
                from statsmodels.stats.proportion import proportions_ztest
                
                counts = [row[0] for row in contingency]
                nobs = [row[0] + row[1] for row in contingency]
                
                if len(counts) == 2:
                    z_stat, p_value = proportions_ztest(counts, nobs)
                    reject_null = p_value < 0.05
                    
                    results['gender_frequency'] = {
                        'method': 'z_test',
                        'z_statistic': z_stat,
                        'p_value': p_value,
                        'reject_null': reject_null,
                        'genders': valid_genders
                    }
                    
                    print(f"   Z-test for proportions: z={z_stat:.2f}, p={p_value:.4f}")
                    print(f"   Decision: {'REJECT' if reject_null else 'FAIL TO REJECT'} H₀")
        else:
            print("   Insufficient gender data for comparison")
    
    # Save simple results
    if results:
        import json
        results_path = os.path.join(output_dir, 'simple_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\n✓ Simple results saved to: {results_path}")
    
    return results

def main():
    """Main execution function."""
    print("\n" + "="*80)
    print("ALPHACARE INSURANCE SOLUTIONS - RISK ANALYTICS")
    print("TASK 3: HYPOTHESIS TESTING FOR RISK SEGMENTATION")
    print("="*80)
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run insurance hypothesis testing')
    parser.add_argument('--data', type=str, default='sample',
                       help='Data source: "sample", "full", or path to CSV file')
    parser.add_argument('--output', type=str, default='outputs/task3_results',
                       help='Output directory for results')
    parser.add_argument('--simple', action='store_true',
                       help='Use simple tests only (skip custom modules)')
    args = parser.parse_args()
    
    # Determine data path
    if args.data == 'sample':
        data_path = check_and_generate_sample_data()
        if not data_path:
            print("ERROR: Could not generate or find sample data")
            return 1
    elif args.data == 'full':
        data_path = "data/02_interim/cleaned_data.csv"
        if not os.path.exists(data_path):
            print(f"ERROR: Full data not found at {data_path}")
            print("Using sample data instead...")
            data_path = check_and_generate_sample_data()
            if not data_path:
                return 1
    else:
        data_path = args.data
    
    # Load and validate data
    df = load_and_validate_data(data_path)
    if df is None:
        return 1
    
    # Create output directory
    create_output_directory(args.output)
    
    # Generate quick insights
    generate_quick_insights(df)
    
    # Run hypothesis tests
    results = None
    if not args.simple:
        results = run_hypothesis_tests(df, args.output)
    
    # If custom modules failed or --simple flag used, run simple tests
    if results is None:
        print("\n" + "="*80)
        print("RUNNING BASIC HYPOTHESIS TESTS")
        print("="*80)
        results = create_simple_hypothesis_tests(df, args.output)
    
    # Generate summary
    print("\n" + "="*80)
    print("TASK 3 SUMMARY")
    print("="*80)
    
    if results:
        print("\n✅ Hypothesis testing completed successfully!")
        
        # Create summary CSV
        summary_data = []
        
        # Check for province results
        if 'province_hypothesis' in results:
            prov = results['province_hypothesis']
            if 'overall_decision' in prov:
                decision = prov['overall_decision']
                summary_data.append({
                    'Hypothesis': 'Province Risk Differences',
                    'Null Hypothesis': 'No risk differences across provinces',
                    'Result': 'REJECT' if decision['reject_null'] else 'FAIL TO REJECT',
                    'p-value': prov.get('frequency_across_provinces', {}).get('p_value', 'N/A'),
                    'Business Implication': 'Consider regional pricing adjustments'
                })
        elif 'province_frequency' in results:
            prov = results['province_frequency']
            summary_data.append({
                'Hypothesis': 'Province Risk Differences',
                'Null Hypothesis': 'No risk differences across provinces',
                'Result': 'REJECT' if prov['reject_null'] else 'FAIL TO REJECT',
                'p-value': prov.get('p_value', 'N/A'),
                'Business Implication': 'Consider regional pricing adjustments' if prov.get('reject_null') else 'No need for regional adjustments'
            })
        
        # Check for gender results
        if 'gender_hypothesis' in results:
            gender = results['gender_hypothesis']
            if 'overall_decision' in gender:
                decision = gender['overall_decision']
                summary_data.append({
                    'Hypothesis': 'Gender Risk Differences',
                    'Null Hypothesis': 'No risk differences between genders',
                    'Result': 'REJECT' if decision['reject_null'] else 'FAIL TO REJECT',
                    'p-value': gender.get('male_vs_female', {}).get('claim_frequency', {}).get('p_value', 'N/A'),
                    'Business Implication': 'Consider gender in risk assessment'
                })
        elif 'gender_frequency' in results:
            gender = results['gender_frequency']
            summary_data.append({
                'Hypothesis': 'Gender Risk Differences',
                'Null Hypothesis': 'No risk differences between genders',
                'Result': 'REJECT' if gender['reject_null'] else 'FAIL TO REJECT',
                'p-value': gender.get('p_value', 'N/A'),
                'Business Implication': 'Consider gender in risk assessment' if gender.get('reject_null') else 'Gender not a significant factor'
            })
        
        # Save summary
        if summary_data:
            summary_df = pd.DataFrame(summary_data)
            summary_path = os.path.join(args.output, 'tables', 'hypothesis_summary.csv')
            summary_df.to_csv(summary_path, index=False)
            print(f"\n✓ Hypothesis summary saved: {summary_path}")
            print("\nSummary Table:")
            print(summary_df.to_string(index=False))
    
    # Final message
    print("\n" + "="*80)
    print("TASK 3 EXECUTION COMPLETE!")
    print("="*80)
    print(f"\nAll outputs saved to: {args.output}")
    print("\nNext steps:")
    print("1. Review the outputs in the directory above")
    print("2. Check outputs/task3_results/tables/ for summary data")
    print("3. Update your README.md with Task 3 results")
    print("4. Commit your changes: git add . && git commit -m 'feat: Complete Task 3'")
    print("5. Push to GitHub: git push origin task-3")
    print("\n" + "="*80)
    
    return 0

if __name__ == "__main__":
    # Run the main function
    exit_code = main()
    sys.exit(exit_code)