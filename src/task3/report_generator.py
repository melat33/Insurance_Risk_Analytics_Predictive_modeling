"""
Generate comprehensive business reports from hypothesis testing results.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any
from datetime import datetime
import json


class ReportGenerator:
    """
    Generate business-oriented reports from statistical test results.
    
    Creates actionable insights and recommendations for ACIS insurance.
    """
    
    def __init__(self, results: Dict[str, Any]):
        """
        Initialize with test results.
        
        Args:
            results: Dictionary containing all hypothesis test results
        """
        self.results = results
        
    def generate_executive_summary(self) -> str:
        """Generate executive summary for business stakeholders."""
        summary_parts = []
        
        summary_parts.append("="*80)
        summary_parts.append("EXECUTIVE SUMMARY - HYPOTHESIS TESTING RESULTS")
        summary_parts.append("="*80)
        summary_parts.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        summary_parts.append("Significance Level: α = 0.05")
        summary_parts.append("\n" + "="*80)
        
        # Overall findings
        summary_parts.append("\nKEY FINDINGS:")
        summary_parts.append("-"*40)
        
        # Check each hypothesis
        hypotheses_summary = []
        
        # Province hypothesis
        if 'province_hypothesis' in self.results:
            prov_result = self.results['province_hypothesis']
            if 'overall_decision' in prov_result:
                decision = prov_result['overall_decision']
                if decision['reject_null']:
                    hypotheses_summary.append(
                        "✓ REJECT: There ARE significant risk differences across provinces"
                    )
                else:
                    hypotheses_summary.append(
                        "✗ FAIL TO REJECT: No significant risk differences across provinces"
                    )
        
        # Zip code hypotheses
        if 'zipcode_hypotheses' in self.results:
            zip_result = self.results['zipcode_hypotheses']
            
            # Risk differences
            if 'hypothesis_2_decision' in zip_result:
                hyp2 = zip_result['hypothesis_2_decision']
                if hyp2['reject_null']:
                    hypotheses_summary.append(
                        "✓ REJECT: There ARE significant risk differences between zip codes"
                    )
                else:
                    hypotheses_summary.append(
                        "✗ FAIL TO REJECT: No significant risk differences between zip codes"
                    )
            
            # Margin differences
            if 'hypothesis_3_decision' in zip_result:
                hyp3 = zip_result['hypothesis_3_decision']
                if hyp3['reject_null']:
                    hypotheses_summary.append(
                        "✓ REJECT: There ARE significant margin differences between zip codes"
                    )
                else:
                    hypotheses_summary.append(
                        "✗ FAIL TO REJECT: No significant margin differences between zip codes"
                    )
        
        # Gender hypothesis
        if 'gender_hypothesis' in self.results:
            gender_result = self.results['gender_hypothesis']
            if 'overall_decision' in gender_result:
                decision = gender_result['overall_decision']
                if decision['reject_null']:
                    hypotheses_summary.append(
                        "✓ REJECT: There ARE significant risk differences between genders"
                    )
                else:
                    hypotheses_summary.append(
                        "✗ FAIL TO REJECT: No significant risk differences between genders"
                    )
        
        # Add all hypothesis findings
        for i, finding in enumerate(hypotheses_summary, 1):
            summary_parts.append(f"{i}. {finding}")
        
        return "\n".join(summary_parts)
    
    def generate_detailed_findings(self) -> str:
        """Generate detailed findings with statistical evidence."""
        detailed_parts = []
        
        detailed_parts.append("\n" + "="*80)
        detailed_parts.append("DETAILED STATISTICAL FINDINGS")
        detailed_parts.append("="*80)
        
        # Province analysis
        if 'province_hypothesis' in self.results:
            detailed_parts.append("\n1. PROVINCE RISK DIFFERENCES:")
            detailed_parts.append("-"*40)
            
            prov_result = self.results['province_hypothesis']
            
            if 'frequency_across_provinces' in prov_result:
                freq_test = prov_result['frequency_across_provinces']
                detailed_parts.append(
                    f"Claim Frequency (ANOVA): F = {freq_test.get('statistic', 'N/A'):.2f}, "
                    f"p = {freq_test.get('p_value', 'N/A'):.4f}"
                )
                detailed_parts.append(
                    f"Decision: {'REJECT' if freq_test.get('reject_null', False) else 'FAIL TO REJECT'} H₀"
                )
            
            if 'severity_across_provinces' in prov_result:
                sev_test = prov_result['severity_across_provinces']
                detailed_parts.append(
                    f"Claim Severity (Kruskal-Wallis): H = {sev_test.get('statistic', 'N/A'):.2f}, "
                    f"p = {sev_test.get('p_value', 'N/A'):.4f}"
                )
            
            if 'province_metrics' in prov_result:
                detailed_parts.append("\nProvince Metrics (sorted by claim frequency):")
                metrics_df = pd.DataFrame(prov_result['province_metrics'])
                metrics_df = metrics_df.sort_values('claim_frequency', ascending=False)
                
                for _, row in metrics_df.iterrows():
                    detailed_parts.append(
                        f"  {row['group']}: "
                        f"Frequency={row['claim_frequency']:.2%}, "
                        f"Severity=R{row['claim_severity_mean']:,.0f}, "
                        f"Loss Ratio={row['loss_ratio']:.2%}"
                    )
        
        # Zip code analysis
        if 'zipcode_hypotheses' in self.results:
            detailed_parts.append("\n\n2. ZIP CODE RISK & MARGIN DIFFERENCES:")
            detailed_parts.append("-"*40)
            
            zip_result = self.results['zipcode_hypotheses']
            
            if 'high_vs_low_risk' in zip_result:
                tests = zip_result['high_vs_low_risk']
                
                # Frequency test
                if 'claim_frequency' in tests:
                    freq = tests['claim_frequency']
                    detailed_parts.append(
                        f"Risk Difference (Frequency): "
                        f"χ² = {freq.get('statistic', 'N/A'):.2f}, "
                        f"p = {freq.get('p_value', 'N/A'):.4f}"
                    )
                    detailed_parts.append(
                        f"  High Risk Group Frequency: {freq.get('group_a_frequency', 0):.2%}"
                    )
                    detailed_parts.append(
                        f"  Low Risk Group Frequency: {freq.get('group_b_frequency', 0):.2%}"
                    )
                    detailed_parts.append(
                        f"  Risk Difference: {freq.get('risk_difference', 0):.4f} "
                        f"(95% CI: [{freq.get('ci_lower', 0):.4f}, {freq.get('ci_upper', 0):.4f}])"
                    )
                
                # Margin test
                if 'margin' in tests:
                    margin = tests['margin']
                    detailed_parts.append(
                        f"\nMargin Difference: "
                        f"t = {margin.get('statistic', 'N/A'):.2f}, "
                        f"p = {margin.get('p_value', 'N/A'):.4f}"
                    )
                    detailed_parts.append(
                        f"  High Risk Group Margin: R{margin.get('mean_a', 0):,.2f}"
                    )
                    detailed_parts.append(
                        f"  Low Risk Group Margin: R{margin.get('mean_b', 0):,.2f}"
                    )
                    detailed_parts.append(
                        f"  Margin Difference: R{margin.get('mean_difference', 0):,.2f}"
                    )
        
        # Gender analysis
        if 'gender_hypothesis' in self.results:
            detailed_parts.append("\n\n3. GENDER RISK DIFFERENCES:")
            detailed_parts.append("-"*40)
            
            gender_result = self.results['gender_hypothesis']
            
            if 'male_vs_female' in gender_result:
                tests = gender_result['male_vs_female']
                
                # Frequency test
                if 'claim_frequency' in tests:
                    freq = tests['claim_frequency']
                    detailed_parts.append(
                        f"Gender Frequency Difference: "
                        f"χ² = {freq.get('statistic', 'N/A'):.2f}, "
                        f"p = {freq.get('p_value', 'N/A'):.4f}"
                    )
                    detailed_parts.append(
                        f"  Male Claim Frequency: {freq.get('group_a_frequency', 0):.2%}"
                    )
                    detailed_parts.append(
                        f"  Female Claim Frequency: {freq.get('group_b_frequency', 0):.2%}"
                    )
                
                # Severity test
                if 'claim_severity' in tests:
                    sev = tests['claim_severity']
                    if 'message' in sev:
                        detailed_parts.append(f"  Severity Test: {sev['message']}")
                    else:
                        detailed_parts.append(
                            f"  Gender Severity Difference: "
                            f"t = {sev.get('statistic', 'N/A'):.2f}, "
                            f"p = {sev.get('p_value', 'N/A'):.4f}"
                        )
                        detailed_parts.append(
                            f"    Male Average Claim: R{sev.get('mean_a', 0):,.2f}"
                        )
                        detailed_parts.append(
                            f"    Female Average Claim: R{sev.get('mean_b', 0):,.2f}"
                        )
        
        return "\n".join(detailed_parts)
    
    def generate_business_recommendations(self) -> str:
        """Generate actionable business recommendations."""
        rec_parts = []
        
        rec_parts.append("\n" + "="*80)
        rec_parts.append("BUSINESS RECOMMENDATIONS FOR ALPHACARE INSURANCE")
        rec_parts.append("="*80)
        
        recommendations = []
        
        # Province recommendations
        if 'province_hypothesis' in self.results:
            prov_result = self.results['province_hypothesis']
            if 'overall_decision' in prov_result:
                decision = prov_result['overall_decision']
                if decision['reject_null']:
                    recommendations.append({
                        'area': 'Geographic Pricing',
                        'recommendation': 'Implement province-based premium adjustments',
                        'action': 'Adjust premiums by ±X% based on province risk levels',
                        'impact': 'Optimize risk-adjusted pricing across regions'
                    })
        
        # Zip code recommendations
        if 'zipcode_hypotheses' in self.results:
            zip_result = self.results['zipcode_hypotheses']
            
            # Risk differences
            if 'hypothesis_2_decision' in zip_result:
                hyp2 = zip_result['hypothesis_2_decision']
                if hyp2['reject_null']:
                    recommendations.append({
                        'area': 'Risk Segmentation',
                        'recommendation': 'Use zip codes for micro-segmentation',
                        'action': 'Create high/medium/low risk zip code clusters',
                        'impact': 'More accurate risk assessment at granular level'
                    })
            
            # Margin differences
            if 'hypothesis_3_decision' in zip_result:
                hyp3 = zip_result['hypothesis_3_decision']
                if hyp3['reject_null']:
                    recommendations.append({
                        'area': 'Marketing Strategy',
                        'recommendation': 'Focus marketing on profitable zip codes',
                        'action': 'Allocate more resources to high-margin areas',
                        'impact': 'Increase overall portfolio profitability'
                    })
        
        # Gender recommendations
        if 'gender_hypothesis' in self.results:
            gender_result = self.results['gender_hypothesis']
            if 'overall_decision' in gender_result:
                decision = gender_result['overall_decision']
                if decision['reject_null']:
                    recommendations.append({
                        'area': 'Risk Factor Consideration',
                        'recommendation': 'Consider gender in risk assessment models',
                        'action': 'Include gender as a variable in pricing algorithms',
                        'impact': 'More accurate individual risk pricing'
                    })
        
        # Format recommendations
        for i, rec in enumerate(recommendations, 1):
            rec_parts.append(f"\n{i}. {rec['area'].upper()}")
            rec_parts.append(f"   Recommendation: {rec['recommendation']}")
            rec_parts.append(f"   Action Required: {rec['action']}")
            rec_parts.append(f"   Expected Impact: {rec['impact']}")
        
        # General recommendations
        rec_parts.append("\n\nGENERAL STRATEGIC RECOMMENDATIONS:")
        rec_parts.append("-"*40)
        rec_parts.append("1. Implement a tiered pricing model based on geographic risk")
        rec_parts.append("2. Develop targeted marketing campaigns for low-risk segments")
        rec_parts.append("3. Regularly monitor and update risk segmentation models")
        rec_parts.append("4. Consider regulatory constraints when implementing risk-based pricing")
        
        return "\n".join(rec_parts)
    
    def generate_full_report(self, output_path: str = None):
        """Generate full comprehensive report."""
        report_parts = []
        
        # Header
        report_parts.append("="*80)
        report_parts.append("FINAL REPORT - INSURANCE RISK HYPOTHESIS TESTING")
        report_parts.append("AlphaCare Insurance Solutions (ACIS)")
        report_parts.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_parts.append("="*80)
        
        # Add all sections
        report_parts.append(self.generate_executive_summary())
        report_parts.append(self.generate_detailed_findings())
        report_parts.append(self.generate_business_recommendations())
        
        # Footer
        report_parts.append("\n" + "="*80)
        report_parts.append("END OF REPORT")
        report_parts.append("="*80)
        
        full_report = "\n".join(report_parts)
        
        # Save to file if path provided
        if output_path:
            with open(output_path, 'w') as f:
                f.write(full_report)
            print(f"Report saved to: {output_path}")
        
        return full_report