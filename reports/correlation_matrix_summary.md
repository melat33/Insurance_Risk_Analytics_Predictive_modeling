# Correlation Matrix Analysis Summary

**Analysis Date:** 2025-12-11 14:16:11

**Dataset:** 50000 rows × 52 columns

**Numerical Variables Analyzed:** 15

## Strongest Correlations Found

The following correlations have |r| > 0.5:

| Variable 1 | Variable 2 | Correlation | Interpretation |
|------------|------------|-------------|----------------|
| underwrittencoverid | policyid | 0.941 | Strong Positive |
| calculatedpremiumperterm | totalpremium | 0.819 | Strong Positive |

## Top 5 Variable Pairs by Correlation Strength

1. **underwrittencoverid vs policyid**  
   - Correlation: **0.941**  
   - Interpretation: Strong Positive  
   - Business Impact: Very strong relationship - these variables move together

2. **calculatedpremiumperterm vs totalpremium**  
   - Correlation: **0.819**  
   - Interpretation: Strong Positive  
   - Business Impact: Very strong relationship - these variables move together


## Statistical Significance

P-values for top correlations:

| Variable Pair | Pearson r | p-value | Significant? |
|---------------|-----------|---------|--------------|
| underwrittencoverid vs policyid | 0.941 | 0.0000 | Yes |
| calculatedpremiumperterm vs totalpremium | 0.819 | 0.0000 | Yes |

**Note:** p-value < 0.05 indicates statistical significance
