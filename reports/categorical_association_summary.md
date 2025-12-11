# Categorical Association Analysis Summary

**Analysis Date:** 2025-12-11 14:16:50

**Dataset:** 50000 rows × 52 columns

**Categorical Variables Analyzed:** 36

## Association Test Results

Chi-square test results for categorical variable pairs:

| Variable 1 | Variable 2 | Chi-square | P-value | Cramer's V | Strength | Significant |
|------------|------------|------------|---------|------------|----------|-------------|
| transactionmonth | citizenship | 0.00 | 1.0000 | 0.000 | Very Weak | Not Significant |
| transactionmonth | legaltype | 613.33 | 0.0000 | 0.111 | Weak | Significant |
| citizenship | legaltype | 0.00 | 1.0000 | 0.000 | Very Weak | Not Significant |

## Interpretation Guide

### Cramer's V Strength:
- **< 0.1:** Very weak association
- **0.1 - 0.3:** Weak association
- **0.3 - 0.5:** Moderate association
- **> 0.5:** Strong association

### Statistical Significance:
- **P-value < 0.05:** Statistically significant association
- **P-value ≥ 0.05:** No statistically significant association

## Business Insights

### Strongest Association: transactionmonth vs legaltype
- **Cramer's V:** 0.111 (Weak association)
- **Statistical significance:** Significant (p = 0.0000)
- **Business implication:** These variables show the strongest relationship among all tested pairs

## Recommendations

1. **Investigate significant associations** for potential business insights
2. **Consider cross-tabulations** in your reporting for these variable pairs
3. **Validate findings** with domain experts
