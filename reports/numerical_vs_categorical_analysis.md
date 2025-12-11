# Numerical vs Categorical Analysis Report

**Analysis Date:** 2025-12-11 14:18:44

**Dataset:** 50000 rows × 52 columns

**Total Pairs Attempted:** 6
**Successfully Analyzed:** 3

## Key Findings

### Significant Differences Found: 3/3 pairs

#### cylinders by transactionmonth
- **F-statistic:** 6.12
- **P-value:** 0.000000
- **Highest mean:** 2014-12-01 (4.10)
- **Lowest mean:** 2013-11-01 (4.00)
- **Difference:** 0.10

#### cubiccapacity by transactionmonth
- **F-statistic:** 46.80
- **P-value:** 0.000000
- **Highest mean:** 2015-08-01 (2563.26)
- **Lowest mean:** 2014-02-01 (2194.12)
- **Difference:** 369.14

#### kilowatts by transactionmonth
- **F-statistic:** 5.75
- **P-value:** 0.000000
- **Highest mean:** 2014-03-01 (103.28)
- **Lowest mean:** 2013-11-01 (75.00)
- **Difference:** 28.28

## Detailed Statistics

| Numerical Variable | Categorical Variable | Groups | Sample Size | F-statistic | P-value | Significant |
|--------------------|----------------------|--------|-------------|-------------|---------|-------------|
| cylinders | transactionmonth | 22 | 50000 | 6.12 | 0.000000 | Yes |
| cubiccapacity | transactionmonth | 22 | 50000 | 46.80 | 0.000000 | Yes |
| kilowatts | transactionmonth | 22 | 50000 | 5.75 | 0.000000 | Yes |

## Files Generated

1. **Summary CSV:** `numerical_vs_categorical_summary.csv`
2. **Visualizations:** PNG files for each variable pair
3. **This Report:** `numerical_vs_categorical_analysis.md`

## Recommendations

1. **Investigate significant differences** found in the analysis
2. **Consider stratifying analysis** by these categorical variables
3. **Validate findings** with business domain experts
