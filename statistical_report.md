# Statistical Significance Report

**Period:** 2020-02-29 to 2025-11-30
**Benchmark:** SPY (S&P 500 ETF)

## Regression Results (vs SPY)
| Metric | Value | Description |
| :--- | :--- | :--- |
| **Alpha (Annualized)** | `0.10%` | Excess return over benchmark |
| **Beta** | `0.0000` | Market sensitivity |
| **R-Squared** | `0.0184` | Correlation with market |
| **Alpha t-stat** | `295.6620` | Significance score |
| **Alpha p-value** | `0.0000` | Probability of random chance |

## Interpretation
> [!SUCCESS]
> The strategy generated statistically significant alpha (p < 0.05). The t-stat of 295.66 indicates strong evidence against the null hypothesis of zero excess returns.

## Model Summary
```
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                  Value   R-squared:                       0.018
Model:                            OLS   Adj. R-squared:                  0.004
Method:                 Least Squares   F-statistic:                     1.237
Date:                Sat, 29 Nov 2025   Prob (F-statistic):              0.270
Time:                        16:36:18   Log-Likelihood:                 790.83
No. Observations:                  68   AIC:                            -1578.
Df Residuals:                      66   BIC:                            -1573.
Df Model:                           1                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const       8.329e-05   2.82e-07    295.662      0.000    8.27e-05    8.39e-05
SPY         6.383e-06   5.74e-06      1.112      0.270   -5.08e-06    1.78e-05
==============================================================================
Omnibus:                       28.286   Durbin-Watson:                   2.934
Prob(Omnibus):                  0.000   Jarque-Bera (JB):               45.386
Skew:                          -1.609   Prob(JB):                     1.39e-10
Kurtosis:                       5.381   Cond. No.                         21.7
==============================================================================

Notes:
[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.
```
