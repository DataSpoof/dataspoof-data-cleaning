# Statistical testing


```python
from mlpilot import stats

stats.compare_groups(df, value="revenue", group="region")
# -> one-way ANOVA (or Welch / Kruskal-Wallis), omega-squared, Tukey pairwise comparisons, plain English
stats.chi_square(df, "plan", "churned")            # Cramér's V; Fisher's exact for small 2x2 tables
stats.correlation_test(df, "price", "demand", method="spearman")   # with a 95% CI
stats.ab_test(df, variant="variant", metric="converted", control="control")   # lift, CI, p-value
stats.required_sample_size(baseline_rate=0.10, mde=0.02)          # -> 3,841 per variant
stats.test_all(df, target="churned")               # every column vs the target, FDR-corrected
```

**How the test is chosen:** `compare_groups` checks each group's normality and the equality
of variances (Levene) first, then chooses between Welch's t-test, Mann-Whitney, ANOVA,
Alexander-Govern and Kruskal-Wallis.

**What every result includes:**
- the test statistic and p-value;
- an effect size with a size label (negligible / small / medium / large);
- a confidence interval where one applies;
- the assumptions that were checked;
- a one-line interpretation.

The tests check the results against reference values (Fisher's exact test, Holm and BH
corrections, sample-size formulas).
