# Insights & reporting


```python
for insight in ac.find_insights(sales_df, "sales"):
    print(insight)                     # e.g.
# Average sales rose about 50% from Jan 2024 to Dec 2024 (steady monthly trend, R² 1.00).
# sales is 19% lower when stock < 24 (average 104 vs 128 otherwise; 499 rows).

report = ac.insights_report(sales_df, "sales", title="Q4 sales review")      # offline template summary
report = ac.insights_report(sales_df, "sales", llm=True)                     # Claude writes the narrative
report.save("sales_review.html")                                             # also .md / .json
```

**What it finds:**
- segments that move the target up or down;
- trends and sudden changes over time;
- strong correlations;
- concentration, such as "20% of stores produce 80% of revenue";
- outliers and data-quality caveats;
- with `model=...`, the model's own top drivers.

Segment findings must pass a significance test corrected for multiple comparisons, and must cover
enough rows. Every finding carries the numbers behind it.

**The LLM narrative.** With `llm=True`, Claude (`claude-opus-5-5` by default) writes the executive
summary:
- It receives only the findings and their aggregate numbers, never raw rows.
- It is told not to introduce numbers that aren't in the findings, and to describe associations
  rather than causes.
- It uses `ANTHROPIC_API_KEY` or an `ant auth login` profile.
- Without credentials, or on any API error, the report falls back to the template summary with a
  warning.
- `llm=False` (the default) makes no network calls.

Install the SDK with `pip install "AutomatedCleaning[llm]"`.
