"""Insights & reporting (module 10): plain-English findings and executive reports.

* ``find_insights(df, target)``      ranked, significance-tested findings: drivers ("sales is 18%
                                     lower when stock < 20"), trends, correlations, concentration,
                                     outliers, data-quality notes, model drivers.
* ``insights_report(df, target, llm=True)``   KPIs + findings + an executive narrative written
                                     by Claude (or a template, offline) -> .html / .md / .json.
"""

from .finder import Insight, find_insights
from .narrate import DEFAULT_MODEL, narrate, template_narrative
from .report import InsightsReport, insights_report

__all__ = ["Insight", "find_insights", "narrate", "template_narrative", "InsightsReport", "insights_report",
           "DEFAULT_MODEL"]
