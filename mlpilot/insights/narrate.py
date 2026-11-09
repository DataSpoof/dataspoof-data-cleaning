"""Turning insights into an executive narrative: an LLM (Claude), or a deterministic template.

Only the insight sentences and their aggregate numbers are sent to the model — never raw
rows. The prompt forbids numbers that aren't in the facts. Without credentials, or on any
API error or refusal, the template narrative is used instead.
"""

from __future__ import annotations

import json
import warnings

from ..core import ServiceWarning

DEFAULT_MODEL = "claude-opus-5-5"

_SYSTEM = """You write short executive summaries of data analyses for business readers.
Rules:
- Use ONLY the facts provided. Every number you write must appear in the facts; never estimate or invent figures.
- The findings are associations in historical data, not proven causes. Say "is associated with" or "tends to", not "causes".
- Lead with the most important finding. Group related findings. Plain language, no jargon, no statistics terms like p-value.
- Format: a 1-2 sentence headline paragraph, then 3-6 bullet points of key findings, then one short "Suggested next steps" paragraph.
- Keep it under 250 words. Plain Markdown, no tables, no headings above level 3."""


def template_narrative(insights: list, context: dict | None = None) -> str:
    """A plain summary built from the insights themselves (no LLM)."""
    if not insights:
        return "No notable patterns were found that pass the significance and size thresholds."
    ctx = context or {}
    about = f" about {ctx['target']}" if ctx.get("target") else ""
    rows = f" of {ctx['rows']:,} rows" if isinstance(ctx.get("rows"), int) else ""
    lines = [f"The analysis{rows} found {len(insights)} notable pattern(s){about}. "
             f"The strongest: {insights[0].headline}", ""]
    lines += [f"- {i.headline}" for i in insights[1:7]]
    lines += ["", "These are associations in historical data, not proven causes; test the most important ones "
                  "(for example with an experiment) before acting on them."]
    return "\n".join(lines).strip()


def narrate(insights: list, *, context: dict | None = None, model: str = DEFAULT_MODEL, client=None,
            max_tokens: int = 4000, effort: str = "medium") -> tuple:
    """Write the executive summary with Claude. Returns ``(text, source)`` where ``source`` is
    ``"llm:<model>"`` or ``"template"`` (used when the API can't be reached or declines).

    ``client``: an ``anthropic.Anthropic`` client (default: one built from the environment —
    ``ANTHROPIC_API_KEY`` or an ``ant auth login`` profile)."""
    facts = [{"finding": i.headline, "kind": i.kind,
              "numbers": {k: v for k, v in i.evidence.items()
                          if isinstance(v, (int, float, str)) and k not in ("label", "p_value", "q_value")}}
             for i in insights]
    prompt = ("Write the executive summary for this analysis.\n\n<context>\n"
              + json.dumps(context or {}, default=str) + "\n</context>\n\n<facts>\n"
              + json.dumps(facts, indent=1, default=str) + "\n</facts>")
    try:
        if client is None:
            import anthropic
            client = anthropic.Anthropic()
        response = client.beta.messages.create(
            model=model, max_tokens=max_tokens, system=_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
            output_config={"effort": effort},
            # If the model declines, the API retries on a fallback model in the same call.
            betas=["server-side-fallback-2026-07-01"], fallbacks="default",
        )
        if response.stop_reason == "refusal":
            raise RuntimeError("the model declined the request")
        text = "".join(b.text for b in response.content if getattr(b, "type", None) == "text").strip()
        if not text:
            raise RuntimeError("empty response")
        return text, f"llm:{getattr(response, 'model', model)}"
    except Exception as exc:              # no key, network, rate limit, refusal: never fail the report
        warnings.warn(f"LLM narrative unavailable ({type(exc).__name__}: {exc}); using the template summary.",
                      ServiceWarning, stacklevel=2)
        return template_narrative(insights, context), "template"
