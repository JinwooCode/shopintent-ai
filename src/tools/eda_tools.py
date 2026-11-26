"""Deterministic EDA: clean_data.csv -> eda_stats.json + eda_report.html (no LLM).

The Insights Analyst agent only ever sees the facts produced here (stats_to_facts), so every number it
writes can be checked against them.
"""
from __future__ import annotations

import base64
import io
import json
from html import escape
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

MONTH_ORDER = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
VISITOR_LABEL = {"Returning_Visitor": "Returning visitors", "New_Visitor": "New visitors", "Other": "Other visitors"}
# Behaviour compared between buyers and non-buyers: column -> (label, unit, scale)
COMPARE = {
    "ProductRelated": ("Product pages viewed", "pages", 1),
    "ProductRelated_Duration": ("Time on product pages", "min", 1 / 60),
    "Administrative": ("Account & admin pages viewed", "pages", 1),
    "Informational": ("Information pages viewed", "pages", 1),
    "BounceRates": ("Bounce rate of pages viewed", "%", 100),
    "ExitRates": ("Exit rate of pages viewed", "%", 100),
}
NUMERIC = ["Administrative", "Administrative_Duration", "Informational", "Informational_Duration",
           "ProductRelated", "ProductRelated_Duration", "BounceRates", "ExitRates", "PageValues", "SpecialDay"]
LEAKY = ["PageValues"]   # excluded from the main model; never offered to the agent as a driver
COLORS = {"main": "#4F46E5", "soft": "#C7D2FE", "buyer": "#059669", "non": "#94A3B8"}


# --------------------------------------------------------------------------- stats
def _rate_table(df: pd.DataFrame, col: str, order: list | None = None, label=None) -> list[dict]:
    g = df.groupby(col)["Revenue"].agg(sessions="size", buyers="sum")
    keys = [k for k in (order or g.index.tolist()) if k in g.index]
    return [{"group": (label(k) if label else str(k)), "sessions": int(g.at[k, "sessions"]),
             "buyers": int(g.at[k, "buyers"]), "rate": round(g.at[k, "buyers"] / g.at[k, "sessions"], 4)}
            for k in keys]


def compute_eda_stats(df: pd.DataFrame) -> dict:
    buyers = df[df["Revenue"]]
    non = df[~df["Revenue"]]
    near = df.assign(near=df["SpecialDay"] > 0)
    traffic = _rate_table(df, "TrafficType")
    traffic = sorted(traffic, key=lambda r: -r["sessions"])[:5]
    for r in traffic:
        r["group"] = f"Traffic source {r['group']}"
    corr = (df[NUMERIC + ["Revenue"]].astype(float).corr()["Revenue"].drop("Revenue")
            .sort_values(key=abs, ascending=False))
    return {
        "dataset": {"sessions": int(len(df)), "buyers": int(df["Revenue"].sum()),
                    "purchase_rate": round(float(df["Revenue"].mean()), 4), "columns": int(df.shape[1])},
        "by_month": _rate_table(df, "Month", MONTH_ORDER),
        "by_visitor_type": _rate_table(df, "VisitorType", list(VISITOR_LABEL), VISITOR_LABEL.get),
        "by_weekend": _rate_table(df, "Weekend", [False, True], lambda k: "Weekend" if k else "Weekday"),
        "by_special_day": _rate_table(near, "near", [False, True],
                                      lambda k: "Close to a holiday" if k else "Not close to a holiday"),
        "top_traffic_sources": traffic,
        "buyer_vs_non_buyer": [
            {"feature": c, "label": lab, "unit": unit,
             "buyer_median": round(float(buyers[c].median()) * k, 1),
             "non_buyer_median": round(float(non[c].median()) * k, 1)}
            for c, (lab, unit, k) in COMPARE.items()],
        "correlation_with_revenue": [{"feature": f, "r": round(float(v), 3), "leaky": f in LEAKY}
                                     for f, v in corr.items()],
    }


def _pct(x: float) -> str:
    return f"{x * 100:.1f}%"


def stats_to_facts(stats: dict) -> list[str]:
    """Plain-language facts handed to the agent. Leaky features are left out on purpose."""
    d = stats["dataset"]
    facts = [f"The dataset has {d['sessions']:,} sessions; {d['buyers']:,} ended in a purchase "
             f"(overall purchase rate {_pct(d['purchase_rate'])})."]
    for key, what in (("by_month", "in {g}"), ("by_visitor_type", "for {g}"), ("by_weekend", "on a {g}"),
                      ("by_special_day", "for visits {g}"), ("top_traffic_sources", "for {g}")):
        for r in stats[key]:
            facts.append(f"Purchase rate {what.format(g=r['group'].lower() if key != 'by_month' else r['group'])}: "
                         f"{_pct(r['rate'])} ({r['sessions']:,} sessions, {r['buyers']:,} buyers).")
    for r in stats["buyer_vs_non_buyer"]:
        u = "%" if r["unit"] == "%" else f" {r['unit']}"
        facts.append(f"{r['label']} (median): buyers {r['buyer_median']}{u}, non-buyers {r['non_buyer_median']}{u}.")
    for r in stats["correlation_with_revenue"]:
        if not r["leaky"]:
            facts.append(f"Correlation of {r['feature']} with purchase: {r['r']}.")
    return facts


# --------------------------------------------------------------------------- report
def _png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight", metadata={"Software": None})
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def _style(ax, title: str):
    ax.set_title(title, loc="left", fontsize=11, fontweight="bold", color="#111827")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.spines["left"].set_color("#D1D5DB")
    ax.spines["bottom"].set_color("#D1D5DB")
    ax.tick_params(colors="#4B5563", labelsize=9)


def _rate_chart(rows: list[dict], title: str, overall: float, horizontal=False) -> str:
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    labels, vals = [r["group"] for r in rows], [r["rate"] * 100 for r in rows]
    top = max(vals)
    colors = [COLORS["main"] if v == top else COLORS["soft"] for v in vals]
    axis_label = f"Purchase rate (%)  ·  dashed line = overall {overall * 100:.1f}%"
    if horizontal:
        ax.barh(labels[::-1], vals[::-1], color=colors[::-1])
        ax.axvline(overall * 100, color="#6B7280", ls="--", lw=1)
        ax.set_xlabel(axis_label, fontsize=9, color="#4B5563")
    else:
        ax.bar(labels, vals, color=colors)
        ax.axhline(overall * 100, color="#6B7280", ls="--", lw=1)
        ax.set_xlabel(axis_label, fontsize=9, color="#4B5563")
        ax.set_ylabel("Purchase rate (%)", fontsize=9, color="#4B5563")
    _style(ax, "")   # the section heading already names the chart
    return _png(fig)


def _corr_chart(rows: list[dict]) -> str:
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    rows = rows[::-1]
    ax.barh([r["feature"] + (" (excluded, leakage)" if r["leaky"] else "") for r in rows], [r["r"] for r in rows],
            color=["#E5E7EB" if r["leaky"] else (COLORS["buyer"] if r["r"] > 0 else "#EF4444") for r in rows])
    ax.axvline(0, color="#9CA3AF", lw=1)
    ax.set_xlabel("Correlation with purchase (Pearson r)", fontsize=9, color="#4B5563")
    _style(ax, "How each numeric feature moves with purchase")
    return _png(fig)


def _table(rows: list[dict]) -> str:
    body = "".join(f"<tr><td>{escape(r['group'])}</td><td>{r['sessions']:,}</td><td>{r['buyers']:,}</td>"
                   f"<td><b>{_pct(r['rate'])}</b></td></tr>" for r in rows)
    return f"<table><tr><th>Group</th><th>Sessions</th><th>Buyers</th><th>Purchase rate</th></tr>{body}</table>"


def build_eda_report(stats: dict, contract: dict, out: Path) -> None:
    d = stats["dataset"]
    overall = d["purchase_rate"]
    cmp_rows = "".join(
        f"<tr><td>{escape(r['label'])}</td><td>{r['buyer_median']} {r['unit']}</td>"
        f"<td>{r['non_buyer_median']} {r['unit']}</td></tr>" for r in stats["buyer_vs_non_buyer"])
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    labels = [r["label"] for r in stats["buyer_vs_non_buyer"] if r["unit"] == "pages"]
    b = [r["buyer_median"] for r in stats["buyer_vs_non_buyer"] if r["unit"] == "pages"]
    n = [r["non_buyer_median"] for r in stats["buyer_vs_non_buyer"] if r["unit"] == "pages"]
    y = range(len(labels))
    ax.barh([i + 0.2 for i in y], b, height=0.4, color=COLORS["buyer"], label="Buyers")
    ax.barh([i - 0.2 for i in y], n, height=0.4, color=COLORS["non"], label="Non-buyers")
    ax.set_yticks(list(y), labels)
    ax.legend(frameon=False, fontsize=8)
    ax.set_xlabel("Median pages per session", fontsize=9, color="#4B5563")
    _style(ax, "Buyers browse far more product pages")
    cmp_png = _png(fig)

    sections = [
        ("Purchase rate by month", _rate_chart(stats["by_month"], "Purchase rate by month", overall),
         _table(stats["by_month"])),
        ("Returning vs new visitors", _rate_chart(stats["by_visitor_type"], "Purchase rate by visitor type",
                                                  overall, True), _table(stats["by_visitor_type"])),
        ("Weekend and holidays", _rate_chart(stats["by_weekend"] + stats["by_special_day"],
                                             "Weekend and holiday proximity", overall, True),
         _table(stats["by_weekend"] + stats["by_special_day"])),
        ("Top traffic sources", _rate_chart(stats["top_traffic_sources"], "Five largest traffic sources",
                                            overall, True), _table(stats["top_traffic_sources"])),
        ("Buyers vs non-buyers", cmp_png,
         f"<table><tr><th>Behaviour (median)</th><th>Buyers</th><th>Non-buyers</th></tr>{cmp_rows}</table>"),
        ("Correlation with purchase", _corr_chart(stats["correlation_with_revenue"]),
         "<p class='note'>PageValues correlates most strongly, but it is computed from pages that led to "
         "purchases, so it leaks the answer. It is excluded from the main model and from the insights.</p>"),
    ]
    blocks = "".join(f"<section><h2>{escape(t)}</h2><img src='data:image/png;base64,{img}' alt='{escape(t)}'>"
                     f"{extra}</section>" for t, img, extra in sections)
    lists = "".join(f"<h3>{k.capitalize()}</h3><ul>{''.join(f'<li>{escape(x)}</li>' for x in contract.get(k, []))}</ul>"
                    for k in ("assumptions", "constraints"))
    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>ShopIntent AI · EDA report</title>
<style>
body {{font-family: Inter, Segoe UI, system-ui, sans-serif; color: #1F2937; background: #F8F9FC; margin: 0;}}
main {{max-width: 920px; margin: 0 auto; padding: 2rem 1.2rem 3rem;}}
header {{background: linear-gradient(120deg, #1E1B4B, #4338CA); color: #fff; border-radius: 18px; padding: 1.6rem 1.8rem;}}
header h1 {{margin: 0 0 .3rem; font-size: 1.8rem;}} header p {{margin: 0; color: #C7D2FE;}}
.kpis {{display: grid; grid-template-columns: repeat(4, 1fr); gap: .8rem; margin: 1.2rem 0;}}
.kpi {{background: #fff; border: 1px solid #E5E7EB; border-radius: 14px; padding: .9rem 1rem;}}
.kpi b {{display: block; font-size: 1.5rem;}} .kpi span {{color: #6B7280; font-size: .82rem;}}
section {{background: #fff; border: 1px solid #E5E7EB; border-radius: 16px; padding: 1.2rem 1.4rem; margin: 1rem 0;}}
section h2 {{margin: 0 0 .8rem; font-size: 1.15rem;}} img {{max-width: 100%; height: auto;}}
table {{border-collapse: collapse; width: 100%; font-size: .88rem; margin-top: .6rem;}}
th, td {{text-align: left; padding: .45rem .6rem; border-bottom: 1px solid #F1F2F6;}}
th {{color: #6B7280; font-weight: 600;}} .note {{color: #6B7280; font-size: .88rem;}}
@media (max-width: 640px) {{.kpis {{grid-template-columns: 1fr 1fr;}}}}
</style></head><body><main>
<header><h1>ShopIntent AI · Exploratory data analysis</h1>
<p>{escape(contract.get('dataset', 'UCI Online Shoppers'))} · generated from artifacts/analyst/clean_data.csv</p></header>
<div class="kpis">
<div class="kpi"><b>{d['sessions']:,}</b><span>sessions</span></div>
<div class="kpi"><b>{d['buyers']:,}</b><span>purchases</span></div>
<div class="kpi"><b>{_pct(overall)}</b><span>purchase rate</span></div>
<div class="kpi"><b>{d['columns']}</b><span>columns</span></div>
</div>
{blocks}
<section><h2>Dataset contract</h2>{lists}</section>
<p class="note">Business interpretation of these numbers: see insights.md.</p>
</main></body></html>
"""
    out.write_text(html, encoding="utf-8", newline="\n")


def save_stats(stats: dict, out: Path) -> None:
    out.write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8", newline="\n")
