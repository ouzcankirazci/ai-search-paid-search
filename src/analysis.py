"""
Analysis for "AI Search Isn't Cutting Your Leads - It's Filtering Them"

Reads data/marketing_data.csv, applies documented cleaning rules, computes the
derived performance metrics, runs the four-step analysis, quantifies the
cut-vs-reallocate business impact, and renders the single hero chart.

Run:
    python src/analysis.py
Outputs:
    charts/hero_chart.png
    console report with every number cited in the README / memo
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "marketing_data.csv"
CHART_PATH = ROOT / "charts" / "hero_chart.png"

COMMERCIAL_INTENT = ("commercial", "transactional")
ROLL = 4  # 4-week rolling window for trend visuals (totals/stats use raw weeks)


# ---------------------------------------------------------------------------
# Cleaning + metrics
# ---------------------------------------------------------------------------
def safe_div(numerator, denominator):
    """Element-wise divide that returns NaN (never 0 or inf) on a zero denominator.

    Cleaning decision: rate metrics are undefined when their denominator is zero
    (e.g. a paused, zero-click week). We mark them NaN so they are excluded from
    rate averages rather than silently poisoning them with 0 or inf.
    """
    num = np.asarray(numerator, dtype="float64")
    den = np.asarray(denominator, dtype="float64")
    out = np.full(num.shape, np.nan, dtype="float64")
    mask = den != 0
    out[mask] = num[mask] / den[mask]
    return out


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, parse_dates=["week"])

    # Cleaning decision: brand is treated separately. Branded demand is largely
    # insulated from AI Overviews; leaving it in would dilute the erosion signal.
    df["is_branded"] = df["query_intent"].eq("branded")

    # Row-level derived metrics (guarded against zero denominators).
    df["ctr"] = safe_div(df["clicks"], df["impressions"])
    df["cpc"] = safe_div(df["cost"], df["clicks"])
    df["conversion_rate"] = safe_div(df["leads"], df["clicks"])
    df["cost_per_lead"] = safe_div(df["cost"], df["leads"])
    df["cost_per_qualified_lead"] = safe_div(df["cost"], df["qualified_leads"])
    df["qualified_lead_rate"] = safe_div(df["qualified_leads"], df["leads"])
    df["cac"] = safe_div(df["cost"], df["customers"])
    df["roas"] = safe_div(df["revenue"], df["cost"])
    df["revenue_per_visitor"] = safe_div(df["revenue"], df["clicks"])
    return df


def weekly_rollup(df: pd.DataFrame, by_ai: bool = False) -> pd.DataFrame:
    """Sum volumes per week, then derive rates from the sums.

    Cleaning decision: aggregate rates are computed from summed numerators and
    denominators (not as the mean of per-row rates) so paused/zero weeks cannot
    distort the averages and segment sizes are properly weighted.
    """
    keys = ["week", "ai_overview_present"] if by_ai else ["week"]
    g = df.groupby(keys, as_index=False).agg(
        impressions=("impressions", "sum"),
        clicks=("clicks", "sum"),
        cost=("cost", "sum"),
        leads=("leads", "sum"),
        qualified_leads=("qualified_leads", "sum"),
        customers=("customers", "sum"),
        revenue=("revenue", "sum"),
    )
    g["qualified_lead_rate"] = safe_div(g["qualified_leads"], g["leads"])
    g["cost_per_qualified_lead"] = safe_div(g["cost"], g["qualified_leads"])
    g["revenue_per_visitor"] = safe_div(g["revenue"], g["clicks"])
    g["roas"] = safe_div(g["revenue"], g["cost"])
    return g.sort_values(keys).reset_index(drop=True)


def pct_change(first: float, last: float) -> float:
    return 100.0 * (last / first - 1.0) if first else float("nan")


def window_means(series: pd.Series, n: int = 8) -> tuple[float, float]:
    """Mean of the first n and last n observations (noise-robust endpoints)."""
    return float(series.head(n).mean()), float(series.tail(n).mean())


# ---------------------------------------------------------------------------
# Analysis steps
# ---------------------------------------------------------------------------
def headline_changes(nonbrand: pd.DataFrame) -> dict:
    wk = weekly_rollup(nonbrand)
    leads_first, leads_last = window_means(wk["leads"])
    qr_first, qr_last = window_means(wk["qualified_lead_rate"])
    ql_first, ql_last = window_means(wk["qualified_leads"])
    rpv_first, rpv_last = window_means(wk["revenue_per_visitor"])
    return dict(
        leads_first=leads_first, leads_last=leads_last,
        leads_change=pct_change(leads_first, leads_last),
        qr_first=qr_first, qr_last=qr_last,
        qr_change=pct_change(qr_first, qr_last),
        ql_first=ql_first, ql_last=ql_last,
        ql_change=pct_change(ql_first, ql_last),
        rpv_first=rpv_first, rpv_last=rpv_last,
        rpv_change=pct_change(rpv_first, rpv_last),
    )


def decomposition(nonbrand: pd.DataFrame) -> dict:
    """Reveal that the drop is concentrated in informational queries, and that the
    mechanism is AI Overviews: where an AI Overview is present, each 1,000
    impressions yields far fewer leads.

    Returns:
      by_intent : leads first-8 vs last-8 weeks, % change, per intent.
      yield_tbl : leads per 1,000 impressions, AI present vs not, per intent
                  (full period) plus the yield penalty AI imposes.
      info_pen  : informational impressions behind an AI Overview, first vs last.
    """
    # (a) Concentration: where did the leads actually go?
    g = nonbrand.groupby(["query_intent", "week"], as_index=False)["leads"].sum()
    recs = []
    for intent, sub in g.groupby("query_intent"):
        sub = sub.sort_values("week")
        first, last = window_means(sub["leads"])
        recs.append(dict(query_intent=intent, leads_first=first, leads_last=last,
                         change_pct=pct_change(first, last)))
    by_intent = pd.DataFrame(recs).sort_values("change_pct").reset_index(drop=True)

    # (b) Mechanism: lead yield (leads per 1,000 impressions) by intent x AI.
    y = nonbrand.groupby(["query_intent", "ai_overview_present"], as_index=False).agg(
        leads=("leads", "sum"), impressions=("impressions", "sum"))
    y["leads_per_1k_impr"] = safe_div(y["leads"], y["impressions"]) * 1000
    yield_tbl = y.pivot(index="query_intent", columns="ai_overview_present",
                        values="leads_per_1k_impr")
    yield_tbl["ai_yield_penalty_pct"] = (yield_tbl["yes"] / yield_tbl["no"] - 1) * 100

    # (c) Penetration: how much informational traffic moved behind an AI Overview.
    info = nonbrand[nonbrand["query_intent"] == "informational"]
    pen = info.groupby(["week", "ai_overview_present"], as_index=False)["impressions"].sum()
    pp = pen.pivot(index="week", columns="ai_overview_present", values="impressions").fillna(0)
    share_yes = pp["yes"] / (pp["yes"] + pp["no"])
    pen_first, pen_last = window_means(share_yes.sort_index())

    return dict(by_intent=by_intent, yield_tbl=yield_tbl,
                info_pen_first=pen_first, info_pen_last=pen_last)


def business_impact(nonbrand: pd.DataFrame) -> dict:
    """Cut-vs-reallocate scenario, built on the most recent 8-week run rate."""
    recent = nonbrand[nonbrand["week"].isin(sorted(nonbrand["week"].unique())[-8:])]

    def agg(frame):
        cost = frame["cost"].sum()
        ql = frame["qualified_leads"].sum()
        rev = frame["revenue"].sum()
        return cost, ql, rev

    info = recent[recent["query_intent"] == "informational"]
    ci = recent[recent["query_intent"].isin(COMMERCIAL_INTENT)]

    cost_info, ql_info, rev_info = agg(info)
    cost_ci, ql_ci, rev_ci = agg(ci)
    cost_all, ql_all, rev_all = agg(recent)

    weeks = recent["week"].nunique()
    # Per-week run rate.
    wk_ql_all, wk_rev_all = ql_all / weeks, rev_all / weeks

    # Efficiency of commercial-intent spend (blended).
    cpql_ci = cost_ci / ql_ci if ql_ci else float("nan")
    rev_per_ql_ci = rev_ci / ql_ci if ql_ci else float("nan")
    cpql_info = cost_info / ql_info if ql_info else float("nan")

    # Reallocation: move 70% of informational spend into commercial-intent at its
    # current blended efficiency; keep 30% for discovery/brand-building.
    shift = 0.70
    moved_spend = cost_info * shift
    ql_lost_from_info = ql_info * shift
    ql_gained_from_ci = moved_spend / cpql_ci if cpql_ci else 0.0
    net_ql_delta = ql_gained_from_ci - ql_lost_from_info
    net_rev_delta = net_ql_delta * rev_per_ql_ci

    return dict(
        weeks=weeks,
        cost_all=cost_all, ql_all=ql_all, rev_all=rev_all,
        wk_ql_all=wk_ql_all, wk_rev_all=wk_rev_all,
        annual_ql_cut=wk_ql_all * 52, annual_rev_cut=wk_rev_all * 52,
        cpql_info=cpql_info, cpql_ci=cpql_ci, rev_per_ql_ci=rev_per_ql_ci,
        moved_spend=moved_spend, shift=shift,
        net_ql_delta_8wk=net_ql_delta, annual_net_ql=net_ql_delta / weeks * 52,
        annual_net_rev=net_rev_delta / weeks * 52,
    )


# ---------------------------------------------------------------------------
# Hero chart
# ---------------------------------------------------------------------------
def make_hero_chart(nonbrand: pd.DataFrame, h: dict) -> None:
    wk_ai = weekly_rollup(nonbrand, by_ai=True)
    wk_all = weekly_rollup(nonbrand)

    pivot = wk_ai.pivot(index="week", columns="ai_overview_present", values="leads").fillna(0)
    for col in ("no", "yes"):
        if col not in pivot:
            pivot[col] = 0.0
    pivot = pivot.sort_index()

    # 4-week rolling smoothing for the visual only (documented in subtitle).
    leads_no = pivot["no"].rolling(ROLL, min_periods=1).mean()
    leads_yes = pivot["yes"].rolling(ROLL, min_periods=1).mean()
    qr = (
        wk_all.set_index("week")["qualified_lead_rate"]
        .rolling(ROLL, min_periods=1)
        .mean()
    )
    weeks = pivot.index

    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": "#999999"})
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.subplots_adjust(top=0.80, bottom=0.16, left=0.085, right=0.90)

    C_NO = "#1f77b4"      # leads without an AI Overview
    C_YES = "#d6604d"     # leads with an AI Overview present
    C_QR = "#1a7f37"      # qualified-lead rate

    ax.stackplot(
        weeks, leads_no, leads_yes,
        labels=["Leads · no AI Overview", "Leads · AI Overview present"],
        colors=[C_NO, C_YES], alpha=0.85, edgecolor="white", linewidth=0.4,
    )
    ax.set_ylabel("Weekly paid-search leads (non-brand, 4-wk avg)", fontsize=11)
    ax.set_ylim(0, (leads_no + leads_yes).max() * 1.18)
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.margins(x=0)
    ax.grid(axis="y", color="#e6e6e6", linewidth=0.8)
    ax.set_axisbelow(True)

    # Secondary axis: qualified-lead rate.
    ax2 = ax.twinx()
    ax2.plot(weeks, qr * 100, color=C_QR, linewidth=3.0, label="Qualified-lead rate")
    ax2.set_ylabel("Qualified-lead rate (%)", color=C_QR, fontsize=11)
    ax2.tick_params(axis="y", colors=C_QR)
    ax2.set_ylim(0, max(60, float((qr * 100).max()) * 1.35))
    for spine in ("top",):
        ax.spines[spine].set_visible(False)
        ax2.spines[spine].set_visible(False)

    # Title + subtitle (with the data caveat).
    fig.suptitle(
        "AI search isn't cutting your leads \u2014 it's filtering them",
        x=0.085, y=0.955, ha="left", fontsize=18.5, fontweight="bold",
    )
    fig.text(
        0.085, 0.885,
        "As AI Overviews spread, paid-search volume fell \u2014 but the leads that "
        "remained were better.\n"
        "Synthetic, research-informed data (52 weeks); brand campaigns excluded. "
        "Volumes 4-week smoothed.",
        ha="left", va="top", fontsize=10.5, color="#444444",
    )

    # The one callout.
    callout = (
        f"Lead volume fell {abs(h['leads_change']):.0f}%, but the qualified-lead "
        f"rate rose {h['qr_change']:.0f}%\n(from {h['qr_first']*100:.0f}% to "
        f"{h['qr_last']*100:.0f}%). Qualified leads held roughly flat "
        f"({h['ql_change']:+.0f}%).\nThe issue was lead MIX, not performance."
    )
    ax.annotate(
        callout,
        xy=(weeks[int(len(weeks) * 0.62)], qr.iloc[int(len(weeks) * 0.62)] * 0 +
            (leads_no + leads_yes).max() * 0.92),
        xytext=(weeks[int(len(weeks) * 0.30)], (leads_no + leads_yes).max() * 1.08),
        fontsize=11.5, fontweight="bold", color="#111111",
        bbox=dict(boxstyle="round,pad=0.5", fc="#fff8e1", ec="#e0b400", lw=1.4),
    )

    # Legends combined.
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(
        lines1 + lines2, labels1 + labels2,
        loc="lower left", fontsize=9.5, frameon=True, framealpha=0.9, ncol=1,
    )

    fig.text(
        0.085, 0.035,
        "Source: simulated advertiser data modeled on Pew (CTR ~15%\u2192~8% with "
        "AI summaries), Similarweb (zero-click ~56%\u2192~69%), and informational "
        "CTR \u221230\u2013\u201360% studies.",
        ha="left", fontsize=8.5, color="#777777",
    )

    CHART_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(CHART_PATH, dpi=150)
    plt.close(fig)
    print(f"\nSaved hero chart -> {CHART_PATH}")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------
def main() -> None:
    df = load_data()
    nonbrand = df[~df["is_branded"]].copy()
    brand = df[df["is_branded"]].copy()

    print("=" * 72)
    print("AI SEARCH & PAID SEARCH  -  ANALYSIS REPORT")
    print("=" * 72)

    # Data quality / cleaning footprint.
    zero_clicks = int((df["clicks"] == 0).sum())
    zero_spend = int((df["cost"] == 0).sum())
    print(f"\n[Cleaning] rows={len(df)} | weeks={df['week'].nunique()} | "
          f"campaigns={df.groupby(['query_intent','campaign_type','landing_page_type']).ngroups}")
    print(f"[Cleaning] zero-click rows={zero_clicks} | zero-spend rows={zero_spend} "
          f"(rate metrics on these are NaN, excluded from rate averages)")
    print(f"[Cleaning] branded rows separated and EXCLUDED from erosion analysis "
          f"({len(brand)} rows)")

    # Step 1 + 3: headline.
    h = headline_changes(nonbrand)
    print("\n--- STEP 1: The alarm (total non-brand paid leads) ---")
    print(f"  Weekly leads: {h['leads_first']:.0f} -> {h['leads_last']:.0f} "
          f"({h['leads_change']:+.1f}%)")
    print("\n--- STEP 3: The twist (quality of remaining leads) ---")
    print(f"  Qualified-lead rate: {h['qr_first']*100:.1f}% -> {h['qr_last']*100:.1f}% "
          f"({h['qr_change']:+.1f}%)")
    print(f"  Qualified leads/wk : {h['ql_first']:.0f} -> {h['ql_last']:.0f} "
          f"({h['ql_change']:+.1f}%)")
    print(f"  Revenue per visitor: ${h['rpv_first']:.2f} -> ${h['rpv_last']:.2f} "
          f"({h['rpv_change']:+.1f}%)")

    # Step 2: decomposition.
    print("\n--- STEP 2: Where the drop lives (concentration + mechanism) ---")
    dec = decomposition(nonbrand)
    print("  Lead change by intent (first 8 vs last 8 weeks):")
    for _, r in dec["by_intent"].iterrows():
        print(f"    {r['query_intent']:<14} {r['leads_first']:7.0f} -> "
              f"{r['leads_last']:7.0f}  ({r['change_pct']:+6.1f}%)")
    print("  Lead yield (leads per 1,000 impressions), AI Overview present vs not:")
    yt = dec["yield_tbl"]
    for intent, r in yt.iterrows():
        print(f"    {intent:<14} no-AI={r['no']:.2f}  AI={r['yes']:.2f}  "
              f"(yield {r['ai_yield_penalty_pct']:+.0f}% under AI)")
    print(f"  Informational impressions behind an AI Overview: "
          f"{dec['info_pen_first']*100:.0f}% -> {dec['info_pen_last']*100:.0f}%")

    # Step 4: business impact.
    print("\n--- STEP 4: Business impact (last 8-week run rate) ---")
    b = business_impact(nonbrand)
    print(f"  Current non-brand run rate: {b['wk_ql_all']:.0f} qualified leads/wk, "
          f"${b['wk_rev_all']:,.0f} revenue/wk")
    print(f"  CPQL informational=${b['cpql_info']:,.0f}  vs  "
          f"commercial-intent=${b['cpql_ci']:,.0f}")
    print(f"  SCENARIO 'CUT': forfeit ~{b['annual_ql_cut']:,.0f} qualified leads "
          f"and ~${b['annual_rev_cut']:,.0f} revenue per year.")
    print(f"  SCENARIO 'REALLOCATE' (shift {b['shift']*100:.0f}% of info spend, "
          f"${b['moved_spend']:,.0f}/8wk):")
    print(f"     net +{b['annual_net_ql']:,.0f} qualified leads/yr and "
          f"+${b['annual_net_rev']:,.0f} revenue/yr at equal spend.")

    make_hero_chart(nonbrand, h)
    print("\nDone.")


if __name__ == "__main__":
    main()
