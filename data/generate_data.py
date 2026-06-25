"""
Synthetic data generator for "AI Search Isn't Cutting Your Leads - It's Filtering Them"

The data is SYNTHETIC but RESEARCH-INFORMED. No clean public advertiser-level
dataset exists for this question, so we simulate one whose dynamics are anchored
to published figures:

  * Pew Research (~68k queries): result CTR roughly halves (~15% -> ~8%) when an
    AI summary is present.
  * Similarweb: zero-click searches rose from ~56% to ~69% in about a year.
  * Multiple studies: informational-query CTR down ~30-60%.
  * Google is moving ads INTO AI Overviews / AI Mode and pushing AI Max.

The point of the project is NOT that these exact numbers are real for any
advertiser. It is to show a defensible analytical model and the decision it
leads to. Every assumption below is listed in the README.

Grain: one row per (week, campaign, ai_overview_present).
A "campaign" is a fixed (query_intent, campaign_type, landing_page_type) combo.

Run:
    python data/generate_data.py
Output:
    data/marketing_data.csv  (~52 weeks)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42
rng = np.random.default_rng(SEED)

N_WEEKS = 52
START_WEEK = pd.Timestamp("2024-06-24")  # Mondays, 52 consecutive weeks

# A realistic Q3 budget freeze: this campaign is paused for 4 weeks, producing
# genuine zero-spend / zero-click rows the cleaning step has to handle.
PAUSES: dict[tuple[str, str, str], set[int]] = {
    ("transactional", "pmax", "demo"): set(range(18, 22)),
}

# ---------------------------------------------------------------------------
# Per-intent assumptions
#
# ai_pen_start/end: share of impressions that sit behind an AI Overview at the
#   start vs end of the year. Rises fastest for informational queries, which is
#   where AI Overviews are most aggressive.
# ai_ctr_mult: how much ad CTR is suppressed when an AI Overview is present
#   (informational ~halved, transactional barely touched), per the Pew effect.
# ctr_decay: additional baseline CTR erosion over the year even WITHOUT an AI
#   Overview (general "answer engine" drift), strongest for informational.
# cpc_rise: CPC inflation over the year as click volume falls / auctions tighten.
# ---------------------------------------------------------------------------
INTENTS = {
    "informational": dict(
        ai_pen_start=0.20, ai_pen_end=0.85,
        base_ctr=0.055, ai_ctr_mult=0.45, ctr_decay=0.35,
        base_cpc=1.20, cpc_rise=0.60,
        close_rate=0.05, avg_deal_value=800,
    ),
    "commercial": dict(
        ai_pen_start=0.08, ai_pen_end=0.42,
        base_ctr=0.045, ai_ctr_mult=0.85, ctr_decay=0.05,
        base_cpc=3.50, cpc_rise=0.35,
        close_rate=0.12, avg_deal_value=3000,
    ),
    "transactional": dict(
        ai_pen_start=0.05, ai_pen_end=0.25,
        base_ctr=0.040, ai_ctr_mult=0.92, ctr_decay=0.02,
        base_cpc=6.00, cpc_rise=0.30,
        close_rate=0.20, avg_deal_value=6000,
    ),
    "branded": dict(
        ai_pen_start=0.03, ai_pen_end=0.12,
        base_ctr=0.100, ai_ctr_mult=0.97, ctr_decay=0.02,
        base_cpc=0.80, cpc_rise=0.10,
        close_rate=0.15, avg_deal_value=2500,
    ),
}

# Landing page -> click->lead conversion rate and base lead->qualified rate.
LANDING_PAGES = {
    "generic":    dict(conv_rate=0.020, base_qual=0.15),
    "comparison": dict(conv_rate=0.050, base_qual=0.40),
    "calculator": dict(conv_rate=0.060, base_qual=0.45),
    "demo":       dict(conv_rate=0.080, base_qual=0.55),
    "case_study": dict(conv_rate=0.040, base_qual=0.35),
}

# Intent multiplier on the lead->qualified rate (transactional leads qualify best).
QUAL_INTENT_MULT = {
    "informational": 0.60,
    "commercial": 1.10,
    "transactional": 1.30,
    "branded": 1.00,
}

# Each campaign: (query_intent, campaign_type, landing_page_type, base weekly impressions).
# Informational is intentionally the largest early lead source -- it is the bucket
# AI Overviews hit hardest, so its decline drives the headline "leads are falling".
CAMPAIGNS = [
    ("informational", "search", "generic",    160_000),
    ("commercial",    "search", "comparison",  60_000),
    ("commercial",    "search", "calculator",  28_000),
    ("transactional", "search", "demo",        35_000),
    ("branded",       "search", "case_study",  40_000),
    ("informational", "pmax",   "generic",      60_000),
    ("commercial",    "pmax",   "comparison",   30_000),
    ("transactional", "pmax",   "demo",         18_000),
]


def _noise(scale: float) -> float:
    """Multiplicative lognormal-ish noise centered near 1.0."""
    return float(np.exp(rng.normal(0.0, scale)))


def generate() -> pd.DataFrame:
    rows: list[dict] = []
    weeks = [START_WEEK + pd.Timedelta(weeks=w) for w in range(N_WEEKS)]

    for w_idx, week in enumerate(weeks):
        t = w_idx / (N_WEEKS - 1)  # 0.0 -> 1.0 across the year
        # Mild seasonality: gentle dip mid-year, lift toward year end.
        season = 1.0 + 0.06 * np.sin(2 * np.pi * (w_idx / N_WEEKS))

        for intent, campaign_type, lp, base_impr in CAMPAIGNS:
            cfg = INTENTS[intent]
            lp_cfg = LANDING_PAGES[lp]

            # Budget freeze -> emit explicit zero-spend rows for paused weeks.
            paused_weeks = PAUSES.get((intent, campaign_type, lp), set())
            if w_idx in paused_weeks:
                for ai_flag in ("yes", "no"):
                    rows.append(_zero_row(week, intent, ai_flag, campaign_type, lp))
                continue

            total_impr = base_impr * season * _noise(0.05)

            # AI Overview penetration for this intent this week.
            ai_pen = cfg["ai_pen_start"] + (cfg["ai_pen_end"] - cfg["ai_pen_start"]) * t
            ai_pen = float(np.clip(ai_pen * _noise(0.04), 0.0, 0.97))

            for ai_flag in ("yes", "no"):
                share = ai_pen if ai_flag == "yes" else (1.0 - ai_pen)
                impressions = int(round(total_impr * share))
                if impressions <= 0:
                    rows.append(_zero_row(week, intent, ai_flag, campaign_type, lp))
                    continue

                # ---- CTR: baseline erosion + AI Overview suppression ----
                ctr = cfg["base_ctr"] * (1.0 - cfg["ctr_decay"] * t)
                if ai_flag == "yes":
                    ctr *= cfg["ai_ctr_mult"]
                ctr = max(ctr * _noise(0.06), 0.0)
                clicks = int(round(impressions * ctr))

                # ---- CPC: inflates over the year; slightly higher under AI ----
                cpc = cfg["base_cpc"] * (1.0 + cfg["cpc_rise"] * t)
                if ai_flag == "yes":
                    cpc *= 1.08
                cpc *= _noise(0.05)
                cost = round(clicks * cpc, 2)

                # ---- Leads: click->lead by landing page ----
                lead_rate = lp_cfg["conv_rate"] * _noise(0.07)
                leads = int(round(clicks * lead_rate))

                # ---- Qualified leads: lead->qualified, improves ~10% over year ----
                qual_rate = (
                    lp_cfg["base_qual"]
                    * QUAL_INTENT_MULT[intent]
                    * (1.0 + 0.10 * t)
                )
                qual_rate = float(np.clip(qual_rate * _noise(0.05), 0.0, 0.95))
                qualified_leads = int(round(leads * qual_rate))

                # ---- Customers + revenue ----
                customers = int(round(qualified_leads * cfg["close_rate"] * _noise(0.08)))
                revenue = round(customers * cfg["avg_deal_value"] * _noise(0.05), 2)

                rows.append(dict(
                    week=week.date().isoformat(),
                    query_intent=intent,
                    ai_overview_present=ai_flag,
                    campaign_type=campaign_type,
                    landing_page_type=lp,
                    impressions=impressions,
                    clicks=clicks,
                    cost=cost,
                    leads=leads,
                    qualified_leads=qualified_leads,
                    customers=customers,
                    revenue=revenue,
                ))

    return pd.DataFrame(rows)


def _zero_row(week, intent, ai_flag, campaign_type, lp) -> dict:
    """A real, kept row with zero activity (e.g. PMax before launch)."""
    return dict(
        week=week.date().isoformat(),
        query_intent=intent,
        ai_overview_present=ai_flag,
        campaign_type=campaign_type,
        landing_page_type=lp,
        impressions=0,
        clicks=0,
        cost=0.0,
        leads=0,
        qualified_leads=0,
        customers=0,
        revenue=0.0,
    )


def main() -> None:
    df = generate()
    out_path = Path(__file__).resolve().parent / "marketing_data.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df):,} rows to {out_path}")
    print(f"Weeks: {df['week'].nunique()} | "
          f"Campaigns: {df.groupby(['query_intent','campaign_type','landing_page_type']).ngroups}")
    print(f"Total leads: {df['leads'].sum():,} | "
          f"Total qualified: {df['qualified_leads'].sum():,} | "
          f"Total revenue: ${df['revenue'].sum():,.0f}")


if __name__ == "__main__":
    main()
