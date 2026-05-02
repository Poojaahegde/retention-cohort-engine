"""
Synthetic cohort data generator for retention analysis.
Generates realistic user behavior with segment-based retention profiles.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta


PLAN_TYPES = ["free", "starter", "pro", "enterprise"]
SOURCES = ["organic", "paid_search", "referral", "social", "direct"]
REGIONS = ["US", "EU", "APAC", "LATAM"]

# Retention probabilities by segment (realistic SaaS benchmarks)
RETENTION_PROFILES = {
      "free": {"d1": 0.45, "d3": 0.30, "d7": 0.22, "d14": 0.18, "d30": 0.14},
      "starter": {"d1": 0.60, "d3": 0.42, "d7": 0.35, "d14": 0.28, "d30": 0.22},
      "pro": {"d1": 0.70, "d3": 0.55, "d7": 0.48, "d14": 0.40, "d30": 0.33},
      "enterprise": {"d1": 0.80, "d3": 0.68, "d7": 0.62, "d14": 0.55, "d30": 0.46},
}

SOURCE_MULTIPLIER = {
      "organic": 1.15,
      "referral": 1.25,
      "direct": 1.05,
      "paid_search": 0.85,
      "social": 0.80,
}


def simulate_last_active_day(plan_type: str, source: str, onboarding: bool) -> int:
      """Simulate how many days a user stays active based on their segment."""
      profile = RETENTION_PROFILES.get(plan_type, RETENTION_PROFILES["free"])
      src_mult = SOURCE_MULTIPLIER.get(source, 1.0)
      onb_mult = 1.3 if onboarding else 0.7

    # Sample from a geometric-like distribution weighted by segment
      base_retention = profile["d30"] * src_mult * onb_mult
      rand = np.random.random()

    if rand < base_retention * 3.0:
              return np.random.randint(30, 91)  # Long-term user
elif rand < base_retention * 5.5:
          return np.random.randint(14, 31)  # Medium retention
elif rand < base_retention * 8.0:
          return np.random.randint(7, 15)   # Short retention
elif rand < base_retention * 12.0:
          return np.random.randint(3, 8)    # Early drop
else:
          return np.random.randint(0, 3)    # Day 1 churn


def generate_cohort_data(n_months: int = 6, users_per_month: int = 300, seed: int = 42) -> pd.DataFrame:
      """
          Generate synthetic user cohort data.
              Each row is a user with signup_date and behavioral attributes.
                  """
      np.random.seed(seed)
      users = []
      base_date = datetime(2025, 11, 1)

    for month_idx in range(n_months):
              cohort_date = base_date + timedelta(days=month_idx * 30)
              # Slight improvement trend over time (simulating product improvements)
              trend_mult = 1.0 + month_idx * 0.02

        n_users = users_per_month + np.random.randint(-30, 30)

        for i in range(n_users):
                      plan = np.random.choice(PLAN_TYPES, p=[0.50, 0.25, 0.15, 0.10])
                      source = np.random.choice(SOURCES, p=[0.35, 0.25, 0.20, 0.12, 0.08])
                      region = np.random.choice(REGIONS, p=[0.45, 0.30, 0.15, 0.10])
                      team_size = np.random.choice([1, 2, 5, 10, 25, 50], p=[0.30, 0.25, 0.20, 0.15, 0.07, 0.03])
                      onboarding = np.random.random() < (0.55 + month_idx * 0.03)  # Improving over time
            first_session_actions = np.random.poisson(3 if onboarding else 1)

            last_active_day = simulate_last_active_day(plan, source, onboarding)

            d1_returned = int(last_active_day >= 1)
            d3_returned = int(last_active_day >= 3)

            # LTV: function of plan, retention, and team size
            plan_arpu = {"free": 0, "starter": 15, "pro": 50, "enterprise": 200}[plan]
            months_active = min(last_active_day // 30, 3)
            ltv_90d = plan_arpu * months_active * (team_size ** 0.3) + np.random.normal(0, 5)
            ltv_90d = max(0, ltv_90d)

            users.append({
                              "user_id": f"u_{month_idx:02d}_{i:04d}",
                              "signup_date": cohort_date + timedelta(days=np.random.randint(0, 28)),
                              "plan_type": plan,
                              "source": source,
                              "region": region,
                              "team_size": team_size,
                              "onboarding_complete": int(onboarding),
                              "first_session_actions": first_session_actions,
                              "d1_returned": d1_returned,
                              "d3_returned": d3_returned,
                              "last_active_day": last_active_day,
                              "ltv_90d": round(ltv_90d, 2),
            })

    return pd.DataFrame(users)
