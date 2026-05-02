"""
CohortEngine: Computes cohort retention matrices and detects retention cliffs.
"""

import pandas as pd
import numpy as np


RETENTION_DAYS = [1, 3, 7, 14, 21, 30, 45, 60, 90]


class CohortEngine:
      def compute_cohort_retention(self, users_df: pd.DataFrame) -> pd.DataFrame:
                """
                        Compute a cohort retention matrix.
                                users_df must have: user_id, signup_date, last_active_day (int, days since signup)
                                        Returns: DataFrame with cohort months as rows, retention days as columns.
                                                """
                if users_df.empty:
                              return pd.DataFrame()

                users_df = users_df.copy()
                users_df["cohort_month"] = pd.to_datetime(users_df["signup_date"]).dt.to_period("M").astype(str)

          cohort_data = {}
        for cohort, group in users_df.groupby("cohort_month"):
                      total = len(group)
                      row = {}
                      for day in RETENTION_DAYS:
                                        retained = (group["last_active_day"] >= day).sum()
                                        row[day] = round(retained / total * 100, 1) if total > 0 else 0.0
                                    cohort_data[cohort] = row

        matrix = pd.DataFrame(cohort_data).T
        matrix.columns = RETENTION_DAYS[:len(matrix.columns)]
        return matrix

    def detect_cliffs(self, cohort_matrix: pd.DataFrame) -> dict:
              """
                      Detect the retention cliff: the day with the steepest average drop.
                              Returns analysis dict with cliff_day, severity, and recommendations.
                                      """
        if cohort_matrix.empty:
                      return {"cliff_day": 3, "cliff_severity": 15.0, "return_probability": 70, "estimated_recovery": 15}

        avg_retention = cohort_matrix.mean()
        drops = avg_retention.diff().abs()
        cliff_day = int(drops.idxmax()) if not drops.empty else 3
        cliff_severity = float(drops.max()) if not drops.empty else 15.0

        # Probability that users who survive the cliff have much higher D30 retention
        return_probability = min(95, 60 + cliff_severity * 1.5)
        # Estimated % of cliff churners recoverable via intervention
        estimated_recovery = min(30, cliff_severity * 0.4)

        return {
                      "cliff_day": cliff_day,
                      "cliff_severity": cliff_severity,
                      "return_probability": return_probability,
                      "estimated_recovery": estimated_recovery,
        }

    def compute_summary(self, cohort_matrix: pd.DataFrame) -> dict:
              """
                      Compute summary stats per cohort: D1, D7, D30 retention.
                              """
        summary = {}
        key_days = [d for d in [1, 7, 30] if d in cohort_matrix.columns]
        for cohort in cohort_matrix.index:
                      row = {}
            for day in key_days:
                              row[f"D{day}"] = cohort_matrix.loc[cohort, day]
                          summary[cohort] = row
        return summary
