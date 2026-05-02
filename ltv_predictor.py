"""
LTVPredictor: Predicts 90-day LTV from early behavioral signals using Gradient Boosting.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


FEATURE_COLS = ["d1_returned", "d3_returned", "first_session_actions", "onboarding_complete",
                                "plan_type_enc", "source_enc", "team_size"]


class LTVPredictor:
      def __init__(self):
                self.model = GradientBoostingRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)
                self.plan_enc = LabelEncoder()
                self.source_enc = LabelEncoder()
                self._trained = False
                self._feature_names = FEATURE_COLS

      def _prepare_features(self, df: pd.DataFrame, fit: bool = False) -> np.ndarray:
                df = df.copy()
                if fit:
                              df["plan_type_enc"] = self.plan_enc.fit_transform(df.get("plan_type", pd.Series(["free"] * len(df))))
                              df["source_enc"] = self.source_enc.fit_transform(df.get("source", pd.Series(["organic"] * len(df))))
else:
            try:
                              df["plan_type_enc"] = self.plan_enc.transform(df.get("plan_type", pd.Series(["free"] * len(df))))
                              df["source_enc"] = self.source_enc.transform(df.get("source", pd.Series(["organic"] * len(df))))
except Exception:
                  df["plan_type_enc"] = 0
                  df["source_enc"] = 0

        features = []
        for col in FEATURE_COLS:
                      if col in df.columns:
                                        features.append(df[col].fillna(0).values)
else:
                  features.append(np.zeros(len(df)))
          return np.column_stack(features)

    def train(self, users_df: pd.DataFrame):
              """Train GBM model on user behavioral features to predict LTV."""
              if "ltv_90d" not in users_df.columns:
                            return
                        X = self._prepare_features(users_df, fit=True)
        y = users_df["ltv_90d"].values
        if len(X) > 20:
                      self.model.fit(X, y)
                      self._trained = True

    def predict(self, users_df: pd.DataFrame) -> np.ndarray:
              """Predict 90-day LTV for each user."""
        if not self._trained:
                      return np.full(len(users_df), 50.0)
                  X = self._prepare_features(users_df, fit=False)
        return np.clip(self.model.predict(X), 0, None)

    def feature_importance(self) -> dict:
              """Return feature importance dict."""
        if not self._trained:
                      return {}
                  importances = self.model.feature_importances_
        return dict(sorted(zip(self._feature_names, importances), key=lambda x: x[1]))
