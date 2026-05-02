import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from cohort_engine import CohortEngine
from ltv_predictor import LTVPredictor
from data_generator import generate_cohort_data

st.set_page_config(page_title="Retention Cohort Engine", page_icon="📈", layout="wide")

st.title("📈 Retention Cohort Engine")
st.markdown("**Deep Cohort Analysis & LTV Prediction** — Build D1/D7/D30/D90 curves, detect cliffs, predict LTV.")
st.markdown("---")

# Sidebar
st.sidebar.header("⚙️ Configuration")
n_months = st.sidebar.slider("Cohort window (months)", 3, 12, 6)
n_users_per_month = st.sidebar.slider("Users per month", 100, 1000, 300, step=50)
benchmark_d30 = st.sidebar.slider("D30 retention benchmark target (%)", 15, 50, 30)
show_segment = st.sidebar.selectbox("Segment by", ["None", "Plan Type", "Acquisition Source", "Region"])

st.sidebar.markdown("---")
st.sidebar.markdown("**Retention Benchmarks (SaaS):**")
st.sidebar.markdown("- D30 median: ~20%")
st.sidebar.markdown("- D30 top quartile: ~35%")
st.sidebar.markdown("- D30 best-in-class: ~50%")

# Generate data
with st.spinner("Generating cohort data..."):
      users_df = generate_cohort_data(n_months=n_months, users_per_month=n_users_per_month)

st.success(f"Loaded {len(users_df):,} users across {n_months} cohorts.")

engine = CohortEngine()
cohort_matrix = engine.compute_cohort_retention(users_df)

# Section 1: Cohort Heatmap
st.header("📊 Cohort Retention Heatmap")
fig_heat = px.imshow(
      cohort_matrix,
      labels=dict(x="Day Since Signup", y="Cohort Month", color="Retention %"),
      color_continuous_scale="Blues",
      text_auto=".0f",
      aspect="auto",
      title="Retention % by Cohort and Day"
)
fig_heat.update_layout(height=400)
st.plotly_chart(fig_heat, use_container_width=True)

# Section 2: Retention Curves
st.header("📉 Retention Curves by Cohort")
fig_lines = go.Figure()
for cohort in cohort_matrix.index:
      fig_lines.add_scatter(
                x=cohort_matrix.columns,
                y=cohort_matrix.loc[cohort],
                name=str(cohort),
                mode="lines+markers"
      )
  fig_lines.add_hline(y=benchmark_d30, line_dash="dash", line_color="red",
                                          annotation_text=f"D30 Target ({benchmark_d30}%)")
fig_lines.update_layout(title="Retention Curves", xaxis_title="Day Since Signup",
                                                yaxis_title="Retention %", height=400)
st.plotly_chart(fig_lines, use_container_width=True)

# Section 3: Cliff Detection
st.header("🚨 Retention Cliff Detection")
cliff_analysis = engine.detect_cliffs(cohort_matrix)

col1, col2, col3 = st.columns(3)
col1.metric("Critical Cliff Day", f"Day {cliff_analysis['cliff_day']}")
col2.metric("Avg Cliff Severity", f"-{cliff_analysis['cliff_severity']:.1f}pp", delta=None)
col3.metric("Intervention Window", f"Day {cliff_analysis['cliff_day']-1}–{cliff_analysis['cliff_day']+1}")

st.warning(f"🚨 **Cliff detected on Day {cliff_analysis['cliff_day']}**: Average drop of {cliff_analysis['cliff_severity']:.1f} percentage points. "
                      f"This is your highest-leverage intervention point. Users who return by Day {cliff_analysis['cliff_day']} have {cliff_analysis['return_probability']:.0f}% higher D30 retention.")

st.info(f"💡 **Recommendation:** Launch a re-engagement campaign targeting users who haven't returned by Day {cliff_analysis['cliff_day']-1}. "
                f"Email, push notification, or in-app nudge at the cliff day can recover an estimated {cliff_analysis['estimated_recovery']:.0f}% of cliff churners.")

# Section 4: Summary Metrics
st.markdown("---")
st.header("📐 Cohort Summary Metrics")
summary = engine.compute_summary(cohort_matrix)
summary_df = pd.DataFrame(summary).T
st.dataframe(summary_df.style.format("{:.1f}"), use_container_width=True)

avg_d30 = cohort_matrix.iloc[:, cohort_matrix.columns.get_loc(30) if 30 in cohort_matrix.columns else -1].mean() if not cohort_matrix.empty else 0

m1, m2, m3 = st.columns(3)
m1.metric("Avg D30 Retention", f"{avg_d30:.1f}%",
                    delta=f"{'Above' if avg_d30 > benchmark_d30 else 'Below'} target ({benchmark_d30}%)")
m2.metric("Best Cohort D30", f"{cohort_matrix.iloc[:, -1].max():.1f}%" if not cohort_matrix.empty else "N/A")
m3.metric("Retention Trend", "📈 Improving" if len(cohort_matrix) > 1 and cohort_matrix.iloc[-1, -1] > cohort_matrix.iloc[0, -1] else "📉 Declining")

# Section 5: LTV Prediction
st.markdown("---")
st.header("🤖 LTV Prediction (ML Model)")

predictor = LTVPredictor()
predictor.train(users_df)

ltv_predictions = predictor.predict(users_df)
users_with_ltv = users_df.copy()
users_with_ltv["predicted_ltv"] = ltv_predictions

st.subheader("LTV by Segment")
if show_segment != "None":
      seg_col = {"Plan Type": "plan_type", "Acquisition Source": "source", "Region": "region"}.get(show_segment, "plan_type")
      if seg_col in users_with_ltv.columns:
                seg_ltv = users_with_ltv.groupby(seg_col)["predicted_ltv"].mean().reset_index()
                fig_ltv = px.bar(seg_ltv, x=seg_col, y="predicted_ltv",
                                 title=f"Predicted 90-day LTV by {show_segment}",
                                 color="predicted_ltv", color_continuous_scale="Greens")
                st.plotly_chart(fig_ltv, use_container_width=True)
else:
      st.metric("Average Predicted 90-day LTV", f"${users_with_ltv['predicted_ltv'].mean():.2f}")
      fig_ltv_dist = px.histogram(users_with_ltv, x="predicted_ltv", nbins=30,
                                  title="LTV Distribution", labels={"predicted_ltv": "Predicted LTV ($)"})
      st.plotly_chart(fig_ltv_dist, use_container_width=True)

# Feature importance
fi = predictor.feature_importance()
if fi:
      fi_df = pd.DataFrame(list(fi.items()), columns=["Feature", "Importance"]).sort_values("Importance", ascending=True)
      fig_fi = px.bar(fi_df, x="Importance", y="Feature", orientation="h",
                      title="LTV Prediction: Feature Importance", color="Importance", color_continuous_scale="Blues")
      st.plotly_chart(fig_fi, use_container_width=True)

st.caption("Retention Cohort Engine | [GitHub](https://github.com/Poojaahegde/retention-cohort-engine)")
