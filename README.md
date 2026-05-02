# Retention Cohort Engine 📈 — Deep Cohort Analysis & LTV Prediction

Don't just track retention — understand it. Build D1/D7/D30/D90 cohort curves, predict LTV with ML, identify retention cliff patterns, and get PM-ready intervention recommendations.

## 🚀 Product Overview

### The Problem
Retention is the most important metric for product health — but most PMs look at it wrong. Aggregate retention rates hide the truth: which cohorts are healthy, which are dying, and whether your latest product changes actually improved retention or just changed the mix of users.

Most dashboards show you a single number ("30-day retention: 34%"). What you actually need is cohort-level analysis: How does the Feb cohort compare to Jan? Which day in the first week is the critical cliff? Which user segments have 3x higher retention, and why?

### The Solution
Retention Cohort Engine generates synthetic (or real) user cohort data, computes D1/D7/D14/D30/D60/D90 retention curves, identifies retention cliff days, predicts LTV per cohort using ML regression, and generates PM-ready intervention recommendations — all in an interactive Streamlit dashboard.

### The Impact
- 📊 **Cohort heatmap** — visualize retention by signup month and day-since-signup
- - 📉 **Cliff detection** — automatically identifies the day where retention drops fastest
  - - 🤖 **LTV prediction** — ML model predicts 90-day LTV from early behavioral signals
    - - 🔍 **Segment comparison** — compare retention across user segments (plan type, source, region)
      - - 💡 **Intervention recommendations** — tells PMs exactly when and how to intervene
       
        - ---

        ## 🎯 Why This Matters (Product Perspective)

        The average SaaS company focuses on D30 retention but misses that **most churn happens in the first 7 days**, and **the critical intervention window is D3–D5** — before the habit is formed or broken.

        This tool demonstrates that I think about retention at the cohort level — comparing across time periods, identifying which product changes moved the needle, and understanding that "average retention" is meaningless without segment breakdown.

        **The key insight:** Retention is not a number. It's a curve, and the shape of that curve tells you everything about your product's health and where to focus.

        ---

        ## 🧠 AI/ML Explanation

        | Component | Technique | Why It Was Chosen |
        |---|---|---|
        | Cohort Retention Curves | Rolling cohort analysis (pandas) | Industry-standard retention calculation — tracks users by signup date |
        | Cliff Detection | First derivative + threshold detection | Identifies the specific day where the retention curve drops most steeply |
        | LTV Prediction | Gradient Boosting Regressor | Predicts 90-day LTV from D1/D7 behavioral signals — proven approach in growth modeling |
        | Feature Engineering | Early behavioral signals (D1 actions, onboarding, referrals) | D1 behavior is the strongest predictor of long-term retention |
        | Segment Analysis | GroupBy aggregation + statistical comparison | Identifies which user segments have significantly different retention profiles |
        | Retention Benchmarking | Industry percentile comparison | Contextualizes your retention vs. SaaS benchmarks (D30: 20% = median, 40% = top quartile) |

        **The Retention Cliff Model:**
        ```
        cliff_day = argmax(d(retention)/dt)  # Day with steepest slope
        cliff_severity = retention[cliff_day-1] - retention[cliff_day]  # % drop at cliff
        ```

        **LTV Prediction Features:**
        - D1 retention (did user return day after signup?)
        - - Actions in first session (depth of first engagement)
          - - Onboarding completion (binary)
            - - Referral source (organic vs. paid)
              - - Plan type at signup
                - - Team size (for B2B)
                 
                  - ---

                  ## 🛠 Tech Stack

                  | Layer | Technology |
                  |---|---|
                  | UI | Streamlit |
                  | Data Generation | NumPy, Pandas (synthetic cohort data) |
                  | ML Model | scikit-learn (GradientBoostingRegressor) |
                  | Visualization | Plotly (heatmaps, line charts, scatter plots) |
                  | Statistical Analysis | SciPy (cohort comparison tests) |
                  | Language | Python 3.8+ |

                  ---

                  ## 📊 Sample Output

                  **Cohort Retention Heatmap (6 months, D1–D30):**

                  | Cohort | D1 | D3 | D7 | D14 | D30 |
                  |---|---|---|---|---|---|
                  | Jan 2026 | 61% | 42% | 38% | 28% | 19% |
                  | Feb 2026 | 65% | 47% | 41% | 32% | 22% |
                  | Mar 2026 | 62% | 44% | 39% | 29% | 21% |
                  | Apr 2026 | 68% | 51% | **46%** | 37% | **26%** |

                  **Observation:** Apr cohort shows +7pp improvement in D7 retention vs. Jan baseline. Correlates with onboarding redesign shipped in late March.

                  **Cliff Detection:**
                  - Cliff day: **Day 3** (steepest drop in all cohorts)
                  - - Cliff severity: avg **-18pp** drop from D1 → D3
                    - - Recommendation: **Critical intervention window is D2–D3**. Users who don't return by D3 have 78% probability of never returning.
                     
                      - **LTV Prediction (GBM model, R² = 0.84):**
                     
                      - | User Segment | Predicted 90-day LTV | Confidence |
                      - |---|---|---|
                      - | Organic + Completed Onboarding | $127 | High |
                      - | Paid Acquisition + Onboarding Skipped | $31 | High |
                      - | Referral + Team Size >5 | $218 | Medium |
                      - | Free trial → No D3 return | $8 | High |
                     
                      - **Top LTV Predictors (Feature Importance):**
                      - 1. D3 retention (0.34 importance)
                        2. 2. Onboarding completion (0.28)
                           3. 3. Actions in first session (0.19)
                              4. 4. Referral source (0.12)
                                 5. 5. Plan type (0.07)
                                   
                                    6. ---
                                   
                                    7. ## 📸 Demo Instructions
                                   
                                    8. ```bash
                                       # 1. Clone the repo
                                       git clone https://github.com/Poojaahegde/retention-cohort-engine.git
                                       cd retention-cohort-engine

                                       # 2. Install dependencies
                                       pip install -r requirements.txt

                                       # 3. Launch the dashboard
                                       streamlit run app.py
                                       ```

                                       Open `http://localhost:8501` in your browser.

                                       The dashboard auto-generates 6 months of synthetic cohort data across 500+ users. Use the sidebar to:
                                       - Adjust cohort window (3–12 months)
                                       - - Filter by user segment (plan type, source, region)
                                         - - Configure LTV prediction features
                                           - - Set retention benchmark targets
                                            
                                             - ---

                                             ## 🎯 Product Thinking Layer

                                             ### 👥 Target Users
                                             - **Product Managers** at subscription-based products tracking monthly cohort health
                                             - - **Growth PMs** identifying which cohort changes correlated with product/marketing initiatives
                                               - - **Startup founders** understanding whether their retention is improving or degrading over time
                                                
                                                 - ### 😣 Pain Points Solved
                                                 - - **Aggregate retention hides cohort trends** — "34% D30 retention" tells you nothing about whether it's getting better or worse, or which users are different
                                                   - - **No cliff detection** — PMs don't know which specific day in the first week is the critical moment to intervene
                                                     - - **LTV guessing** — teams estimate LTV anecdotally; ML prediction from early signals provides an evidence-based number
                                                       - - **No segment comparison** — understanding that paid-acquisition users have 2x lower retention than organic users is critical for CAC decisions
                                                        
                                                         - ### 🧩 Key Product Decisions Made
                                                        
                                                         - **Cohort-based (not aggregate) retention as default:** Aggregate retention is a lagging indicator. Cohort retention shows you whether you're improving, and when. This is how Netflix, Duolingo, and Slack actually measure retention.
                                                        
                                                         - **GBM over linear regression for LTV:** Retention prediction has non-linear relationships (e.g., users who skip onboarding AND come from paid channels have dramatically lower LTV — the interaction term matters). GBM captures this; linear regression misses it.
                                                        
                                                         - **Cliff detection as a core feature:** Most retention charts show the curve. Almost none surface the specific day that matters most. This is the insight that drives intervention design.
                                                        
                                                         - **Synthetic data by default:** A retention tool requires months of data to be meaningful. Synthetic data lets PMs experience the full analysis without needing production data — the portfolio value is in the methodology, not the data.
                                                        
                                                         - ### 🗺 Future Roadmap
                                                        
                                                         - | Priority | Feature | Expected Impact |
                                                         - |---|---|---|
                                                         - | P0 | CSV/database connection for real cohort data | Transform from demo to production tool |
                                                         - | P0 | Cohort comparison: pre/post product change | Measure whether a specific release improved retention |
                                                         - | P1 | Survival analysis (Kaplan-Meier curves) | Model time-to-churn probability distributions |
                                                         - | P1 | Intervention simulator — "if we add D3 email, retention improves by X%" | Connect insight to action |
                                                         - | P2 | Amplitude/Mixpanel API integration | Live cohort data ingestion |
                                                         - | P2 | Cohort anomaly detection — alert when a cohort underperforms baseline | Proactive PM monitoring |
                                                         - | P3 | Revenue retention (net dollar retention) | Track expansion revenue within cohorts |
                                                        
                                                         - ---

                                                         ## 📁 Project Structure

                                                         ```
                                                         retention-cohort-engine/
                                                         ├── app.py              # Main Streamlit dashboard
                                                         ├── cohort_engine.py    # Cohort retention calculation + cliff detection
                                                         ├── ltv_predictor.py    # GBM LTV prediction model
                                                         ├── data_generator.py   # Synthetic user cohort data generator
                                                         ├── requirements.txt    # Python dependencies
                                                         └── README.md           # This file
                                                         ```

                                                         ---

                                                         ## 🔗 Related Projects in This Portfolio
                                                         - [Churn Prediction Dashboard](https://github.com/Poojaahegde/churn-prediction-dashboard) — ML-powered churn prediction with explainable AI
                                                         - - [Product Metrics Dashboard](https://github.com/Poojaahegde/product-metrics-dashboard) — DAU, retention & conversion tracker
                                                           - - [A/B Test Analyzer](https://github.com/Poojaahegde/ab-test-analyzer) — Statistical significance for PM experiments
                                                            
                                                             - ---

                                                             *Built as part of an AI PM portfolio — demonstrating that retention analysis requires cohort thinking, cliff detection, and predictive modeling, not just aggregate metrics.*
