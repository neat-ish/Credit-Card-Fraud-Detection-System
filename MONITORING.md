# Model Monitoring & Deployment Guide

**Project:** Credit-Card-Fraud-Detection-System
**Dataset:** IEEE-CIS Fraud Detection (Kaggle)
**Champion Model:** XGBoost (Tuned) -- selected via A/B test in notebook 04

**Deployment context:** Batch fraud scoring pipeline -- transaction-level binary classification for real-time fraud flag generation

This pipeline is designed to score incoming payment transactions against a trained LightGBM model and apply a business-cost-optimised decision threshold to flag suspected fraud. The model runs in batch (hourly or daily), consuming feature-engineered transaction data and producing fraud probability scores that feed into a downstream rule engine.

The model does not serve real-time per-transaction predictions. Scores are written to a results table post-batch, and flags are applied by a threshold lookup against `champion_meta.json`. Monitoring runs on a confirmed-label window with a ~30-day lag, reflecting the time required for chargeback resolution to establish ground truth.

---

## 1. Introduction & Context

Credit card fraud causes billions of dollars in losses annually for financial institutions. Traditional rule-based fraud systems flag transactions based on fixed thresholds and known patterns, but struggle to adapt to evolving fraud behaviour. This project builds a supervised 
ML pipeline on the IEEE-CIS dataset (590K real-world transactions, 3.5% fraud rate) to score each transaction with a fraud probability and apply a business-cost-optimised decision threshold that minimises total operational cost -- balancing missed fraud losses against the friction cost of wrongly blocking legitimate customers. The pipeline is designed to be integrated into a batch scoring workflow where transaction features are engineered post-authorisation, scored against the champion model, and flagged for review by a fraud operations team. Model selection and threshold deployment decisions are governed by a structured A/B test with both statistical and practical significance checks, ensuring changes to the production system are data-driven and operationally justified.

## 2. Business KPIs

### 2.1 Primary KPIs

| KPI | Definition | Target |
|---|---|---|
| Fraud Detection Rate (FDR) | TP / (TP + FN) -- fraction of actual fraud caught | > 85% |
| False Positive Rate (FPR) | FP / (FP + TN) -- legitimate transactions wrongly blocked | < 3% |
| Precision | TP / (TP + FP) -- of all flagged, how many are actual fraud | > 70% |
| Total Business Cost ($) | (FN x $120) + (FP x $5) | Minimised -- see cost sweep in notebook 03 |
| Net Fraud Loss Avoided ($) | Baseline loss (no model) minus model-adjusted loss | Track weekly |

Cost parameters are configurable. Current assumptions: $120 average fraud loss per missed case (FN), $5 customer friction cost per false block (FP).

### 2.2 Operational KPIs

| KPI | Definition | Alert Threshold |
|---|---|---|
| Daily Fraud Flag Rate | % of transactions flagged per day | Alert if > +/- 1.5x rolling baseline |
| Score PSI | PSI on model output probabilities vs. training distribution | Alert > 0.10 / Retrain > 0.20 |
| Avg Fraud Amount Missed | Mean transaction value of FN cases | Alert on sustained upward trend |
| Dispute Resolution Rate | FP disputes upheld by customers | < 2% of total flags |

### 2.3 Monitoring Cadence

| Frequency | What to Check | Owner |
|---|---|---|
| Daily | FDR, FPR, flag volume, score PSI | Data Science / Fraud Ops |
| Weekly | Business cost delta, PR-AUC trend | Data Science |
| Monthly | Feature PSI (all features), full model eval on labelled window | Data Science |
| Quarterly | Threshold recalibration, retraining decision | Data Science + Product |

---

## 3. Model Monitoring

### 3.1 Score Distribution Drift (PSI)

PSI is computed on model output probabilities and on all input features monthly, comparing the current period against the training distribution.

| PSI | Interpretation | Action |
|---|---|---|
| < 0.10 | Stable | No action |
| 0.10 - 0.20 | Moderate shift | Monitor closely; cross-reference with feature importance |
| > 0.20 | Significant shift | Plan retrain if a high-importance feature is affected |

A PSI > 0.20 on a top-10 feature by importance is treated as a hard retrain trigger. PSI > 0.20 on low-importance features is lower urgency.

### 3.2 Model Performance Monitoring

Evaluate on a labelled holdout window -- transactions with confirmed fraud outcomes, typically available with a 30-day lag due to chargeback resolution.

- PR-AUC on a rolling 30-day window -- alert if drop > 5% from deployment baseline
- ROC-AUC on the same window -- secondary signal
- F1-score at the operational threshold -- alert if drop > 0.03 absolute
- Confusion matrix delta -- watch for asymmetric degradation (FDR dropping while FPR holds, or vice versa)

PR-AUC is the primary tracking metric because the dataset is highly imbalanced (~3.5% fraud). ROC-AUC is retained as a secondary check only.

### 3.3 Decision Thresholds

Two thresholds are maintained and stored in `models/champion_meta.json`:

| Threshold | Optimised For | Review Cadence |
|---|---|---|
| F1-optimal | Balanced precision-recall on validation set | Quarterly or on distribution shift |
| Cost-optimal | Minimise total dollar cost: (FN x $120) + (FP x $5) | Quarterly or on fraud pattern change |

The cost-optimal threshold is the one used in the A/B test (notebook 04) as the treatment arm.

### 3.4 Retraining Criteria

Retrain when any two of the following are true simultaneously:

- Score PSI > 0.20
- PR-AUC drops > 5% from the deployment baseline on a confirmed-label window
- FDR drops below 80% on the same window
- A material population event occurs (new card product, geography expansion)

---

## 4. Deployment Notes

This section is a lightweight handoff reference. The pipeline is batch-only by design.

### 4.1 Serving Pattern

- **Batch scoring:** Run XGBoost on all transactions from the prior period (hourly or daily). Write fraud probability scores to a results table.
- **Threshold application:** A downstream rule engine reads the scores and applies the operational threshold from `champion_meta.json` to generate binary flags.
- **Real-time scoring:** Not supported by this pipeline. If sub-second per-transaction scoring is needed, a separate lightweight model (e.g. doubly robust or DML) should be trained on the same feature set and served as a REST endpoint.

### 4.2 Artifacts for Versioning

Each deployment should store:

- `lgbm_tuned.pkl` -- champion model
- `champion_meta.json` -- thresholds, validation and test metrics, training date
- `feature_names.txt` -- exact ordered feature list from notebook 02
- A/B test decision summary from notebook 04


### 4.3 Dependency Pinning

The scoring environment must match training exactly.

| Package | Risk if unpinned |
|---|---|
| lightgbm | Score values can shift across minor versions |
| xgboost | Same |
| scikit-learn | LabelEncoder output may differ across versions |
| pandas / numpy | Data type defaults can change |

---

*Nitish Patnaik | github.com/neat-ish*