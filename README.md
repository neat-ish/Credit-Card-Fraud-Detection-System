# Credit-Card-Fraud-Detection-System

End-to-end ML pipeline for credit card fraud detection on the IEEE-CIS Fraud dataset. Covers exploratory analysis, feature engineering, model training with Bayesian hyperparameter tuning, and a threshold A/B test with statistical and practical significance evaluation. Champion model: XGBoost (Tuned), selected via structured A/B test.

---

## Business Context

Credit card fraud causes billions of dollars in losses annually for financial institutions. Traditional rule-based fraud systems flag transactions based on fixed thresholds and known patterns, but struggle to adapt to evolving fraud behaviour. This project builds a supervised 
ML pipeline on the IEEE-CIS dataset (590K real-world transactions, 3.5% fraud rate) to score each transaction with a fraud probability and apply a business-cost-optimised decision threshold that minimises total operational cost -- balancing missed fraud losses against the friction cost 
of wrongly blocking legitimate customers. The pipeline is designed to be integrated into a batch scoring workflow where transaction features are engineered post-authorisation, scored against the champion model, and flagged for review by a fraud operations team. Model selection and threshold deployment decisions are governed by a structured A/B test with both statistical and practical significance checks, ensuring changes to the production system are data-driven and operationally justified.

## Architecture

```
+--------------------+
|    00_setup        |
|                    |
|  Kaggle API auth   |
|  Download +        |
|  extract IEEE-CIS  |
|  dataset to        |
|  data/raw/         |
+--------------------+
         |
         v
+--------------------+
|    01_eda          |
|                    |
|  Fraud rate:       |
|  ~3.5% (1:28)      |
|  434 raw features  |
|  Missing value     |
|  audit             |
|  Correlation       |
|  analysis          |
|  Temporal          |
|  stability check   |
+--------------------+
         |
         v
+--------------------+
|  02_feature_       |
|  engineering       |
|                    |
|  Drop >90%         |
|  missing (35 cols) |
|  Time features (6) |
|  Amount features   |
|  Card combos       |
|  Email flags       |
|  Velocity features |
|  Target encoding   |
|  Label encoding    |
|  70/15/15 split    |
|  (time-based)      |
+--------------------+
         |
         v
+--------------------+      +--------------------+
|  03_modelling_     |      |  models/           |
|  and_evaluation    | ---> |                    |
|                    |      |  lr_model.pkl      |
|  Baselines:        |      |  rf_model.pkl      |
|  - Logistic Reg    |      |  xgb_default.pkl   |
|  - Random Forest   |      |  xgb_tuned.pkl     |
|                    |      |  lgbm_default.pkl  |
|  Advanced:         |      |  lgbm_tuned.pkl    |
|  - XGBoost         |      |  champion_meta     |
|  - LightGBM        |      |  .json             |
|                    |      +--------------------+
|  Optuna tuning     |
|  (50 trials each)  |
|  Objective: PR-AUC |
|                    |
|  Threshold sweep:  |
|  F1-optimal +      |
|  cost-optimal      |
|  thresholds saved  |
+--------------------+
         |
         v
+--------------------+
|  04_ab_testing     |
|                    |
|  Control:   t=0.50 |
|  Treatment: t=cost-|
|  optimal           |
|                    |
|  Statistical:      |
|  - Z-test recall   |
|  - Z-test FPR      |
|  - Power + MDE     |
|                    |
|  Practical:        |
|  - Cohen's h       |
|  - Cost delta ($)  |
|                    |
|  Verdict:          |
|  DEPLOY /          |
|  INCONCLUSIVE /    |
|  RETAIN CONTROL    |
+--------------------+
```

---

## Dataset

**Source:** [IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection) -- Kaggle

| Property | Value |
|---|---|
| Training transactions | 590,540 |
| Raw features | 434 |
| Fraud rate | ~3.5% |
| Class imbalance ratio | ~1:28 |
| Train / Val / Test split | 70 / 15 / 15 |

Two files merged on `TransactionID`: transaction data (amounts, card info, email domains, anonymised V-columns) and identity data (device, browser, network info).

---

## Notebook Summaries

### 00_setup.ipynb -- Environment and Data Download

Authenticates against the Kaggle API, downloads the IEEE-CIS competition zip (~118MB) to `data/raw/`, and extracts the four CSVs. Also creates the `data/processed/` directory used by downstream notebooks.

### 01_eda.ipynb -- Exploratory Data Analysis

**Goal:** Understand the data before touching features or models.

Key findings that directly shaped downstream decisions:

- **Class imbalance (1:28):** Established that accuracy is a meaningless metric. All modelling uses PR-AUC as the primary objective and `scale_pos_weight` to handle imbalance.
- **Temporal stability check:** Fraud rate over time shows low variance, validating a time-based train/val/test split. A random split would leak future patterns into training.
- **Missing value audit:** 35 features have >90% missing values. These are flagged for removal before any modelling.
- **Correlation analysis:** Top-35 features correlated with `isFraud` identified -- primarily V-columns (Vesta engineered features), card features, and transaction amount. This informed feature priority during engineering.
- **Transaction amount:** Fraud transactions show a different distribution from legitimate ones. Log transformation and decimal extraction were identified as useful transformations.
- **Categorical features:** `ProductCD`, `card4`, `card6`, and email domains show meaningful variation in fraud rate by category -- justifying their inclusion and specific encoding strategies.

### 02_feature_engineering.ipynb -- Feature Engineering and Splits

**Goal:** Transform raw data into a modelling-ready feature set without leaking future information.

**Key decisions:**

- **Dropped >90% missing (35 columns):** Retaining these would require heavy imputation with very little signal. Dropped before any further processing.
- **Time-based 70/15/15 split (not random):** Transactions are sorted chronologically and split at fixed time boundaries. This simulates real deployment where the model scores future data it has never seen. A random split would allow future transactions to appear in training.
- **Target encoding with smoothing (factor=10):** High-cardinality columns (card1, card2, card3, card5, email domains, card combinations) are target-encoded rather than one-hot encoded. Smoothing prevents overfitting to rare categories that appear only a handful of times in training.
- **D-columns filled with -1:** D-columns represent time deltas. Missing values indicate the feature was not applicable (e.g. first transaction on a card), not that the value is zero. -1 distinguishes these semantically from actual zero-delta values.
- **C-columns filled with 0:** C-columns are counts. A missing count is logically zero.
- **V-columns filled with median:** Vesta's anonymised features have no known semantics, so median imputation is the safest default.
- **Velocity features:** Transaction count per card, time since last transaction, average amount per card, and deviation from card average. These capture behavioural signals that individual transaction features cannot.

Features created:

| Category | Features |
|---|---|
| Time | Transaction_hour, Transaction_day, Transaction_weekday, Time_of_day, Is_business_hours, Is_weekend |
| Amount | TransactionAmt_log, TransactionAmt_decimal, Is_round_amount, Amount_bin |
| Card combinations | card1_card2, card1_card5, card3_card5, card4_card6 |
| Email | Email_domain_match, P_email_is_common, R_email_is_common |
| Velocity | card1_transaction_count, time_since_last_transaction, card1_avg_amount, amount_deviation, amount_deviation_ratio |

### 03_modelling_and_evaluation.ipynb -- Modelling, Tuning, and Evaluation

**Goal:** Train and select the best model, then find the optimal operating threshold.

**Model selection rationale:**

Four models are trained in a deliberate progression from simple to complex:

- **Logistic Regression and Random Forest** serve as baselines to establish a performance floor. If a complex model does not significantly beat them, added complexity is not justified.
- **XGBoost and LightGBM** are the primary candidates. Both are gradient boosted tree ensembles well-suited to tabular data with mixed feature types and class imbalance. LightGBM uses a leaf-wise (best-first) growth strategy versus XGBoost's depth-wise approach, which tends to make it faster and more accurate on large tabular datasets. Both are compared to let the data decide.

**Why PR-AUC as the tuning objective (not ROC-AUC):**

With a 1:28 class imbalance, ROC-AUC can appear high even for a model that performs poorly on the minority (fraud) class, because it is dominated by the large legitimate class. PR-AUC focuses specifically on the model's behaviour for positive (fraud) predictions and is more informative under imbalance. All 50 Optuna trials for both models optimise PR-AUC on the validation set.

**Hyperparameter tuning:**

Optuna with MedianPruner (10 startup trials, 20 warmup steps) runs 50 trials per model. The pruner cuts unpromising trials early, concentrating budget on the search regions that have produced good results. Nine hyperparameters are tuned for XGBoost and nine for LightGBM, including learning rate (log scale), tree depth, regularisation, and subsampling ratios.

**Threshold optimisation:**

A single probability threshold is not appropriate for all operating conditions. Two thresholds are derived from a full sweep (0.05 to 0.95):

- **F1-optimal threshold:** Maximises F1-score on the validation set. Balances precision and recall equally.
- **Cost-optimal threshold:** Minimises total expected dollar cost using $120 per missed fraud (FN) and $5 per false block (FP). This threshold is lower than F1-optimal because the asymmetric costs ($120 vs $5) make missing fraud much more expensive than a false alarm.

Both thresholds are stored in `models/champion_meta.json` and passed to the A/B test.

### 04_ab_testing.ipynb -- Threshold A/B Test

**Goal:** Determine whether the cost-optimal threshold materially outperforms the naive default (0.50), with statistical rigour.

**Why A/B test a threshold (not two models):**

The model is fixed (XGBoost Tuned). The A/B test isolates the effect of the threshold decision alone. Control uses 0.50 (the naive default). Treatment uses the cost-optimal threshold from notebook 03. Using the same model and same test set eliminates model variance as a confound -- any metric difference is attributable solely to the threshold.

**Statistical tests:**

| Test | What it measures |
|---|---|
| Two-proportion Z-test on recall | Is the fraud detection rate difference larger than sampling noise? |
| Two-proportion Z-test on FPR | Is the false alarm rate difference larger than sampling noise? |
| Achieved power + MDE | Is the test adequately powered to detect the observed effect? |

**Practical significance:**

Statistical significance alone is not sufficient for a deployment decision. Cohen's h quantifies the effect size of the recall difference. The business cost delta converts the metric difference into a dollar figure. Both must clear their thresholds for a Deploy verdict.

**Decision logic:**

```
Criteria evaluated (3 total):
  1. Z-test on recall: p < 0.05
  2. Z-test on FPR:    p < 0.05
  3. Cohen's h on recall: |h| >= 0.05 (non-negligible)

n_criteria met:
  3/3  ->  Deploy: threshold change is statistically and practically justified
  2/3  ->  Inconclusive: gather more data or run live traffic experiment
  0-1  ->  Retain Control: treatment shows no meaningful improvement
```

---

## Key Design Decisions Summary

| Decision | Rationale |
|---|---|
| Time-based split (not random) | Prevents future data leaking into training; mirrors production deployment |
| PR-AUC as tuning objective | ROC-AUC is misleading under 1:28 imbalance; PR-AUC focuses on the minority class |
| Smoothed target encoding | Prevents rare-category overfitting in high-cardinality columns |
| Two thresholds (F1 + cost) | Different operating conditions require different tradeoffs |
| A/B test on threshold, not model | Isolates the threshold effect; model variance is not a confound |
| Velocity features (card-level) | Individual transaction features cannot capture behavioural patterns across time |

---

## Results

| Model | ROC-AUC | PR-AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|
| Logistic Regression | 0.7683 | 0.1773 | 0.0756 | 0.6831 | 0.1362 |
| Random Forest | 0.9104 | 0.5250 | 0.2240 | 0.7459 | 0.3445 |
| XGBoost (Default) | 0.9476 | 0.6460 | 0.2932 | 0.7867 | 0.4272 |
| XGBoost (Tuned) | 0.9502 | 0.7080 | 0.8057 | 0.5710 | 0.6683 |
| LightGBM (Default) | 0.8921 | 0.3761 | 0.0000 | 0.0000 | 0.0000 |
| LightGBM (Tuned) | 0.9391 | 0.6865 | 0.7720 | 0.5631 | 0.6512 |


**Champion:** XGBoost (Tuned) -- selected by PR-AUC on validation set, confirmed by A/B test in notebook 04.

---

## Project Structure

```
Credit-Card-Fraud-Detection-System/
|
|-- data/
|   |-- raw/                         # IEEE-CIS CSVs (gitignored)
|   |-- processed/                   # Engineered splits (gitignored)
|
|-- models/
|   |-- lr_model.pkl
|   |-- rf_model.pkl
|   |-- xgb_default.pkl
|   |-- xgb_tuned.pkl
|   |-- lgbm_default.pkl
|   |-- lgbm_tuned.pkl
|   |-- champion_meta.json
|
|-- 00_setup.ipynb
|-- 01_eda.ipynb
|-- 02_feature_engineering.ipynb
|-- 03_modelling_and_evaluation.ipynb
|-- 04_ab_testing.ipynb
|-- MONITORING.md
|-- requirements.txt
|-- .gitignore
|-- README.md
```

---

## Setup

```bash
git clone https://github.com/neat-ish/Credit-Card-Fraud-Detection-System.git
cd Credit-Card-Fraud-Detection-System
pip install -r requirements.txt
```

Run notebooks in order: 00 -> 01 -> 02 -> 03 -> 04. Kaggle API credentials required for notebook 00.

**requirements.txt**
```
pandas
numpy
matplotlib
seaborn
scikit-learn
lightgbm
xgboost
optuna
joblib
statsmodels
scipy
kaggle
```

---

## Monitoring

See `MONITORING.md` for the full monitoring guide: business KPIs, PSI-based drift detection, retraining criteria, and deployment notes.

---

*Nitish Patnaik | github.com/neat-ish*
