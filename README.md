# Time-Series ICU Patient Deterioration Predictor

***Hybrid Machine Learning System for Early Warning in Critical Care***

---

## Executive Summary

**Tech stack:** *Python, PyTorch, Scikit-learn, LightGBM, pandas, NumPy*

This project implements a dual-architecture early warning system comparing gradient-boosted decision trees (LightGBM) against temporal convolutional networks (TCN) for predicting ICU patient deterioration, across three NEWS2-derived clinical-risk outcomes (maximum risk attained, average sustained risk, % time spent in high-risk state). 

Models were trained on the MIMIC-IV Clinical Demo v2.2 dataset (100 patients), using dual feature engineering pipelines: 171 timestamp-level temporal features (24-hour windows) for TCN, and 40 patient-level aggregated features for LightGBM.

The hybrid approach reveals complementary strengths: LightGBM achieves superior calibration and regression fidelity (68% Brier reduction, +17% AUC, +44% R²) for sustained risk assessment, while TCN demonstrates stronger acute event discrimination (+9.3% AUC, superior sensitivity) for detecting rapid deterioration. Together, they characterise short-term instability and longer-term exposure to physiological risk.

The complete pipeline includes clinically validated NEWS2 preprocessing (CO₂ retainer logic, GCS mapping, supplemental O₂ protocols), comprehensive feature engineering, robust evaluation, and model-specific interpretability (SHAP for LightGBM; gradient×input saliency for TCN).

A deployment-lite inference system supports batch and per-patient predictions for reproducible, end-to-end use.

| Target           | Best Model | Key Metric(s)             | Notes |
|------------------|------------|--------------------------|-------|
| Maximum Risk     | TCN        | ROC AUC: 0.923           | Strong acute detection, high sensitivity |
| Median Risk      | LightGBM   | ROC AUC: 0.972, Brier: 0.065 | Superior sustained risk calibration |
| Percentage Time High | LightGBM | R²: 0.793                | Better regression fidelity for high-risk exposure |

**Key Contributions:**
- Full clinical-validity pipeline with robust NEWS2 computation
- Dual feature engineering workflow (patient-level vs timestamp)
- Dual-model training with model-specific hyperparameter tuning
- Transparent interpretability validated against domain knowledge
- Deployment-lite inference pipeline demonstrating end-to-end usability

---

## Table of Contents
1. [Introduction](#introduction)
2. [Clinical Motivation](#clinical-motivation)
3. [Data Pipeline Overview](#data-pipeline-overview)
4. [Phase 1: CO₂ Retainer Identification & NEWS2 Tracker](#phase-1-co2-retainer-identification--news2-tracker)
5. [Phase 2: ML-Ready Feature Engineering](#phase-2-ml-ready-feature-engineering)
6. [Phase 3: LightGBM Training & Validation](#phase-3-lightgbm-training--validation)
7. [Next Steps](#next-steps)

---


## 1. Clinical Background & Motivation

### The Problem
ICU patient deterioration often emerges through subtle physiological changes hours before critical events. The National Early Warning Score 2 (NEWS2) is widely used in UK hospitals to detect and escalate care for deteriorating patients. Accurate, real-time scoring and risk stratification can:
- Enable earlier intervention and ICU escalation
- Support clinical decision-making with actionable, interpretable metrics
- Provide a foundation for advanced ML-based early warning systems

![NEWS2 API Diagram](https://developer.nhs.uk/apis/news2-1.0.0-alpha.1/images/NEWS2chart.png)

***Figure: NHS Digital, NEWS2 API guide (Open Government Licence v3.0)***  

Although the national standard for deterioration detection, NEWS2 has well-recognised constraints:
- **No temporal modelling:** Although observations are charted sequentially, the scoring algorithm treats each set of vitals independently and does not incorporate trend, slope, variability, or rate-of-change.
- **Discrete scoring limitations:** continuous physiological signals are combined into discrete scores and does not model interactions between multiple variables, limiting sensitivity to subtle multivariate deterioration patterns.
- **Escalation overload:** Threshold-based triggers produce high false-positive rates in elderly and multimorbid cohorts, contributing to alert burden and escalation fatigue.
- **Limited predictive horizon:** Escalation only occurs after thresholds are breached, restricting early-warning capability.

##
### Clinical Escalation Context
NEWS2 scoring bands map directly to clinical monitoring frequency and escalation actions; these operational consequences define the clinical targets we aim to predict:

| NEWS2 Score                      | Clinical Risk | Monitoring Frequency                                  | Clinical Response                                                                 |
|-----------------------------------|-----------------|---------------------------------------------------------------|------------------------------------------------------------------------------------|
| **0**                             | Low           | Minimum every 12h                             | Routine monitoring by registered nurse.                                            |
| **1–4**                           | Low           | Minimum every 4-6h                            | Nurse to assess need for change in monitoring or escalation.                       |
| **Score of 3 in any parameter**   | Low–Medium    | Minimum every 1h                               | Urgent review by ward-based doctor to decide monitoring/escalation.            |
| **5–6**                           | Medium        | Minimum every 1h                               | Urgent review by ward-based doctor or acute team nurse; consider critical care team review.   |
| **≥7**                            | High          | Continuous monitoring                              | Emergent assessment by clinical/critical-care team; usually transfer to HDU/ICU. |

#### Why this matters
- Transitions between these risk categories directly influence clinical workload and resource allocation, including urgent reviews and ICU involvement.  
- Predicting imminent transitions into these categories (e.g., entering high risk within the next 4–6 hours) enables earlier intervention, reducing delayed escalations and improving critical-care resource planning.  

#### Why NEWS2 is the reference standard
- NEWS2 is the national standard for ward-based clinical deterioration assessment and provides a clinically validated ground-truth for model training and evaluation. 
- The ML models predict summary outcomes derived from NEWS2 clinical-risk categories:
  - `max_risk`: Maximum risk attained during stay 
  - `median_risk`: Average sustained risk across the stay  
  - `pct_time_high`: Percentage of time spent in high-risk state  
- Evaluating ML predictions against these NEWS2-derived outcomes allows assessment of **predictive horizon**, **sensitivity**, and the ability to anticipate **clinically actionable deterioration trends** before standard escalation would occur.

##
### Why Machine Learning?
ICU deterioration involves complex and often subtle, multivariate temporal patterns that standard threshold-based systems cannot fully capture. Machine learning enables prediction of clinically meaningful NEWS2-derived outcomes using both static and temporal representations of patient physiology.

| Model | Type   | Input Features   | Modelling Type   | Strengths     | Weaknesses     | Interpretability |
|-------|------|-------------------|------------------|----------------|-----------------|---------------|
| **LightGBM** | Gradient-Boosted Decision Tree (GBDT) | Aggregated patient-level | Static | Fast, interpretable, good calibration | Cannot capture sequential dynamics | SHAP |
| **TCN** | Temporal Convolutional Network | Timestamp-level sequential | Temporal | Captures temporal trends, slopes, variability | Requires high-resolution data, slower to train | Saliency (grad×input) |

#### LightGBM (classical machine learning)
- Strong baseline for tabular clinical data
- Captures nonlinear interactions between vital signs  
- Fast to train and tune, handles missing data robustly
- Highly interpretable via SHAP  
- Often competitive or superior when temporal structure is weak  

#### TCN (temporal deep learning)
- Models long-range temporal context and time-dependent patterns
- Robust to irregular sampling  
- Potentially detects subtle deterioration earlier than threshold-based approaches  

#### Why compare both?
- LightGBM evaluates performance on aggregated, low-frequency data.
- TCN uses temporal modelling to capture complex, sequential patterns.
- Comparison reflects realistic deployment: classical ML may suffice for lower-frequency ward data, whereas temporal models exploit high-resolution ICU monitoring to detect early deterioration.
- This identifies where temporal modelling adds value, where classical ML suffices, and the trade-offs between performance and interpretability.

This project therefore systematically evaluates temporal vs. non-temporal ML approaches for predicting ICU deterioration using clinically meaningful NEWS2-derived targets.

---

## 2. Project Goals & Contributions

### Primary Objectives
1. Establish a clinically validated deterioration-prediction framework by using NEWS2-derived risk categories as the reference standard for all model targets and evaluation.
2. Compare classical vs. deep learning to quantify the added predictive value of temporal modelling by systematically comparing a static gradient-boosted decision tree model (LightGBM) against a temporal convolutional network (TCN) on the same patients and targets.
3. Design dual feature engineering pipelines (timestamp-level for TCN, patient-level for LightGBM) incorporating temporal statistics, imputation, and missingness flags; aligned with real hospital data quality constraints.
4. Evaluate model performance across 3 risk horizons to ensure fair model comparison, demonstrating how different architectures capture acute vs. long-term physiological instability.
5. Implement transparent model-specific interpretability pathways (SHAP vs. saliency) to ensure outputs remain clinically aligned and defensible, supporting clinician trust during decision-making and escalation.
6. Develop an end-to-end, deployment-lite inference system capable of running batch and per-patient predictions, enabling direct applicability to real-world ICU or ward settings.

### Key Technical Contributions
- **Dual-Model Feature Engineering:** Built a reproducible pipeline from raw EHR data, including feature extraction, NEWS2 computation (GCS→LOC mapping, CO₂ retainer and supplemental O₂ rules), and clinically interpretable timestamp- and patient-level features (LOCF, missingness flags, rolling windows, summary statistics).
- **LightGBM Baselines:** Developed robust patient-level models with binary target handling for sparse classes, stratified cross-validation, and hyperparameter tuning for stable, reproducible performance.
- **TCN Architecture:** Designed multi-task temporal convolutional network for time-series data, combining causal dilated convolutions, residual connections, and masked mean pooling.
- **Evaluation & Diagnostics:** Built unified evaluation utility for reproducible metrics; diagnosed and corrected TCN metric misalignment to restore true ROC-AUC.
- **Model Retraining & Stability:** Implemented targeted TCN retraining with class-weighted BCE and log-transform regression, preserving architecture and hyperparameters while improving convergence and predictive reliability.
- **Comprehensive Analysis Pipeline:** Executed unbiased test set inference, TCN calibration/threshold tuning, and cross-model comparative analysis for LightGBM vs. TCN.
- **Interpretability & Deployment:** Delivered global SHAP feature explanations (LightGBM) and gradient×input saliency maps (TCN), and packaged a unified, lightweight inference pipeline supporting batch and interactive per-patient predictions with deterministic, dataset-agnostic outputs.

---

## 3. Phase 1 - Data Extraction & NEWS2 Computation

### 3.1 Data Source: MIMIC-IV Demo (v2.2)
- **Overview:** Medical Information Mart for Intensive Care (MIMIC)-IV database is comprised of deidentified patient electronic health records, taken from PhysioNet.org.
- **Contents:** 26 tables of structured vital signs, labs, and admission data; excludes free-text clinical notes.
- **Patients:** 100 ICU admissions (de-identified subset)
- **Tables used:** `chartevents`, `patients`, `admissions`, `d_items`
- **Limitations:** Small sample size (full dataset contains >65,000 ICU admissions), limited high-risk events.

##
### 3.2 NEWS2 Pipeline Overview
**Goal:** Extract relevant vital signs, encode clinical rules, compute NEWS2 scores per timestamp and per patient producing reproducible outputs for feature engineering.

```text 
                          Raw MIMIC-IV CSVs
                    (chartevents, patients, etc.)
                                  │
                        extract_news2_vitals.py
                                  │                               
                                  ▼                               
          news2_vitals_with_co2.csv + co2_retainer_details.csv
                                  │
                          compute_news2.py
                                  │
                  ┌───────────────┴───────────────┐
                  ▼                               ▼
  news2_scores.csv (timestamp-level)   news2_patient_summary.csv (patient-level)
```

##
### 3.3 Clinical Feature Extraction & NEWS2 Computation

The pipeline extracts all NEWS2-relevant physiological variables using universal encoding labels, including custom CO2 retainer logic implementation. NEWS2 parameter scoring and total scoring is implemented, with both timestamp-level and patient-level files being created.

#### Core Vital Parameters
| Parameter               | Range                        | NEWS2 Points |
|-------------------------|-----------------------------|--------------|
| Respiratory Rate        | ≤8 → ≥25                    | 0–3          |
| SpO₂ (Scale 1)          | ≤91 → ≥96                   | 0–3          |
| SpO₂ (Scale 2, hypercapnic) | ≤83 → ≥97               | 0–3          |
| Supplemental O₂         | No / Yes                     | 0 / 2        |
| Temperature             | ≤35°C → ≥39.1°C             | 0–3          |
| Systolic BP             | ≤90 → ≥220                  | 0–3          |
| Heart Rate              | ≤40 → ≥131                  | 0–3          |
| Level of Consciousness  | Alert / Not Alert        | 0 / 3        |

(Discrete NHS NEWS2 bands are applied in code; the table above shows compressed min–max ranges.)

#### Clinical Logic Implementation
1. **CO₂ Retainer Identification**
```text 
All ABG measurements in `chartevents.csv` examined. Criteria (all must be met):
- PaCO₂ > 45 mmHg (chronic hypercapnia)
- pH 7.35-7.45 (compensated respiratory acidosis)  
- ABG measurements ±1 hour apart
→ Triggers SpO₂ Scale 2 (target 88-92% vs 94-98%). No patients in current dataset met criteria.
```
2. **GCS → Level of Consciousness (LOC) Mapping**
```text
Combine column scores (`GCS - Eye Opening`, `GCS - Verbal Response`, `GCS - Motor Response`) to create `gcs_total`. 
- GCS ≥15: Alert (LOC = 0)
- GCS <15: Not Alert (LOC = 3)
→ Non-alert states add +3 to NEWS2
```
3. **Supplemental O₂ Detection**
```text 
FiO₂ can be identified via `Inspired O2 Fraction` in CSV and converted to binary supplemental O₂ indicator. 
- ≤0.21 = not on supplemental O₂ (0)
- >0.21 = on supplemental O₂ (1)
→ Any supplemental O₂ adds +2 to NEWS2
```
#### Output Format
**Timestamp-level: `news2_scores.csv`**
- One row per observation timestamp
- Raw vitals, individual parameter scores, total NEWS2 score
- Supplemental O₂, CO₂ retainer, and consciousness/GCS labels.
- Escalation risk category (low/medium/high), monitoring frequency, response.

**Patient-level: `news2_patient_summary.csv`**
- **Per-patient aggregations:** min, max, mean, median, total number of timestamps
- Summary statistics per patient.

---

## 4. Phase 2 - Feature Engineering (Timestamp & Patient-Level)
**Purpose:** Transform NEWS2 timestamp-level data into ML-ready features for tree-based models (LightGBM) and Neural Networks (TCN). Handle missing data, temporal continuity, and encode risk labels while preserving clinical interpretability

```text
                              news2_scores.csv
                                       │
             ┌─────────────────────────┴─────────────────────────┐
             │                                                   │
  make_patient_features.py                           make_timestamp_features.py
             │                                                   │
             ▼                                                   ▼
  Patient-Level Feature Engineering                  Timestamp-Level Feature Engineering
  - Median, mean, min, max per vital                 - Missingness flags 
  - Imputation using patient-specific median         - Last Observation Carried Forward (LOCF)
  - % Missingness per vital                          - Carried-forward flags
  - Encode risk labels and summary target stats      - Rolling windows 1/4/24h (mean, min, max, std, slope, AUC)
      • max_risk                                     - Time since last observation (staleness)                                  
      • median_risk                                  - Encode risk labels
      • pct_time_high                                            │
             │                                                   │
             ▼                                                   ▼
  news2_features_patient.csv                        news2_features_timestamp.csv
             │                                                   │
             ▼                                                   ▼
  LightGBM Model (Classical ML)                     Temporal Convolutional Network (TCN)                         
```

- **Timestamp-level features** = richest representation, essential for sequence models / deep learning
- **Patient-level features** = distilled summaries, useful to quickly test simpler models, feature importance or quick baseline metrics.

##
### 4.1 Timestamp-Level Features (for TCN)
**Purpose:** Capture temporal dynamics for sequential modeling

#### Imputation Strategy
- **Missingness flags:** Binary indicators (1/0) for each vital parameter (1 if value was carried forward) so models can learn from missing patterns.
- **LOCF (Last Observation Carried Forward) flags:** Propagate previous valid measurement, and create a carried-forward flag (binary 1/0).

Missingness flag before carried-forward fill so that it is known which values were originally missing

#### Rolling Window Features (1/4/24h)
- **Mean:** Average value
- **Min/Max:** Range boundaries
- **Std**: Variability
- **Slope:** Linear trend coefficient
- **AUC (Area Under Curve):** Integral of value over time

Captures short-, medium-, and long-term deterioration patterns

#### Other Features
- **Staleness:** Time since last observation (staleness per vital)
- **Numeric risk encoding:** Encoded as 0 (low), 1 (medium-low), 2 (medium), 3 (high)

#### Output Format
- **Timestamp-level:** `news2_features_timestamp.csv`
- ML-ready per patient per timestamp features (ordered by `subject_id` and by `charttime`).
- Full time-series per patient allows modelling of temporal patterns, trends, dynamics
```text 
(total_timestamps, n_features) = (20,814 timestamps, 136 features)
``` 
- **136 features =**
  - Rolling windows → 5 vitals x 3 windows x 6 stats = 90
  - 8 vitals x (staleness + missingness flag + LOCF flag) = 24
  - 14 raw clinical signals + 3 derived clinical labels = 17
  - NEWS2 → NEWS2 score + risk label + monitoring frequency + response + numeric risk = 5

##
### 4.2 Patient-Level Features (for LightGBM)
**Purpose:** Aggregated risk profile for interpretable tree-based modeling

#### Feature Computation
- **Vital sign aggregations (per parameter):** Median, mean, min, max
- **Patient-specific median imputation**: Fill missing values for each vital (preserves patient-specific patterns), if a patient never had a vital recorded, fall back to population median.
- **% Missingness per vital:** track proportion of missing values pre-imputation, data quality indicator, signal from incomplete measurement pattern.
- **Escalation summary metrics:** convert risk label to numeric then compute
  - `max_risk` = maximum risk attained
  - `median_risk`= average sustained risk
  - `pct_time_high` = % time spent in high-risk state

#### Output Format
- **Patient-level:** `news2_features_patient.csv`
- ML-ready per patient aggregated summary features (ordered by `subject_id`)
- One row per patient (fixed-length vector).
```text
(n_patients, n_features) = (100, 43)
```
- **43 features =**
  - 8 vitals x (median + mean + min + max + % missing) = 40
  - Summary features → `max_risk` + `median_risk` + `pct_time_high` = 3

---

## 5. ML Modelling Strategy

### Bimodal Architecture Rationale

### Why This Matters Clinically
- Acute detection: Requires sensitivity to short-term trend changes (TCN advantage)
- Sustained risk: Requires calibrated probability estimates over full stay (LightGBM advantage)
- Ensemble potential: Combine both signals for comprehensive early warning system
---

ML Model Selection
-	**Options considered**:
  -	Logistic Regression → easy to deploy and explainable but underpowered, tends to underperform on raw time-series vitals.
  -	Deep learning (LSTMs/Transformers) → overkill, prone to overfitting with moderate datasets.
  -	Boosted Trees (XGBoost / LightGBM / CatBoost) → robust for tabular ICU data, handle NaNs, train fast, interpretable.
-	**Decision: LightGBM (Gradient Boosted Decision Tree (GBDT) library)**
  - State-of-the-art for structured tabular data (EHR/ICU vitals is tabular + time-series).
  -	Handles missing values natively (NaNs) → no additional imputation required (simpler pipeline).
  -	Provides feature importances → interpretability for clinical review.
  -	Easy to train/evaluate quickly → allows multiple experiments.

**Model Roadmap Finalised**:
   - **V1: LightGBM (Gradient Boosted Decision Trees)**  
     - **Input**: `news2_features_patient.csv`.  
     - **Output**: news2_features_patient.csv → LightGBM → AUROC, feature importances.
     - One row per patient, interpretable, strong baseline.
     - Very interpretable for clinicians (median HR, % missing SpO₂, % time high risk).  
   - **V2: Neural Network (TCN – Temporal Convolutional Network)**  
     - **Input**: `news2_features_timestamp.csv`.  
     - **Output**: news2_features_timestamp.csv → TCN → sequence classification (predict escalation).
     - Full time-series per patient, captures sequential deterioration patterns.
     - Demonstrates modern advanced deep learning sequence modeling.  
     - Shows can move from tabular ML → time-series DL progression.
     - More impressive to interviewers / academics (future-proof).

4. **Neural Network Model Selection**:
   - **Considered**: LSTM/GRU, Transformers, TCN.  
   - **Decision: TCN** because it handles long sequences efficiently, avoids vanishing gradients, and trains faster than RNNs.  
   - **Requirements**: sequence padding, normalisation, masking for missingness. 

### Neural Network Model Selection
- **Options considered**:
  - **Recurrent Neural Networks (LSTM / GRU)** → well-suited for sequences but prone to vanishing gradients on long ICU stays, slower to train.
  - **Transformers** → powerful for long sequences, but overkill for moderate dataset size, computationally intensive.
  - **Temporal Convolutional Networks (TCN)** → convolutional sequence modeling, parallelizable, captures long-term dependencies efficiently.
- **Decision: TCN (Temporal Convolutional Network)**
  - Ideal for time-series vitals data with sequential trends.
  - Can handle long sequences without vanishing gradient issues like recurrent neural networks (RNN).
  - Parallel convolutional operations → faster training than sequential RNNs.
  - Compatible with timestamp-level features and missingness flags.
- **Preprocessing requirements**:
  - Sequence padding to unify input lengths.
  - Normalisation of continuous vitals.
  - Optional interpolation or masking for missing values.
  - One-hot encoding of categorical labels if required.
- **Strengths**:
  - Captures temporal patterns and trends across patient stays.
  - Expressive for sequence modeling where LightGBM may miss temporal dynamics.
  - Empirically outperforms LSTM/GRU for moderate-length clinical sequences.
- **Weaknesses / Limitations**:
  - More computationally intensive than tree-based models.
  - Less interpretable than LightGBM feature importances.
  - Requires careful tuning of hyperparameters (kernel size, dilation, layers).
- **Use case in pipeline**:
  - Secondary model after LightGBM to capture fine-grained temporal trends.
  - Useful for sequences where timestamp-level patterns predict escalation more accurately.

 **Future-proofing with both feature sets ensures robustness and flexibility**:
  - **LightGBM (V1)** → clinician-friendly, interpretable baseline.  
  - **TCN (V2)** → modern DL, captures dynamics.  
- **Timestamp-level features** = richest representation, essential for sequence models / deep learning
- **Patient-level features** = distilled summaries, useful to quickly test simpler models, feature importance or quick baseline metrics.
- Keeping both pipelines means we can mix (hybrid approaches) if needed (e.g., summary features + LSTM on sequences). 
- LightGBM is often deployed first because it’s fast, robust, and interpretable, while the neural network is a v2 that might improve performance. 

### Portfolio story
- **LightGBM (v1)**: We started with patient-level aggregation to establish a baseline model that is interpretable and fast to train. This gives clinicians an overview of which vitals and risk patterns matter most.
- **Neural Network (TCN)(v2)**: Once we had a solid baseline, we moved to a temporal convolutional network to directly learn time-dependent deterioration patterns from patient trajectories. This captures dynamics that aggregated features can’t.


---

## 6. Phase 3 — LightGBM Pipeline: Baseline Modelling & Hyperparameter Tuning

**Purpose:** Establish classical ML baseline, identify dataset properties, and produce stable hyperparameters for downstream evaluation.

```text
                                      news2_features_patient.csv
                                                  │
              ┌──────────────────────────┬────────┴───────────────┬───────────────────────┐
  complete_train_lightgbm.py    feature_importance.py   train_final_models.py       tune_models.py
              │                          │                        │                       │
              ▼                          ▼                        ▼                       ▼
  Preprocessing + CV Training   Feature Importance      Final Models               Hyperparameter Tuning
  (baseline models)             (per-fold)              (3 deployed LGBMs)         (only retained output)
  - Class collapse +            - Aggregated across     - Full-dataset models      - 5-fold CV grid search
  binarisation                  folds                   - Deployment-ready .pkls   - Tuning run logs per target
  - 5-fold CV (15 models)       - Bar plots + CSVs                │                - best_params.json
  - Fold metrics                       │                          │                       │
              │                        │                          │                       │
              ▼                        ▼                          ▼                       ▼
    Superseded by Phase 5     Superseded by Phase 6     Superseded by Phase 5    Retained for Phase 5
    (evaluation pipeline)     (SHAP interpretability)   (retraining)             (retraining and evaluation) 
```

### 6.1 Overview
**Pipeline Summary**
- **Class handling:** Identified severe class imbalance and rare-events carcity → applied target binarisation and stratified 5-fold CV to stabilise evaluation.
- **Baseline modelling:** Trained initial LightGBM classifiers/regressors on patient-level features (5-fold CV → 15 baseline models + metrics).
- **Hyperparameter tuning:** Per-target tuning of `learning_rate`, `max_depth`, `n_estimators`, `min_data_in_leaf` → produced `best_params.json` used in later phases.
-	**Feature importance:** Generated fold-averaged LightGBM feature importance to confirm model signal (later superseded by Phase 6 SHAP).
-	**Final models:** Trained full-cohort deployment-style final models using tuned hyperparameters (later superseded by Phase 5 retraining).

### 6.2 Baseline Cross-Validation 
- A 5-fold CV LightGBM baseline was run to validate pipeline correctness, expose dataset issues, and establish a minimal working baseline before tuning.
- **Models were trained on `news2_features_patient.csv` for 3 targets:**
  - Classification: `max_risk` + `median_risk` → 5-fold StratifiedKFold
  - Regression: `pct_time_high` → 5-fold KFold
- **Baseline runs revealed major dataset properties:**
	- Strong class imbalance and rare-event sparsity.
	- Clinically expected: severe deterioration (max_risk=3) and sustained high-risk burden are uncommon.
- **Required class consolidation + binarisation:**
	-	`max_risk`: (0/1/2 → 0 [not high risk]), (3 → 1 [high risk])
	-	`median_risk`: (0/1 → 0 [low risk]), (2 → 1 [medium risk]), (3 removed [no high risk])
	-	Binarisation improves model stability, simplifies clinical interpretation, and aligns with binary alert systems.
- StratifiedKFold was essential to ensure each fold contained minority-class examples, avoid folds with absent classes, and reduce evaluation variance; critical with only 100 patients.
- Identifying these issues early ensured consistent binarisation and stratified sampling throughout later phases, making subsequent pipelines robust and reproducible on small, imbalanced clinical datasets.

### 6.3 Hyperparameter Tuning
- This is the only Phase 3 component used in later phases (Phase 5 LightGBM evaluation).
- Tuned the four parameters with the highest impact on stability and generalisation for small tabular datasets:
	- `learning_rate` → controls step size; balances speed vs overfitting.
	- `max_depth` / `num_leaves` → limits tree complexity; prevents overfitting small dataset.
	- `n_estimators` → total number of trees.
	-	`min_data_in_leaf` → ensures each leaf has enough samples, 
-	**5-fold CV evaluation:**
	-	Classification (`max_risk`, `median_risk`) → AUROC / Accuracy
	-	Regression (`pct_time_high`) → RMSE
- **Why tuning was essential:**
	-	Only 100 patients → high overfitting risk.
	-	These parameters dominate LightGBM performance on small data.
	-	Ensures LightGBM is fairly compared against TCN models in later phases.
- Generates stable, reproducible, validated parameter sets that optimise baseline performance without overfitting, especially critical with only 100 patients.

#### Best hyperparameters

| Target         | learning_rate | max_depth | n_estimators | min_data_in_leaf |
|----------------|---------------|-----------|--------------|-------------------|
| **max_risk**       | 0.1           | 5         | 50           | 20                |
| **median_risk**    | 0.05          | 5         | 100          | 5                 |
| **pct_time_high**  | 0.1           | 3         | 200          | 5                 |


### 6.4 Feature Importance
- Fold-level LightGBM feature importance was generated to confirm that models were learning meaningful clinical signal and to verify early pipeline sanity.
- Provided an initial interpretability check and highlighted key predictors per target.
- These plots served as exploratory, legacy interpretability and were later superseded by SHAP analysis in Phase 6.

### 6.5 Deployment-style LightGBM models 
-	Trained one final model per target (3 total) on the full cohort using tuned hyperparameters.
-	Maximised use of all available data and produced deployment-ready artefacts for demonstration of real-world deployment practice.
- These models served as complete baseline artefacts, although final benchmarking used the unified evaluation pipeline in phase 5, not these models.
- These final models validate pipeline completeness and full ML workflow coverage (baseline → tuning → deployment) even if later superseded by retraining in phase 5.

### 6.6 Summary 
#### Retained Outputs

| Item                                | Rationale                                                     |
|-------------------------------------|----------------------------------------------------------------|
| **Hyperparameters (`best_params.json`)** | Used in Phase 5 LightGBM–TCN fair comparison                  |

#### Superseded Outputs

| Item                                | Replaced By / Reason                                          |
|-------------------------------------|----------------------------------------------------------------|
| Baseline 5-fold models              | Phase 5 retrained models on same split as TCN                   |
| Fold-level feature importance       | Phase 6 SHAP (model-consistent interpretability)              |
| Full-cohort final models            | Phase 5 retrained models on same split as TCN                 |

### Why Phase 3 Matters
- Phase 3 established the first credible classical ML baseline, exposed the dataset’s true behaviour, and produced the validated LightGBM hyperparameters reused in final benchmarking. 
- It delivered the two foundations that shaped all later phases:
	-	Target binarisation (after diagnosing rare-event imbalance).
	-	Stable tuned hyperparameters (ensuring fair LightGBM vs TCN comparison).
- Even though most intermediate outputs were superseded, Phase 3 remains essential because it validated the problem framing, confirmed learnable signal, and grounded the project in rigorous baseline modelling before moving to temporal architectures.

