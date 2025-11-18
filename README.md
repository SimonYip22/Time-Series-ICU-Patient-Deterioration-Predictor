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
2. Quantify the added predictive value of temporal modelling by systematically comparing a static gradient-boosted decision tree model (LightGBM) against a temporal convolutional network (TCN) on the same patients and targets,  highlighting the explicit classical vs. deep learning comparison.
3. Design dual feature engineering pipelines (timestamp-level for TCN, patient-level for LightGBM) incorporating temporal statistics, imputation, and missingness/carried-forward flags; aligned with real hospital data quality constraints, enabling dual-pipeline evaluation.
4. Evaluate model performance across complementary risk horizons (maximum risk, median sustained risk, percentage time in high-risk) to ensure fair, like-for-like model comparison, demonstrating how different architectures capture acute vs. long-term physiological instability.
5. Implement transparent interpretability pathways (SHAP and gradient×input saliency) to ensure outputs remain clinically defensible, aligned with established physiological understanding, supporting clinician trust during decision-making and escalation.
6. Develop an end-to-end, deployment-lite inference system capable of running batch and per-patient predictions, enabling direct applicability to real-world ICU or ward settings.

### Key Technical Contributions
- **ML Pipeline Construction:** Built a reproducible pipeline from raw EHR data, including feature extraction, NEWS2 computation (GCS→LOC mapping, CO₂ retainer identification, supplemental O₂ rules), and clinically interpretable timestamp- and patient-level features (LOCF, missingness flags, rolling windows, summary statistics).
- **LightGBM Baselines:** Developed robust patient-level models with binary target handling for sparse classes, stratified cross-validation, and hyperparameter tuning for stable, reproducible performance.
- **TCN Architecture:** Designed a multi-task temporal convolutional network for sequential ICU data, combining causal dilated convolutions, residual connections, and masked mean pooling for joint classification (max/median risk) and regression (% time high).
- **Data-Driven Optimisations:** Applied target binarisation, log-transformed regression, threshold tuning, and missingness-aware features to handle sparse, irregular clinical datasets while preserving interpretability.
- **Evaluation & Diagnostics:** Built unified evaluation utilities for reproducible, bias-free metrics; diagnosed and corrected subtle TCN metric misalignment to restore true ROC-AUC.
- **Model Retraining & Stability:** Implemented targeted TCN retraining with class-weighted BCE and log-transform regression, preserving architecture and hyperparameters while improving convergence and predictive reliability.
- **Comprehensive Analysis Pipeline:** Executed unbiased test set inference, calibration/threshold tuning, structured CSV/JSON outputs, and cross-model comparative analysis for LightGBM vs TCN across classification and regression tasks.
- **Interpretability & Deployment:** Delivered global SHAP feature explanations (LightGBM) and gradient×input saliency maps (TCN), aggregated top-10 features per target, and packaged a unified, lightweight inference pipeline supporting batch and interactive per-patient predictions with deterministic, dataset-agnostic outputs.

---

## 3. Phase 1: Data Extraction & NEWS2 Computation

### 3.1 Data Source: MIMIC-IV Demo (v2.2)
- **Overview:** Medical Information Mart for Intensive Care (MIMIC)-IV database is comprised of deidentified electronic health records for patients admitted to the Beth Israel Deaconess Medical Center, taken from PhysioNet.org.
- **Contents:** 26 tables of structured vital signs, labs, and admission data; excludes free-text clinical notes.
- **Patients:** subset 100 ICU admissions (synthetic/de-identified subset)
- **Tables used:** `chartevents`, `patients`, `admissions`, `d_items`
- **Limitations:** Small sample size (full dataset contains >65,000 ICU admissions), limited high-risk events, synthetic distribution.

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
2. **GCS → Level of Consciousness Mapping**
```text
Combine column scores (`GCS - Eye Opening`, `GCS - Verbal Response`, `GCS - Motor Response`) to create `gcs_total`. 
- GCS ≥15: Alert
- GCS <15: Not Alert
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
- Raw vitalis, ndividual parameter scores, total NEWS2 score
- Supplemental O₂, CO₂ retainer, and consciousness/GCS labels.
- Escalation risk category (low/medium/high), monitoring frequency, response.

**Patient-level: `news2_patient_summary.csv`**
- **Per-patient aggregations:** min, max, mean, median, total number of timestamps
- Summary statistics per patient.

---

## 4. Phase 2: Feature Engineering (Timestamp & Patient-Level)
**Goal:** Transform NEWS2 timestamp-level data into ML-ready features for tree-based models (LightGBM) and Neural Networks (TCN). Handle missing data, temporal continuity, and encode risk labels while preserving clinical interpretability

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
- **Patient-level:** `make_patient_features.py`
- ML-ready per patient aggregated summary features (ordered by `subject_id`)
```text
(n_patients, n_features) = (100, 43)
```
- **43 features =**
  - 8 vitals x (median + mean + min + max + % missing) = 40
  - Summary features → `max_risk` + `median_risk` + `pct_time_high` = 3

---