# Time-Series ICU Patient Deterioration Predictor

## *Hybrid Machine Learning System for Early Warning in Critical Care*

---

## Executive Summary

**Tech stack:** *Python, PyTorch, Scikit-learn, LightGBM, pandas, NumPy*

This project implements a dual-architecture early warning system comparing gradient-boosted decision trees (LightGBM) against temporal convolutional networks (TCN) for predicting ICU patient deterioration, across three NEWS2-derived clinical-risk outcomes (maximum risk attained, average sustained risk, % time spent in high-risk state). 

Models were trained on the MIMIC-IV Clinical Demo v2.2 dataset (100 patients), using dual feature engineering pipelines: 171 timestamp-level temporal features (24-hour windows) for TCN, and 40 patient-level aggregated features for LightGBM.

The hybrid approach reveals complementary strengths: LightGBM achieves superior calibration and regression fidelity (68% Brier reduction, +17% AUC, +44% R²) for sustained risk assessment, while TCN demonstrates stronger acute event discrimination (+9.3% AUC, superior sensitivity) for detecting rapid deterioration. Together, they characterise short-term instability and longer-term exposure to physiological risk.

The complete pipeline includes clinically validated NEWS2 preprocessing (CO₂ retainer logic, GCS mapping, supplemental O₂ protocols), comprehensive feature engineering, robust evaluation, and model-specific interpretability (SHAP for LightGBM; gradient×input saliency for TCN).

A deployment-lite inference system supports batch and per-patient predictions for reproducible, end-to-end use.

| Target           | Best Model | Key Metric(s)             | Notes |
|-----------------|------------|--------------------------|-------|
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

### Clinical Escalation Context
NEWS2 scoring bands map directly to clinical monitoring frequency and escalation actions; these operational consequences define the clinical targets we aim to predict:

| NEWS2 Score.                      | Clinical Risk | Monitoring Frequency                                  | Clinical Response                                                                 |
|-----------------------------------|---------------|--------------------------------------------------------|------------------------------------------------------------------------------------|
| **0**                             | Low           | Minimum every **12 hours**                             | Routine monitoring by registered nurse.                                            |
| **1–4**                           | Low           | Minimum every **4–6 hours**                            | Nurse to assess need for change in monitoring or escalation.                       |
| **Score of 3 in any parameter**   | Low–Medium    | Minimum every **1 hour**                               | **Urgent** review by ward-based doctor to decide monitoring/escalation.            |
| **5–6**                           | Medium        | Minimum every **1 hour**                               | **Urgent** review by ward-based doctor or acute team nurse; consider critical care team review.   |
| **≥7**                            | High          | **Continuous** monitoring                              | **Emergent** assessment by clinical/critical-care team; usually transfer to HDU/ICU. |

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

### Why Machine Learning?
ICU deterioration involves complex and often subtle, multivariate temporal patterns that standard threshold-based systems cannot fully capture. Machine learning enables prediction of clinically meaningful NEWS2-derived outcomes using both static and temporal representations of patient physiology.

| Model | Type | Input Features | Modelling Type | Strengths | Weaknesses | Interpretability |
|-------|------|----------------|------------------|-----------|------------|----------------|
| LightGBM | Gradient-Boosted Decision Tree (GBDT) | Aggregated patient-level | Static | Fast, interpretable, good calibration | Cannot capture sequential dynamics | SHAP |
| TCN | Temporal Convolutional Network | Timestamp-level sequential | Temporal | Captures temporal trends, slopes, variability | Requires high-resolution data, slower to train | Saliency (grad×input) |

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

```text
                               Raw EHR Data
                   (vitals, observations, lab results)
                                       │
             ┌─────────────────────────┴─────────────────────────┐
             │                                                   │
             ▼                                                   ▼

   Patient-Level Feature Engineering                        Timestamp-Level Feature Engineering
   (make_patient_features.py → news2_features_patient.csv)  (make_timestamp_features.py → news2_features_timestamp.csv)
   - Median, mean, min, max per vital                       - Missingness flags 
   - Imputation using patient-specific median               - Last Observation Carried Forward (LOCF)
   - % Missingness per vital                                - Carried-forward flags
   - Encode risk labels and summary target stats            - Rolling windows 1/4/24h (mean, min, max, std, slope, AUC)
      • max_risk                                            - Time since last observation (staleness)                                  
      • median_risk                                         - Encode risk labels
      • pct_time_high

             ▼                                                   ▼

     LightGBM Model (Classical ML)                   Temporal Convolutional Network (TCN)
   - One fixed-length vector per patient             - Full multivariate sequence per patient per timestamp
   - Fast, interpretable (SHAP)                      - Learns trends, slopes, sub-threshold drift
   - Strong baseline for tabular data                - Handles irregular sampling & long-range context
   - Cannot model sequences                          - Requires sequential data
```