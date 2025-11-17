# Time-Series ICU Patient Deterioration Predictor

## *Hybrid Machine Learning System for Early Warning in Critical Care*

---

## Executive Summary

**Tech stack:** *Python, pandas, NumPy, LightGBM, PyTorch, Scikit-learn*

This project implements a dual-architecture early warning system comparing gradient-boosted decision trees (LightGBM) against temporal convolutional networks (TCN) for predicting ICU patient deterioration, across three risk horizons (maximum risk atained, average sustained risk, % time spent in high risk). Built on MIMIC-IV Clinical Demo v2.2 dataset (100 patients), the system processes 171 temporal features across 24-hour windows and 40 aggregated patient-level features, to support continuous monitoring and escalation decisions.

```text
                               Raw EHR Data
                   (vitals, observations, lab results)
                                       │
             ┌─────────────────────────┴─────────────────────────┐
             │                                                   │
             ▼                                                   ▼

   Patient-Level Feature Engineering                 Timestamp-Level Feature Engineering
(make_patient_features.py → news2_features_patient.csv)     (make_timestamp_features.py → news2_features_timestamp.csv)
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

The hybrid approach reveals complementary strengths: LightGBM achieves superior calibration and regression fidelity (68% Brier reduction, +17% AUC, +44% R²) for sustained risk assessment, while TCN demonstrates stronger acute event discrimination (+9.3% AUC, superior sensitivity) for detecting rapid deterioration. 

The complete pipeline includes NHS-validated NEWS2 preprocessing with CO₂ retainer logic, GCS mapping, and supplemental O₂ protocols; extensive evaluation metrics and model-specific interpretability methods for clinical validation (SHAP for LightGBM, absolute gradient×input saliency for TCN); and a deployment-ready dual inference system (batch and per-patient) for end-to-end usability.

| Target           | Best Model | Key Metric(s)             | Notes |
|-----------------|------------|--------------------------|-------|
| Maximum Risk     | TCN        | ROC AUC: 0.923           | Strong acute detection, high sensitivity |
| Median Risk      | LightGBM   | ROC AUC: 0.972, Brier: 0.065 | Superior sustained risk calibration |
| Percentage Time High | LightGBM | R²: 0.793                | Better regression fidelity for high-risk exposure |

**Key Contributions:**
- Clinical validity pipeline with robust NEWS2 computation
- Dual feature engineering (patient-level vs timestamp) for both classical and deep learning models
- Duel model training with hyperparameter tuning
- Rigorous refinement and model evaluation
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
ICU patient deterioration manifests through subtle vital sign changes hours before critical events. The National Early Warning Score 2 (NEWS2) is widely used in UK hospitals to detect and escalate care for deteriorating patients. Accurate, real-time scoring and risk stratification can:
- Enable earlier intervention and ICU escalation
- Support clinical decision-making with actionable, interpretable metrics
- Provide a foundation for advanced ML models to improve patient outcomes

![NEWS2 API Diagram](https://developer.nhs.uk/apis/news2-1.0.0-alpha.1/index.html)

Figure: NHS Digital, NEWS2 API guide (Open Government Licence v3.0)  

Although NEWS2 is the national standard for deterioration detection, it has well-recognised constraints:
- **No true temporal modelling:** Although observations are charted sequentially, the scoring algorithm treats each set of vitals independently and does not incorporate trend, slope, variability, or rate-of-change.
- **Discrete scoring limitations:** NEWS2 discretises continuous physiological signals into coarse bands and does not model interactions between multiple variables, which limits sensitivity to subtle multivariate deterioration patterns.
- **Escalation overload:** Threshold-based scoring generates many false positives in elderly and multimorbid cohorts, contributing to alert burden and escalation fatigue.
- **Limited predictive horizon:** NEWS2 typically identifies deterioration only after thresholds are crossed, offering limited early-warning capability compared with models that can detect sub-threshold physiological drift.

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
- Transitions between risk bands (especially into medium or high) drive clinical workload and resource allocation, including urgent reviews and ICU involvement.  
- Predicting imminent transitions into these categories (e.g., entering high risk within the next 4–6 hours) enables earlier intervention, reducing delayed escalations and improving critical-care resource planning.  

#### Why NEWS2 is used as the reference standard
- NEWS2 is the nationally accepted standard for ward-based clinical deterioration assessment. Using it as the ground-truth ensures that ML models are trained and evaluated against a clinically validated reference.  
- ML models predict summary outcomes derived from NEWS2 clinical-risk categories:
  - `max_risk`: Maximum risk attained during the observation window  
  - `median_risk`: Average sustained risk across the stay  
  - `pct_time_high`: Percentage of time spent in high-risk state  
- Evaluating ML predictions against these NEWS2-derived outcomes allows assessment of **predictive horizon**, **sensitivity**, and the ability to anticipate **clinically actionable deterioration trends** before standard escalation would occur.

### Why Machine Learning?
ICU deterioration is complex and often subtle, involving multivariate temporal patterns that standard threshold-based systems cannot fully capture. ML models allow us to go beyond static scoring by predicting summary outcomes derived from NEWS2 clinical-risk categories.

| Model | Type | Input Features | Modelling Type | Strengths | Weaknesses | Interpretability |
|-------|------|----------------|------------------|-----------|------------|----------------|
| LightGBM | Gradient-Boosted Decision Tree | Aggregated patient-level | Static | Fast, interpretable, good calibration | Cannot capture sequential dynamics | SHAP |
| TCN | Temporal Convolutional Network | Timestamp-level sequential | Temporal | Captures temporal trends, slopes, variability | Requires high-resolution data, slower to train | Saliency (|grad×input|) |

#### LightGBM (classical, non-temporal ML)
- LightGBM, a gradient-boosted decision tree (GBDT) algorithm, provides a strong baseline for tabular clinical data
- Captures nonlinear interactions between vital signs  
- Fast to train and tune, handles missing data robustly
- Highly interpretable via SHAP  
- Often competitive or superior when temporal structure is weak  

#### Temporal Convolutional Network (TCN) (temporal deep learning)
- TCN captures time-dependent patterns, slopes, and variability  
- Models long-range temporal context  
- Robust to irregular sampling  
- Potentially detects subtle deterioration earlier than threshold-based approaches  

#### Why compare both?
- LightGBM provides a robust classical-ML baseline for tabular clinical data.
- TCN evaluates whether temporal modelling yields measurable gains by capturing sequential patterns and slopes in vital signs.
- This comparison reflects realistic deployment: classical ML may suffice for lower-frequency ward data, whereas temporal models exploit high-resolution ICU monitoring to detect early deterioration.
- The evaluation clarifies where temporal modelling adds value, where classical ML is sufficient, and the trade-offs between interpretability and predictive performance.

This project therefore systematically evaluates temporal vs. non-temporal ML approaches for predicting ICU deterioration, using clinically meaningful NEWS2-derived summary outcomes as targets.