ICU Patient Deterioration Predictor
Python, PyTorch, scikit-learn, LightGBM

Developed early warning system comparing deep learning (TCN) versus classical ML (LightGBM) for predicting
ICU patient deterioration across multiple risk horizons on MIMIC-IV demo (100 patients), benchmarked against
NEWS2; supporting patient triage, continuous monitoring, and escalation decisions.
Engineered hybrid approach: TCN processing 171 temporal-features across 24-hour windows (8 vital parameters)
for acute detection; LightGBM analysing 40 aggregated patient-features for sustained risk assessment.
Implemented clinical-validity-aware preprocessing and missing data handling using NEWS2-derived ground truth
with GCS mapping, CO2 retainer and supplemental O2 logic validated against NHS protocols.
Achieved complementary performance: TCN superior for acute events with rapid detection (+9.3% AUC, superior
sensitivity); LightGBM superior for long-term sustained deterioration exposure (68% Brier reduction, +17% AUC,
+44% R², better calibration and accuracy), suggesting ensemble approach for production deployment.
Validated interpretability methods (SHAP vs saliency maps) for clinical transparency and deployed reproducible,
auditable end-to-end pipeline with comprehensive documentation.

### Phase 1: Baseline NEWS2 Tracker (Steps 1-4)
**Goal: Extract, clean, and compute NEWS2 scores from raw synthetic EHR data. Establish a reproducible pipeline for clinical scoring and patient-level summarisation.**
1. **Dataset Preparation**
	-	Downloaded synthetic MIMIC-IV demo dataset (`mimic-iv-clinical-database-demo-2.2.zip`) and unzipped CSV files.
	-	Explored `chartevents.csv` and other relevant CSVs to identify required vitals for NEWS2 scoring.
	-	**Reasoning**: Understanding raw data structure and units is critical for accurate downstream scoring.
2. **Data Extraction**
	-	Wrote `extract_news2_vitals.py` to extract NEWS2-relevant vitals from all CSVs.
	-	Standardised headers and extracted only required columns.
  - Implemented CO₂ Retainer identification code to identify patients meeting CO₂ retainer criteria:
    -	PaCO₂ > 45 mmHg
    -	pH 7.35–7.45
    -	ABG measurements ±1 hour apart
	-	Updated vitals dataset `news2_vitals.csv` to include CO₂ retainer status `news2_vitals_with_co2.csv`.
  - **Created helper/inspection script**: `check_co2_retainers.py` to identify patients meeting CO₂ retainer criteria and check logic of `extract_news2_vitals.py`. 
  -	**Defensive Coding & Sanity Checks**:
    -	Added checks for missing columns before merges.
    -	Assigned default values for missing vitals (0 or False).
    -	Handled duplicates and avoided _x/_y conflicts.
	-	**Generated**:
    - `news2_vitals.csv` → original extracted vitals before retainer logic implemented (file not used)
    -	`news2_vitals_with_co2.csv` → vitals with retainer flags 
    -	`co2_retainer_details.csv` → patient-specific CO₂ retainer information
	-	**Reasoning**: 
    - Creates a clean, consistent dataset while preventing KeyErrors, duplicates, or missing column issues during NEWS2 computation. Clean input for NEWS2 computation and CO₂ retainer checks.
    - `check_co2_retainers.py` ensures accurate oxygen scoring according to NEWS2 rules; future-proof for real datasets.
    - `co2_retainer_details.csv` makes sure that retainer patients actually do/don't exist.
4. **NEWS2 Scoring**
	-	**Implemented `compute_news2.py` with**:
    -	Dictionaries defining threshold-to-score mappings for each vital
    -	Functions to compute individual vital scores
    -	Aggregated to total NEWS2 per timestamp
	-	Pivoted wide-format CSV with all expected vitals (`respiratory_rate, spo2, supplemental_o2, temperature, systolic_bp, heart_rate`)
	-	Safely merged GCS components to compute `gcs_total` and `level_of_consciousness`
  - **Human-readable labels**: `consciousness_label`, `supplemental_o2_label`, `co2_retainer_labelv added for clarity and safety.
	-	**Outputs**:
    -	`news2_scores.csv` → per-timestamp NEWS2 scores
    -	`news2_patient_summary.csv` → per-patient aggregates (min, max, mean, median scores)
	-	**Reasoning**: Produces ready-to-use datasets for later baselines or predictive modelling.
**End Products of Phase 1**
-	`news2_scores.csv` → timestamp-level NEWS2 scores
-	`news2_patient_summary.csv` → patient-level aggregate scores
-	`co2_retainer_details.csv` → CO₂ retainer info
-	**Clean Python scripts**:
	-	`extract_news2_vitals.py`
	-	`check_co2_retainers.py`
	-	`compute_news2.py`
-	**Documentation**: 
  - Notes on scoring rules, defensive coding practices, GCS and FiO₂ handling, and pipeline reproducibility
  - Safe pipeline capable of handling missing columns, duplicate merges, and incomplete clinical data.

### Phase 2: Preprocess Data for ML-Models (Steps 1-4)
**Goal: Transform NEWS2 timestamp and patient-level data into ML-ready features for tree-based models (LightGBM) and Neural Networks (TCN). Handle missing data, temporal continuity, and encode risk labels while preserving clinical interpretability.**
1. **Validating NEWS2 Scoring**
  - **Action**: Ran `validate_news2_scoring.py` on test dictionaries.
  - **Findings**:
    - Low GCS cases initially produced incorrect scores.
    - The scoring function ignored consciousness because row.get("level_of_consciousness", pd.NA) returned pd.NA.
    -	Other special cases (SpO₂, supplemental O₂) were correctly scored because their thresholds were handled explicitly.
  - **Fixes**: Moved `if pd.isna(value): return 0` **to the end of the function**.
  - **Outcome**: All unit tests passed, learned the importance of understanding intermediate variables in scoring pipelines.
  - The main pipeline did not have these problems as `gcs_total` is converted into `level_of_consciousness` before the scoring is called, so there was no missing keys.
2. **Missing Data Strategy**
  - **Timestamp-level features**:
    - Use LOCF (Last Observation Carried Forward) to maintain temporal continuity.
    - Add missingness flags (1 if value was carried forward) so models can learn from missing patterns.
    - **Justification**: mimics clinical reality; preserves trends; Tree-based models like LightGBM handle NaNs naturally.
  - **Patient-level summary features**:
    - Use median imputation per patient timeline if a vital is missing across some timestamps.
    -	Include % of missing timestamps as a feature.
    -	**Justification**: balances robustness with bias avoidance; prevents skewing min/max/mean statistics.
  -	**Key decisions**:
    -	Do not fill population median at timestamp-level (would break temporal continuity).
    -	Only fill median at patient summary level if some timestamps exist; otherwise, leave as NaN or optionally fallback to population median.
3. **Prepare Timestamp-Level Features `make_timestamp_features.py`**:
  1. Start from `news2_scores.csv` (all vitals + NEWS2 + escalation labels).
    - Parse `charttime` as `datetime`.
    - Sort by `subject_id` & `charttime`.
  2. Create missingness flags for each vital (before fills).
  3. LOCF forward-fill per subject (optionally backward-fill for initial missingness or leave as NaN), do not use population median.
  4. Create carried-forward flags (binary indicator - 1 if the value came from LOCF). Helps ML distinguish between observed vs assumed stable, exploit missingness patterns (e.g. vitals measured more frequently when patients deteriorate).
  5. **Compute rolling windows (1h, 4h, 24h)**: mean, min, max, std, count, slope, AUC.
  6. Compute time since last observation (`time_since_last_obs`) for each vital (staleness per vital).
  7. **Encode risk/escalation labels**: Convert textual escalation/risk labels → numeric ordinal encoding (`Low=0, Low-Medium=1, Medium=2, High=3`) for ML. Keeps things simple - one column, easy to track in feature importance
  8. Save `news2_features_timestamp.csv` (ML-ready). 
  **Rationale**:
  - Trees can leverage trends and missingness.
  -	Rolling windows capture short-, medium-, and long-term deterioration patterns.
  -	Timestamp features feed ML models like LightGBM directly without further preprocessing.
4. **Preparing Patient-Level Features `make_patient_features.py`**:
  1. Load input file `news2_scores.csv`. 
  2. **Group by patient**: Aggregate vitals per patient timeline (median, mean, min, max per vital).
  3. **Per-Patient Median imputation**: Fill missing values for each vital using patient-specific median (so their profile isn’t biased by others), if a patient never had a vital recorded, fall back to population median.
  4. **Compute % Missingness per vital**: Track proportion of missing values per vital before imputation (HR missing in 30% of their rows = 0.3), missingness itself may signal clinical patterns (e.g. some vitals only measured in deteriorating patients).
  5. **Encode risk/escalation labels**: Ordinal encoding (Low=0, Low-Medium=1, Medium=2, High=3), calculate summary stats per patient: max risk (highest escalation they reached), median risk (typical risk level), % time at High risk (what fraction of their trajectory was spent here).
  6. **Output**: Save `news2_features_patient.csv` (compact, one row per patient, ML-ready summary).
  **Rationale**:
  -	Median imputation preserves patient-specific patterns without introducing bias from other patients.
  -	% Missing captures signal from incomplete measurement patterns.
  -	Ordinal risk encoding simplifies downstream ML model input while retaining interpretability. Together, these three summary features summarise a patient’s escalation profile across their stay. Proportion features (like % high) are standard numeric features (not encoded categories).
  -	This is enough for model; don’t need optional metrics like streaks, AUC, or rolling windows for the patient summary.
**Outputs of Phase 2**
- **Scripts**:
  -	`news2_features_timestamp.csv` → ML-ready timestamp-level dataset with features, rolling windows, missingness flags, and encoded risk.
  -	`news2_features_patient.csv` → ML-ready patient-level summary dataset with aggregated features and escalation metrics.
-	Validated NEWS2 scoring function and pipeline, ensuring correct handling of GCS, FiO₂, supplemental O₂, CO₂ retainers, and missing data.
-	Defensive coding practices for merges, missing columns, and idempotent transformations.

### Overview
**For timestamp-level ML features (news2_features_timestamp.csv)**:

```text
raw long vitals (from MIMIC/ICU)  
    ↓ compute_news2.py  
news2_scores.csv               ← "clinical truth" (all vitals + NEWS2 + escalation labels)  
    ↓ make_timestamp_features.py
news2_features_timestamp.csv   ← "ML ready" (numeric features, missingness flags, encodings)  
```

**For patient-level summary features (news2_features_patient.csv)**:

```text
raw long vitals  
    ↓ compute_news2.py  
news2_scores.csv                ← news2_patient_summary.csv not needed
    ↓ make_patient_features.py  
news2_features_patient.csv      ← ML ready (patient-level aggregates, imputed medians, missingness %)
```

**The difference**:
- Timestamp pipeline → preserves row-by-row dynamics (LOCF, staleness, rolling windows).
-	Patient pipeline → collapses timeline into patient-level summaries (medians, % missing, escalation profile).

#### Step 5: ML Model Selection
-	**Options considered**:
  -	Logistic Regression → easy to deploy and explainable but underpowered, tends to underperform on raw time-series vitals.
  -	Deep learning (LSTMs/Transformers) → overkill, prone to overfitting with moderate datasets.
  -	Boosted Trees (XGBoost / LightGBM / CatBoost) → robust for tabular ICU data, handle NaNs, train fast, interpretable.
-	**Decision: LightGBM (Gradient Boosted Decision Tree (GBDT) library)**
  - State-of-the-art for structured tabular data (EHR/ICU vitals is tabular + time-series).
  -	Handles missing values natively (NaNs) → no additional imputation required (simpler pipeline).
  -	Provides feature importances → interpretability for clinical review.
  -	Easy to train/evaluate quickly → allows multiple experiments.

timestamp level data 
- End-to-end workflow complete:
  1. Load & sort data.  
  2. Add missingness flags.  
  3. Apply LOCF.  
  4. Add carried-forward flags.  
  5. Add rolling window features.  
  6. Add time since last obs.  
  7. Encode risk labels.  
  8. Save final dataset → `news2_features_timestamp.csv`.

### Overview 
```text
  Raw EHR Data (vitals, observations, lab results)
         │
         ▼
Timestamp Feature Engineering (news2_scores.csv)
 - Rolling statistics (mean, min, max, std)
 - Slopes, AUC, time since last observation
 - Imputation & missingness flags
         │
         ├─────────────► Neural Network Models (v2)
         │              - Input: full time-series per patient
         │              - Can learn temporal patterns, trends, dynamics
         │
         ▼
Patient-Level Feature Aggregation (make_patient_features.py → news2_features_patient.csv)
 - Median, mean, min, max per vital
 - Impute missing values
 - % missing per vital
 - Risk summary stats (max, median, % time at high risk)
 - Ordinal encoding for risk/escalation
         │
         ▼
LightGBM Model (v1)
 - Input: one row per patient (fixed-length vector)
 - Uses aggregated statistics only
 - Cannot handle sequences or variable-length time series
 ```

3. **Model Roadmap Finalised**:
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

### Extra insights
- **Future-proofing with both feature sets ensures robustness and flexibility**:
  - **LightGBM (V1)** → clinician-friendly, interpretable baseline.  
  - **TCN (V2)** → modern DL, captures dynamics.  
- **Timestamp-level features** = richest representation, essential for sequence models / deep learning
- **Patient-level features** = distilled summaries, useful to quickly test simpler models, feature importance or quick baseline metrics.
- Keeping both pipelines means we can mix (hybrid approaches) if needed (e.g., summary features + LSTM on sequences). 
- LightGBM is often deployed first because it’s fast, robust, and interpretable, while the neural network is a v2 that might improve performance. 

### Portfolio story
- **LightGBM (v1)**: We started with patient-level aggregation to establish a baseline model that is interpretable and fast to train. This gives clinicians an overview of which vitals and risk patterns matter most.
- **Neural Network (TCN)(v2)**: Once we had a solid baseline, we moved to a temporal convolutional network to directly learn time-dependent deterioration patterns from patient trajectories. This captures dynamics that aggregated features can’t.

### Phase 3: LightGBM Training + Validation (Steps 1–7)
**Goal: Train, validate, and document a LightGBM model on patient-level features, producing a polished, interpretable baseline and deployment-ready models for all three targets (`max_risk, median_risk, pct_time_high`).**
1. **Dataset Preparation `prepare_patient_dataset.py`**
  - Load processed patient-level features `news2_features_patient.csv`.
  - **Binary target conversion**:
    - `max_risk`: 0–2 → 0 (not high risk), 3 → 1 (high risk)
    -	`median_risk`: 0–1 → 0 (low risk), 2 → 1 (medium risk)
  - **Define modelling types**:
    -	`max_risk` → binary classifier: high-risk vs not high-risk.
    -	`median_risk` → binary classifier: medium-risk vs low-risk.
    -	`pct_time_high` → continuous regression target.
  - Separate features (X) and targets (y):
  - Verify datatypes, shapes, and missing values.
  - **K-Fold Setup**:
    - 5-fold cross-validation for all targets.
    -	`StratifiedKFold` for classification (`max_risk, median_risk`) to preserve minority class distribution.
    -	`Standard KFold` for regression (`pct_time_high`) as class balance not relevant.
	-	**Reasoning**: 
    - This step prepares X_train, y_train, X_test, y_test for CV.
    - CV + binary conversion preserves minority class distribution, giving stable estimates.
    - Ensures stable and reproducible evaluation on a small dataset (100 patients).
2. **Model Initialisation & Quick Test `initial_train_lightgbm.py`**
  - **Initialise LightGBM models**:
    - `LGBMClassifier` → `max_risk, median_risk`
    - `LGBMRegressor ` → `pct_time_high`
  - Set baseline parameters and `random_state=42` for reproducibility.
	-	Skipped full training for initial tests; ran quick fit on small subset (10) to check pipeline, feature-target alignment, and X/y shapes.
	-	**Reasoning**: 
    - Ensures pipeline works end-to-end before full CV training. 
    - Catches misalignments, missing values, or type errors early.
3. **Model Training and Validation (Cross-Validation) `complete_train_lightgbm.py`**
  - Loop through all targets (creates 3 models)
  - **Train models on 4 folds, validate on 1**:
    -	Fit with early stopping on training folds.
    -	**Evaluate on validation folds**
      - Load saved models, generate predictions on validation folds.
      - **Compute fold-wise metrics**:
        -	ROC-AUC (primary) and Accuracy (fallback) for classifiers.
        -	RMSE/MSE for regression.
      - Aggregate feature importance across folds per target.
	-	**Store per-fold outputs**: trained models, fold metrics, and feature importance.
  - **Outputs (`baseline_models/`)**:
  	-	3 targets × 5 folds = 15 trained models → `{target}_fold{n}).pkl`
	  -	15 feature importance CSVs → `{target}_fold{n}_feature_importance.csv`
	  -	3 CV result CSVs (scores per fold per target) → `{target}_cv_results.csv`
	  - Training summary (target name, dataset shape (100,40), mean CV score, top 10 features) → `training_summary.txt`
	-	**Reasoning:** 
    - Cross-validation provides unbiased performance estimates and reproducible models.
    - Fold-wise outputs allow reproducibility and further analysis.
    - Feature importance per fold supports interpretability.
4. **Hyperparameter Tuning `tune_models.py`**
  - **Tune key parameters for stability and performance**: `learning_rate`, `max_depth` / `num_leaves`, `n_estimators`, `min_data_in_leaf`.
	-	**Evaluate using 5-fold CV**:
    -	ROC-AUC / Accuracy for classifiers.
    -	RMSE for regression.
  - **Outputs (`hyperparameter_tuning_runs/`)**:
    -	`best_params.json` (per target)
    -	`*_cv_results.csv` (fold-wise CV scores)
    -	`tuning_logs/` (all tuning parameter sets and scores)
  -	**Rationale**: 
  	-	Optimises baseline performance without overfitting.
	  -	Produces reproducible, interpretable models.
	  -	Ensures portfolio credibility.
5. **Feature Importance Aggregation `feature_importance.py`**
	- Load per-fold feature importance CSVs.
	-	Aggregate across folds, compute mean importance.
	-	Generate bar plots of top 10 features per target.
  - **Outputs (`feature_importance_runs/`)**:
    -	3x `{target}_feature_importance.csv`
    -	3x `{target}_feature_importance.png`
  - **Reasoning**:
    -	Highlights which features drive predictions.
    -	Supports interpretability for clinical stakeholders.
    -	Visual outputs are portfolio-ready.
6. **Final Model Training (Deployment-Style Models) `train_final_models.py`**
	- Train one final model per target using the entire 100-patient dataset.
  - Apply best hyperparameters from tuning.
  - Save models for deployment/demo purposes.
  - **Outputs (`deployment_models/`)**: 3x `{target}_final_model.pkl`
	-	**Reasoning**:
    - Produces the best possible trained model using all available data.
    -	**Matches real-world practice**: once validated, you don’t throw away data, you train on the full cohort.
    -	Produces 3 clean, reproducible, deployment-ready final models.
    - Allows demonstration of classifier + regressor outputs in portfolio.
7. **Portfolio-Ready Summary `summarise_results.py`**
  - Compile CV scores, best hyperparameters, feature importance.
	-	Save plain-text portfolio-ready summary.
  - **Output**:`deployment_models/training_summary.txt` (portfolio-ready summary including CV scores, hyperparameters, top features).
  - **Reasoning**:
    - Transparent reporting of methodology.
    -	Provides a reproducible baseline for comparison with Neural Networks (Phase 4).
**Why Not Go Further**
- Phase 3 makes LightGBM phase complete, credible, and deployment-worthy without unnecessary over-optimisation.
- **Ensembling (stacking, blending, bagging multiple LightGBM models)**: adds complexity without new insights → not unique for a portfolio.
- **Nested CV**: more statistically rigorous, but overkill for 100 patients; doesn’t change credibility.
- **Bayesian optimisation / AutoML**: looks flashy, but to recruiters it signals you know how to use a library, not that you understand the fundamentals.
- **Overfitting risk**: with 100 patients, “chasing” tiny gains just makes results unstable and less reproducible.
- **Time sink**: delays me getting to Neural Nets (the unique, impressive part of your project).
**Phase 3 Pipeline**:
- **Pipeline credibility**: flly reproducible, stratified CV for imbalanced classification, deployment-ready models.
- **Portfolio messaging**:
  - Small datasets require pragmatic design choices (binary targets, fold selection) to produce credible results.
	- Phase 3 demonstrates handling messy, small clinical datasets.
	-	**Shows robust ML pipeline**: data prep → CV → training → tuning → feature importance → final models.
	-	Explains methodological pivot (from rare-event classification to trend/regression) as a real-world research adaptation.


### Purpose of Baseline Classical LightGBM ML Model
1. Show I can prepare patient-level data for ML.
2. Provides a baseline classical ML benchmark for patient deterioration prediction.
3. Demonstrates an end-to-end ML workflow, and a credible, well-structured pipeline (data prep → CV → training → saving → validation → documentation → final deployment models).
4. Ensures reproducibility and robustness with cross-validation, and deployment readiness (final models).
5. Adds interpretability through feature importance, crucial in clinical healthcare settings.
6. Establishes a strong baseline Performance benchmark for later comparison with Neural Networks, showing their added value.

### How Gradient Boosted Decision Tree Model Works
-	Trees split on feature thresholds.
-	Each split improves the model’s predictions.
- Build trees sequentially, each correcting previous errors.
-	Feature importance = how often (or how much) the model used each feature to reduce errors.

### Pipeline Visualisation of `complete_train_lightgbm.py` 
```text
news2_features_patient.csv  (patient-level dataset)
         │
         ▼
Preprocessing
 - Collapse rare classes:
    • max_risk: (0,1,2 → 2 [not high risk], 3 → 3 [high risk])
    • median_risk: (0,1 → 1 [low risk], 2 → 2 [medium risk], 3 removed)
 - Prepare features (exclude subject_id & target columns)
         │
         ▼
Binary Conversion for LightGBM Classification
 - max_risk: 2 → 0 (not high risk), 3 → 1 (high risk)
 - median_risk: 1 → 0 (low risk), 2 → 1 (medium risk)
         │
         ▼
Cross-Validation Setup
 - 5-fold StratifiedKFold (max_risk, median_risk → binary classification)
 - 5-fold KFold (pct_time_high → regression)
         │
         ▼
Model Training Loop (for each target)
 ├── max_risk (binary classifier)
 │     • Metric: ROC-AUC / Accuracy
 ├── median_risk (binary classifier)
 │     • Metric: ROC-AUC / Accuracy
 └── pct_time_high (regressor)
       • Metric: MSE / RMSE
         │
         ▼
Per-Fold Processing (5 folds per target)
 - Train LightGBM model with early stopping
 - Predict on validation fold
 - Compute score (ROC-AUC or RMSE)
 - Save model (.pkl)
 - Save feature importance (.csv)
         │
         ▼
Per-Target Outputs
 - CV results file: scores per fold + mean/std
 - Append per-target summary to training_summary.txt
         │
         ▼
Final Output: 34 files in saved_models/
 - 15 trained models (.pkl) → 3 targets × 5 folds
 - 15 feature importance CSVs → 3 targets × 5 folds
 - 3 CV results CSVs → one per target
 - 1 training summary log (training_summary.txt)
 ```

### Conclusions On Patient-Level Data
**Max Risk**
- Data not well distributed at all. The class imbalance reflects the clinical reality that most patients requiring intensive monitoring are indeed high-acuity cases.
- The fundamental issue is that with only 1 patient in the minority class for max_risk, any ML approach will fail.
- Dataset doesn't contain enough diversity in the max_risk variable to support multiclass learning. This is a data collection issue, not a modeling limitation.
- We must change from 'three risk levels' to 'high risk vs not high risk', which is clinically relevant and often more actionable than granular risk stratification. 
**Median Risk**
- It makes clinical sense that nobody’s median is high-risk, most patients don’t sit at high risk their whole stay. That’s clinically plausible.
- But from a modeling perspective it’s still quite imbalanced (76 vs 24).
- We must stratify KFold in order to make sure there is even distribution.
- Median risk never reached 3 in this dataset, so we must restrict the class set to [1,2].
**Percentage Time High**
- pct_time_high is continuous and has a good spread across patients.
- No need to transform pct_time_high for current tree-based model pipeline.
- Skew and zero inflation are not a major problem in this context.
- Doing extra preprocessing would add work but minimal benefit.
**Overall Conclusions**
- Data distribution is realistic. pct_time_high is continuous and well-distributed, so orginal 5 folds is perfectly fine, no need for special handling as regression will be stable. And for max_risk and median_risk, chnaging to binary class sets allows us to keep our original 5-fold StratifiedKFold.
- Imbalance reflects real clinical distributions, we can still use all three variables, but document this insight into the clinical dataset limitations.
- The model can learn patterns from this dataset effectively.
- No longer need to code for new variables and replan outcomes for LightGBM and Neural Network models as we previously thought.
- Overall we keep all three targets, have both classification and regression tasks, can explain different fold choices per target as a **thoughtful design decision based on data distribution.**

### Changes We Will Make To `complete_train_lightgbm.py`
1. **Simplify to binary classifciation (max_risk):**
  - Binary classification is most pragmatic, and should produce a robust, clinically interpretable model. 
  - Converting to "high risk (3) vs not high risk (0,1,2)" gives you 86 vs 14 samples, which is workable for 5-fold CV and potentially more clinically relevant (identifying highest-risk patients) and actionable than trying to distinguish between three risk levels when you barely have data for the lowest category.
  - Binary risk classification clinically gives decision-making clarity (eliminates intermediate categories), helps prioritise scarce resource allocation, and matches matches hospital binary alert systems.
  - Statistically, they improve model performance and interpretability:
    - Binary models typically achieve better discriminative performance than multiclass, ROC-AUC interpretation is straightforward for clinical audiences, sensitivity/specificity trade-offs are easier to optimise for clinical priorities
    - Feature importance directly answers "what predicts highest risk?", clinical staff can understand model decisions more easily, fewer false positive categories to explain.
  - This approach transforms a data limitation into a focused, clinically relevant research question.
1. **Combining 0+1 into a single “low-risk” class (median_risk):**
  - Reasonable simplification for modeling, especially given how 0 patients originally had score 0.
  - NEWS2 scoring has risk split into low-risk, medium-risk and high-risk, however there is an extra sub-risk within low. If the total NEWS2 score lies within the low-risk range, but any single vital scored a 3, then the risk would be low-medium.
  - Preserves clinical reasoning: low-risk and low-medium risk are merged, while medium and high risk remain distinct.
  - Reduces the chance of empty-class folds that would break training.
2. **Use StratifiedKFold for classification only (max_risk, median_risk), keep plain KFold for regression (pct_time_high):**
  - Cleanest solution. This avoids crashes, ensures every fold sees all classes, and doesn’t complicate pipeline with LabelEncoder or LightGBM params (forcing global class encoding)
  - Regression target pct_time_high uses plain KFold because class distribution isn’t relevant.
	- Stratification aligns with the small minority classes: even rare events appear in validation, preventing folds without examples of certain classes.
3. **Keep 5-Fold CV Strategy**:
  - Why keep 5-fold: statistical reliability, reduce variance, more meaningful evalusation, standard practice. 
  - max_risk (2, 3) → 5-fold CV (binary classification, minority class adequate to support this).
  - median_risk (1, 2 only, 3 absent) → 5-fold CV (binary classification, minority class adequate to support this).
  - pct_time_high (continuous regression) → 5-fold CV (enough data, no class imbalance problem).
4. **Explicitly encode that only the discrete values [2,3] or [1,2] exist:**
	- Forces LightGBM to always expect all classes.
	-	Ensures predict_proba outputs arrays of consistent length across folds.
	-	Avoids downstream code errors when using np.argmax or other evaluation steps.
	-	Without this, LightGBM might give inconsistent output shapes.
**Overall** 
  - StratifiedKFold introduces slightly artificial folds (slightly less “natural”), but the benefits—avoiding unseen class errors and maintaining output consistency—far outweigh the downside.
	- This setup makes the pipeline robust and reproducible, even with small or imbalanced clinical datasets.

### 4 Key Parameters for Hyperparameter Tuning 
**Decision**:
- We have a small dataset (100 patients). With complex models or too many trees, overfitting is easy, the model could “memorise” the patients instead of generalising to other patients. That’s why tuning parameters is critical.
- These 4 parameters are the only ones we tune, because they have the largest impact on performance and stability for our dataset size. 
- Other parameters (like regularisation terms) are left at defaults to avoid overcomplicating tuning and risking overfitting (learning the training data too well, including noise or random fluctuations, rather than the underlying patterns).
**4 Key parameters (built in arguments)**:
1. `learning_rate`
	-	Controls the step size at each iteration when building trees.
	-	Balances training speed vs overfitting: too high → may overshoot minima (unstable), too low → slow convergence.
2. `max_depth` / `num_leaves`
	-	Limits tree size/complexity.
	-	Prevents overfitting due to overly complex trees, which is critical for a small dataset (100 patients).
3. `n_estimators`
	-	Total number of trees in the ensemble.
	-	More trees improve model capacity but risk overfitting; fewer trees risk underfitting.
4. `min_data_in_leaf`
	-	Minimum samples (number of patients) required in a leaf node.
	-	Stabilises predictions by preventing leaves with very few samples (avoiding noisy splits).


### Pipeline For Day 12
```text
news2_features_patient.csv (raw patient-level features)
         │
         ▼
Script 1: tune_models.py (performs hyperparameter tuning & cross-validation)
         │
         ├─► CV Results CSVs per target (hyperparameter_tuning_runs/ )
         ├─► Best Hyperparameters JSON (hyperparameter_tuning_runs/ )
         └─► Logs of each tuning run for debugging / record-keeping (hyperparameter_tuning_runs/tuning_logs/)
         │
         ▼
Script 2: feature_importance.py
 - Aggregates feature importance across folds
 - Produces visualisations
         │
         ├─► Feature Importance CSVs per target (feature_importance_runs/)
         └─► Bar plots of top features per target (feature_importance_runs/)
         │
         ▼
Script 3: train_final_models.py (trains 1 final model per target using best hyperparameters)
         │
         ├─► 3 Deployment-Ready Models (.pkl) (deployment_models/)
         └─► Optional training logs (deployment_models/)
         │
         ▼
Script 4: summarise_results.py
 - Compiles CV scores, best hyperparameters, top features
 - Produces portfolio-ready summary
         │
         └─► training_summary.txt (deployment_models/)
```

### Phase 4: Temporal Convolutional Network (TCN) Training + Validation (Steps 1-5)
**Goal: Build, validate, and document a temporal deep learning model (TCN) on sequential EHR features. Deliver a trained model (`tcn_best.pt`) that generalises well on validation data.**
1. **Dataset Preparation (Sequential Features) `prepare_tcn_dataset.py`**
  - **Input**: timestamp-level EHR data (vitals, labs, obs) `news2_features_timestamp.csv`, `news2_features_patient.csv`
  - **Output**: padded sequences ready for temporal modelling.
  -	**Steps**:
    - Patient-level stratified splits (train/val/test).
    - Continuous-only z-score scaling (fit on train, applied to val/test).
    - Binary features preserved
    - Drop of irrelevant categorical text features.
    - Conversion to tensors with padding + masks.
    - Saving scaler + config JSON for reproducibility.
  - **Reasoning**: handling realistic, messy temporal data is my clinical-technologist edge.
2. **Model Architecture (TCN) `tcn_model.py`**
  - **Base**: Temporal Convolutional Network (stacked causal dilated 1D convolutions).  
  - **Design**:
    - **Input**: `(batch, sequence_length, features)` → permuted to `(batch, channels, sequence_length)` for Conv1d.  
    - **Residual blocks**: 3 TemporalBlocks, each with:  
      - 2 causal convolutions (dilated, length-preserving).  
      - LayerNorm → stabilises training.  
      - ReLU activation → non-linearity.  
      - Dropout → regularisation.  
      - Residual/skip connection → stable gradient flow.  
    - **Stacking with dilation**: each block doubles dilation (1, 2, 4) → exponentially increasing receptive field.  
    - **Masked mean pooling**: collapses variable-length sequences into a single patient-level vector, ignoring padding.  
    - **Optional dense head**: Linear → ReLU → Dropout → mixes pooled features before output.  
    - **Task-specific heads**:  
      - Classification: `classifier_max`, `classifier_median` (binary logits).  
      - Regression: `regressor` (continuous `pct_time_high`).  
  - **Targets**:  
    - Binary classification → `max_risk`, `median_risk`.  
    - Regression → `pct_time_high`.  
  - **Reasoning**:  
    - TCNs are causal by design → no future leakage.  
    - Dilated convolutions give long temporal memory without very deep stacks.  
    - Residual connections + LayerNorm = stable training, even with many blocks.  
    - Chosen as a modern, efficient alternative to RNNs (LSTM/GRU) and Transformers, showing a deliberate design choice.  
3. **Model Training `tcn_training_script.py`**
	- **Loss functions**: Binary cross-entropy (for classification heads), MSE (regression).
	-	**Optimiser**: Adam with learning-rate scheduler (reduce on plateau).
	-	**Regularisation**: dropout within TCN, gradient clipping, early stopping (patience=7).
	-	**Training loop logic**: forward → compute loss for all 3 tasks → backward → gradient clipping → optimiser update → validation → LR schedule.
  - **Reproducibility controls**: Fixed seeds for Python/NumPy/PyTorch, enforced deterministic CuDNN ops, saved hyperparameter config (`config.json`) and training/validation loss history (`training_history.json`) to ensure bit-for-bit reproducibility.
  - **Reasoning**: This phase ensures the model learns from patient sequences in a stable, controlled way. Shows deep learning maturity (correct loss functions, imbalance handling, monitoring).
4. **Validation (during training)**
	-	**Setup**: Patient-level validation split (not seen during training).
	-	**Metrics tracked**: Validation loss per epoch.
	-	**Logic**:
    -	When validation loss improves (validation loss ↓) → save checkpoint (`tcn_best.pt`).
    -	When validation loss stagnates/gets worse (validation loss ↑) → patience counter increases.
    -	Training stops early when overfitting begins (after 7 epochs of no improvement).
  - **Reasoning**: Validation ensures the model generalises and doesn’t just memorise training data.
5. **Generate Visualisations `plot_training_curves.py`**
  - **Input**: `trained_models/training_history.json`
  - **Output**: `plots/loss_curve.png`
  - **Features**:
    -	Plots Training vs Validation loss curves across epochs.
    -	Highlights the best epoch (red dashed line + dot).
    -	Text annotation shows epoch and validation loss value.
    -	Optional “overfitting region” annotation marks where validation loss rises.
    -	Grid and layout optimised for clarity and interpretability.
  - **Reasonings**: 
    - Transforms numerical loss logs into a visual understanding of model learning behaviour.
    - Focus on training behaviour and convergence, show whether the model converged, generalisation, where early stopping kicked in, whether overfitting started.
**Why Not Go Further**
- Skip ensembling, hyperparameter sweeps, AutoML, Transformers.
- Adds weeks of complexity with minimal recruiter payoff.
- The aim is a robust, interpretable, and reproducible baseline, not research-grade complexity.
- Phase 4 delivers a scientifically sound foundation for model evaluation and comparison.
**End Products of Phase 4**
- `trained_models/tcn_best.pt`→ best-performing model checkpoint.
- `trained_models/config.json` → hyperparameter and architecture record.
- `trained_models/training_history.json` → epoch-wise loss tracking.
- `plots/loss_curve.png` → visualisation of training vs validation loss.
- Debugged and reproducible training + validation pipeline.

### Goals
-	Build a robust, reproducible temporal dataset for TCN training.
-	**Handle messy timestamp-level EHR data**: missingness, variable-length sequences, mixed-type columns.
-	Ensure patient-level splits are stratified, reproducible, and free of leakage.
- Fix prior issues with categorical columns (e.g. consciousness_label) and dtype conversions that caused PyTorch crashes.

### Why Temporal Convolutional Network (TCN)?
- TCN is a modern sequence model that is complex enough to impress recruiters but not so niche or exotic that it looks gimmicky.
- **Why not other neural networks?**
  - **LSTM/GRU**: older, sequentially unrolled models → training is slow, vanishing gradients, weaker for long sequences.
  - **Transformers (BERT-style, GPT-style)**: dominant in NLP, too heavy for our dataset (100 patients, not millions of tokens). Would look like overkill and raise “did we really need this?” questions.
- **Why not more niche/exotic neural networks?**
  - **Neural ODEs (Ordinary Differential Equations)**: continuous-time dynamics models. Very niche, rarely used in production.
	- **Graph Neural Networks (GNNs)**: great if we are model hospital networks or patient similarity graphs, but not necessary for ICU vitals.
	- **WaveNet-style autoregressive models**: very heavy, Google’s original audio model, impractical for our dataset size.
	- **Attention-only architectures**: flashy but raise “did he just copy a paper?” questions.
- These are the ones that would look impressive to a PhD audience but gimmicky / overkill to recruiters, they won’t credit more for using these. They’ll think we're chasing buzzwords instead of showing clinical + ML maturity.
- **TCN is advanced, technically impressive, clinically relevant, and justified for the EHR time-series dataset**:
	-	**Causal convolutions** → predictions at time t only depend on past, not future.
	-	**Dilated convolutions** → exponential receptive field, captures long ICU sequences.
	-	**Parallel training** → faster and more scalable than RNNs.
	-	**Strong benchmark in clinical time-series papers** → credible.

### Purpose of Advanced Deep Learning TCN ML Model
1. Show I can handle sequential EHR data, including missingness, imputation, rolling features, and time alignment.
2. Provides a state-of-the-art deep learning benchmark for patient deterioration prediction.
3. Demonstrates mastery of temporal modelling architectures (causal dilated convolutions, residual blocks, pooling).
4. Captures temporal dynamics that LightGBM misses, such as deterioration trends and escalation patterns.
5. Handles long ICU stays efficiently without vanishing gradient problems.
6. **Portfolio-ready contrast**: proves I can go beyond baseline classical ML to advanced sequence-level modelling.
7. **Clinician-technologist edge**: shows I can not only build powerful models but also interpret them (via saliency maps).

### How Temporal Convolutional Network (TCN) Works
**Temporal awareness**:
-	Causal convolutions → at each time step, the model only looks backwards in time (no data leakage from the future).
-	Dilated convolutions → skip connections expand the receptive field exponentially, letting the model capture long patient histories without deep stacking (without needing hundreds of layers).
**Stable training**:
- Residual blocks → stabilise training, prevent vanishing/exploding gradients, making the deep temporal model easier to optimise.
**From sequences to predictions**:
-	Global pooling → compresses the sequence into a single fixed-length representation.
-	Dense output layer → produces prediction:
  -	Sigmoid activation → binary classification (max_risk, median_risk).
  -	Linear activation → regression (pct_time_high).
-	**Interpretability**: 
  - Saliency maps (e.g., gradient-based attribution) highlight which time periods and features most influenced the model’s prediction.

### Step 1: Dataset Preparation Pipeline
```text
Input: news2_features_timestamp.csv, news2_features_patient.csv
        │
        ▼
Preprocessing: prepare_tcn_dataset.py
- Setup and Directory Creation
- Load Dataset and Sort Chronologically
- Merge Patient-Level Outcomes
- Convert Targets to Binary for Stratification
- Patient-Level Stratification & Train/Val/Test Split
- Feature/Target Separation
- Normalisation (Z-Score for Continuous Features)
- Grouping into Per-Patient Sequences
- Padding/Truncation and Mask Creation
        │
        ├───────────────────────────────────────────────┐
        │                                               │
        ▼                                               ▼
Save Tensors + Masks (prepared_datasets/)          Save Padding & Feature Configuration (deployment_models/preprocessing/)
- train.pt (tensor: sequences for training)        - standard_scaler.pkl (mean/std from training set)
- val.pt (tensor: sequences for validation)        - padding_config.json (max_seq_len, padding rules)
- test.pt (tensor: sequences for testing)          - patient_splits.json (dictionary of patient train/val/test split)
- corresponding masks.pt                                │                
        │                                               ▼
        ▼                                          Used only at inference:
Used only during training for:                     - Apply same scaling to new patient sequences
- Model fitting                                    - Apply same padding/truncation rules
- Validation.                                      - Ensure input format matches trained TCN
- Testing
```

### Dataset Preparation Outputs & Purpose
```text
┌─────────────────────────────┐
│   Training / Validation     │
│   Usage: during TCN fitting │
└─────────────────────────────┘
        │
        ▼
src/ml_models_tcn/prepared_datasets/
├── train.pt          # Tensor: training sequences (batch, seq_len, num_features)
├── train_mask.pt     # Mask tensor: ignore padded values during training
├── val.pt            # Validation sequences
├── val_mask.pt       # Validation mask
├── test.pt           # Test sequences
└── test_mask.pt      # Test mask

┌─────────────────────────────┐
│ Deployment / Inference      │
│ Usage: for scoring new data │
└─────────────────────────────┘
        │
        ▼
src/ml_models_tcn/deployment_models/preprocessing/
├── standard_scaler.pkl      # Z-score scaler (continuous features)
├── padding_config.json      # Max sequence length, padding value, feature order
└── patient_splits.json      # dictionary of patient train/val/test split
```

| Artifact               | File Name              | Folder                      | Shape / Type                                   | Purpose                                              |
|------------------------|------------------------|-----------------------------|------------------------------------------------|------------------------------------------------------|
| Training tensor        | `train.pt`              | `prepared_datasets/`          | `(num_train_patients, max_seq_len, num_features)` | Model input for training patients                    |
| Validation tensor      | `val.pt`                | `prepared_datasets/`          | `(num_val_patients, max_seq_len, num_features)`   | Model input for validation patients                  |
| Test tensor            | `test.pt`               | `prepared_datasets/`          | `(num_test_patients, max_seq_len, num_features)`  | Model input for test patients                        |
| Training mask          | `train_mask.pt`         | `prepared_datasets/`          | `(num_train_patients, max_seq_len)`               | Mask padded timesteps (1 = real, 0 = padding)        |
| Validation mask        | `val_mask.pt`           | `prepared_datasets/`          | `(num_val_patients, max_seq_len)`                 | Mask padded timesteps                                |
| Test mask              | `test_mask.pt`          | `prepared_datasets/`          | `(num_test_patients, max_seq_len)`                | Mask padded timesteps                                |
| Scaler object          | `standard_scaler.pkl`   | `deployment_models/preprocessing/`  | sklearn StandardScaler object                   | Z-score scaling params (fit on training set only)    |
| Padding configuration  | `padding_config.json`   | `deployment_models/preprocessing/`  | JSON dict (max_seq_len, feature ordering, etc.) | Reproducibility of preprocessing pipeline            |
| Patient splits         | `ppatient_splits.json`  | `deployment_models/preprocessing/`  | JSON dict (train/validate/test) | Reproducibility of preprocessing pipeline            |
**Important**: All sequence tensors + masks are in prepared_datasets/.
Scalars/configs/splits are in deployment_models/scalers/ because they are needed for inference/deployment, not training data itself.

### Why Patient-Level Outcomes for Current TCN
#### What we’re doing
- **Features**: timestamp-level sequences (vitals, labs, missingness) → TCN sees full temporal dynamics.  
- **Targets**: patient-level outcomes (`max_risk`, `median_risk`, `pct_time_high`).  
- **Training**: every timestep in a patient’s sequence inherits the same label → TCN maps *whole sequence → patient-level label*.  
- This is stronger than LightGBM because it is sequence-aware, it can detect temporal patterns (spikes, trends, trajectories) that simple patient-level aggregates cannot.  
#### Why not per-timestep prediction
- True sequence-to-sequence TCN would label each timestamp with its own escalation risk (e.g., 1-hour lookahead).  
- **Delivers richer early-warning outputs, but comes with major challenges**:  
  - Requires fine-grained labels at each timepoint (rare in ICU data).  
  - Needs complex evaluation (per-timestep metrics, handling overlapping windows).  
  - Makes training fragile with only 100 patients → risk of severe overfitting.  
#### Why patient-level labels are the right choice now
- **Clinical alignment**: clinicians care if escalation ever happened (max/median risk), not the exact timestamp.  
- **Data constraints**: per-timestep outcomes are too sparse and noisy for reliable modelling.  
- **Portfolio clarity (easy-to-explain contrast)**:
  - **LightGBM**: aggregated features.  
  - **TCN**: full sequences, but still patient-level labels.  
- **Recruiter value**: demonstrates temporal modelling without overcomplicating.  
- **Feasibility**: avoids weeks of extra coding/debugging that add little portfolio benefit.  
#### Overall summary
- **TCN adds value**: captures patterns that LightGBM misses, while staying clinically relevant.  
- **Not full potential**: a sequence-to-sequence setup could deliver per-timestep risk scores, but is impractical here.  
- **Strategic choice**: safe with sparse data, clean comparison against baseline, portfolio-ready and recruiter-friendly.  
#### Strategic decision
- Stick with patient-level TCN outcomes. 
- **Include in README**: “This model could be extended to per-timestep risk prediction for richer early-warning capability. Due to dataset size and label sparsity, we instead demonstrate the temporal advantage at patient-level outcomes, already surpassing classical ML baselines.”  
- This shows both awareness of the full TCN potential and strategic judgement in using the right level of complexity for the dataset and portfolio.  

### Model Comparison Summary

| Aspect                | LightGBM (Classical ML)                   | Current TCN                               | Full TCN Potential                            |
|-----------------------|-------------------------------------------|-------------------------------------------|-----------------------------------------------|
| Input data format     | Patient-level aggregate stats              | Timestamp sequences (padded)               | Timestamp sequences (padded)                  |
| Input source          | news2_features_patient.csv                 | news2_features_timestamp.csv               | news2_features_timestamp.csv                  |
| Target variables      | max_risk, median_risk, % time high risk    | max_risk, median_risk, % time high risk    | Per-timestep risk (hourly escalation)         |
| Features used         | Mean, median, min, max, % missingness      | Vitals/labs with LOCF + missingness flags  | Same + explicit per-timestep outcomes         |
| Temporal handling     | None (static aggregates)                   | Temporal dynamics via dilated convolutions | Temporal dynamics + early warning predictions |
| Interpretability      | Feature importance (tabular, transparent) | Saliency maps over features & time         | Saliency + sequence-level attributions        |
| Complexity            | Low                                        | Moderate                                   | High                                          |
| Advantage             | Strong, credible baseline                  | Captures deterioration trajectories        | Captures trajectories + predicts escalation   |
| Portfolio story       | Shows baseline competency                  | Shows deep learning maturity               | Shows advanced temporal modelling edge        |

#### Chronological flow
**Forward Pass**
1. Input tensor `(batch, channels, seq_len)` → **BCL format**.
2. Layer 1 (Conv1 → LayerNorm → ReLU → Dropout): 
  - **Causal conv**: extract first layer of temporal patterns.
  - **LayerNorm**: normalise across channels at each timestep (convert to BLC format just for this step).
  - **ReLU activation**: add non-linearity.
  - **Dropout (during training only)**: randomly zero some outputs.
3. Layer 2 (Conv2 → LayerNorm → ReLU → Dropout): combine previous patterns into more complex ones.
4. **Downsample (1×1 conv)**: reshapes channels for residual addition.
5. **Residual addition**: adds original input (or downsampled input) to output. Give gradients a shortcut path back.
6. Output tensor `(batch, out_channels, seq_len)` → ready for next block or pooling or final classifier.
**Backward pass (not in this script)** 
1. Compute loss at the very end (prediction vs label).
2. Backprop starts: compute gradients of loss wrt outputs.
3. Gradients flow backward through residual add, then through conv2, conv1, etc.
4. Optimiser updates the weights (kernels, 1×1 conv, etc.) a little bit.
**Then you repeat this whole forward+backward cycle many times over the dataset.**
Forward pass → Loss → Backward pass (gradients) → Weight update (repeat many epochs until convergence)
**Intuition**:
- Each block learns a set of temporal detectors.
- Stacking multiple blocks increases the **effective receptive field**, combining short-term and long-term patterns.
- Works with **causal convolutions** to ensure predictions never “peek into the future”.

#### Why it matters for our TCN
- Builds **robust temporal features** from sequential ICU data.
- Prepares sequences for **masked pooling** and downstream **patient-level predictions** (classification/regression).
- Together with **causal convolutions**, ensures model captures **short-term spikes** and **long-term trends** across multiple vitals, improving predictive performance over classical models like LightGBM.

### TCN Model with Masked Pooling
#### Overview
- This step defines the full **Temporal Convolutional Network (TCN) model** that takes variable-length ICU time series data and produces **patient-level predictions**.  
- It stacks **TemporalBlocks** (causal convolutions + residual connections) to extract temporal patterns, then converts the sequence into a fixed-size vector using **masked mean pooling**.  
- Finally, the pooled vector is passed through optional **dense head(s)** and **task-specific heads** for classification and regression.

#### Key Concepts
1. **Temporal Blocks (Feature Extractors)**
- Each block = 2 causal convolutions + residual connection.
- **Dilation doubles** each block (1, 2, 4, …), exponentially increasing the **receptive field** (how far back in time the model can "see").
- Purpose: capture both short- and long-term dependencies in ICU sequences.

2. **Stacking Temporal Blocks**
- Blocks are stacked in `nn.Sequential`:
  - Output of one → input of the next.
- Final output shape after the TCN stack: `(B, C_last, L)`  
  - `B`: batch size (patients)  
  - `C_last`: number of channels (features learned by last block)  
  - `L`: sequence length  

3. **Masked Mean Pooling**
- ICU sequences vary in length → we pad them for batching.
- **Problem**: padding timesteps are not real, should not influence averages.
- **Solution**:  
  - Multiply by `mask` → zero out padding.  
  - Sum only valid timesteps.  
  - Divide by number of valid timesteps.  
- Result: a **single vector per patient** `(B, C_last)` summarising the whole sequence.

4. **Optional Dense Head**
- Purpose: add an extra small fully-connected layer before the outputs.
- Structure:  
  - Linear → ReLU → Dropout.  
- Role:  
  - Mixes learned features across channels.  
  - Adds extra non-linearity.  
  - Provides regularisation (dropout).
- Optional because sometimes you want this richer representation, sometimes you prefer direct pooled features.
- Controlled by `head_hidden`:  
  - If set (e.g. `64`) → dense head is used.  
  - If `None` → skip dense head, use pooled features directly.

5. **Task-Specific Heads**
- Separate linear layers for each prediction task.  
- Each head outputs **one number per patient**:
  - `classifier_max`: logit for maximum-risk classification.  
  - `classifier_median`: logit for median-risk classification.  
  - `regressor`: continuous regression (e.g. fraction of time high-risk).  

6. **Logits and Squeezing**
- **Logit** = raw score from classifier before sigmoid.  
  - Used with `BCEWithLogitsLoss` for numerical stability.  
- After linear layer, shape = `(B, 1)`.  
- `.squeeze(-1)` → `(B,)`, removes the trailing dimension of size 1.  
  - Needed because loss functions expect `(B,)` not `(B,1)`.

**Flow of Data**
1. Input: `(B, L, F)` (batch, sequence length, features per timestep).  
2. Permute → `(B, F, L)` for Conv1d.  
3. Pass through stacked TemporalBlocks → `(B, C_last, L)`.  
4. Permute back → `(B, L, C_last)`.  
5. Apply **masked mean pooling** → `(B, C_last)`.  
6. Optional dense head (if enabled) → `(B, head_hidden)`.  
7. Pass to **task-specific heads**:  
   - `classifier_max`: `(B,)`  
   - `classifier_median`: `(B,)`  
   - `regressor`: `(B,)`.  
8. Return dictionary of predictions (ready for loss functions).

#### Reasoning
- **Temporal blocks**: capture multi-scale temporal dependencies.  
- **Masked pooling**: ensures variable-length sequences map to fixed-size patient representations.  
- **Dense head**: optional flexibility to enrich representations.  
- **Task-specific heads**: handle multi-task learning cleanly.  
- **Squeeze**: makes outputs compatible with PyTorch loss functions.  

### Design Choices
1. **Device**
	-	GPU → massively faster for convolutions, especially with batches of sequences and multiple channels
	-	CPU is fine for small-scale testing, but for 3 TCNs on 96-hour sequences, it will be slow, limited parallelism.
  - **Decision**: Use GPU if available (cuda), fallback to CPU. Speedup is huge; batch training and multi-epoch runs become feasible.
2. **Loss Functions**
  - Measures the difference/error between predictions (3 model output numbers) and true labels (correct values for that bacth).
	-	Classification: BCEWithLogitsLoss → binary cross-entropy that takes raw logits from the model instead of probabilities (handles logits directly, which are numerically stable; avoids overflow in sigmoid).
    - Optionally handle class imbalance using pos_weight = (# negative samples / # positive samples).
	-	Regression: MSELoss (Mean Squared Error) → penalizes larger errors more than smaller ones. Standard for continuous values (pct_time_high).
  - **Decision**:
    - Classification → BCEWithLogitsLoss(pos_weight=...) if imbalance is significant, ensures the model does not ignore rare high-risk patients..
    - Regression → MSELoss().
3. **Optimiser**
	-	Adam → good default for deep networks, adaptive learning rates, handles sparse gradients well.
	-	Learning rate (LR) = 1e-3 → standard starting point for TCNs. Can reduce later if needed.
  - **Decision**: Adam(lr=1e-3)
  - **Reasoning**: Stable, widely used, doesn’t require manual LR decay initially.
4. **Scheduler (Optional)**
	-	ReduceLROnPlateau → reduces LR if validation metric stalls and stops improving.
	-	StepLR → reduces LR every N epochs.
	-	For efficiency: Start without scheduler for initial baseline training. Add ReduceLROnPlateau only if validation plateaus.
  - **Decision**: None initially (optional later: ReduceLROnPlateau)
  - **Reasoning**: Keeps pipeline simple; avoids premature optimisation while debugging.
5. **Batch Size & Epochs**
	-	Batch size: 16–64 depending on memory. Start: 16–32 (manageable on typical laptop GPU or CPU).
	-	Epochs (one full pass through the entire training dataset): start small (10–20) for testing, increase to 50–100 once stable.
	-	Gradient clipping (optional) if exploding gradients in deep TCNs with long sequences: torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
  - **Decision**: Batch Size 16–32, Epochs 10–20 initially → scale to 50–100, Gradient Clipping max_norm=1.0 optional
  - **Reasoning**: Balances stability, memory, and speed.
6. **Validation / Early Stopping**
	-	Monitor validation metric:
    - Classification → ROC-AUC (focus on ranking), F1 (balance precision/recall), accuracy.
    -	Regression → RMSE (mean squared error), R² (proportion of variance explained).
	-	Early stopping → stop if metric doesn’t improve for patience epochs (5–10).
  - **Decision**: Early Stopping Patience 5–10 epochs
	-	**Reasoning**: Allows stopping training if metrics do not improve → prevents overfitting, saves training time, ensures best model on unseen data.
7. **Dataset / Dataloader**
	-	Load tensors + masks: train.pt, val.pt, test.pt + masks.
	-	Create TensorDataset and DataLoader → handles batching, shuffling.
  - Mask used to ignore padded timesteps during loss calculation.
  - **Reasoning**: Ensures correct sequence-level learning. Avoids padding noise.
8. **Forward Pass Logic**
	-	Model input: (B, L, F)
	-	Permute to (B, F, L) for PyTorch Conv1d
	-	Forward through stacked TCN blocks → masked mean pooling → optional dense head → task heads → outputs
  - **Reasoning**: Preserves causal structure, produces single patient-level vector per task (classification/regression).
9. **Backpropagation**
  - Compute loss per task:
    - loss_max = criterion_max(logit_max, target_max) → How wrong the model is for max_risk classification.
    - loss_median = criterion_median(logit_median, target_median) → How wrong for median_risk classification.
    - loss_reg = criterion_reg(regression, target_pct) → How wrong for pct_time_high regression.
    - total_loss = loss_max + loss_median + loss_reg → Combine all tasks into a single loss.
  - Backward pass (backprop):
    -	total_loss.backward() → PyTorch calculates gradients, i.e., how much each weight contributed to the error.
    -	optimizer.step() → Update the weights according to the gradients (reduce the error).
    -	optimizer.zero_grad() → Clear old gradients; otherwise, PyTorch accumulates them, ensures that each batch’s gradients are computed independently.
  - **Reasoning**:
    - Loss functions (BCE, MSE) measures the difference/error between model output predictions and true labels. 
    - PyTorch automatically computes the gradient of the loss with respect to every trainable parameter in the model.
    - A gradient tells the direction and magnitude by which each parameter (weight) should change to reduce the loss.
    - By computing gradients on total_loss, the model learns jointly across all tasks, joint optimisation across all three tasks.
    -	Gradients tell model how to adjust kernels to reduce error to make predictions closer to true values.
10. **Metrics**
	-	Classification: 
    - ROC-AUC → ranks predictions correctly (high vs low risk). Best metric for imbalanced data.
    -	F1-score → balances precision & recall, useful if false positives/negatives matter.
    -	Accuracy → simple overall correctness; less sensitive to imbalance.
	-	Regression: 
    - RMSE → standard deviation of errors. Penalizes large mistakes.
    - R² → how much variance in target is captured by model.
	-	**Reasoning**:
    -	Allows direct comparison with LightGBM baseline.
    -	Multiple metrics ensure a thorough, clinically-relevant evaluation.
  

### TCN Forward & Backward Pass (Joint Multi-Task)

```text
             Forward pass
     Input sequences (B, L, F)
                 │
                 ▼
     Stacked TemporalBlocks (TCN)
   ┌───────────────┐
   │ Causal Conv1D │  → extracts temporal features
   │  + ReLU       │  non-linear patterns
   │  + LayerNorm  │  stabilises training
   │  + Dropout    │  prevents overfitting
   └───────────────┘
                 │
                 ▼
       Masked mean pooling
   (collapses variable-length sequences → patient-level vector)
                 │
                 ▼
        Optional dense head
   (Linear → ReLU → Dropout)
   → combines features across channels before outputs
                 │
                 ▼
        Task-specific heads
 ┌────────────┬─────────────┬─────────────────┐
 │ Max        │ Median      │ Regression      │
 │ classifier │ classifier  │ (pct_time_high) │
 │ logits     │ logits      │ continuous      │
 └────────────┴─────────────┴─────────────────┘
                 │
                 ▼
           Compute Losses
loss_max = BCEWithLogitsLoss(logit_max, target_max)
loss_median = BCEWithLogitsLoss(logit_median, target_median)
loss_reg = MSELoss(regression, target_pct)
total_loss = loss_max + loss_median + loss_reg

             Backward pass
total_loss.backward()    # compute gradients
        │
        ▼
Gradients flow backward:
- Task heads → Dense head → Masked pooling → TCN blocks
- Each weight receives gradient: ∂Loss/∂weight
- Indicates direction to adjust weights to reduce error
        │
        ▼
optimizer.step()         # update weights using gradients
optimizer.zero_grad()    # reset gradients for next batch
```
**Why It Works**
1. **Joint optimisation**: 
  - The three tasks contribute to the same network’s weights. 
  - The network learns shared temporal patterns useful for all tasks.
2. **Gradients**: 
  - A gradient tells the direction and magnitude by which each parameter should change to reduce the loss.
  - Measure how much each weight contributed to the error.
    -	Large gradient → weight needs bigger adjustment.
    -	Small gradient → minor adjustment.
3. **Iterative improvement**: 
  - Repeating forward + backward passes over many batches (epochs) gradually reduces loss → better predictions.

### Output Directory finalised
```text
ml_models_tcn/
│
├── tcn_model.py
├── tcn_training_script.py
├── plot_training_curves.py
│
├── trained_models/
│   ├── tcn_best.pt                 ← best weights
│   ├── training_history.json       ← per-epoch loss log
│   ├── config.json                 ← hyperparameters + settings
│
├── prepared_datasets/              ← .pt tensors
├── deployment_models/preprocessing/
│   └── patient_splits.json, scalers, padding_config.json
└── plots/
    └── loss_curve.png              ← visualisation output
```

#### What Happened in the Run
**Epoch 1–3:**  
  - Both training and validation loss decreased steadily (Train: 1.46 → 0.98, Val: 1.12 → 0.96).  
  - The model was learning generalisable temporal patterns early in training.
**Epoch 4–5:**  
  - Training loss continued to fall (0.93 → 0.88), but validation loss began to rise (0.96 → 0.98).  
  - Marks the onset of overfitting — model starting to specialise too much on the training distribution.
**Epoch 6–11:**  
  - Training loss kept decreasing (0.79 → 0.53) while validation loss rose consistently (1.00 → 1.26).  
  - Confirms strong overfitting beyond epoch 4–5, the model is memorising rather than generalising.
**Early Stopping:**  
  - Triggered at epoch 11 after no improvement in validation loss for several epochs.  
  - Automatically preserved the best model weights (around epoch 3–4), ensuring the optimal generalising model was saved.

#### Reproducibility and Determinism
**To verify the stability and scientific integrity of the experiment, this final run locked all sources of stochasticity and documented expected behaviours of inherently random processes:**

| Source                     | Controlled via / Expected Behaviour                                           | Purpose                                                   |
|-----------------------------|-------------------------------------------------------------------------------|---------------------------------------------------------------------------|
| Python built-in random      | `random.seed(SEED)`                                                           | Ensures deterministic behaviour in any Python random operations (e.g., shuffling IDs). |
| NumPy                       | `np.random.seed(SEED)`                                                        | Fixes randomness in array sampling and preprocessing steps.              |
| PyTorch (CPU)               | `torch.manual_seed(SEED)`                                                     | Fixes model weight initialisation and CPU tensor operations.             |
| PyTorch (GPU)               | `torch.cuda.manual_seed(SEED)` + `torch.cuda.manual_seed_all(SEED)`          | Fixes GPU computation order and initialisation.                          |
| CuDNN backend               | `torch.backends.cudnn.deterministic = True`<br>`torch.backends.cudnn.benchmark = False` | Forces deterministic convolution algorithms and disables autotuning variance. |
| Random weight initialisation | Different starting points → slightly different convergence                  | Prevents model from getting stuck in one local minimum                   |
| Shuffled mini-batches       | Each epoch sees data in a new order                                           | Better generalisation                                                      |
| Dropout                     | Randomly deactivates neurons                                                  | Regularisation; reduces overfitting                                       |
| GPU numerical nondeterminism| Minor floating-point differences                                              | Harmless                                                                  |
**This combination eliminates stochastic variance, ensuring that**:
- The model, losses, and saved weights are identical across reruns on the same machine.
- Experimental results can be reliably replicated, meeting scientific and academic reproducibility standards.
- Any remaining variability in unconstrained runs (e.g., weight initialisation, mini-batch shuffling, dropout) is now controlled for, producing a true reproducible baseline.

#### Why the Final Run Was Locked
- Prior runs showed nearly identical training dynamics despite random variation, implying the underlying model and data are stable.
- To make this baseline scientifically defensible and comparable, it was essential to remove the remaining stochastic noise (due to random initialisation and shuffling).
- **Locking all seeds and enforcing deterministic computation ensures**:
  - Every future run on the same code yields the exact same losses.
  - Changes in results can now only come from intentional modifications (architecture, hyperparameters, or data).
  - **We have a true reference baseline**: a frozen, reproducible experiment for paper-level reporting or future benchmarking.

#### Conclusion
- The final reproducible run confirms that the entire TCN pipeline (data loading → model → training loop → early stopping) is working correctly and deterministically.
- Overfitting behaviour remains stable across all runs, reinforcing that the model and hyperparameters are sound.
- The experiment is now fully traceable, interpretable, and repeatable.
- This final configuration can be safely tagged as the reproducible baseline version of the TCN training script.

### Summary
**Phase 4 completed successfully, the TCN model training pipeline is now**: 
  - Deterministic, auditable, scientifically reproducible, visualisable, interpretable, and ready for formal evaluation.
**Final Outputs:**
1. `trained_models/tcn_best.pt` → best-performing model weights.  
2. `trained_models/config.json` → hyperparameters and model configuration.  
3. `trained_models/training_history.json` → full epoch-wise training record.  
4. `plots/loss_curve.png` → visualised training vs validation loss curves.  
5. Console logs → with early stopping, sanity checks, and tensor validation.
**This marks the completion of Phase 4**:  
- From reproducible model training to visual interpretability.

## Phase 5: Evaluation & Baselines (Steps 1-4)
**Goal: complete all model evaluation scripts and establish scientifically valid baselines for comparison. Phase 5 marks the end of model retraining, validation, and metrics generation; setting the stage for direct comparison between LightGBM (non-temporal) and TCN (temporal) models under identical conditions.**

1. **Centralised Metrics Utility (`evaluation_metrics.py`)**
  - **Purpose:**  
    - Before evaluation, establish a reusable, reproducible, unified metrics framework.
    - Create a single module defining all key metric functions to ensure that every model (NEWS2, LightGBM, TCN) uses identical computation. methods.
  - **Implements:**
    - `compute_classification_metrics(y_true, y_prob, threshold=0.5)`
    - `compute_regression_metrics(y_true, y_pred)`
  - **Reasoning:**
    - Provides a single trusted source of truth for performance metrics, guarantees consistency across model evaluations.  
    - Prevents metric drift or implementation bias.  
    - Simplifies later scripts by standardising output format across scripts → metrics are imported, not duplicated.  
    - Ensures LightGBM, TCN, and any future models are evaluated on identical statistical definitions.  


2. **Final TCN Evaluation on Test Set (`evaluate_tcn_testset.py` + `evaluate_tcn_testset_refined.py`)**
  - **Purpose:** 
    - Run the **baseline TCN (Phase 4)** and the **refined TCN (Phase 4.5)** on the **same held-out test split** to ensure a fair, unbiased comparison.  
    - Quantify performance across classification and regression outputs while maintaining full reproducibility.  
    - Introduce **scientific transparency**, **calibration**, and **threshold tuning** into the evaluation pipeline.
  - **Process:**
    - **Test Data Reconstruction**
      - Rebuild test targets (`y_test`) dynamically from patient CSV + ordered JSON split.
      - Load preprocessed test tensors (`x_test.pt`, `mask_test.pt`).
    - **Model Loading**
      - Load model architecture from `tcn_model.py` using hyperparameters from respective config files (`config.json`, `config_refined.json`).
      - Load corresponding model trained weights (`tcn_best.pt`, `tcn_best_refined.pt`) into the model.
      - Move model to device (CPU/GPU) and set `model.eval()` for deterministic inference.
    - **Inference**
      - Run inference under `torch.no_grad()` for deterministic, memory-efficient evaluation. 
      - **Evaluate on test set**: Completely unseen patients (15), final unbiased check.
      - **Collect predictions for**:
        - `logit_max` (max-risk binary classification)
        - `logit_median` (median-risk binary classification)
        - `Regression` (`pct_time_high`, continuous prediction), **`log-space` for refined model evaluation ONLY**
  - **Post-processing:**  
    -	Convert logits → probabilities using `torch.sigmoid(logits)` for binary tasks.
    - Save raw predictions and probabilities (e.g. `results/tcn_predictions.csv` + `tcn_predictions_refined.csv`) for reproducibility and traceability.
	- **Apply inverse regression transform (refined model `evaluate_tcn_testset_refined.py` only)**:
    - Original y values were tranformed using log1p during training, so we must convert back from log-transformed quantities to percentages.
    - Apply `expm1()` to NumPy arrays at the end of evaluation flow, right before computing regression metrics:
    ```python
    y_pred_reg = np.expm1(y_pred_reg)
    y_true_reg = np.expm1(y_true_reg)
    ```
    - The model was trained on `log1p(pct_time_high)` targets → this reverses the transformation.  
    - True test values remain in raw form.  
    - Metrics are thus computed on both:
      - **Log-space:** internal model correctness  
      - **Raw-space:** clinical interpretability 
  - **Reasoning:**
    - Ensures **reproducible, deterministic inference** on unseen patients (no dropout or randomness).  
    - Provides both **scientific validation (log-space)** and **clinically interpretable metrics (raw-space)**.  
    - Enables direct, fair comparison between:
      - **Baseline TCN (Phase 4)**  
      - **Refined TCN (Phase 4.5)**  
      - **LightGBM and NEWS2** baselines (later in Phase 5).  
    - Creates reproducible, publication-ready evaluation outputs with transparent calibration evidence.
    - Forms the analytical bridge between model training behaviour and real-world predictive reliability.

3. **Compute Metrics (`evaluate_tcn_testset.py` + `evaluate_tcn_testset_refined.py`)**
  - **Purpose:** 
    - Quantitatively evaluate classification and regression performance with full transparency.  
    - Diagnose and correct bias using correlation, calibration, and threshold tuning.  
  - **Process:**
    - Call the functions `compute_classification_metrics` and `compute_regression_metrics` from `evaluation_metrics.py` for consistent evaluation across models when computing metrics.
      -	**Classification targets (`max_risk_binary, median_risk_binary`)**:
        - ROC-AUC → ranking ability (threshold-independent)
        - F1-score → balance between precision & recall
        - Accuracy, Precision, Recall → diagnostic insight
      -	**Regression target (`pct_time_high`)**:
        - RMSE → absolute prediction error  
        - R² → explained variance
    - Save metric outputs into structure JSON (`tcn_metrics.json`) containing:
      - Classification: max-risk, median-risk
      - Regression: pct_time_high
      - Inference time and reproducibility info
    - **Refined model `evaluate_tcn_testset_refined.py` only:** 
      - **Inspect regression predictions:**
        - Prints the range (min–max) and mean of regression outputs in log-space.
        - Helps verify:
          - The numerical stability of model predictions (no extreme outliers)
          - That predictions are within a plausible range consistent with log1p-transformed targets
          - Whether the model is over- or under-predicting on average before calibration
      - **Threshold Tuning (Median-Risk Head Only):**
        - Although `pos_weight` handled imbalance during training, post-training threshold tuning was needed to correct decision bias.  
        - Validation set used to find optimal threshold:
          ```python
          best_thresh_median, best_f1 = find_best_threshold(y_val_median, val_prob_median)
          ```
        - Prints optimal threshold for F1 metric on validation set
        - Max-risk head retained threshold = 0.5 (performance already optimal).
      - **Correlation Analysis:** 
        - Compute Pearson correlation in both log- and raw-space
        - Differentiate between trend accuracy (correlation) and scale bias (R²). 
        - Indicates whether model is directionally correct but biased, and thus calibration can fix the issue without retraining
        ```python
        corr_log = np.corrcoef(y_true_log, y_pred_log)[0,1]
        corr_raw = np.corrcoef(y_true_raw, y_pred_raw)[0,1]
        ```
      - **Post-hoc Calibration:**
        - Apply simple linear regression on log-space predictions: `y_true_log ≈ a * y_pred_log + b`
        - Corrects systematic over- or under-estimation.
        - Generates calibrated predictions:
        ```python
        y_pred_reg_log_cal = a * y_pred_reg_log + b
        y_pred_reg_raw_cal = np.expm1(y_pred_reg_log_cal)
        ```
      - **Visual Outputs**
        - Two diagnostic plots to visualise **regression calibration quality** and confirm that numerical bias has been corrected post-hoc.  
        - These plots provide **visual evidence**: pre-calibration is used to diagnose whether the model’s errors are random or biased, post-calibration verifys that the model’s predictions are now aligned with ground truth without requiring retraining.
        - **Pre-calibration (`tcn_regression_calibration_logspace.png`):** 
          - Scatter plot of true vs predicted log-space regression values, red dashed diagonal represents perfect calibration (`y = x`).
          - Highlights any consistent offset or slope mismatch, indicating whether the model systematically over- or under-predicts.
        - **Post-calibration (`tcn_regression_calibration_comparison_logspace.png`):** 
          - Overlays predictions before (blue) and after (orange) calibration against the ideal diagonal. 
          - Shows how the linear correction (`y_true_log ≈ a*y_pred_log + b`) realigns predictions with the ground truth.
          - Validates the effectiveness of the calibration step. 
        - Both plots provide **transparent diagnostic evidence** of model validity and calibration success.  
        - They complement the quantitative metrics (RMSE, R², correlation) by showing the same improvement visually.  
      - **Metric Outputs**
        - All results compiled into structured JSON (`tcn_metrics_refined.json`) containing:
          - Classification: max-risk, median-risk, **median-risk (tuned)**
          - Regression: **pre- and post-calibration (log + raw)** 
          - Inference time and reproducibility info
        - Calibration plots (`tcn_regression_calibration_logspace.png` + `tcn_regression_calibration_comparison_logspace.png`) saved as visual interpretability for publication or audit documentation.
  - **Reasoning:**  
    - Both scripts output evaluation metrics ready for comparison in the rest of Phase 5.
    - In the refined evaluation script:
      - **Correlation + Calibration + Tuning** create a complete diagnostic chain:
        - Correlation → detects structural validity  
        - Calibration → fixes numeric bias  
        - Threshold tuning → optimises classification decision boundary  
      - Prevents misinterpretation of R² or F1 as signs of model failure.  
      - Completes the **analytical lifecycle of the refined TCN**, preparing it for comparative analysis with LightGBM and NEWS2 in the final benchmarking phase.
    - Establishes a **scientifically interpretable and reproducible evaluation pipeline** ready for direct comparison with baseline models.

4. **LightGBM Baseline**
  - **Purpose:**  
    - Retrain and evaluate the final LightGBM models using patient-level NEWS2 features on the **same 70/15 train/test split** used for the TCN model.  
    - Establish a **controlled, production-grade baseline** for non-temporal models (classical ML) that is **directly comparable** to TCN predictions.  
    - Ensures reproducibility, transparency, and consistent evaluation by using the **best validated hyperparameters from Phase 3**, while aligning patient inclusion and feature inputs with the TCN evaluation.  
    - Converts the original full-dataset deployment models into a **scientifically controlled experimental baseline** for fair cross-model comparison.
  - **Steps:**
    - Load patient-level NEWS2 features (`news2_features_patient.csv`) and fixed train/test splits (`patient_splits.json`).  
    - Load best hyperparameters from Phase 3 (`best_params.json`) for each target (`max_risk`, `median_risk`, `pct_time_high`).  
    - Recreate binary targets (`max_risk_binary`, `median_risk_binary`) for classification tasks.  
    - Retrain LightGBM models on the 70-patient training set using the tuned hyperparameters:
      - Classification: `lgb.LGBMClassifier(**params, class_weight="balanced")`
      - Regression: `lgb.LGBMRegressor(**params)`
      - Save each model as `.pkl` in `lightgbm_results/`.  
    - Evaluate retrained models on the 15-patient test set:
      - Classification → `model.predict_proba(X_test)[:, 1]`
      - Regression → `model.predict(X_test)`
      - Store true vs predicted values in `preds_dict`.  
    - Save predictions as `lightgbm_predictions.csv`.  
    - Compute performance metrics using `compute_classification_metrics()` and `compute_regression_metrics()`.  
    - Save metrics as `lightgbm_metrics.json` and write concise evaluation summary to `training_summary.txt`.
  - **Outputs and Explanation:**  
    - `*_retrained_model.pkl` → Contains the trained LightGBM model objects for each target; can be loaded later for inference or evaluation.  
    - `lightgbm_predictions.csv` → A combined table of true labels and predicted outputs for all test patients; ensures alignment between predictions and patient IDs.  
    - `lightgbm_metrics.json` → Stores detailed performance metrics for classification (AUROC, F1, Accuracy, Precision, Recall) and regression (RMSE, R²) in a machine-readable format for reproducibility.  
    - `training_summary.txt` → Provides a human-readable summary of dataset characteristics, feature count, per-target metrics, and hyperparameters; serves as a full record for audit and reporting.
  - **Reasoning:**  
    - Original Phase 3 LightGBM models were trained on all 100 patients (deployment models) and are **not directly comparable** to the TCN evaluation, which uses a 70/15/15 split.  
    - Retraining on the same split ensures:
      - **Identical patient inclusion/exclusion** across models.  
      - **Consistent feature inputs** for each patient.  
      - **Direct comparability** of classification and regression metrics.  
    - Using the **best hyperparameters from Phase 3** ensures that the LightGBM models are in their **strongest validated configuration**, analogous to how the TCN was optimised (class weighting, log-transform, threshold tuning, calibration).  
    - Produces clean, aligned prediction outputs ready for direct cross-model analysis.
    - This approach produces a **controlled, reproducible experimental baseline**, enabling fair head-to-head comparison with TCN across all targets, while maintaining transparency and methodological rigor.

**End Products of Phase 5**
- **Fully evaluated multi-task TCN models (baseline + refined):** Both models trained and validated with two classification heads (`max_risk`, `median_risk`) and one regression head (`pct_time_high`), evaluated on an unseen test set to confirm generalisation and calibration.
- **Retrained LightGBM baselines:** Finalised LightGBM models retrained on the *same train/test split* as the TCN to ensure **direct, fair comparison** across all targets using identical data conditions.
- **Structured metrics and outputs:** Comprehensive JSON and CSV outputs for **baseline TCN**, **refined TCN**, and **LightGBM**, enabling transparent comparison across all tasks and ensuring reproducibility of results.
- **Pre- vs Post-calibration visual diagnostics:** Visual evidence demonstrating calibration effectiveness for regression tasks in the refined TCN; proving numerical bias correction without retraining.
- **Unified evaluation utilities and summaries:** Standardised metric computation (`evaluation_metrics.py`) and consistent reporting pipelines across models, ensuring reproducibility and methodological integrity.

**Why Not Further**
- **Phase Completion:** Phase 5 marks the **end of all evaluation scripts** → all models have been trained, evaluated, and benchmarked under controlled and reproducible conditions.
- **Purpose Fulfilled:** This phase ensures **scientific rigour**, **consistency**, and **fair comparability** of outputs across different model types (classical ML vs deep temporal).
- **No Need for Further Tuning:** 
  - Additional parameter tuning, feature importance, or interpretability steps are unnecessary at this stage.
  - The **LightGBM models** were retrained purely for **comparability**, not deployment, since their deployment-ready versions (trained on all 100 patients) already exist from **Phase 3**.  
  - The **TCN models** were already **fully optimised** → weighted classes, log-transform for regression, post-hoc calibration, and threshold tuning (only for median-risk) were applied in a **pragmatic and disciplined** way.  
  - Max-risk threshold was deliberately left unchanged to preserve integrity and avoid overfitting.  
  - These adjustments balanced performance optimisation with scientific caution → avoiding unnecessary complexity or bias introduction.
- **Scientific Justification:**
  - Both models now represent their **best validated configurations**:
    - LightGBM: best hyperparameters reused from Phase 3 → efficient, interpretable, strong baseline.  
    - TCN: refined through controlled calibration, not arbitrary tuning → accurate yet stable.  
  - Both evaluated under identical conditions → **methodologically fair and statistically valid** comparison.
- **Readiness for Next Phase:** The pipeline is now ready for **Phase 6: Visualisation, Comparison, and Finalisation**, which will generate comparative plots, interpretability analyses, and deployment-lite demonstration.

**Unique Technical Story**
- A coherent, evidence-driven progression **(Clinical Baseline → Tabular ML → Deep Temporal Model):** 
  1. **NEWS2:** The established clinical ground truth baseline.  
  2. **LightGBM:** Classical, interpretable ML capturing patient-level risk patterns.  
  3. **TCN:** Advanced temporal deep learning model capturing sequential deterioration dynamics.  
- **Narrative Significance:** This evolution demonstrates **scientific discipline**, **methodological transparency**, and **applied clinical ML expertise** → moving from traditional scoring systems to modern AI, while maintaining interpretability, reproducibility, and fairness in every comparison.

**Metric Table:**

| Metric       | Type          | Purpose / Reasoning |
|--------------|---------------|-------------------|
| ROC-AUC      | Classification | Measures model’s ability to rank positives above negatives; threshold-independent. Useful for imbalanced datasets. |
| F1-score     | Classification | Harmonic mean of precision and recall; balances false positives & false negatives. |
| Accuracy     | Classification | Proportion of correct predictions; simple overall performance measure. |
| Precision    | Classification | TP / (TP + FP); indicates how often predicted positives are correct. |
| Recall       | Classification | TP / (TP + FN); indicates how well actual positives are captured. |
| RMSE         | Regression     | Root Mean Squared Error; quantifies average prediction error magnitude. |
| R²           | Regression     | Coefficient of determination; proportion of variance in true labels explained by predictions. |
**Reasoning / Benefits:**  
- Keeps metric logic consistent across TCN, LightGBM, and NEWS2.
- Guarantees reproducibility and comparability across all models (if you compute metrics differently in each script, results can’t be compared fairly).  
- Prevents metric drift or subtle implementation biases.  
- **Simplifies later evaluation scripts**:
  - Metrics are imported, not reimplemented
  - Prevents code duplication
- Makes maintenance easy if later want to add more metrics (e.g., AUPRC or MAE).

**Model Outputs and Metric Mapping:**
- Each model head corresponds to a task:
| Model head         | Purpose                                           | Output key      | Task Type         | Metrics Used         |
|------------------|-------------------------------------------------|----------------|-------------|----------------|
| classifier_max    | Predicts severe deterioration (max risk)       | `logit_max`     | Binary (logit) | ROC-AUC, F1, Accuracy     |
| classifier_median | Predicts moderate deterioration (median risk)  | `logit_median`   | Binary (logit) | ROC-AUC, F1, Accuracy    |
| regressor         | Predicts fraction of time in high-risk zone   | `regression`     | Continuous   | RMSE, R²        |

**Metric Interpretation & Diagnosis**

| Task | Metrics | Key Finding | Likely Cause | Severity | Fixable? |
|------|----------|--------------|---------------|-----------|-----------|
| **Max Risk** | AUC = 0.577, F1 = 0.929, Acc = 0.867 | Predicts positives well (F1 = 0.929), but AUC suggests imbalance and possible overfitting to dominant class. | Moderate class imbalance | Medium | Yes |
| **Median Risk** | AUC = 0.722, F1 = 0.000, Acc = 0.800 | Model outputs only negatives, no positive predictions (F1 = 0). | Severe imbalance (minority class ignored) | Medium–High | Yes |
| **Regression (pct_time_high)** | RMSE = 0.135, R² = −1.586 | Model predicts near-mean values (R² ~ -1.5); fails to capture variance. | Weak feature–target correlation, skewed distribution | High | Potentially |

**Technical Interpretation**
**Classification (Max & Median Risk)**
- **F1 imbalance pattern:**  
  - Max Risk → high F1 but low AUC → model predicts mostly positives.  
  - Median Risk → F1 = 0 → model predicts all negatives.  
- Both indicate **imbalanced dataset effects**, where the Binary Cross-Entropy loss is dominated by majority classes.  
- The small dataset size exacerbates this, as the model can minimise loss simply by ignoring rare outcomes.
**Regression (`pct_time_high`)**
- **Negative R²** means model predictions cluster around the mean → low variance → negative R² (worse than predicting the mean).
- Model outputs are nearly constant → underfitting due to conservative modelling.  
- **Likely due to**:  
  - **Skewed / zero-inflated target** → target variable not evenly distributed.
  - **Low variance in data** → all patients are within a narrow range, even minor prediction deviations appear large relative to that variance.
  - **MSE loss penalising outliers** → MSE loss penalises large errors disproportionately, in a skewed dataset, a few high-value outliers dominate the loss → model avoids over-predicting high-risk cases and predicts near the mean → “risk-averse” behaviour makes the model conservative, producing underfitted, low-variance outputs.

**Recommended Fixes**
| Problem | Corrective Strategy | Expected Effect |
|----------|---------------------|-----------------|
| **Max Risk (low AUC despite high F1)** | Calibrate decision threshold (e.g. 0.3–0.4) or apply **Platt scaling / isotonic calibration** | Improves discrimination (AUC) without sacrificing recall |
| **Median Risk imbalance** | Add `class_weight` to BCE loss or use oversampling (e.g. SMOTE / positive upweighting) | Boosts recall & F1 for minority class |
| **Regression underfitting** | Apply log/sqrt transform or switch to **Huber / MAE** loss | Increases R² stability, reduces penalty from outliers |

**Summary**
- The TCN is functioning correctly, it loads, predicts, and saves reproducibly.  
- The failures are statistical, not architectural.  
- These results provide a clear baseline for retraining and comparison against LightGBM/NEWS2.  
- With class weighting and loss adjustments, both classification and regression heads should improve.

## Phase 4.5: Diagnostics and Retraining (Steps 1-4)
**Goal: Address performance issues from Phase 4 through targeted, data-level corrections; improving class balance and regression stability; while preserving full reproducibility and comparability for Phase 5 evaluation.**

1. **Full TCN Diagnostic Analysis (`tcn_diagnostics.py`)**
  - **Purpose:** Identify the root causes of poor median-risk and regression performance in Phase 4.  
  - **Process:**  
    - Loaded Phase 4 model and patient-level data; reproduced predictions on validation and test sets.  
    - Conducted threshold sweeps, probability histograms, and regression diagnostics (RMSE/R²).  
    - Verified dataset imbalance and regression skew using label distribution plots.  
  - **Outputs:**  
    - **Summary file:** `tcn_diagnostics_summary.json` → consolidated summary of classification and regression metrics
    - **12 diagnostic plots saved to `/plots/`, grouped as follows:**  
      - **Probability Histograms for classification (4):**  
        - `prob_hist_max_val.png`, `prob_hist_max_test.png` → distribution of predicted probabilities for Max Risk on validation/test set (confirms strong class separation and calibration stability). 
        - `prob_hist_median_val.png`, `prob_hist_median_test.png` → median-risk probability spread on validation/test set (shows prediction collapse near 0 due to class imbalance).
      - **Regression Diagnostics scatter plots (4):**  
        - `val_reg_scatter.png`, `test_reg_scatter.png` → predicted vs. true regression values on validation/test set (demonstrates modest alignment, signal capture due to positive R²≈ 0.2)
        - `val_reg_residuals.png`, `test_reg_residuals.png` → residuals vs. predictions on validation/test set; checks for bias or heteroscedasticity (errors evenly distributed around zero, stable residuals).  
      - **Label Distribution histograms (4):**  
        - `dist_test_max_risk_labels.png`, `dist_test_median_risk_labels.png` → shows label distribution for Max Risk (mostly positives; highlights class skew) and Median Risk (confirms minority-positive class imbalance ~20%).   
        - `dist_test_pct_time_high_true.png`, `dist_test_pct_time_high_pred.png` → true regression target distribution (visualises heavy right skew in raw values) and predicted regression output distribution (shows compressed range consistent with underfitting to skewed target)
  - **Reasoning:**  
    - Revealed that performance issues were **data-driven (imbalance, skew)**, not architectural; confirming the need for targeted reweighting and transformation in retraining.

2. **Fix Evaluation Script (`evaluate_tcn_testset.py`)**
  - **Purpose:** Correct ROC-AUC inconsistencies caused by misaligned test patient ordering.  
  - **Process:**  
    - Replaced unordered `set()` indexing with ordered `.loc[test_ids]` for perfect alignment.  
    - Recomputed metrics with corrected label order.  
  - **Outputs:**  
    - Accurate and reproducible metrics (ROC-AUC), consistent across all scripts:  
      - Max Risk: AUC = 0.923, F1 = 0.929  
      - Median Risk: AUC = 0.778, F1 = 0.000  
      - Regression: RMSE = 0.077, R² = 0.166 
  - **Reasoning:**  
    - Ensured **true one-to-one matching** between predictions and ground truths; restoring metric validity and confirming earlier diagnostic interpretations.

3. **Retrain TCN Model (`tcn_training_script_refined.py`)**
  - **Purpose:** Implement minimal, scientifically controlled fixes while keeping architecture constant.  
  - **Process:**  
    - **Regression fix:** Applied `log1p(y)` before tensor creation to stabilise variance.  
    - **Classification fix:** Computed dynamic `pos_weight = neg/pos` for median-risk BCE loss.  
    - Updated config and training outputs with metadata for reproducibility.  
  - **Outputs:**  
    - All outputs saved to `/prediction_diagnostics/trained_models_refined/`
    - `tcn_best_refined.pt` → best model weights.  
    - `config_refined.json` → metadata with transformations, loss setup, and metrics.  
    - `training_history_refined.json` → epoch-wise train/val loss for post-hoc analysis.  
  - **Reasoning:**  
    - Log-transform reduces regression skew (deterministic), while class weighting corrects imbalance (dynamic).  
    - Both fixes target the observed weaknesses from Phase 4 without altering model architecture.

4. **Plot Training vs Validation Loss Curve (`plot_training_curves_refined.py`)**
  - **Purpose:** Visualise retraining improvement and convergence dynamics after applying **median-risk weighted BCE** and **log-transformed regression target**; compare Phase 4 baseline vs Phase 4.5 refined model to confirm data-level fixes improved training behaviour.
  - **Process:**
    - **Define paths** to Phase 4 (`training_history.json`) and Phase 4.5 (`training_history_refined.json`) losses; create `loss_plots/`.
    - **Load JSON histories** for both phases and extract `train_loss` / `val_loss`.
    - **Identify best validation epochs** for both models for annotation.
    - **Plot Phase 4.5 refined curves** (`loss_curve_refined.png`), annotate minima and overfitting region, add note under title explaining loss axis being log-transformed.
    - **Plot comparison curves** (`loss_curve_comparison.png`) overlaying Phase 4 baseline vs Phase 4.5 refined, annotate minima for both phases, add not below title and disclaimer under plot that direct numerical comparison is not possible, only trends can be compared.
    - **Save plots** and print console confirmations.
  - **Outputs:**  
    - `loss_curve_refined.png` → Phase 4.5 standalone learning curve.  
    - `loss_curve_comparison.png` → overlay showing baseline vs refined convergence for trend comparison.
  - **Interpretation:**  
    - Phase 4.5 training shows **faster early convergence**, lower minimum validation loss, and earlier overfitting, reflecting **enhanced learning for minority class and stabilised regression**.  
    - Comparison overlay highlights **accelerated learning dynamics** due to weighted BCE + log-transform, while baseline shows slower, steadier loss decline.  
    - Comparison is for visual trends, not direct numerical comarisons due to scale differences between plots.
    - Confirms **controlled improvements in convergence** without modifying architecture, validating retraining interventions.

**End Products of Phase 4.5**
| Output File / Folder | Purpose |
|-----------------------|----------|
| `tcn_best_refined.pt` | Retrained TCN weights at best validation epoch (early-stopping checkpoint). |
| `config_refined.json` | Complete experimental record: includes phase, log-transform, class weighting, `pos_weight_median_risk`, and `final_val_loss`. |
| `training_history_refined.json` | Epoch-wise training/validation loss for post-hoc visualisation and overfitting detection. |
| `tcn_diagnostics_summary.json` | Consolidated summary of classification and regression metrics identifying Phase 4 issues. |
| `/plots/` (12 diagnostic plots) | Visual evidence of class imbalance, regression skew, and post-fix stability (classification histograms, regression scatter/residuals, label distributions). |
| `loss_plots/loss_curve_refined.png` | Refined Phase 4.5 learning curve for standalone inspection. |
| `loss_plots/loss_curve_comparison.png` | Overlay of Phase 4 vs Phase 4.5, demonstrating retraining improvements. |

**Summary**
Phase 4.5 implements a controlled **diagnose → correct → retrain** loop:
- Diagnosed systemic dataset issues (imbalance, skew).  
- Applied minimal, reproducible fixes (log-transform, class weighting).  
- Preserved architecture and hyperparameters for scientific comparability.  
This phase bridges **Phase 4 (baseline)** and **Phase 5 (evaluation)**, producing a validated, documented, and reproducible refined model.

**Portfolio Framing**
Phase 4.5 exemplifies **rigorous ML pipeline practices**:
- Separation of diagnostics, retraining, and evaluation.  
- Transparent metadata and auditable fixes.  
- **Traceable lineage:** *Phase 4 → Phase 4.5 → Phase 5*.  
Demonstrates real-world iterative model refinement on messy, imbalanced clinical datasets.

**Interpretation of Results**
| Task | Metric | Observation | Interpretation |
|------|---------|--------------|----------------|
| **Max Risk Classification** | F1 = 0.929, ROC-AUC = 0.923 | Stable across thresholds | Strong separability; head functioning correctly |
| **Median Risk Classification** | F1 = 0.000, ROC-AUC = 0.778 | Predicts almost all 0s | Class imbalance and poor calibration; needs loss reweighting |
| **Regression (pct_time_high)** | R² = 0.166 (test), 0.2207 (val) | Positive R² values | Captures moderate signal; still underfits; skewed target limits variance explained |

**Diagnostic Conclusions:**
- **Max Risk head:** functioning correctly (stable calibration and high seperability), robust and interpretable; no corrective action needed.  
- **Median Risk head:** failing due to severe imbalance → retraining with class weighting required.  
- **Regression head:** improving but limited by target skew → apply target transformation (e.g., log/sqrt) or Huber/MAE loss.  
- Confirms **data-driven performance limitations**, not architectural errors.

**Why Include Validation Diagnostics**
- Validation evaluation verifies whether the same weaknesses persist beyond the test set.
-	Confirms systematic imbalance (median risk) and consistent regression behaviour (R² ≈ 0.2).
-	Shows model generalises stably but remains data-limited, not code-limited.

**In summary**:
- This diagnostic phase validates the original pipeline, identifies data-driven weaknesses, and forms a reproducible baseline for targeted model improvement
- It confirms systemic issues are dataset-derived, not implementation errors.
- Directly informs targeted reweighting, rescaling, and retraining in Phase 4.5.
- Hallmark of rigorous, research-grade machine learning work.

**fix evaluation script - What Changed**
- **Max Risk ROC-AUC:** increased from **0.577 → 0.923** because the test labels are now correctly aligned with the patient split JSON.  
- **Median Risk ROC-AUC:** improved slightly from **0.722 → 0.778**, confirming consistent label alignment.  
- **Regression metrics:** now match the diagnostic script exactly (**RMSE = 0.077**, **R² = 0.166**) after both scripts were re-ran.
**Why This Is Correct**
- Both scripts (`evaluate_tcn_testset.py` and `tcn_diagnostics.py`) now use the **ordered JSON split** to define patient IDs.  
- **Predictions and ground-truth labels** are fully aligned, ensuring **reproducible metrics** across runs.  
- The **Max Risk head** is now correctly evaluated; previous underestimation was due to **misaligned patient ordering**, not overfitting.

### TCN Diagnostics and Model Validation `tcn_diagnostics.py`
**Overview**
- This script (`tcn_diagnostics.py`) performs a **comprehensive diagnostic evaluation** of the trained Temporal Convolutional Network (TCN) from **Phase 4**.  
- Its purpose is to **validate model behaviour, interpretability, and consistency** before progressing to final evaluation and baseline comparison in Phase 5.
- **By running this diagnostic pipeline, we ensure that**:
  - Predictions are not constant (e.g., all 0s or all 1s).
  - Classification heads (`max_risk`, `median_risk`) produce meaningful probability distributions.
  - Regression head (`pct_time_high`) outputs continuous and variable predictions.
  - F1, RMSE, and R² values align with expected model behaviour.
  - Validation and test sets show consistent patterns.
  - Diagnostic plots and summaries are reproducible and versioned for auditability.
**Why This Script Exists**
- **After Phase 4 training, the TCN showed underperformance in all 3 heads**:
  - **Max Risk**: F1 = 0.929, AUC = 0.577 → possible overfitting
  - **Median Risk:** AUC = 0.722, F1 = 0.000 → signal present but failed thresholding, model predicted almost all negatives.
  - **Regression:** RMSE: 0.135, R² = −1.586 → model predictions collapsed toward mean values (underfitting).
- **To determine if these failures were due to data distribution or implementation, this diagnostic script**:
  1. Reloads the model and identical preprocessed tensors.  
  2. Recreates binary and continuous labels directly from `news2_features_patient.csv`.  
  3. Reproduces evaluation metrics using stored predictions (`tcn_predictions.csv`).  
  4. Extends analysis to **validation data**, confirming whether issues generalise beyond the test set.
- **Therefore systematically diagnosing**:
  1. Whether each head learned any real signal.  
  2. Whether probability distributions are saturated or skewed.  
  3. Whether regression outputs correlate with ground truth.  
  4. Whether imbalance or label noise explains poor test metrics.
- Acts as a **verification checkpoint** before retraining and enables evidence-based correction in **Phase 4.5**.

### Saved Diagnostic Plots
**All plots saved under `prediction_diagnostics/plots/`**
- 8 from test/validation diagnostics (4x probabilities, 2x scatter, 2x residuals)
- 4 from label distribution (2x) and regression comparison visualisations (2x)

| Plot Type | Example Filename | Description |
|------------|------------------|--------------|
| **Max Risk Probability Histogram (Test)** | `prob_hist_max_test.png` | Distribution of Max Risk predicted probabilities on the test set — checks for saturation or collapse |
| **Median Risk Probability Histogram (Test)** | `prob_hist_median_test.png` | Detects skew or collapse in Median Risk predictions on the test set |
| **Max Risk Probability Histogram (Validation)** | `prob_hist_max_val.png` | Distribution of Max Risk probabilities on the validation set — confirms generalisation consistency |
| **Median Risk Probability Histogram (Validation)** | `prob_hist_median_val.png` | Detects class imbalance or saturation in validation predictions |
| **Regression Scatter (Test)** | `test_reg_scatter.png` | Predicted vs true `pct_time_high` values — assesses correlation strength |
| **Regression Residuals (Test)** | `test_reg_residuals.png` | Residuals vs predictions — reveals bias, variance, and underfitting patterns |
| **Regression Scatter (Validation)** | `val_reg_scatter.png` | Predicted vs true values on validation data — checks for overfitting vs generalisation |
| **Regression Residuals (Validation)** | `val_reg_residuals.png` | Residual analysis on validation data — consistency check for regression head |
| **Label Distribution — Max Risk (Test)** | `dist_test_max_risk_labels.png` | Class distribution for Max Risk binary labels |
| **Label Distribution — Median Risk (Test)** | `dist_test_median_risk_labels.png` | Class distribution for Median Risk binary labels |
| **True Regression Distribution (Test)** | `dist_test_pct_time_high_true.png` | True distribution of regression targets |
| **Predicted Regression Distribution (Test)** | `dist_test_pct_time_high_pred.png` | Predicted regression target distribution |

**Why Save Visualisations**
- Verify model behaviour before/after retraining.
- Enables visual comparison before/after retraining.  
- Documents model behaviour for transparency and auditability.  
- Visual proof of dataset skew, collapse, or calibration drift.
- Supports inclusion in technical reports or repository README.  

### Diagnostics vs Evaluation Metric Output Misalignment
**Patient ID Misalignment**
- Initial evaluation (`evaluate_tcn_testset.py`) used CSV filtering without enforcing JSON split order.  
- ROC-AUC metrics were sensitive to this ordering, causing artificially low Max Risk AUC (0.577 → 0.923 after fix).  
- Threshold-based metrics (F1, Accuracy) were less affected.

**Original Metric Comparison: Evaluation vs Diagnostics Scripts**
| Task | Metric | `evaluate_tcn_testset.py` | `tcn_diagnostics.py` | Notes |
|------|--------|---------------------------|---------------------|-------|
| **Max Risk Classification** | F1 (0.5) | 0.929 | 0.929 | F1 consistent; threshold-independent ROC-AUC differs due to patient ID misalignment in CSV |
| | ROC-AUC | 0.577 | 0.923 | Diagnostics script uses JSON split for ordering → correct ranking |
| **Median Risk Classification** | F1 (0.5) | 0.000 | 0.000 | Consistent; class imbalance issue persists |
| | ROC-AUC | 0.722 | 0.778 | Slight difference due to ordering; corrected by JSON split |
| **Regression (pct_time_high)** | RMSE | 0.135 | 0.1351 | Consistent |
| | R² | -1.586 | -1.5859 | Consistent; negative R² indicates underfitting on test set |
| **Validation Regression** | RMSE | N/A | 0.0744 | Diagnostic script only |
| | R² | N/A | 0.2207 | Partial signal on validation |

**Key Steps**
1. **Corrected Evaluation Pipeline**:  
  - Update `evaluate_tcn_testset.py` to always use JSON split ordering. 
  - **Rebuild test targets using the JSON split**:
    ```python
    test_ids = splits["test"]  
    test_df = features_df.set_index("subject_id").loc[test_ids].reset_index()
    ```
  - Ensures predicted probabilities align correctly with ground truth labels.  
  - ROC-AUC should now matche `tcn_diagnostics.py`.
2. **Re-run `tcn_diagnostics.py` and `evaluate_tcn_testset.py`**
  - Re-run both script to get updated metrics. 
  - The diagnostic script reads `tcn_predictions.csv` from `evaluate_tcn_testset.py`.
	-	If that CSV was generated before the evaluation script was fixed, it still contains misaligned test labels.
  - Re-running the diagnostic script after fixing and re-running the evaluation script will output the correct RMSE and R².
3. **Documented Final metrics**
  - Terminal outputs for both scripts are documented for auditibility.
  - All metric outputs now match. 

**Updated Final Metrics Comparison**
1. **Interpretation of Metrics Post-Fix**
  - **Max Risk:** F1 stable at 0.929, ROC-AUC = 0.923 → no overfitting.  
  - **Median Risk:** F1 = 0, ROC-AUC ~0.778 → confirms class imbalance problem; model underfits.  
  - **Regression (pct_time_high):** R² = 0.166, RMSE = 0.077 → moderate accuracy, captures some signal but leaves most variance unexplained.
2. **Data-Driven Insights**
  - Poor Median Risk performance is **dataset-limited**, not a code or architecture issue.  
  - Regression underfitting is influenced by **skewed targets and low variance**.  
  - Max Risk can be considered **robust and reliable**; no corrective action required.

**Updated Final Metrics Comparison: Diagnostics vs. Evaluation Script**
| Task | Script | F1 / RMSE | ROC-AUC / R² | Accuracy |
|------|--------|------------|---------------|----------|
| **Max Risk** | Diagnostics | 0.929 | 0.923 | 0.867 |
|              | Evaluation  | 0.929 | 0.923 | 0.867 |
| **Median Risk** | Diagnostics | 0.000 | 0.778 | 0.800 |
|                 | Evaluation  | 0.000 | 0.778 | 0.800 |
| **Regression (pct_time_high)** | Diagnostics | 0.077 (RMSE) | 0.166 (R²) | - |
|                               | Evaluation  | 0.077 (RMSE) | 0.166 (R²) | - |
**Note:** Post-fix metrics are now consistent between the diagnostics and evaluation scripts, confirming alignment and reproducibility.

### Overall Summary 
**Overview**
- Today marked the foundation of **Phase 4.5: Diagnostics & Re-training**, a pivotal milestone in the project’s ML lifecycle.  
- This session confirmed that the **model architecture and weights were sound, the fault lay within the evaluation pipeline**, specifically in data alignment and ordering.  
- By enforcing consistent dataset handling, metric computation, and validation across scripts, today’s work **stabilised the project’s foundation**, ensuring that all reported metrics now reflect genuine model performance.  
- This established a **verified, reproducible baseline** for retraining both **TCN** and **LightGBM** models under improved preprocessing (class weighting and regression target transformation), while reinforcing the **importance of data alignment, metric integrity, and reproducibility** as core principles of reliable ML engineering and deployment.  
**Key Technical Outcomes**
1. **Technical Depth**
  - Identified a **non-obvious evaluation-ordering bug** that caused ROC-AUC inconsistencies while F1 and Accuracy remained stable.  
  - Diagnosed the issue at the intersection of **data reconstruction and metric computation**, ensuring predictions and labels were perfectly matched.  
  - Implemented an **ordering-preserving fix** using `.loc[test_ids]` to align patient IDs exactly with the JSON split sequence.
2. **Scientific Rigor**
  - Conducted **independent validation** through diagnostic scripts to confirm that evaluation metrics reflect genuine model behaviour.  
  - Ensured **methodological consistency** across all scripts, every model now rebuilds `y_test` using the same source of truth (`patient_splits.json`).  
  - Adopted reproducible debugging practices, re-running both evaluation and diagnostic scripts to verify identical outputs.
3. **Engineering Reliability**
  - Unified the **evaluation** and **diagnostics** frameworks, producing consistent metrics across both processes.  
  - Strengthened pipeline robustness — now every component yields verifiable and reproducible results.  
  - Laid the groundwork for fair model comparison and retraining in later phases.  
**Impact on the Project**
- **Verified Baseline:** Established the definitive, reproducible metrics for the TCN before retraining.  
- **Metric Integrity:** Eliminated discrepancies caused by misalignment, ensuring AUC/F1 values are trustworthy.  
- **Pipeline Credibility:** Reinforced the core principle that *model reliability depends on data integrity and evaluation validity*.  
- **Future Readiness:** Created a clean foundation for fair retraining of both TCN and LightGBM under improved preprocessing.
**Key Lessons**
- **Order matters:** Misaligned patient IDs silently invalidate ranking-based metrics like ROC-AUC.  
- **Separate diagnostics from evaluation:** Enables independent verification and faster debugging and ensures unbiased results.  
- **Trace all metrics to dataset lineage:** Reproducibility begins with deterministic, consistent data handling.  
- **True ML engineering:** Extends beyond model architecture and performance, it’s about maintaining data, metric, and evaluation integrity end-to-end. It safeguards the entire evaluation ecosystem.


### Portfolio Framing 
**Overview**
- Through systematic debugging and scientific validation, this session transformed the project from **uncertain evaluation** to **verified reliability**.  
- **The fix unified the diagnostic and evaluation pipelines, confirming that**: the model was sound, the data alignment was not.
- **This reinforced one of the most crucial lessons in applied machine learning**: a trustworthy model begins with a trustworthy evaluation pipeline.
**Why This Matters in ML Pipelines**
1. **Reproducibility is Foundational**  
  - Without consistent data alignment, even well-performing models cannot be trusted.  
  - Real-world ML pipelines must yield **identical results under identical conditions**, which depends on deterministic dataset handling.  
2. **Metric Validity Underpins Trust**  
  - Misaligned ground truths can silently produce **misleading metrics**, creating the illusion of poor or inflated model performance.  
  - This debugging phase ensured all metrics genuinely reflect model capacity and dataset characteristics.  
3. **Diagnostics as Scientific Tools**  
  - The diagnostics script functioned as an independent **model integrity testing framework**, revealing hidden issues invisible during standard evaluation.  
  - This mirrors **industry-grade ML validation practices**, where evaluation reproducibility is just as critical as model accuracy.  
**Portfolio Narrative**
- **“This phase was pivotal in transitioning from experimental modelling to reliable ML engineering.”**  
- By diagnosing and fixing a subtle **evaluation-ordering bug**, I ensured that all future retraining and model comparisons are grounded in **reproducible, validated metrics**.  
- This work demonstrates a deep understanding of **data integrity, metric reproducibility, and scientific validation** — essential competencies in real-world ML engineering.  
- It highlights the mindset of a practitioner who not only builds models but verifies the **entire evaluation ecosystem**, ensuring that every performance number is scientifically defensible.  

**Summary of Changes**
| Change | Applied To | Timing / Location | Tensor(s) Affected | Purpose / Rationale |
|--------|------------|-----------------|------------------|------------------|
| **Log1p Transform** | Regression target (`pct_time_high`) | Inside `get_targets()` before tensor creation | Changes `y_train_reg`, `y_val_reg`, `y_test_reg` | Reduces skew, stabilises variance, prevents extreme gradients |
| **Class Weighting** | Median-risk BCE loss | After tensor creation, when defining `criterion_median` | No change to tensors; affects loss calculation only | Corrects class imbalance (~3:1), amplifies minority class gradients, faster early learning |
| **Configuration metadata** | JSON config (`config_refined.json`) | Before training + updated after training | N/A | Records dynamic values (`pos_weight_median_risk`, `final_val_loss`), experiment notes, reproducibility info |
| **Training history JSON** | `training_history_refined.json` | After training loop | Stores train/val losses | Enables post-hoc analysis of learning dynamics and overfitting patterns |
| **Output paths** | Model & history | `trained_models_refined/` | N/A | Ensures reproducibility and traceability; keeps Phase 4 and Phase 4.5 outputs separate. |

**Dynamic vs Deterministic Outputs**
| Value | Type | How It Varies | Printed? |
|-------|------|---------------|----------|
| `pos_weight` | Dynamic | Depends on class ratio in current training split | Yes, each run may differ |
| `final_val_loss` | Dynamic | Depends on random initialisation, convergence and early stopping | Yes, reflects model’s best validation performance; logged for reproducibility |
| `log1p(y)` | Deterministic | Same `np.log1p()` transformation applied every run | No, only min/max printed for verification (sanity check) |

**Effects on Training & Validation**
- **Note:** 
  - We cannot compare absolute magnitudes between Phase 4 vs Phase 4.5, because the loss units are incompatible (raw-scale vs log-scale regression component).
  - Even though both overall losses are scalar numbers, one phase includes a regression loss measured in MSE(raw), and the other in MSE(log), they’re not numerically comparable.
  - We can however can claim behavioural improvement (faster convergence, earlier overfit point, improved stability).
- **Early Training**:
  - Faster decrease in training loss for median-risk classification due to weighted loss.  
  - Regression loss more stable because of log-transform.  
- **Validation Dynamics**:
  - Validation loss reaches minimum earlier (epoch ~3) and rises sooner, reflecting faster overfitting of weighted minority-class patterns.  
  - Log-transform prevents extreme regression errors from dominating total loss.  
- **Overall**:
  - Training is reproducible and deterministic.
  - Early stopping preserves the epoch with the best validation generalisation.
  - JSON outputs provide both static configuration and dynamic metrics (`pos_weight`, `final_val_loss`) for post-hoc comparison.

**Summary**
- Phase 4.5 introduced **two controlled refinements** over Phase 4:
  1. Regression log-transform (`log1p`)  
  2. Median-risk class weighting (`pos_weight`)  
- These changes improve early learning stability, correct class imbalance, and allow scientifically reproducible retraining.  
- Training history and configuration files provide full auditability, enabling downstream comparisons between Phase 4, Phase 4.5, and other models (LightGBM, NEWS2).  
- The best model checkpoint is automatically saved at the epoch with **lowest validation loss**, ensuring optimal generalisation.

### Model Retraining Plan — Fixing Median Risk and Regression Heads
**Overview**
- Following the first step of Phase 4.5 (Diagnostics & Validation), we confirmed that the model architecture itself was not faulty, the poor metrics originated from data imbalance and target skew.  
- This retraining step applies targeted, minimal, and principled corrections to address these issues while preserving the stability of all well-performing components.

**Diagnostic Findings Recap**
| Head | Observed Metric | Issue Identified | Root Cause |
|------|-----------------|------------------|-------------|
| **Max Risk** | F1 = 0.929, ROC-AUC = 0.923 | Excellent performance | None: stable calibration, strong separability |
| **Median Risk** | F1 = 0.000, ROC-AUC = 0.778 | Model predicts all zeros | Severe class imbalance (only 20% positives) |
| **Regression (pct_time_high)** | R² = 0.166 (test), 0.220 (val) | Low R², flat predictions | Skewed, zero-heavy regression target (27% zeros, right tail) |
- **Conclusion:**  
  - The classification collapse and regression underperformance stem from data distributional issues, not architecture, code, or training instability.  
  - Hence, retraining focuses solely on rebalancing and stabilising these heads.

**Data-Level Insights**
- **Median Risk:**  
  - Label ratio → 3 positives : 12 negatives (20% positive rate).  
  - Model optimised BCE loss by always predicting 0, achieving high accuracy but zero recall.  
- **Regression (pct_time_high):**  
  - Highly skewed, with 27% zeros and long right tail up to ~0.44.  
  - Ordinary MSE loss penalises outliers disproportionately, leading to collapse around the mean (≈0.11).  
- These issues directly explain the low F1 and R² observed in Phase 4.5 diagnostics.

**Potential Fixes Considered**
| Problem | Potential Fixes | Evaluation | Final Decision |
|----------|-----------------|-------------|----------------|
| **Median Risk imbalance** | (a) Class weighting  (b) Oversampling  (c) SMOTE / synthetic examples | (a) Weighting directly corrects BCE loss dominance.  (b, c) Risk overfitting and data duplication. | **Use class weighting only.** |
| **Regression skew** | (a) Log-transform target  (b) Huber loss  (c) MAE loss  (d) Resampling zeros | (a) Log-transform stabilises variance and preserves scale.  (b, c) Require new hyperparameters and loss tuning.  (d) Alters distribution artificially. | **Use log1p transformation only.** |

**Final Retraining Plan (Concrete Implementation)**
- **Max Risk Head**
  - No changes.
  - **Rationale:** Metrics demonstrate correct calibration and separability. Modifying this head risks destabilising a functioning component.
- **Median Risk Head** 
  - Add Class Weighting.
  - **Problem:** BCE loss dominated by negatives → model minimises loss by predicting all 0s.
  - **Fix:** Compute dynamic `pos_weight` based on training label proportions.
  - **Effect:**
    -	Positives contribute more strongly to the loss.
    -	Model learns to recognise minority “1” class, increasing recall and F1 without harming calibration.
  - **Expected outcome:**
    -	Median Risk F1 improves from 0.0 → 0.3–0.5 depending on separability.
    -	ROC-AUC remains ~0.75–0.8, confirming signal presence.
- **Regression Head** 
  - Apply Log-Transform to Target.
  - **Problem:** Skewed, zero-inflated target leads to near-constant predictions and low R².
  - **Fix:** Apply log1p transform before training and invert with expm1 at inference.
  - **Rationale**
    -	Log1p compresses the long tail, reduces variance, and linearises target relationships.
    -	Keeps zero safely mapped (log1p(0) = 0).
    -	Requires no loss function change or architectural modification.
  - **Expected outcome:**
    -	Regression variance improves; R² increases to 0.2–0.5.
    -	Residuals become more normally distributed, improving stability and interpretability.

**Why Nothing Else Was Changed**
 **Component** | **Decision** | **Justification** |
|----------------|--------------|--------------------|
| **Architecture** | Unchanged | Model topology already validated; prior failures were data-level, not structural. |
| **Max Risk Head** | Unchanged | Demonstrated high separability and stability; no imbalance or skew. |
| **Loss Functions (other heads)** | Unchanged | BCE and MSE remain theoretically sound once data distributions are corrected. |
| **Hyperparameters** | Unchanged | Learning rate, batch size, and optimiser not linked to prior instability. |
| **Random Seeds & Data Splits** | Unchanged | Maintains perfect reproducibility and comparability with Phase 4 results. |
- **Reasoning:** Keeping all constants ensures that any metric improvement in Phase 4.5 can be attributed **solely to the two targeted data-level fixes**, not confounded by unrelated architectural or hyperparameter changes.

**Rationale Summary**
| **Head** | **Change Implemented** | **Why It’s Necessary** | **Why It’s Sufficient** |
|-----------|------------------------|--------------------------|---------------------------|
| **Max Risk** | None | Already optimal | Modifying a stable head risks unnecessary destabilisation |
| **Median Risk** | Weighted BCE Loss | Corrects severe class imbalance (20% positives) | Restores recall and stabilises F1 without altering feature distribution |
| **Regression (`pct_time_high`)** | Log1p Target Transform | Reduces skew and compresses long-tailed variance | Normalises target distribution, stabilises loss, improves fit consistency |

**Expected Outcomes**
| **Metric** | **Phase 4 (Before)** | **Phase 4.5 (After Retraining, Expected)** | **Improvement** |
|-------------|----------------------|-------------------------------------------|-----------------|
| **Max Risk F1** | 0.929 | 0.929 | Stable — no change expected |
| **Median Risk F1** | 0.000 | 0.300–0.500 | Significant improvement in recall and F1 |
| **Regression R²** | 0.166 (test), 0.220 (val) | 0.300–0.500 | Improved correlation and variance stability |

**Summary**
This retraining phase introduces **two minimal yet sufficient interventions**, both derived from diagnostic evidence:
- **Median Risk Head:** Implemented **class-weighted BCE loss** to counteract class imbalance and prevent collapse into all-negative predictions.  
- **Regression Head:** Applied **log-transform (`log1p`)** to stabilise regression targets and improve variance representation.  
- **Max Risk Head:** Remains **unchanged**, as metrics confirm robust performance.  
- **All other parameters:** Retained unchanged for controlled, interpretable retraining.



### Potential Fixes and Rationale for Final Choice
#### 1. Median Risk (Class Imbalance)
**Problem Identified:**  
- Diagnostics revealed the **Median Risk head consistently predicted all zeros**, giving an F1 score of 0 despite 80% accuracy.  
- This indicated that the model had learned to ignore positive cases due to the class imbalance (20% positive, 80% negative).
**Fixes Considered:**
- 1. **Class Weighting:** Adjust the BCE loss function to penalise misclassified positives more heavily.  
  - **Pros:** Corrects imbalance *within* the loss function; no data duplication or architecture change required.  
  - **Mechanism:** Uses `pos_weight = num_neg / num_pos` in `BCEWithLogitsLoss`, directly scaling the loss contribution of minority positives.  
  - **Effect:** Forces the model to give positive cases sufficient gradient signal during training, improving recall and F1.  
- 2. **Oversampling Positives:** Duplicate minority-class samples to balance the dataset.  
  - **Cons:** Introduces duplicate sequences, causing overfitting; distorts sequence variance in small datasets.  
  - Used in large datasets, but inappropriate for limited ICU data.  
- 3. **SMOTE / Synthetic Samples:** Interpolates new minority samples between existing ones.  
  - **Cons:** Invalid for time-series or structured clinical data, as synthetic patients can generate biologically implausible trajectories.
**Why not transform the target?** 
- Median risk is binary; the only meaningful adjustment is weighting the loss. 
- Transforming 0/1 values is nonsensical.
**Final Decision:** 
- **Use class weighting only**: Change the loss function →  BCEWithLogitsLoss → use pos_weight.
- It’s the most controlled, architecture-neutral correction that preserves data integrity and prevents overfitting.  
- This ensures that any observed F1 improvement reflects true model learning, not artificial dataset inflation.

#### 2. Regression Head (Skewed Target Distribution)
**Problem Identified:**  
- **The regression target `pct_time_high` showed**:
  - ~27% zeros (zero-inflated distribution)
  - Heavy right skew (most values clustered near 0.1–0.2)
  - Very low variance → leading to negative R² on test data.
**Fixes Considered:**
- 1. **Log-transform Target (`log1p`)**  
  - **Pros:** Compresses high-end outliers and normalises variance while preserving zero values (`log1p` keeps 0 → 0).  
  - **Mechanism:**  
    - During training: `y_reg = log1p(y_reg)`  
  - **Effect:** Stabilises regression learning and improves generalisation by giving the model a smoother functional mapping to learn.  
- 2. **(b) Huber Loss**  
  - **Cons:** Balances MSE and MAE but requires careful tuning of the delta parameter.  
  - **Risk:** Introduces an extra hyperparameter, potentially destabilising controlled retraining comparison.  
- 3. **(c) MAE Loss**  
  - **Cons:** More robust to outliers but less sensitive to small errors, which harms fine-grained calibration.  
  - **Outcome:** Would not address skewness directly; only reduces sensitivity to it.  
- 4. **Resampling or Filtering Zeros**  
  - **Cons:** Artificially changes data distribution and removes meaningful low-end observations.  
  - Breaks dataset realism and comparability with Phase 4.
**Why not just change the loss?**
- **You could switch to Huber or MAE, but**:
  - Adds new hyperparameters (δ for Huber)
  - May require careful learning rate adjustments
  -	Makes comparison with Phase 4 less direct
-	Log-transform is simpler, deterministic, and preserves the meaning of the target — a minimal sufficient change.
**Final Decision:**  
- Use log1p transformation only to the target tensors.
- It’s mathematically grounded, non-destructive, and reproducible.  
- This preserves the data’s natural distribution while directly targeting the skew that caused poor R² values.

#### Summary
1. **Median Risk:**
  - **Loss function controls learning priorities:** pos_weight introduces run-specific, dataset-dependent bias that helps learning but increases overfitting risk.
  - Weighted BCE → model “pays attention” to rare positives.
2. **Regression head:**
  - **Target transformations change data scale/distribution:** log1p purely stabilising and deterministic, doesn’t affect variance across runs, only numerical stability.
  - log1p → stabilises regression learning for skewed, zero-inflated targets. 
  - Stabilises variance, reduces extreme gradient contributions.
- In Phase 4.5, these two interventions address the exact data-level issues without touching architecture, hyperparameters, or other heads.

### Refined Retraining Run (with Class Weighting + Log Transform)
#### Terminal Console Output
```bash
[INFO] All refined training outputs will be saved to: /Users/simonyip/Neural-Network-TimeSeries-ICU-Predictor/src/prediction_diagnostics/trained_models_refined
[INFO] Base configuration initialised.
[INFO] Log-transform applied to regression target: min=0.000, max=0.365
[INFO] Log-transform applied to regression target: min=0.000, max=0.241
[INFO] Log-transform applied to regression target: min=0.000, max=0.244
[INFO] Computed pos_weight for median_risk = 2.889
[INFO] Targets loaded (Refined Phase 4.5):
 - train: torch.Size([70]) torch.Size([70]) torch.Size([70])
 - val: torch.Size([15]) torch.Size([15]) torch.Size([15])
 - test: torch.Size([15]) torch.Size([15]) torch.Size([15])
Epoch 1: Train Loss = 1.7663, Val Loss = 1.4496
Epoch 2: Train Loss = 1.4777, Val Loss = 1.3791
Epoch 3: Train Loss = 1.3206, Val Loss = 1.3700
Epoch 4: Train Loss = 1.1545, Val Loss = 1.4158
Epoch 5: Train Loss = 1.0083, Val Loss = 1.5500
Epoch 6: Train Loss = 0.8223, Val Loss = 1.7551
Epoch 7: Train Loss = 0.7313, Val Loss = 2.1282
Epoch 8: Train Loss = 0.6785, Val Loss = 2.2766
Epoch 9: Train Loss = 0.6171, Val Loss = 2.3606
Epoch 10: Train Loss = 0.5767, Val Loss = 2.4318
Early stopping at epoch 10
[INFO] Final refined training configuration saved to /Users/simonyip/Neural-Network-TimeSeries-ICU-Predictor/src/prediction_diagnostics/trained_models_refined/config_refined.json
[INFO] Refined training history saved to /Users/simonyip/Neural-Network-TimeSeries-ICU-Predictor/src/prediction_diagnostics/trained_models_refined/training_history_refined.json
[INFO] Best refined model saved to /Users/simonyip/Neural-Network-TimeSeries-ICU-Predictor/src/prediction_diagnostics/trained_models_refined/tcn_best_refined.pt
```
#### What Happened in the Run
**Epoch 1–3:**  
  - Both training and validation losses decreased steadily (Train: 1.77 → 1.32, Val: 1.45 → 1.37).  
  - Indicates that the model was successfully adapting to both classification and regression tasks under the refined setup.  
  - Best generalisation observed around **epoch 3**, when validation loss reached its minimum (1.37).  
**Epoch 4–5:**  
  - Training loss continued to fall (1.15 → 1.01), but validation loss began to rise (1.42 → 1.55).  
  - Marks the onset of overfitting — the model starts specialising on the weighted median-risk patterns rather than generalising to unseen patients.  
**Epoch 6–10:**  
  - Training loss kept improving (0.82 → 0.58) while validation loss increased consistently (1.76 → 2.43).  
  - Confirms strong overfitting beyond epoch 3; the model continues to fit the training distribution too tightly.  
  - **Early stopping** was triggered at **epoch 10**, preserving the weights from the best epoch (epoch 3).
#### Interpretation
- The overall learning dynamics mirror Phase 4, but overfitting began earlier, suggesting that:
  - **Class weighting** (pos_weight = 2.889) successfully amplified minority-class gradients, improving early learning speed and accelerating early convergence but introducing higher variance and faster overfitting.
  - **Log-transforming the regression target** stabilised numerical gradients and prevented extreme / exploding regression errors, keeping total loss bounded.  
- The rapid drop in training loss with simultaneous validation rise reflects tighter fitting due to the new weighting, not architectural instability.  
- Early stopping functioned as expected, halting training when validation loss plateaued, and preserving the **best model checkpoint at epoch 3**.
- The model trained reproducibly and deterministically, confirming the stability of the refined preprocessing pipeline.
#### Refinement Insights
- **Class Weighting (`pos_weight = 2.889`)**  
  - Balanced the median-risk binary head, correcting for ≈ 3:1 class imbalance (2.889 negative samples for every positive sample). 
  - Loss function multiplies errors on the minority class (positives) by 2.889, so the model “pays more attention” to learning them correctly. 
  - Without this, the model could trivially predict the majority class and achieve deceptively low loss.
  - This printout confirms successful dynamic weight computation.
- **Regression Log-Transform (`Log-transform applied to regression target:..`)**  
  - Printout helps with debugging and validation, confirms confirms per-split transformation consistency, and that regression targets have been transformed as intended.
  - `log1p(y)` Applied automatically during target preparation to reduce heteroscedasticity (variance scaling) thus stabilising variance.  
  - The transform is deterministic (does not vary per training run) and thus not printed, unlike `pos_weight`, so there’s no need to print a run-specific value to verify beyond an initial min/max check for debugging.
#### Why This Affects Early Learning and Validation Loss Dynamics
| Component   | Purpose | Early Training | Validation Dynamics |
|--------------|----------|----------------|----------------------|
| **pos_weight** | Amplifies the contribution of minority-class examples in BCE loss for `median_risk_binary`. | Accelerates learning for minority class; training loss drops faster. | Validation loss rises sooner due to overfitting on weighted patterns, especially with a small dataset. |
| **log1p** | Stabilises regression head by compressing target range and reducing heteroscedasticity. | Smoothes regression contribution to total loss; prevents extreme gradient spikes. | Keeps regression loss bounded, so validation loss mainly reflects classification overfitting rather than extreme regression errors. |
| **Combined Effect** | Integrates both refinements into multi-task learning. | Total training loss decreases rapidly across early epochs. | Validation loss reaches minimum earlier (epoch 3) and rises sharply after, reflecting early minority-class overfitting and stable regression contribution. |
#### Summary
- Phase 4.5 reproduced expected convergence and overfitting patterns, confirming stable integration of both refinements.  
- The model learned faster initially, generalised best around epoch 3, and diverged after that due to small-sample effects.  
- **Best validation loss achieved**: 1.3700 (epoch 3), compared to 0.9587 in Phase 4 → higher due to weighting and reduced bias but improved minority sensitivity.
- **Overall**: refinements were correctly applied, training pipeline remained reproducible, and the model checkpoint at epoch 3 represents the optimal balance between learning and generalisation.

### Phase 4 vs Phase 4.5 - Training vs Validation Loss Comparison
**Overview**
- Phase 4's `training_history.json` contains all baseline training and validation losses per epoch
- Phase 4.5's `training_history_refined.json`contains all refined training and validation losses per epoch to assess the impact of **median-risk weighted BCE** and **log-transformed regression target** on learning dynamics.
- Phase 4.5's `loss_curve_comparison.png` overlays loss curves for both baseline and refined outputs for visual comparison.
**Analysis**
- **Training Loss**
  - Phase 4: Steady decline from 1.46 → 0.53 over 11 epochs.
  - Phase 4.5: Faster initial decrease from 1.77 → 0.58 over 10 epochs.
  - **Interpretation:** Refined model converges more aggressively in early epochs, reflecting stronger gradient signals from weighting and target transformation.
- **Validation Loss**
  - Phase 4: Gradual decline 1.12 → 0.96 by epoch 3, slow increase to 1.26 by epoch 10 → stable, smooth convergence with mild overfitting.
  - Phase 4.5: Drops 1.45 → 1.37 by epoch 3, then rises sharply to 2.43 by epoch 10 → faster learning but earlier and more pronounced overfitting.
  - **Interpretation:** Data-level interventions accelerated early learning but also amplified overfitting, consistent with increased gradient contribution from minority class weighting.
**Key Takeaways**
- **Phase 4.5 training curves** demonstrate improved early learning efficiency for both classification and regression heads.
- **Validation curves** confirm that weighted BCE and log-transform improve convergence but require careful early stopping to prevent rapid overfitting.
- Trend-level comparison confirms measurable improvements in convergence and learning dynamics, even though absolute loss magnitudes differ due to the log-transformed regression component
- Overlay plot (`loss_curve_comparison.png`) clearly shows the **gap between baseline and refined curves**, providing visual evidence that retraining interventions produced measurable improvements without modifying model architecture.
**Outputs**
- Provides a quantitative reference of training and validation loss progression for reproducibility.
- Documents effect of Phase 4.5 retraining interventions on learning behavior and convergence efficiency.
- Provides visual confirmation of improved convergence, early validation loss minimum, and expected overfitting onset.
- **Note:** Loss magnitudes are not directly comparable across phases due to the log-transformation applied to regression targets in Phase 4.5.

### Phase 4 vs Phase 4.5 — JSON Configuration Comparison
**Overview**
- Phase 4’s `config.json` was a minimal operational setup for the original training model.  
- Phase 4.5’s `config_refined.json` became a **reproducibility-grade record**, documenting not just parameters but also the rationale, transformations, and dynamically computed values.
**Key Differences**
| **Aspect** | **Phase 4** | **Phase 4.5 (Refined)** |
|-------------|--------------|--------------------------|
| **Purpose** | Basic run config | Full metadata + audit trail |
| **Phase Tag** | None | `"phase": "4.5 - Refined Retraining"` |
| **Loss Functions** | BCE + MSE | Weighted BCE + log-transformed MSE |
| **Transformations** | Not recorded | `"data_transformations"` field for `log1p` + `pos_weight` |
| **Dynamic Values** | None | `"pos_weight_median_risk"` = 2.889 , `"final_val_loss"` = 1.37 |
| **Outputs Logged** | Implicit paths | `"outputs"` + `"outputs_confirmed"` with full paths |
| **Notes** | None | Added explanation of refinements |

**Why the New Fields Matter**
| **Field** | **Purpose** |
|------------|-------------|
| `phase` | Labels experiment version for traceability |
| `data_transformations` | Records applied preprocessing (`log1p`, weighting) |
| `pos_weight_median_risk` | Captures actual computed class-imbalance ratio (~3:1) |
| `final_val_loss` | Stores best validation loss for direct comparison |
| `notes` | Explains experiment intent |
| `outputs_confirmed` | Confirms saved file paths post-training |

Only `pos_weight_median_risk` and `final_val_loss` are dynamic run outputs; the rest provide reproducibility metadata.

**Summary**
- **Phase 4** → original operational baseline  
- **Phase 4.5** → scientifically traceable updated version  
- Adds weighting, log-transform, and explicit documentation for reproducibility.  
- Enables comparison across runs while maintaining the same model architecture and training logic.

Continue phase 5 after completing phase 4.5

**Overall Summary**
- The **refined evaluation pipeline** finalises Phase 4.5 by integrating threshold tuning, bias diagnosis, and post-hoc calibration.  
- Ensures that:
  - Negative R² reflected scale bias, not model failure.  
  - Simple **linear calibration** restored accurate scaling and interpretability.  
  - **Threshold tuning** optimised median-risk classification without retraining.  
- Produces a **fully reproducible, auditable, and publication-ready evaluation module** bridging *machine learning performance* and *clinical interpretability*.

**Analytical Impact**
- The refined evaluation is now a **robust, auditable analysis framework** that includes:
  - **Threshold tuning** (median-risk) performed on validation to optimise F1 for an imbalanced target.
  - **Calibration** (regression) that corrects a linear scale bias without retraining.
  - **Dual-scale reporting** (log vs raw) so both internal validity and clinical utility are visible.
- Together the changes:
  - Fix regression scaling and produce clinically meaningful raw estimates.
  - Improve classification recall/precision trade-off for median-risk via validation-derived threshold tuning.
  - Preserve model architecture and weights; only post-processing (threshold + calibration) changes — therefore comparisons to other models should use the same “best-practice” post-processing.
- **In essence:** `evaluate_tcn_testset_refined.py` completes the analytical lifecycle of the refined TCN, evolving from raw inference to **calibrated, explainable, and clinically interpretable performance evaluation**, bridging the gap between machine learning validation and clinical applicability.

**Practical guidance for fair comparison with LightGBM / NEWS2**
- **Compare best-to-best:** apply the same validation-based tuning and calibration workflow to each model you compare (TCN, LightGBM, NEWS2). That means:
  - If you tune median-risk threshold for TCN, do the same validation-based tuning for LightGBM and NEWS2 (where applicable).
  - If you calibrate TCN regression outputs, consider equivalent calibration strategies for baselines (or explicitly justify why not).
- **This ensures fairness:** each model is evaluated at its **practically usable** operating point.

### Evaluation Script Outputs — Refined TCN Test Set
**Overview**
- This section summarises the **primary evaluation outputs** of `evaluate_tcn_testset_refined.py`, excluding plots. 
- It details **when each output is generated, why it exists, and what insights can be derived**.
**Outputs**
1. **Predictions + Ground Truth CSV — `tcn_predictions_refined.csv`**
  - **Purpose:**  
    - Provides a **transparent mapping** of model predictions to ground truth across all tasks.  
    - Facilitates **reproducibility**, debugging, and downstream analyses (e.g., calibration checks or error audits).
  - **Creation Point:**  
    - Generated **after inference**, before metric computation and calibration.  
    - Combines:
      - Raw model outputs (log-space regression, classification logits/probabilities)
      - Ground-truth labels
      - Inverse-transformed regression predictions (raw %)
  - **Contents:**  
    - Each row corresponds to a **single patient** in the held-out test set (n = 15).  

| Column | Description |
|--------|-------------|
| `y_true_max` | Ground-truth label for max-risk classification (1 = high risk, 0 = not). |
| `prob_max` | Model-predicted probability for max-risk head (sigmoid of logit). |
| `y_true_median` | Ground-truth label for median-risk classification. |
| `prob_median` | Model-predicted probability for median-risk head. |
| `y_true_reg` | True raw percentage of time in high-risk state (`pct_time_high`). |
| `y_true_reg_log` | Log-transformed ground truth (`log1p(y_true_reg)`). Used for **internal log-space metrics**. |
| `y_pred_reg_log` | Model output regression predictions in log-space (native network output). |
| `y_pred_reg_raw` | Inverse-transformed predictions (`expm1(y_pred_reg_log)`), restoring raw % for **clinical interpretation**. |
  
  - **What We Can Learn:**  
    - Direct inspection of per-patient errors (prediction vs ground truth).  
    - Distribution and plausibility of predictions before and after calibration.  
    - Useful for verifying **monotonic trends** and identifying outliers.

2. **All Computed Evaluation Metrics — `tcn_metrics_refined.json`**
  - **Purpose:**  
    - Stores **summary performance metrics** in a structured, reproducible format. 
    - Captures results for both **classification** and **regression** tasks across **pre- and post-calibration**, as well as **post-tuning** for median-risk classification.   
    - Enables automated benchmarking, reproducibility, and integration into reports or comparisons with baseline models.
    - **Creation Point:**  
      - Generated **after all inference, calibration, and metric computations**.  
      - Combines metrics for:
        - **Classification heads**: max risk, median risk (pre- and post-tuning)
        - **Regression head**: log-space, raw-space, pre- and post-calibration
  - **Contents & Structure:**
  | Key | Description |
  |-----|-------------|
  | `max_risk` | Classification metrics for max-risk head (`AUC`, `F1`, `Accuracy`, `Precision`, `Recall`). |
  | `median_risk` | Classification metrics for median-risk head (same metrics as above). |
  | `median_risk_tuned` | Classification metrics for the **median-risk head** after **validation-based threshold tuning** (optimal F1 threshold). Enables fairer comparison on imbalanced data. |
  | `pct_time_high_log` | Regression metrics on log-space predictions (pre-calibration). |
  | `pct_time_high_raw` | Regression metrics on raw-space predictions (pre-calibration). |
  | `pct_time_high_log_cal` | Regression metrics on log-space predictions (post-calibration). |
  | `pct_time_high_raw_cal` | Regression metrics on raw-space predictions (post-calibration). |
  | `inference_time_sec` | Time taken for inference across the test set. |
  
  - **What We Can Learn:**  
    - **Classification (max vs median)**  
      - `max_risk`: confirms strong high-risk discrimination (AUC ≈ 0.9+).  
      - `median_risk`: typically underperforms at threshold 0.5 due to class imbalance.  
      - `median_risk_tuned`: improves F1 substantially via validation-based threshold adjustment → demonstrates the **importance of threshold optimisation** for clinical tasks.  
    - **Regression (pre vs post calibration)** 
      - Pre-calibration metrics highlight **systematic scale bias** in regression (negative R² despite good correlation).  
      - Post-calibration metrics demonstrate **improved alignment** with ground truth, validating linear correction in log-space. Effectively restores numerical validity (R² > 0.5) and halves RMSE. 
    - Serves as a **fully audit-ready summary** of model performance, suitable for reproducibility or reporting.

**Summary**
| Output | Role in Evaluation Pipeline | Insights Provided |
|--------|----------------------------|-----------------|
| `tcn_predictions_refined.csv` | Stores per-patient predictions and ground-truths before metrics | Enables inspection of individual prediction accuracy, monotonicity, and plausibility; supports debugging and calibration |
| `tcn_metrics_refined.json` | Stores aggregated metrics dictionary including **median-risk threshold tuning** and **post-hoc regression calibration** results. | Provides reproducible performance summaries, highlights threshold optimisation and calibration gains, and distinguishes internal (log-space) vs external (raw-space) validity. |

- Both outputs are **critical for reproducibility**, transparency, and auditability of the refined TCN evaluation.  
- They allow separation of **internal training fidelity** (log-space metrics) from **clinical interpretability** (raw-space metrics), bridging ML validation and applied use.

**Key Takeaways**
1. The refined TCN’s evaluation metrics are now **comprehensive**, covering classification, tuning, regression, and calibration.  
2. Threshold tuning for **median-risk** improved recall and F1 without retraining, demonstrating effective post-hoc adjustment.  
3. Regression calibration corrected linear scale bias and restored meaningful R² values.  
4. The `tcn_metrics_refined.json` file now serves as a **definitive quantitative record** of the model’s final, validated performance → suitable for comparison with LightGBM or NEWS2 baselines.

#### Interpretation of Results
**Metric Comparison: Default vs Tuned Threshold**
| **Metric** | **Default (0.5)** | **Tuned (0.43)** | **Interpretation** |
|-------------|-------------------|------------------|--------------------|
| **AUC** | 0.833 | 0.833 | Ranking ability unchanged → AUC reflects separability, not the chosen cutoff. |
| **F1** | 0.286 | **0.545** | Increased by +0.259 absolute (~91% relative improvement); reflects far better balance between precision and recall. |
| **Accuracy** | 0.667 | 0.667 | Unchanged → accuracy remains insensitive to improvements on minority-class predictions. |
**Meaning:**
- The model already understood which samples were riskier (high AUC).
- The low F1 (0.286) came from a mismatch between probability calibration and cutoff point, not from model weakness.
- Meaning it was either missing too many positives (low recall) or making too many false positives (low precision).
- After threshold tuning, F1 improved to 0.545, showing a better precision–recall balance, improving classification balance without retraining.

### Understanding Log-Transformed Regression Targets and Metric Interpretation
#### Why Log-Transform Regression Targets
**Log-transform:**
- During Phase 4.5, the regression target `pct_time_high` was **log-transformed** before training: `y_train_log = np.log1p(y_train)`
- This transformation was applied because `pct_time_high` is **heavily skewed** (many low values, few high values).  
- The log-transform compresses large values and expands small ones, making the distribution more normal-like and stabilising training.
**Result:**  
- The TCN model’s regression head learns to predict values in **log-space**, i.e.`y_pred ≈ log1p(y_true)`

#### What the Model Actually Outputs
Because it was trained on `log1p(y_true)`, the model outputs **log-space predictions**:
- During training: MSE is computed on `log1p(y_true)` vs `y_pred_log`
- During validation: the same (log-space validation loss)
- During inference (test): the model produces `y_pred_log`
For clinical interpretability, we invert this transformation: 
- Reverse the model output prediction back into raw-scale → `y_pred_raw = np.expm1(y_pred_log)`

#### The Four Key Vectors
When evaluating the model, we now have four aligned arrays:

| Variable | Description | Scale |
|-----------|--------------|--------|
| `y_true_reg` | True test targets | Raw clinical scale |
| `y_true_reg_log` | Log-transformed true targets | Log-space |
| `y_pred_reg_log` | Model predictions (native output) | Log-space |
| `y_pred_reg_raw` | Back-transformed predictions (`expm1`) | Raw scale |


#### Why Compute Both Log-Space and Raw-Space Metrics

| Type | Purpose | What It Shows | Comparability |
|------|----------|---------------|----------------|
| **Log-space metrics** | Internal validation | Confirms model learned the intended training objective (MSE on log targets) | ✅ Valid only within this model version |
| **Raw-space metrics** | Clinical interpretability | Shows how well the model performs in the real, clinical scale | ✅ Comparable across all models (Phase 4, LightGBM, NEWS2, etc.) |

**Interpretation:**
- Log metrics → “Did my model learn the right thing?”
- Raw metrics → “Are my predictions meaningful in practice?”
**Both are essential:**
- Without **log metrics**, you can’t verify the model trained correctly.
- Without **raw metrics**, you can’t compare across models or assess real-world performance.

#### Why Log and Raw Metrics Differ
**The log transform is nonlinear**:
- Large values are compressed.
- Small values are expanded.
**Thus, errors are weighted differently**:
- A small absolute difference in log-space can translate to a large raw-space error after exponentiation.
- Consequently, metrics like RMSE or R² differ numerically between log and raw scales.
**They are not supposed to match:**
- Different scales → different error weighting → different interpretations.


#### Interpreting Metric Combinations

| Log Metrics | Raw Metrics | Interpretation | Comparability | Clinical Interpretability |
|--------------|--------------|----------------|----------------|-----------------------------|
| ✅ Good | ✅ Good | Model trained correctly **and** clinically reliable. Safe for comparison with other models. | ✅ | ✅ |
| ✅ Good | ❌ Bad | Model learned well in log-space, but raw predictions diverge → unstable after exponentiation. | ❌ | ⚠️ |
| ❌ Bad | ✅ Good | Highly unlikely; indicates pipeline or evaluation bug. | ❌ | ⚠️ |
| ❌ Bad | ❌ Bad | Model failed entirely. | ❌ | ❌ |

**Only when both are reasonable can you compare to Phase 4, LightGBM, or NEWS2.**


#### Practical Rules for Evaluation
1. Always compute **both log and raw regression metrics** for log-trained models.  
2. If **log metrics are bad**, training was unstable → raw metrics cannot be trusted.  
3. If **log metrics are good but raw metrics are poor**, model generalisation breaks after exponentiation → review data scale or heteroscedasticity.  
4. Only use **raw-space metrics for cross-model comparison** (across Phase 4, Phase 4.5, LightGBM, NEWS2).  
5. **Report both in tables and specify**:
   - Log metrics = internal validation only  
   - Raw metrics = clinically interpretable and cross-model comparable

#### Conceptual Summary
| Dimension | Log-Space | Raw-Space |
|------------|------------|------------|
| Objective | Minimise MSE(log(y)) | Interpret true scale predictions |
| Evaluates | Technical correctness | Clinical utility |
| Sensitivity | Equal weighting of small % differences | Higher penalty for large raw errors |
| Use Case | Model debugging, internal analysis | Comparison and deployment |

#### Key Takeaway
- **Log metrics** prove training pipeline is mathematically correct.  
- **Raw metrics** prove model’s outputs are *clinically useful and comparable*.  
- Both are required for scientific validity.  
- Comparisons with baseline models (Phase 4, LightGBM, NEWS2) must use **raw-space metrics only**.

# Phase 6: Visualisation, Comparison & Finalisation

---

## Phase 6: Visualisation, Comparison & Finalisation (Steps 1-4)
### Goal 
- To synthesise all evaluation outputs from Phase 5 into summary metrics, numerical data, visualisations, and interpretability artefacts. 
- This phase transforms raw metrics into human-readable scientific insights, allowing for quantitative analysis and interpretatability, completing the machine-learning pipeline ready for deployment.

### 1. **Comparative Analysis: Create Summary Metrics (`performance_analysis.py`)**
  - **Purpose**
    - Provides the primary empirical benchmark between **LightGBM** and **TCN_refined** across three ICU deterioration targets.
    - This step consolidates classification + regression metrics into one structured file, combining pre-computed JSON metrics (AUC, F1, RMSE, R²) with newly calculated calibration diagnostics (Brier score, ECE). 
    - It isolates the numerical differences that matter most (AUC, RMSE, ECE, etc.) before adding any visual context, it is therefore the **primary evidence base** for model evaluation.
    - It ensures numerical alignment, consistency, and traceability, serving as the reference backbone for the entire comparative analysis.
  - **Process (Summary)**  
    1. **Setup & Data Integrity**
      - Loads LightGBM and TCN predictions + metrics JSONs.  
      - Confirms identical row alignment and ground-truth parity.  
      - Defines explicit column mappings to prevent mis-referencing (e.g., TCN `y_pred_reg_raw`).
    2. **Utility Functions**
      - `expected_calibration_error()` → computes bin-wise reliability gaps (10 bins over [0, 1]); weighted mean gap = ECE. Lower ECE = better calibration.
      - `kde_1d()` → 1-D Gaussian kernel density estimator for residuals.  
        Provides smooth residual curves used later in regression plots.
    3. **Metric Collection**
      - `collect_class_metrics()` → merges pre-computed ROC/F1 metrics with in-script Brier & ECE.  
      - `collect_regression_metrics()` → extracts RMSE & R² from JSON.  
      - Output dictionaries merged into one comparison DataFrame.
    4. **Output**
      - Saves unified `comparison_table.csv` to `comparison_etrics/` containing all model/target metrics:
        | Discrimination | Calibration | Regression Fidelity |
        |----------------|--------------|---------------------|
        | ROC AUC, F1 | Brier, ECE | RMSE, R² |
  - **Reasoning**: 
    - To perform the **primary, quantitative comparison** between LightGBM and TCN_refined across all three targets.  
    - This step unifies all scalar performance indicators discrimination (AUC, F1), calibration (Brier, ECE), and regression fidelity (RMSE, R²) into one structured, validated table.
      - **Discrimination:** how accurately each model ranks high-risk vs low-risk patients (ROC AUC, F1, Accuracy, Precision, Recall).  
      - **Calibration:** how well the predicted probabilities reflect actual event frequencies (Brier, ECE).  
      - **Regression fidelity:** how precisely each model predicts continuous deterioration exposure (`pct_time_high`).  
    - It provides the **most important and interpretable layer of analysis**, establishing which model performs better and by how much, based purely on objective summary metrics.  
    - **All subsequent visualisation work (Step 2) exists to support and contextualise these quantitative findings.**

### 2. **Comparative Analysis: Generate Visualisations & Numeric Plot Data (`performance_analysis.py`)**
  - **Purpose**:
    - Extend Step 1’s scalar metrics and builds into visual and numerical diagnostics:
      - Step 1 → establishes who performs better overall* (numerical summary).  
      - Step 2 → explains how and why those performance differences arise (shape-level analysis). 
    - Generates all classification and regression visualisations, each paired with machine-readable CSVs containing the arrays used to render the figures, ensuring **full reproducibility without relying on PNG inspection**.
  - **Process (Summary)**  
    1. **Classification Visualisations**
      - **Plots and Data Generated**
      | Plot Type | CSV Output | Insight Provided |
      |------------|-------------|------------------|
      | ROC Curve | `roc_<target>.csv` | Discrimination across thresholds (FPR/TPR + AUC). |
      | Precision–Recall | `pr_<target>.csv` | Positive-class sensitivity vs precision (AP). |
      | Calibration Curve | `calibration_<target>.csv` | Reliability of predicted probabilities (Brier, ECE, bin counts). |
      | Probability Histogram | `prob_hist_<target>.csv` | Distribution + summary stats (mean, std, skew, kurtosis). |

      - All curves are saved both as PNGs (`roc_*.png`, `pr_*.png`, `calibration_*.png`, `prob_hist_*.png`) and CSVs with aligned arrays for both models.
    2. **Regression Visualisations**
      - **Plots and Data Generated**
      | Plot Type | CSV Output | Description / Diagnostic Value |
      |------------|-------------|--------------------------------|
      | True vs Predicted Scatter | `scatter_pct_time_high.csv` | Alignment to identity line → accuracy + bias. |
      | Residual Histogram + KDE | `residuals_pct_time_high.csv` + `residuals_kde_pct_time_high.csv` | Error distribution shape (mean, spread, skew) via histogram and smoothed KDE curves. |
      | Error vs Truth Scatter | `error_vs_truth_pct_time_high.csv` | Conditional bias patterns (residual vs actual). |

      - All regression plots saved in `comparison_plots/` as:  
      `scatter_pct_time_high.png`, `residuals_pct_time_high.png`, `error_vs_truth_pct_time_high.png`.
    3. **Metric Comparison Charts**
      - Grouped bar charts summarising model differences per target:
        - `metrics_comparison_max_risk.png` → AUC, F1, Brier, ECE  
        - `metrics_comparison_median_risk.png` → same metrics (TCN uses `median_risk_tuned`)  
        - `metrics_comparison_pct_time_high.png` → RMSE, R²
  - **Outputs Summary**
    - **Numeric Data → `comparison_metrics/`**
      - `comparison_table.csv` + 12 plot-level CSVs  
      - All include side-by-side model values with padded NaNs for length alignment.  
    - **Plots → `comparison_plots/`**
      - 14 PNGs covering classification, regression, and summary bar charts.
    - **Together:**
      - Step 1 → aggregated summary metrics.  
      - Step 2 → numeric and visual diagnostics for interpretation.  
      - Both form the first half of Phase 6: the **complete comparative evaluation layer** preceding interpretability (Steps 3 onwards).
  - **Reasoning**
    - To perform the **second part of the comparative analysis**, extending Step 1’s scalar summary metrics into **fine-grained, numeric + visual diagnostics**.  
    - This step provides detailed evidence of *why* one model performs differently → showing trends, distributions, and calibration shapes that scalar metrics alone cannot capture.
    - While this analysis is secondary in interpretive hierarchy, it provides essential supporting evidence → particularly for understanding calibration behaviour, residual patterns, and threshold-dependent discrimination.
  - **Analytical Hierarchy**
    | Layer | Step | Analytical Role | Interpretation Weight |
    |-------|------|------------------|------------------------|
    | **1. Summary Metrics** | Step 1 | Core numerical comparison — establishes direction and magnitude of model differences. | **Primary (quantitative evidence)** |
    | **2. Plot Numerics + Visuals** | Step 2 | Explains why metrics differ via shape-level behaviour and calibration structure. | **Secondary (interpretive support)** |
    
    - Together, Steps 1 + 2 provide a complete and scientifically rigorous comparative framework:
      - **Step 1** → definitive quantitative benchmark (numbers).  
      - **Step 2** → diagnostic context (visual patterns). 
  - **Overall Summary**
    - This completed script (steps 1 and 2) collectively:
      - Transform raw evaluation metrics into a structured scientific comparison.  
      - Quantify **discrimination**, **calibration**, and **error behaviour**.  
      - Provide both **numerical reproducibility** (CSV) and **visual interpretability** (PNG).  
      - Ensure that every figure corresponds to precise, traceable numeric data.  
    - Produces a comprehensive, validated foundation → merging rigorous quantitative benchmarking with interpretable visual analytics
    - Explainability and inference analyses (Step 3) will build upon this first stage of analysis.

### 3. **LightGBM SHAP Interpretability (`shap_analysis_lightgbm.py`)**
  - **Purpose**: 
    - Script provides **final interpretability** for all trained LightGBM models, by quantifying each clinical feature’s contribution to predictions using **SHAP (SHapley Additive exPlanations)** values.
    - Deliver transparent, reproducible feature-level insights for all three targets
  - **Process (Summary)**:
    1. **Data + Model Loading**
      - Load processed patient-level features (`news2_features_patient.csv`) and training IDs (`patient_splits.json`).
      - Recreate binary target definitions for consistency.
      - Exclude non-feature columns to define the exact feature set used in training.
    2. **Model Iteration Loop**
      - For each target (`max_risk`, `median_risk`, `pct_time_high`):
        - Load corresponding trained LightGBM model (`.pkl`).
        - Extract training features (`X_train`).
        - Compute SHAP values using:
          ```python
          explainer = shap.TreeExplainer(model)
          shap_values = explainer.shap_values(X_train)
          ```
        - Handle classification vs regression outputs:
          - Use `shap_values[1]` for binary classifiers (positive/high-risk class).
          - Use single array for regression (`pct_time_high`).
    3. **Feature Importance Aggregation**
      - Convert SHAP matrix → global importances:
        ```python
        mean_abs_shap = np.abs(shap_array).mean(axis=0)
        ```
      - Summarise in ranked DataFrame (`feature`, `mean_abs_shap`) and save as CSV.
    4. **Visualisation**
      - Generate bar plots of top 10 features per target for quick visual interpretation.
      - Save plots and CSVs to `interpretability_lightgbm/`.
    5. **Diagnostics**
      - Shape checks (`X_train`, `shap_array`) and aggregation tests ensure correct dimensionality (70×40) and non-collapsed SHAP arrays.
  - **Reasoning**
    - **Why SHAP:** SHAP decomposes each model prediction into additive feature contributions, giving a mathematically rigorous explanation of model behaviour.  
    - **Why TreeExplainer:** Exact, fast computation for tree-based models, avoids approximations used by model-agnostic explainers, ensures feature attributions align with actual LightGBM decision paths.  
    - **Why Mean Absolute SHAP:** Stable, direction-agnostic summary of each feature’s average influence → enables clean, comparable feature ranking across targets.  
    - **Why on Training Set:** Provides global interpretability of what the model actually learned → avoids instability from small test-set SHAP computation. 
  - **Outputs**
    - All outputs are written to: `src/results_finalisation/interpretability_lightgbm/`
    - Each CSV provides precise quantitative feature importance values; plots offer visual interpretability for reporting and documentation.
    | **File** | **Description** |
    |-----------|-----------------|
    | 3x `<target>_shap_summary.csv` | Ranked table of mean |SHAP| values (global feature importance) per target classifier/regressor. |
    | 3x `<target>_shap_summary.png` | Top 10 feature bar plot per target. |
  - **Summary**
    - Script produces the definitive interpretability layer for the final LightGBM models (the exact models evaluated in Phase 5) by computing per-patient, per-feature SHAP attributions and summarising them into stable global rankings. 
    - Gives us transparent explanations: what features drive predictions (global ranking) and how strongly (mean |SHAP|). 
    - It closes the loop between numeric performance (Steps 1–2) and clinical interpretation, enabling evidence-backed statements about why one model behaves better or worse on particular targets.
    - For interpretability, these results are then saved into csv files for reproducibility, and saved as png files for easy visualisations.
    - These results explain why the LightGBM models produce the predictions used in the comparative analysis (Steps 1–2).

### 4. **TCN Saliency Interpretability (`saliency_analysis_tcn.py`)**
  - **Purpose**
    - Compute gradient × input saliency maps (|∂y/∂x × x|) for the refined TCN on the held-out test set to quantify how each input feature and timestep contributes to driving predictions for the three output heads (`max_risk`, `median_risk`, `pct_time_high`).
    - Extends interpretability beyond LightGBM SHAP (Phase 6 Step 3) to temporal reasoning in the sequence model.
    - Produce reproducible numerical and visual interpretability artefacts, this provides a temporal interpretability layer complementing the static SHAP analysis, allowing us to compare saliency vs SHAP.
  - **Process (Summary)**
    - Import the trained **TCNModel** from `tcn_model.py`.
    - Validate environment and required files, and create `interpretability_tcn/` for outputs. The script confirms presence of:
      - Trained model checkpoint: `tcn_best_refined.pt`
      - Model configuration: `config_refined.json`
      - Test tensors: `test.pt`, `test_mask.pt`
      - Preprocessing metadata: `padding_config.json`, `standard_scaler.pkl`, `patient_splits.json`
    - Load device (CPU or GPU), load model configuration (architecture and parameters), load padding configuration (`feature_cols`, `max_seq_len`, `target_cols`), and load test tensors.
    - Rebuild the TCN architecture from `config_refined.json` and load weights from `tcn_best_refined.pt`. Move model and tensors to GPU if available and set `model.eval()`.
    - Define targets mapped to model heads:
      - `max_risk` (`logit_max`)
      - `median_risk` (`logit_median`)
      - `pct_time_high` (`regression`)
    - Compute saliency via helper function `compute_saliency_for_batch()`:
      - Inputs: trained model, test tensor (`x_batch`), mask tensor (`mask_batch`), and target head name.  
      - Enables gradient tracking (`.requires_grad`) on inputs `(B, T, F)`, performs forward pass, extracts target head tensor `(B,)`.
      - Loops through patients to compute gradients for each one, backpropagating to obtain gradients `(T, F)` of the chosen output w.r.t. inputs.
      - Combines gradients with input values and takes absolute value (`|grad × input|`).
      - Returns per-batch saliency tensors of shape `(B, T, F)`.
    - Iterate through the test set in batches (`batch_size=4`). For each target head, compute and concatenate per-patient saliency across all batches to form a full `(n_test, T, F)` saliency array.
    - Generate outputs for each target:
      - **Feature-level mean + std CSV:** mean and standard deviation of saliency per feature.
      - **Temporal mean profile CSV:** mean saliency per timestep.
      - **Top-5 features temporal CSV:** temporal saliency profiles of top features.
      - **Top-10 features global mean heatmap PNG:** visual summary of mean saliency over time.
    - Export all numeric summaries and visualisations from the aggregated saliency array.
  - **Reasoning**
    - Absolute gradient×input combines sensitivity (∂y/∂x) with observed activity (x). This yields a magnitude-based, local measure of influence that is model-specific and comparable across features and timesteps.
    - Per-target head analysis isolates which inputs drive each clinical prediction. Batch processing balances GPU/CPU memory with reproducibility.
    - Aggregation (feature means/std, timestep means, top-feature temporal profiles, top-10 heatmaps) provides layered interpretability: global feature importance, temporal sensitivity, and focused temporal behaviour for the most influential predictors.
  - **Outputs**
    - For each target head the script writes four reproducible files under `interpretability_tcn/`:
      1. `{target}_feature_saliency.csv` — columns: `feature`, `mean_abs_saliency`, `std_abs_saliency` (mean and variability across patients and timesteps).
      2. `{target}_temporal_saliency.csv` — columns: `timestep`, `mean_abs_saliency` (mean across patients and features for each timestep).
      3. `{target}_top_features_temporal.csv` — `timestep` plus the top 5 feature columns showing mean saliency per timestep for each top feature.
      4. `{target}_mean_heatmap.png` — log1p-scaled heatmap of the top 10 features across timesteps (color scale clipped to 5th–95th percentiles for stability).
    - Terminal diagnostics printed per head: NaN count, global mean/max saliency, and inter-head output correlations to confirm head distinctiveness.
  - **Summary**
    - `saliency_analysis_tcn.py` is a modular, reproducible interpretability stage. It rebuilds the trained TCN, computes absolute gradient×input saliency per patient/timestep/feature, and exports concise numeric and visual artefacts for reporting and comparative analysis with SHAP. 
    - The outputs quantify which features the TCN uses and when those features matter in the patient timeline, adding temporal context that complements LightGBM SHAP explanations, allowing for the greatest clinical interpretability.

### End Products of Phase 6
**Summary**
- By completion of Phase 6, the project achieved full comparative and interpretability finalisation for both models (LightGBM and refined TCN). 
- All results were fully analysed, visualised, and interpreted quantitatively and qualitatively.  
- This phase completes the **core analytical cycle**:  
  - Raw metrics → Comparative analysis → Interpretability → Reflection and insight generation.  
  - All results are traceable, interpretable, and publication-ready.
**Deliverables and Artefacts:**
| Category | Outputs | Description |
|-----------|----------|--------------|
| **Performance Metrics** | `comparison_table.csv` | Unified metrics table for discrimination (AUC, F1), calibration (Brier, ECE), and regression fidelity (RMSE, R²). |
| **Classification Plots** | `roc_*.png`, `pr_*.png`, `calibration_*.png`, `prob_hist_*.png` + CSVs | Complete classification visualisations and underlying numeric data. |
| **Regression Plots** | `scatter_pct_time_high.png`, `residuals_pct_time_high.png`, `error_vs_truth_pct_time_high.png` + CSVs | Regression diagnostics for residuals and calibration. |
| **Summary Charts** | `metrics_comparison_*.png` | Grouped comparison plots summarising all metrics per model and target. |
| **LightGBM Interpretability** | `*_shap_summary.csv`, `*_shap_summary.png` | Global feature-level importance for each target via mean |SHAP| values. |
| **TCN Interpretability** | `*_feature_saliency.csv`, `*_temporal_saliency.csv`, `*_top_features_temporal.csv`, `*_mean_heatmap.png` | Gradient×input saliency outputs showing feature and temporal influence across TCN output heads. |
| **Diagnostics and Analysis** | Reflections, comparisons, and interpretive summaries (`notes.md`) | Detailed write-up describing model behaviour, interpretability alignment (SHAP vs Saliency), and performance conclusions. |
**Analytical Endpoints:**
- Completed quantitative analysis across discrimination, calibration, and regression fidelity.
- Completed interpretability synthesis linking SHAP (global, static) with saliency (temporal, global).
- Documented reflection on interpretability consistency, divergence, and model behaviour patterns.
- Provided full reproducibility through paired PNG–CSV artefacts.

### Why Not Further
**Scope rationale:**
1. **Analytical sufficiency:**  
  - Metrics computed already cover the full standard evaluation spectrum:
    -	**Discrimination:** ROC AUC, F1, Precision–Recall, Accuracy, Precision, Recall.
    -	**Calibration:** Brier score, Expected Calibration Error (ECE).
    -	**Regression fidelity:** RMSE, R².
	-	These together provide a complete empirical description of model performance.
	-	Extending further (e.g., MCC, Cohen’s κ, AUROC confidence intervals) would add redundancy without providing new interpretive value beyond what has already been demonstrated through ROC and PR curves.
2. **Interpretability completeness:**  
	-	SHAP captures static/global feature attributions for LightGBM.
	-	Gradient×Input Saliency captures temporal/local attributions for TCN.
	-	Together, they already reveal both what features are important and when they influence predictions; the two fundamental interpretability dimensions in clinical sequence modelling.
	-	Introducing more interpretability methods (e.g., Integrated Gradients via Captum, occlusion tests, or LIME) would only replicate insights already established by the SHAP–Saliency combination.
3. **Design economy:**  
  - Per-patient saliency maps and raw tensor saves were removed to avoid redundancy and noise. Aggregated outputs and temporal CSVs already convey clinically meaningful and reproducible results.
  - Visual scaling refinements (percentile clipping, log scaling, improved colormaps) ensured interpretable visual artefacts without excessive re-engineering.
4. **Pipeline integrity:**  
  - Phase 6 already integrates data processing → model training → evaluation → interpretability → synthesis.  
  - Introducing further layers (e.g., autoencoders, ensemble interpretability, or SHAP–saliency fusion) would break the scope of a clean ML pipeline demonstration, since existing outputs already show all relevant performance behaviour.
**Conclusion:**  
- All analyses implemented are deliberate, sufficient, and aligned with standard ML evaluation pipelines.
- Every model behaviour dimension—performance, calibration, regression fidelity, and interpretability—has been covered with full reproducibility and clarity.
- Further additions would only increase redundancy, not insight.

### Next Steps / Project Completion Pathway
**Overview**
- With Phase 6 complete, the analytical and interpretive work of the project is fully concluded. 
- The next and final stage transitions from research and analysis to deployment and operationalisation.
**Upcoming Stages**
| Phase | Component | Description | Outcome |
|-------|------------|--------------|----------|
| **Phase 7A** | **Inference Demonstration (Deployment-Lite)** | Implement a command-line inference pipeline (`run_inference.py`). Loads saved models, preprocessors, and returns risk predictions for given patient inputs. | Demonstrates full end-to-end pipeline usability, showing ability to package ML models into reproducible inference workflows. |
| **Phase 7B** | **Cloud Deployment (Render)** | Host a lightweight live version via FastAPI (or similar) with containerisation. Exposes endpoints for inference, showing MLOps and cloud deployment skill. | Demonstrates deployment engineering, API design, and CI/CD readiness. |
**Rationale for This Transition**
- **End-to-end demonstration:** Deployment validates reproducibility and model usability beyond training.  
- **Industry relevance:** MLOps and deployment skillsets (packaging, inference, API exposure) are essential to professional ML engineering roles.  
- **Strategic skill signalling:** The staged deployment (inference now → live API later) mirrors professional pipelines: local validation → cloud production.  
**Target Deliverables for Phase 7**
- `run_inference.py`(Deployment-Lite)
- Packaged model assets (`.pt`, `.pkl`, `config_refined.json`, `padding_config.json`)
- Later cloud API on Render for live demonstration (Phase 7B)
- Updated `README.md` detailing the full ML lifecycle: preprocessing → training → evaluation → interpretability → deployment.

### Summary
- Phase 6 completes the analytical and interpretability phase of the project:
  - Both models (LightGBM and TCN_refined) have been trained, validated, and explained.
  - All evaluation artefacts, metrics, and interpretability outputs have been analysed and summarised.
  - All code modules are modular, reproducible, and pipeline-aligned.
  - The project now holds all necessary evidence of technical depth, analytical rigour, and interpretability awareness.
- Next steps:
  - Proceed to **Phase 7 – Deployment**, where the focus shifts from analysis to reproducible inference and operational demonstration. 
  - This final stage will complete the project lifecycle, integrating ML development with deployable engineering practice.


### Comparative Analysis Framework — Analytical Methodology
#### Purpose
- The comparative analysis determines **how and why** LightGBM and TCN_refined differ in performance across the three ICU deterioration targets → `max_risk`, `median_risk`, and `pct_time_high`.
- It is divided into two structured analytical layers:
  1. **Step 1 – Quantitative Summary Metrics Analysis:** Defines the core evidence of model performance using statistically grounded metrics.  
  2. **Step 2 – Numerical Diagnostic & Visualisation Analysis:** Explores why those differences occur by examining calibration shape, residual structure, and prediction behaviour.
- This layered design balances **quantitative robustness** with **diagnostic interpretability**, ensuring conclusions remain objective even with a limited dataset.

#### Step 1: Quantitative Summary Metric Analysis
**Analytical Aim**
- Step 1 establishes the **baseline comparative performance** of both models across classification and regression tasks.  
- It computes a unified, multi-metric framework to quantify each model’s strengths and weaknesses across three performance dimensions:

| Dimension | Metrics | Purpose |
|------------|----------|----------|
| **Discrimination** | ROC AUC, F1, Accuracy, Precision, Recall | Measures ability to separate deteriorating vs stable cases; identifies overall classification competence. |
| **Calibration** | Brier Score, Expected Calibration Error (ECE) | Evaluates reliability of predicted probabilities; critical for clinical decision-making. |
| **Regression Fidelity** | RMSE, R² | Quantifies how closely continuous predictions (proportion of time in high-risk state) match ground truth. |

**Analytical Weighting**
- **Threshold-independent metrics (AUC, Brier, ECE)** are prioritised since they are **robust to threshold choice** and less distorted by small-sample instability.  
- **Threshold-dependent metrics (F1, Precision, Recall)** are supportive diagnostics, providing insight into event sensitivity but less reliable when case counts are low.  
- **Regression metrics (RMSE, R²)** form the direct fidelity check on the continuous target.
**Analytical Role**
- This layer produces the **main quantitative comparison** → the definitive statement of which model performs better overall.  
- It shows:
  - Which model discriminates deterioration better (AUC, F1).  
  - Which produces more calibrated probability estimates (ECE, Brier).  
  - Which predicts continuous risk proportions more faithfully (RMSE, R²).
- Because these metrics are statistical aggregates over all patients, they are **stable, interpretable, and comparable**.  
- They represent the **primary analytical evidence** in this comparative study.

#### Step 2: Numerical Diagnostic and Visualisation Analysis
**Analytical Aim**
- Step 2 builds upon the Step 1 results to investigate **why** those metric differences exist.  
- It analyses **finer-grained numeric patterns** and model behaviours using the underlying data of each plot, not just visual images, to ensure precise, quantitative interpretability.
**Diagnostic Coverage**
- All visualisation CSVs are numerically exhaustive. 
- Each contains **every value and derived statistic** required to reproduce and analyse the trends without inspecting PNGs.
**Classification Diagnostics**
| Aspect | Numeric Content | What It Reveals |
|--------|----------------|----------------|
| **ROC Curves** | `fpr`, `tpr`, AUC | Discrimination at all thresholds; sensitivity–specificity trade-offs. |
| **Precision–Recall Curves** | `precision`, `recall`, Average Precision (AP) | Sensitivity to positive events; highlights impact of class imbalance. |
| **Calibration Curves** | Mean predicted probability, fraction of positives | Probability reliability; identifies under- or over-confidence regions. |
| **Probability Histograms** | Predicted probability distributions + descriptive stats (`mean`, `median`, `std`, `min`, `max`, `skew`, `kurtosis`) | Overall confidence spread and skewness of prediction certainty. |
**Regression Diagnostics**
| Aspect | Numeric Content | What It Reveals |
|--------|----------------|----------------|
| **True vs Predicted Scatter** | `y_true`, `y_pred` pairs + stats | Closeness of predictions to ideal y = x line. |
| **Residual Distributions + KDE** | Residual values + smoothed KDE curves | Bias direction, variance, and shape of error spread. |
| **Error vs Truth Scatter** | `y_true`, `residual` | Whether error magnitude varies systematically with true value (heteroscedasticity). |
| **Descriptive Stats** | `mean`, `median`, `std`, `min`, `max`, `skew`, `kurtosis` per residual set | Quantifies distribution asymmetry and outlier impact. |
**Analytical Role**
- Enables **precise, statistical inspection** of visual patterns without subjective interpretation of irregular plots.  
- Reveals **underlying behavioural causes** of metric differences, e.g.:
  - Why LightGBM achieved lower ECE (more evenly spread probabilities).  
  - Why TCN_refined exhibited higher RMSE (sensitivity to temporal outliers).  
- Provides redundancy: all trends can be verified directly from numeric CSVs even if plot quality is poor.
**Why Numerical Diagnostics Are Prioritised Over Visuals**
- Due to the small dataset (n = 15 patients) and limited variation, visual plots often appear **sharp, irregular, or step-like**, especially calibration and PR curves.  
- Thus, relying solely on PNGs risks **visual misinterpretation**.  
- The numerical CSVs, however, allow for:
  - Accurate statistical computation of trends.  
  - Objective comparison of curve shapes, slopes, and biases.  
  - Full analysis reproducibility without subjective visual estimation.
- In essence, **Step 2 translates visual diagnostics into measurable evidence**, refining but not redefining Step 1’s conclusions.

#### Why the Two-Step Structure
| Step | Role | Contribution |
|------|------|---------------|
| **1. Summary Metrics** | Quantitative foundation | Defines which model performs better on core dimensions using robust scalar metrics. |
| **2. Numerical Diagnostics** | Explanatory deep-dive | Explains *why* those metric differences exist through detailed numeric curve analysis. |

- This design ensures:
  - **Objectivity** → comparisons grounded in data, not visuals.  
  - **Completeness** → global performance plus local behavioural understanding.  
  - **Transparency** → every figure and trend can be traced back to numeric data.  
  - **Interpretability under constraints** → robust even with sparse, noisy datasets.

#### Integrated Interpretation Strategy
1. **Begin with Step 1** to quantify core differences (AUC, Brier, ECE, RMSE, R²).  
2. **Use Step 2** to diagnose reasons (e.g., probability skew, residual asymmetry).  
3. **Contextualise findings** given dataset limitations e.g.,  
   - TCN_refined underperformed partly due to limited temporal diversity and smaller sample size.  
   - LightGBM’s patient-level aggregation handled sparsity better, yielding smoother calibration.  
4. Synthesise both layers to conclude under what conditions each model excels and how they could complement each other clinically.

#### Summary
- **Step 1 - Summary Metrics:** primary, defines model performance hierarchy using robust, threshold-independent metrics.  
- **Step 2 - Numerical Diagnostics:** secondary, validates and explains Step 1 trends through comprehensive numeric analysis (no visual guessing).  
- **Combined:** deliver a complete, evidence-driven comparative evaluation—quantitatively decisive, diagnostically transparent, and resilient to dataset limitations.

---

### Comparison Metrics Definitions
#### Overview
- This section defines all key metrics used for model comparison between **LightGBM** and **TCN_refined**, explaining what each measures, why it was chosen, and how it was computed within the project workflow.  
- Metrics are grouped by task type: **Classification** (binary risk prediction) and **Regression** (continuous deterioration percentage prediction).

#### Threshold-dependent vs Threshold-independent Metrics
- Some performance metrics depend on the choice of a decision threshold (typically 0.5 in binary classification), while others evaluate performance across all possible thresholds.
- **Threshold-independent metrics (e.g. ROC AUC, Brier Score, Expected Calibration Error):** 
  - Assess a model’s overall discrimination and calibration and are generally more robust, especially on small test sets.
  - ROC AUC measures the model’s ability to correctly rank positive vs negative cases irrespective of any specific cut-off, Brier Score quantifies overall probabilistic accuracy, ECE captures calibration quality by evaluating how well predicted probabilities correspond to observed event frequencies.
  - These metrics are more stable on small datasets and provide the most reliable signal of generalisable model performance.
- **Threshold-dependent metrics (e.g. F1, accuracy, precision, recall):** 
  - Depend on a fixed decision boundary (commonly 0.5).
  - They reflect classification behaviour after binarisation of probabilities.
  - Are sensitive to class balance and individual prediction shifts; with only 15 patients, a single misclassified case can shift these values substantially.
  - Moreover, if two models cross the threshold in similar patterns — even with different raw probabilities — they can yield identical scores.
- **Given these constraints:**
  -	Threshold-independent metrics (AUC, Brier, ECE) form the primary evidence base for model comparison.
  -	Threshold-dependent metrics (F1, Accuracy, Precision, Recall) are included only as supportive indicators of discrete decision behaviour.
  -	The TCN (median_risk) case is an exception, as its threshold was explicitly tuned → hence differences in F1 and Accuracy are meaningful and reflect genuine model optimisation, not noise.
  - Therefore, in this analysis, threshold-independent metrics form the primary basis for quantitative comparison, while threshold-dependent metrics provide supportive, descriptive context.

#### Classification Metrics
**Purpose**
-	These metrics evaluate model performance for binary outcomes derived from two per-patient summaries of predicted deterioration risk:
	-	`max_risk` → the maximum predicted probability across all timestamps for a patient (reflects their peak deterioration risk).
	-	`median_risk` → the median predicted probability across all timestamps for a patient (reflects their typical or baseline level of risk).
-	Each summary is then binarised using a clinically meaningful threshold derived from the NEWS2 framework:
	-	`max_risk` → 0 = did not reach high risk, 1 = reached high risk at any point.
	-	`median_risk` → 0 = low-risk profile, 1 = moderate-risk profile (i.e. typical sustained risk elevation).
- Thus, `max_risk` highlights transient acute risk spikes, while `median_risk captures` persistent elevation → both representing complementary clinical dimensions of deterioration.
**Computation**
- Each metric was originally computed in **Phase 5**:
  - For LightGBM: `evaluate_lightgbm_testset.py`
  - For TCN_refined: `evaluate_tcn_testset_refined.py`
- Both scripts import their metric functions from `evaluation_metrics.py`.
- However, **Brier score** and **Expected Calibration Error (ECE)** were computed later in **Phase 6** (`performance_analysis.py`) because they require raw per-patient probability predictions, which are not stored in the phase 5 JSONs.
- **TCN_refined details**
- A threshold-tuning step was added to adjust for the effect of class weighting (`pos_weight`).
- The optimal classification threshold was selected (threshold=0.43) to maximise F1 score on the validation set, rather than using the default 0.5.
- This tuning was applied specifically to the `median_risk` target, as its output distribution was more imbalanced and required calibrated threshold adjustment to avoid excessive false negatives.
- In contrast, LightGBM and TCN (`max_risk`) used the standard 0.5 threshold.

1. **ROC AUC (Area Under the Receiver Operating Characteristic Curve)**
- **Definition:** Measures the model’s ability to discriminate between positive (1) and negative (0) classes across all probability thresholds.
- **Purpose:** Indicates how well the model separates high-risk from low-risk patients independent of the decision threshold.
- **Rationale for inclusion:**  
  - Threshold-independent, robust to class imbalance.  
  - A clinically interpretable measure of ranking quality (how well higher probabilities correspond to sicker patients).  
  - Used as the core discriminative metric across all binary classification tasks.
- **Implementation:**  
  - Computed in Phase 5 using `roc_auc_score(y_true, y_prob)` from scikit-learn.  
  - The final values were stored in JSON files for both models.

2. **F1 Score**
- **Definition:** 
  - Harmonic mean of Precision and Recall: `F1 = 2 * (Precision * Recall) / (Precision + Recall)`
- **Purpose:** Balances false positives and false negatives into a single measure.  
- **Rationale for inclusion:**  
  - Especially meaningful in healthcare where both sensitivity and specificity matter.  
  - Penalises models that achieve high recall at the cost of many false alarms.  
  - Serves as a decision-threshold-dependent summary of classification quality.
- **Implementation:** 
  - Calculated in Phase 5 via `f1_score(y_true, y_pred)`, where `y_pred` is the binary label obtained from a chosen probability threshold (often 0.5).
  - `median_risk` F1 was calulated with its own calculated best probability threshold.

3. **Accuracy**
- **Definition:** 
  - Proportion of total predictions that are correct: `(TP + TN) / (TP + FP + FN + TN)`
- **Purpose:** Provides a simple view of overall correctness.
- **Rationale for inclusion:**  
  - Widely understood and easy to interpret.  
  - However, less informative under class imbalance, hence paired with AUC and F1 for balance.
- **Implementation:** Computed in Phase 5 via `accuracy_score(y_true, y_pred)`.

4. **Precision**
- **Definition:** Fraction of predicted positives that are true positives: `TP / (TP + FP)`
- **Purpose:** Measures reliability of positive predictions → i.e., when the model says “high risk,” how often is it right?
- **Rationale for inclusion:**  
  - High precision indicates low false alarm rate → clinically useful for avoiding alarm fatigue.  
  - Complements recall for decision threshold tuning.
- **Implementation:** Calculated in Phase 5 using `precision_score(y_true, y_pred)`.

5. **Recall (Sensitivity)**
- **Definition:** Fraction of actual positives correctly identified: `TP / (TP + FN)`
- **Purpose:** Reflects how sensitive the model is to true deterioration events.
- **Rationale for inclusion:**  
  - Critical in medical triage: missing a deteriorating patient (false negative) is more harmful than a false alarm.  
  - Ensures patient safety emphasis in evaluation.
- **Implementation:** Computed in Phase 5 via `recall_score(y_true, y_pred)`.

6. **Brier Score**
- **Definition:**  
  - Mean squared difference between predicted probability and actual outcome: `Brier = mean((y_prob - y_true)²)`
- **Purpose:** 
  - Quantifies calibration error: how close the predicted probabilities are to true outcome frequencies.
- **Rationale for inclusion:**  
  - Unlike AUC, it directly evaluates the probabilistic accuracy of predictions.  
  - Lower values = better calibration and reliability.  
  - Important when probabilities are to be used for clinical decision thresholds or risk stratification.
- **Implementation:**  
  - Computed **only in Phase 6** (`performance_analysis.py`) using: `brier_score_loss(y_true, y_prob)`  
  - This required raw per-patient probabilities available in the prediction CSVs, not just aggregated metrics.

7. **Expected Calibration Error (ECE)**
- **Definition:** Measures average deviation between predicted probability and observed frequency, averaged across discrete probability bins.
- **Purpose:** Quantifies calibration performance more intuitively than Brier, reflecting expected discrepancy between model confidence and reality.
- **Rationale for inclusion:**  
  - Clinically crucial → overconfident models can lead to misinformed treatment decisions.  
  - Provides a complementary, interpretable calibration metric.  
  - Common in deep learning evaluation literature.
- **Implementation:**  
  - Computed **only in Phase 6**, within `performance_analysis.py` using a custom manual implementation (`expected_calibration_error` function).  
  - This was necessary because scikit-learn does not include a built-in ECE, and it relies on the actual probability distribution from the predictions CSV.

#### Regression Metrics
**Purpose**
- These metrics evaluate model performance on continuous predictions, specifically the **percentage of time a patient spent above the NEWS2 high-risk threshold (`pct_time_high`)**.
- This continuous target quantifies how long a patient remained at clinically concerning levels, providing a finer-grained assessment than binary classification metrics.
**Computation**
- Regression metrics were precomputed in Phase 5 for each model and stored in their respective JSON files, since both `y_true_reg` and `y_pred_reg` arrays were already available at that stage.
- In Phase 6 (`performance_analysis.py`), these metrics were not recomputed, only visualised and compared between models for consistency and interpretability.
- **Refined TCN details**
	-	For the refined TCN, the regression output was log-transformed prior to training to correct for heavy right-skew (many patients having near-zero deterioration time).
	-	Predictions were made in log space, then back-transformed to raw space; ground-truth values were also converted to log form, enabling both log-space and raw-space evaluations.
	-	A calibration step was then applied, producing calibrated raw metrics to correct systematic prediction bias.
	-	For final comparison, only the calibrated raw metrics (`"pct_time_high_raw_cal"`) were used because:
	  - Only the raw-space predictions are directly comparable to LightGBM.
	  -	Calibration ensures the fairest and most clinically reliable comparison.
	-	The log-space metrics are retained internally for consistency checks and auditability but are not used for cross-model benchmarking.

1. **RMSE (Root Mean Squared Error)**
- **Definition:**  
  - Square root of the mean squared difference between predicted and true values.  
  - `RMSE = sqrt(mean((y_pred - y_true)²))`
- **Purpose:** Measures average magnitude of prediction error → how far predictions deviate from true deterioration percentage.
- **Rationale for inclusion:**  
  - Sensitive to large errors → penalises outliers strongly.  
  - Provides a clear measure of overall fit accuracy.  
  - A standard regression metric directly comparable across models.
- **Implementation:** Computed in Phase 5 via `mean_squared_error(y_true, y_pred, squared=False)` and stored in JSON under `"pct_time_high"` (LightGBM) and `"pct_time_high_raw_cal"` (TCN).

2. **R² (Coefficient of Determination)**
- **Definition:**  
  - Represents proportion of variance in the true values explained by the model’s predictions.  
  - `R² = 1 - (Σ(y_true - y_pred)² / Σ(y_true - mean(y_true))²)`
- **Purpose:** Measures explanatory power → how well the model captures overall variability.
- **Rationale for inclusion:**  
  - Easy to interpret (closer to 1 = better fit).  
  - Complements RMSE by indicating relative fit quality rather than absolute error magnitude.
- **Implementation:**  
  - Computed in Phase 5 using `r2_score(y_true, y_pred)` and stored in JSON.  
  - Not recomputed in Phase 6 since it depends only on regression predictions already summarised numerically.

#### Why Brier and ECE Were Deferred to Phase 6
- **Brier** and **ECE** require raw continuous probability outputs (`prob_max`, `prob_median`), not binary thresholded predictions.  
- Phase 5 (`evaluate_*_testset.py`) scripts only output **aggregated metrics** → the per-patient prediction data were saved separately as CSVs.  
- Therefore, Phase 6 (`performance_analysis.py`) reloaded the raw predictions and computed these calibration metrics directly, ensuring accuracy and traceability.
- This design maintains **pipeline modularity**:
  - **Phase 5** = core model evaluation (discrimination & accuracy).
  - **Phase 6** = cross-model analysis (calibration + interpretability + plotting).

#### Summary

| Metric | Type | Threshold Dependence | Computed In | Purpose |
|:--|:--|:--|:--|:--|
| **ROC AUC** | Classification | **Independent** | Phase 5 | Measures overall discrimination ability across all thresholds. |
| **F1** | Classification | **Dependent** | Phase 5 | Balances precision and recall at the chosen threshold. |
| **Accuracy** | Classification | **Dependent** | Phase 5 | Proportion of correct classifications (true positives + true negatives). |
| **Precision** | Classification | **Dependent** | Phase 5 | Reliability of positive predictions (how many predicted positives are correct). |
| **Recall (Sensitivity)** | Classification | **Dependent** | Phase 5 | Proportion of true positives correctly identified. |
| **Brier Score** | Classification (Calibration) | **Independent** | Phase 6 | Measures probabilistic accuracy → mean squared error of predicted probabilities. |
| **Expected Calibration Error (ECE)** | Classification (Calibration) | **Independent** | Phase 6 | Quantifies how well predicted probabilities match observed outcome frequencies. |
| **RMSE** | Regression | **Independent** | Phase 5 | Root mean squared error → magnitude of prediction error. |
| **R²** | Regression | **Independent** | Phase 5 | Proportion of variance in true values explained by predictions. |

---

### Quantitative Analysis (Summary Metrics)
#### Overview
- This section interprets the summary performance metrics for **LightGBM** and **TCN_refined** across all evaluated tasks:  
  - **Classification (`max_risk`)** → whether the patient ever reached high deterioration risk during their stay.  
  - **Classification (`median_risk`)** → the patient’s typical or central risk level over their admission.  
  - **Regression (`pct_time_high`)** → the proportion of total admission time spent in high-risk states.  
- These definitions are essential for interpreting the numbers correctly:  
  - `max_risk` captures **peak event detection**, `median_risk` reflects **overall physiological stability**, and `pct_time_high` quantifies **temporal exposure to high-risk states**.  
  - The analysis focuses on **numerical indicators** only, without visual trends.

#### Metrics Comparison 
1. **Classification (`max_risk`)**
| Model | ROC AUC | F1 | Accuracy | Precision | Recall | Brier | ECE |
|:--|--:|--:|--:|--:|--:|--:|--:|
| LightGBM | 0.846 | 0.929 | 0.867 | 0.867 | 1.000 | **0.097** | **0.116** |
| TCN_refined | **0.923** | 0.929 | 0.867 | 0.867 | 1.000 | 0.101 | 0.149 |
**Interpretation**
- **Discrimination (ROC AUC):**  
  - `max_risk` asks whether a patient ever reached high deterioration risk during admission.  
  - TCN_refined achieves a **9.1% higher AUC** (0.923 vs 0.846), meaning it ranks patients who experienced deterioration more distinctly than LightGBM.  
  - This aligns with the TCN’s temporal sensitivity → it captures transient peaks, making it particularly well-suited to event-based outcomes.
- **Threshold-dependent performance (F1, Accuracy, Precision, Recall):**  
  - Both models produced identical scores (F1 = 0.93, Accuracy = 0.87, Precision = 0.87, Recall = 1.0).  
  - This uniformity arises because all positive cases were consistently detected at threshold 0.5 → not because both models behave identically probabilistically.  
  - It simply means both flagged the same patients as “ever high risk.”
- **Calibration (Brier, ECE):**  
  - LightGBM shows slightly better alignment between predicted and observed probabilities (Brier ↓4%, ECE ↓22%), achieves marginally better calibration.
  - TCN_refined’s predictions are marginally overconfident, typical of temporal models trained on short sequences.
**Statistical reliability:**  
- Because both models produced identical threshold metrics on a **tiny n = 15 test set**, these values lack statistical reliability and are dominated by rounding effects.  
- In this regime, **only threshold-independent metrics (AUC, Brier, ECE)** retain meaning. They show that:
  - **TCN_refined** better separates classes (higher AUC),  
  - **LightGBM** is better calibrated (lower Brier/ECE).
**Conclusion (max_risk):**  
- Both models perfectly identify which patients ever reached high deterioration risk.  
- **TCN_refined** demonstrates stronger discriminative ability → consistent with its design for capturing dynamic peaks and transitions.  
- **LightGBM** is slightly better calibrated and more probabilistically stable, reflecting its non-temporal averaging of features.  
- Clinically, this means:
  - **TCN_refined** excels at detecting deterioration events.  
  - **LightGBM** provides more trustworthy probability scaling for event likelihood.  
- Both achieve near-ceiling effectiveness for this binary target.


2. **Classification (`median_risk`)**

| Model | ROC AUC | F1 | Accuracy | Precision | Recall | Brier | ECE |
|:--|--:|--:|--:|--:|--:|--:|--:|
| LightGBM | **0.972** | **0.857** | **0.933** | **0.750** | 1.000 | **0.065** | **0.093** |
| TCN_refined | 0.833 | 0.545 | 0.667 | 0.375 | 1.000 | 0.201 | 0.251 |

**Interpretation**
- **Discrimination:**  
  - Here, `median_risk` reflects the *typical risk state* of a patient throughout admission.  
  - LightGBM achieves a **16.7% higher AUC** (0.97 vs 0.83), showing a far stronger ability to rank stable versus chronically unstable patients.  
  - This difference reflects structural alignment: LightGBM uses aggregate summary features that correspond directly to this averaged-risk target.
- **Threshold-dependent performance:**  
  - LightGBM surpasses TCN_refined on all threshold-dependent scores:  
    - **F1:** +57% higher (0.86 vs 0.55)  
    - **Accuracy:** +40% higher (0.93 vs 0.67)  
    - **Precision:** +100% higher (0.75 vs 0.38)  
  - Both achieve perfect Recall = 1.0, but TCN_refined’s low precision indicates **systematic over-prediction of positives**, producing inflated probabilities even for stable patients.  
  - **Threshold tuning:**  
    - TCN_refined used a tuned threshold of **0.43**, optimised via F1-maximisation to correct for class imbalance (`pos_weight`).  
    - Despite this tuning, its metrics remain weaker, meaning the issue lies not in threshold choice but in **underlying probability distribution** and **class separability**.  
    - LightGBM used the standard 0.5 threshold yet achieved higher balanced performance, indicating more stable probability scaling.
- **Calibration:**  
  - LightGBM demonstrates markedly better probabilistic reliability:  
    - **Brier score:** 68% lower (0.065 vs 0.201)  
    - **ECE:** 63% lower (0.093 vs 0.251)  
  - This means its output probabilities are far closer to true event frequencies.  
  - TCN_refined’s outputs are compressed into midrange (≈0.3–0.6) — overconfident but uninformative, typical when temporal variance is low.

**Why TCN_refined Underperformed**
1. **Label–model mismatch:**  
  - `median_risk` represents a patient’s average risk state across their entire admission.  
  - The TCN architecture is optimised to detect temporal transitions or spikes (e.g., short-term deterioration), not average-level patterns.  
  - Since most patients remain clinically stable over long periods, temporal features add noise rather than signal.
2. **Limited temporal contrast between classes:**  
  - For median-level patterns, where most patients are stable, temporal convolutions introduce noise rather than signal.  
  - Both “low-median-risk” and “medium-median-risk” patients exhibit overlapping short-term sequences, differing mainly in overall average values.  
  - TCN’s convolutional filters cannot easily separate such subtle differences, leading to poorly discriminative embeddings.
3. **Calibration drift from dynamic inputs:**  
  - TCN’s probabilities are derived from the final sigmoid activation after multiple temporal convolutions.  
  - In low-variance sequences, these activations can become saturated or poorly scaled, producing overconfident probabilities → consistent with the high Brier (0.20) and ECE (0.25) values.
4. **Structural advantage of LightGBM:**  
  - LightGBM operates on **aggregated tabular features** (means, medians, last values), which directly encode the same concept as `median_risk`.  
  - It therefore aligns structurally with the target definition → explaining its higher AUC and superior calibration.

**Statistical Reliability**
- Unlike `max_risk`, where identical threshold metrics arose from the small sample size (n = 15), the divergence here is **genuine** → it reflects true model-behavioural differences rather than statistical noise.  
- However, with only 15 samples, even these metrics must be interpreted cautiously; **AUC and calibration** remain the most reliable indicators of performance stability.

**Conclusion (median_risk):**
- **LightGBM** decisively outperforms **TCN_refined** across discrimination, accuracy, and calibration.  
- This reflects a clear **task–architecture mismatch**:  
  - TCN excels at transient events, not long-term stability states. Even at the optimised threshold (0.43), its probability outputs remain compressed, overlapping, and poorly calibrated.
  - LightGBM aligns natively with averaged, patient-level features, yielding both higher discrimination and better probability scaling. 
- Clinically:  
  - LightGBM is better suited to monitoring a patient’s typical level of physiological risk across an admission.  
  - TCN_refined’s output probabilities are narrowly distributed, producing mid-range predictions for nearly all patients. This compression hides meaningful distinctions between moderately and persistently high-risk individuals, reducing its usefulness for tracking sustained clinical instability.
- **Overall:** LightGBM provides more interpretable, calibrated, and clinically coherent estimates for median-level risk assessment.
- Therefore, the performance gap here illustrates a key insight: 
  - **Temporal networks excel for dynamic event detection (max-risk).**
  - **Tabular learners dominate for static or aggregate-state classification (median-risk).**

---

**3. Regression (`pct_time_high`)**

| Model | RMSE | R² |
|:--|--:|--:|
| LightGBM | **0.038** | **0.793** |
| TCN_refined | 0.056 | 0.548 |

**Interpretation:**
- `pct_time_high` quantifies how much of a patient’s total admission time was spent above the high-risk threshold → a continuous indicator of sustained deterioration exposure.  
- **Error magnitude (RMSE):**  
  - LightGBM predictions deviate from ground truth by an average of 0.038 percentage units.  
  - TCN_refined predictions deviate by 0.056 units, which is approximately **48% higher error** than LightGBM.
- **Explained variance (R²):**  
  - LightGBM explains ~79% of variance in deterioration percentage.  
  - TCN_refined explains only ~55% (~24 percentage points lower), reflecting weaker overall fit.
  - This shows that LightGBM captures the distribution of “risk exposure time” more precisely.
- **Data transformations for TCN:**  
  - TCN predictions were initially computed in **log-space**, then transformed back and calibrated before metric calculation.  
  - Despite these adjustments improving numerical alignment, accuracy did not, TCN’s predictions remain less precise and less stable than LightGBM’s, indicating inherent limitations in capturing the continuous deterioration percentage.
- **Overall regression fit:**  
  - LightGBM produces more accurate and reliable predictions with smaller residuals and better variance explanation.  
  - TCN_refined’s higher RMSE and lower R² suggest that temporal modelling adds less value for median-level continuous deterioration in this dataset, particularly given the small test set.
  - LightGBM’s static summarisation captures global deterioration duration better than TCN’s short-term focus.

**Statistical reliability:**  
- Metrics are descriptive for a small test set (n = 15), but relative differences are substantial enough to confidently indicate superior regression performance for LightGBM.  
- Continuous-valued metrics like RMSE and R² are more informative here than threshold-dependent classification metrics.

**Conclusion (pct_time_high):**
- LightGBM is **clearly superior** for estimating continuous deterioration exposure, on both absolute (RMSE) and relative (R²) measures.  
- It produces smaller residuals and explains substantially more variance, indicating it generalises better for regression-style clinical risk modelling.  
- TCN_refined’s temporal formulation contributes less value when the task measures proportion of time, not discrete events.
- LightGBM provides the most reliable and precise quantitative estimates of patient deterioration percentages across the test set.

---

#### Overall Quantitative Summary

| Dimension | Winner | Notes |
|:--|:--|:--|
| Discrimination (ROC AUC) | **TCN_refined (max_risk)**, **LightGBM (median_risk)** | TCN excels at short-term event detection; LightGBM at sustained-state separation. |
| Threshold Accuracy (F1/Accuracy/Precision) | **LightGBM overall** | Especially superior for median risk. |
| Calibration (Brier/ECE) | **LightGBM** | More reliable probability scaling across all targets. |
| Regression Fit (RMSE/R²) | **LightGBM** | Substantially lower error and higher explained variance. |

**Integrated Interpretation:**  
- **TCN_refined** demonstrates clear strength in dynamic event detection → its temporal filters capture sharp, transient spikes corresponding to acute deterioration (`max_risk`).  
- **LightGBM**, by contrast, dominates in aggregate or sustained-risk estimation (`median_risk`, `pct_time_high`), reflecting its structural advantage in modelling patient-level summaries.  
- Across calibration and probabilistic reliability metrics, LightGBM consistently outperforms → a key consideration for deployment in systems where calibrated probabilities drive downstream alerts or thresholds.  
- **Clinically,** this distinction mirrors real-world roles:  
  - TCN-like models could flag acute deteriorations in real time,  
  - LightGBM-like models could provide daily risk stratification and long-term prognosis.  
- Overall, **LightGBM offers the most stable, generalisable quantitative performance** across all targets, whereas **TCN_refined adds incremental value specifically for event-based deterioration detection.**

---

### Limitations and Contextual Analysis: TCN Performance vs LightGBM
**Background**
- Deep learning models, such as **Temporal Convolutional Networks (TCN)**, are generally expected to outperform classical machine learning on complex sequential data due to their ability to:
  - Capture temporal dependencies across patient time-series.
  - Model subtle, nonlinear interactions in multi-feature sequences.
  - Potentially discover latent patterns not obvious in summary statistics.
- In contrast, **LightGBM** is a gradient-boosted tree ensemble designed for tabular data:
  - It operates on aggregated features (means, medians, last-observation values).
  - It excels in small-data regimes and for targets that are themselves aggregates (like `median_risk`).
**What We Did to Optimise TCN**
1. **Phase 4.5 (TCN refinement):**
  - Trained and validated on timestamp-level sequences of patient vitals.
  - Implemented positional encoding, dropout, and hyperparameter tuning to stabilise learning.
  - Used class weighting (`pos_weight`) to address severe imbalance in deterioration events.
  - TCN training/validation inputs for `pct_time_high` were log-transformed.
2. **Phase 5 (Evaluation):**
  - Evaluated per-patient predictions at threshold 0.5 for max-risk; tuned threshold 0.43 for median-risk to maximise F1.
  - Metrics computed: ROC AUC, F1, Accuracy, Precision, Recall, RMSE, R².
  - TCN outputs for `pct_time_high` were in log-space, calibrated, and then evaluated to ensure comparability with LightGBM.
3. **Phase 6 (Calibration & Comparison):**
  - Brier score and Expected Calibration Error (ECE) calculated from raw probability predictions.
  - Comparison against LightGBM performed using identical test patients (n = 15) for all tasks.
**Why TCN Underperformed Despite Refinement**
1. **Small Test Set**
   - n = 15 patients is far below the scale typically needed for deep learning to generalise.
   - Even well-trained TCN weights are unstable; metrics (especially threshold-dependent metrics) are noisy and unreliable.
2. **Target Definition Misalignment**
   - `median_risk` reflects the **average risk state across a patient’s stay**, binarised for evaluation:
     - **0 = low-median-risk** → originally 0 or 1
     - **1 = medium-median-risk** → originally 2
   - Temporal fluctuations matter less for median risk; TCN focuses on short-term dynamics rather than long-term averages.
   - LightGBM’s tabular aggregation of patient-level summary features (means, medians, last values) aligns naturally with this binary definition, giving it a structural advantage over TCN for this task.
3. **Limited Temporal Contrast**
   - Although patient risk varies over time, the **overall sequences of vital signs for low- and medium-median-risk patients are very similar** in magnitude and pattern.
   - The TCN is designed to detect **temporal patterns and transitions**, but when the sequences overlap heavily, its convolutional filters cannot extract meaningful differences between the two classes.
   - As a result, the learned embeddings (internal representations) **fail to separate low vs medium median-risk patients**, reducing discrimination and lowering classification metrics like AUC and precision.
4. **Calibration and Probability Compression**
   - The TCN produces a predicted probability for each patient after the final sigmoid activation. Because the input sequences for median-risk patients have **low variability over time**, the temporal convolutions produce very similar activations across patients.
   - Sigmoid activation then **maps these similar activations into a narrow probability range**, often close to 0 or 1, leading to **overconfident predictions** even when the true risk is intermediate.
   - This overconfidence is reflected in **high Brier score (0.201) and ECE (0.251)**, indicating that predicted probabilities are misaligned with observed outcomes.
   - Post-hoc calibration can adjust probabilities somewhat, but when the model’s raw outputs are highly compressed or misaligned with the true risk distribution, **calibration cannot fully recover reliable probabilities**.  
   - In contrast, LightGBM’s aggregated features produce outputs that naturally scale with the observed median risk, resulting in better-calibrated probabilities.
5. **Log-Transformation in Regression (`pct_time_high`)**
   - To stabilise variance and reduce the effect of extreme predictions, TCN outputs for `pct_time_high` were initially computed in **log-space**.  
   - After model inference, predictions were **transformed back** to the original scale and **calibrated** before computing RMSE and R².  
   - While this process improves numerical stability and mitigates extreme outliers, it does **not change the underlying predictive limitations** of the TCN.  
   - Specifically, `pct_time_high` represents the **percentage of a patient’s time spent at high risk** throughout their entire stay → a long-term, aggregated measure.  
   - TCNs are designed to detect **dynamic, short-term temporal spikes**, not cumulative or slowly varying signals.  
   - Consequently, even after log transformation and calibration, the TCN predictions remain **less precise and less aligned** with the true high-risk time percentage compared to LightGBM, which leverages aggregated tabular features that naturally capture this target.
**Key Takeaways**
- **Deep learning is not guaranteed to outperform classical ML on small datasets**, especially when targets are aggregate measures rather than dynamic events.
- TCN excels for **max-risk detection**, where temporal patterns and spikes are meaningful.
- LightGBM excels for **median-risk** and **pct_time_high** because:
  - Its input features directly summarise patient-level statistics over time, which are naturally predictive of these aggregate targets.
  - It is robust in small-data regimes.
  - Its probability estimates are naturally better calibrated for these tasks.
- Threshold tuning (0.43 for median-risk TCN) improves metric alignment but cannot compensate for intrinsic limitations in feature–target alignment and small sample size.
**Implications for Interpretation**
- **Quantitative metrics alone** do not fully explain model behaviour.
  - AUC and calibration metrics provide the most reliable indicators for small n = 15 test set.
- **Threshold-dependent metrics** (F1, Accuracy, Precision, Recall) are highly sensitive to rounding and small sample effects.
- This analysis highlights the importance of **task–model alignment** and **sample size** when interpreting performance differences between deep learning and classical ML methods.

---

### Methodological Rationale and Design Reflection
**Overview**
- This section outlines the rationale behind the chosen modelling pipeline and the methodological decisions shaping the comparison between **LightGBM** and the **Temporal Convolutional Network (TCN)**.  
- The design prioritised **comparability, interpretability, and applied insight** over purely technical optimisation.  
- Although this constrained the TCN’s full temporal potential, it enabled both models to be evaluated on **identical, real-world patient-level prediction tasks**, a critical consideration for **applied healthcare machine learning**.

**Project Goals and Rationale**
1. **Comparability Over Complexity**
  - The overarching goal was not to build two different models for two different tasks, but to **directly compare** a classical tabular learner (LightGBM) and a deep temporal model (TCN) under **identical predictive conditions**.  
  - A shared design allowed:
    - Direct quantitative comparison of discrimination, calibration, and regression metrics.  
    - A clear test of whether deep learning provides measurable benefit over classical methods in small, patient-level datasets.  
  - This **comparative framework** was central to the project’s scientific validity.
2. **Applied Machine Learning Perspective**
  - The approach reflects **applied ML thinking**, prioritising:
    - Comparability over maximum performance.  
    - Interpretability over black-box optimisation.  
    - Practical insight over theoretical idealism.  
  - In real-world healthcare settings, models must operate under **limited data, constrained resources, and high interpretability requirements**, making this a deliberately realistic study design.
3. **Critical Thinking and Trade-offs**
  - Every model choice introduces structural biases.  
  - By constraining both models to **identical patient-level prediction granularity**, the project isolated **architectural differences** rather than confounding them with task-level variation.
  - This is a hallmark of sound experimental design: controlled constraint to ensure **methodological fairness**.
4. **Why Not a Multi-Outcome or Multi-Granularity Design**
  - A dual or hybrid pipeline (e.g., timestamp-level TCN + patient-level LightGBM) would have demonstrated engineering versatility,  
    but **not** answered the methodological question of whether deep temporal models actually outperform classical ones on the same clinical prediction task.  
  - This project’s aim was **comparative insight**, not mere technical diversity.  
  - The chosen approach therefore provided a **clean, interpretable benchmark** of model suitability under identical constraints.

**Why the Pipeline Was Designed This Way**
1. **Ensuring Direct Comparability**
  - Both models were trained and evaluated on identical **patient-level targets**:
    - `max_risk`  
    - `median_risk`  
    - `pct_time_high`  
  - This made it possible to measure key metrics (ROC AUC, F1, Brier, ECE, RMSE, R²) in a strictly like-for-like manner.  
  - If the models were trained on different temporal resolutions, the results would have been **qualitatively incomparable**, invalidating the comparison.
2. **The Alternative: Fully Temporal Supervision**
  - A theoretically optimal TCN design would have predicted deterioration probabilities at each **timestamp**, allowing direct modelling of short-term risk dynamics.  
  - These predictions could then be aggregated (e.g., by taking the maximum or median per patient).
  - However:
    - LightGBM cannot operate on timestamp-level labels, so direct comparison would have been impossible.  
    - The two models would effectively represent **two different tasks** — dynamic forecasting vs static patient-level classification — rather than two solutions to the same task.  
    - With a dataset of only **15 test patients**, timestamp-level supervision would have been statistically fragile and computationally unstable.
  - Hence, the **patient-level prediction structure** was a deliberate, controlled constraint designed to keep the comparison fair.
3. **Pragmatic and Computational Constraints**
  - Timestamp-level supervision requires **hundreds or thousands of patients** to learn stable temporal representations.  
  - With a small dataset, patient-level aggregation was essential to:
    - Stabilise training,  
    - Prevent overfitting, and  
    - Produce interpretable, reproducible results.  
  - Implementing timestamp-level labels would have required major architectural changes and computational resources beyond this project’s practical scope.  
  - The final pipeline therefore represents a **methodologically grounded trade-off** between **comparability** and **temporal expressiveness**.

**Consequences of This Design**
1. **Structural Bias Toward LightGBM**
  - All three outcome targets (`max_risk`, `median_risk`, `pct_time_high`) are **aggregate, patient-level summaries** of risk across an admission.
  - **LightGBM** naturally consumes aggregated tabular inputs (e.g., patient-level means, medians, and latest values), which directly mirror the structure of these targets.  
  - In contrast, the **TCN** was designed for timestamp-level reasoning*, but in this project it had to compress full temporal sequences into a single scalar output per patient, effectively **neutralising its key temporal advantage**.
  - This created an inherent **alignment bias** that favoured LightGBM, because the target definition matched LightGBM’s static input structure more closely than the TCN’s dynamic processing architecture.
2. **Loss of Timestamp-Level Supervision**
  - Although the **TCN was trained on timestamp-level features**, its **supervision signal (labels)** was still at the patient level — i.e., one label per full sequence.
  - This means that while the model saw detailed temporal variation in vitals, labs, and observations, it was only taught to predict a **single patient-level summary outcome** (e.g., overall max or median deterioration).
  - Consequently, only the **final pooled sequence embedding** contributed to the loss function.  
    - Gradients flowed back from one scalar label through all timesteps.  
    - This diluted temporal sensitivity → the model could not learn which time segments were most predictive of deterioration.
  - In practice, this forced the TCN to behave less like a true sequence forecaster and more like a **temporal feature summariser**, collapsing its temporal depth into a static representation.
  - This setup did not make the model “non-temporal,” but it **weakened temporal gradient flow** and restricted its ability to exploit timestamp-level dependencies → the exact strength that normally allows deep temporal models to outperform tabular ones.
3. **Different Model Strengths by Design**
  - **LightGBM**: excels at aggregate state recognition → its feature engineering (aggregates, medians, last values) directly aligns with the target structure of all patient-level outcomes.
  - **TCN**: excels at dynamic event detection and timestamp-level forecasting, where risk transitions occur over short timescales.
  - Because this project’s evaluation was designed around patient-level targets, the TCN’s inherent advantage in temporal prediction was **underutilised by design**.
  - The comparison, therefore, was **methodologically fair but structurally biased**:
    - It allowed direct, one-to-one metric comparison between both models on identical targets.
    - But it inherently favoured LightGBM’s architecture, which was already aligned with the outcome definition.
    - TCN, in contrast, had to self-compress temporal richness to remain comparable, effectively operating under a structural handicap.

**Clinical and Practical Context**
1. **Realistic Data Constraints**
  - In real-world hospitals:
    - An ICU typically has **10–20 patients** at any time.  
    - Even large hospitals rarely exceed **~100 high-dependency or ICU-level patients** across all wards.  
  - This means applied ML in healthcare operates in a **small-n, high-frequency** regime:
    - Each patient has thousands of timepoints.  
    - But there are few independent patients overall.
2. **Implications for Real-World Deployment**
  - Large public datasets like MIMIC-IV (10,000+ patients) help research benchmarking,  
    but deployment scenarios involve far fewer patients, limiting model generalisability.  
  - This project’s **small-patient test set (n = 15)** therefore **mirrors real deployment conditions**, not an artificial benchmark.  
  - In such settings:
    - **LightGBM** is well-suited for robustness and interpretability.  
    - **TCNs** cannot reach their potential due to insufficient patient diversity.

**Key Insights from This Design Choice**
1. **Comparative Validity**  
  - By enforcing a shared target granularity, both models were benchmarked on **exactly the same predictive question** → predicting patient-level outcomes rather than timestamp-level ones.  
  - This design ensured **scientific validity** and methodological fairness: both models received identical inputs and produced comparable outputs, allowing a like-for-like evaluation.  
  - Although this choice constrained the TCN’s temporal capabilities, it preserved the integrity of the **comparative framework**, which was the project’s primary goal.
2. **Task–Model Alignment**  
  - The observed performance differences stem from **target–architecture alignment**, not algorithmic superiority.  
  - **LightGBM** is optimised for **static, tabular representations**, where each feature summarises a patient’s physiological state (e.g., mean HR, max NEWS2, last SpO₂).  
  - **TCN**, in contrast, is optimised for **temporal event detection**, where labels vary dynamically across time (e.g., risk transitions or deterioration spikes).  
  - Because all targets (`max_risk`, `median_risk`, `pct_time_high`) were **aggregated at the patient level**, the LightGBM model was structurally aligned with the target definition, while the TCN was forced to compress temporal data into a single static prediction.  
  - The resulting differences in performance therefore reflect **task suitability**, not model inferiority.
3. **Data Regime Dependency**  
  - In small, low-variance datasets like this one (n = 15 patients), classical models often outperform deep learning architectures due to differences in **inductive bias** and **data efficiency**. 
  - **Inductive Bias** is the set of built-in assumptions a model makes about data structure and how it behaves. 
    - **LightGBM** has a **strong inductive bias**:  
      - LightGBM’s bias stems from its **decision-tree structure**, which learns patterns through **if–then splits (threshold-based decision splits)**, e.g.:  
        - If NEWS2 > 5 → higher deterioration risk 
        - If age > 80 and HR > 110 → high risk
      - This threshold-based reasoning mirrors clinical thinking, where risk is defined by interpretable cut-points rather than continuous temporal trends.  
      - As a result, LightGBM is naturally suited for **static, tabular, rule-based risk prediction**, allowing robust learning even from very small samples.
      - The model’s hierarchical structure and decision rules act as built-in **regularisers**, preventing overfitting when data are sparse or noisy. 
    - **TCN**, by contrast, has a **weak inductive bias**:  
      - It assumes very little about the data’s structure, and makes almost no prior assumptions about how features relate. 
      - Instead, it learns dependencies directly from raw temporal sequences through **1D convolutions**, detecting evolving patterns over time.  
      - This flexibility allows powerful pattern recognition in large datasets but makes the model highly **data-hungry** → it needs extensive, diverse sequences to learn stable patterns to generalise effectively.  
      - With limited data, the TCN’s convolutional filters cannot reliably distinguish signal from noise, producing **unstable and poorly generalisable** temporal representations.
  - **Data Efficiency**  
    - **LightGBM** is highly **data-efficient**:  
      - It generalises well even in small datasets because its structure and learning process rely on simple, interpretable transformations of tabular features.  
      - Fewer parameters and clear feature–outcome mappings make it robust under data scarcity.  
    - **TCN** is inherently **data-intensive**:  
      - Its large number of learnable parameters and complex layer structure require substantial data diversity to stabilise training.  
      - When trained on small datasets, it tends to memorise local fluctuations rather than learning general clinical relationships.  
  - **Implication for This Project**  
    - In this data regime — **small sample size, low temporal variance, and aggregate targets** — LightGBM’s strong inductive bias and efficiency gave it a decisive advantage.  
    - TCN’s theoretical strengths (capturing long-range dependencies and complex dynamics) could not manifest because the dataset was too small to support high-dimensional temporal learning.  
    - Thus, LightGBM’s superior performance reflects a **data–model mismatch**, not algorithmic inferiority.
4. **Potential Under Full Supervision**  
  - With a **larger dataset** and **timestamp-level supervision**, the TCN would likely outperform LightGBM.  
  - Proper timestamp-level training would allow the TCN to:  
    - Capture **fine-grained temporal patterns**, such as gradual deterioration or recovery.  
    - Learn **causal transitions** between physiological states instead of static averages.  
    - Exploit **multi-scale temporal features** (both short-term fluctuations and long-term trends).  
  - LightGBM, by design, cannot model such temporal dependencies → it treats each patient as a single independent sample.  
  - Therefore, under full temporal supervision and sufficient data, a well-tuned TCN (or similar deep temporal model) would likely achieve **superior discrimination, generalisation, and calibration** across clinically relevant timescales.  

**Would a Dual-Pipeline Design Have Been Better?**
- A dual-pipeline design could have included:
  - **LightGBM** for static patient-level classification, and  
  - **TCN** for timestamp-level event forecasting.  
- This would have demonstrated both models’ strengths in their native domains.  
- However, it would have become a **multi-objective project**, not a comparative one → shifting focus away from methodological evaluation toward model engineering.  
- For the current project’s aims; **applied, comparative, and interpretive ML in healthcare**; the shared patient-level framework was the optimal design.  
- It demonstrated:
  - Methodological discipline,  
  - Awareness of bias and constraint, and  
  - Alignment with **real-world clinical applicability**, not academic idealism.

**Final Perspective**
- The chosen design reflects an intentional methodological trade-off:
  - Enables direct cross-model benchmarking on identical tasks.  
  - Restricts TCN’s full temporal learning potential.
- This was **not a limitation by mistake**, but a **controlled experimental choice** to isolate the variable of interest (architecture) under equal conditions.
- **The resulting findings are meaningful:** Deep learning does not inherently outperform classical ML; its advantage depends on data scale, label granularity, and task–model alignment.
- In larger datasets with timestamp-level outcomes, TCNs would likely achieve superior generalisation and temporal understanding.  
  However, under realistic data constraints and applied evaluation goals, **LightGBM’s simplicity, calibration, and robustness** make it the more effective model for practical deployment.

**Key Takeaways**
- **Applied focus:** The study reflects real-world ML practice → prioritising comparability, interpretability, and efficiency over theoretical performance.  
- **Transparency:** Every trade-off was explicit, ensuring reproducibility and honest benchmarking.  
- **Insightful outcome:** Model suitability depends jointly on data regime, target semantics, and deployment context.  
- **Practical impact:**  
  - LightGBM’s calibration, reliability, and simplicity make it the preferred model for small-cohort hospital settings.  
  - Deep temporal architectures like TCNs remain powerful for large, timestamp-rich datasets → but their advantages emerge only when data scale supports temporal generalisation.

---

### Plots and CSV Interpretability Guide
#### Overview 
- This explains the plots generated for LightGBM and TCN models, the purpose of each plot type, what the plots are intended to show, and the extra metrics included in the CSVs to allow full interpretation without the need for visual inspection.
- All numeric arrays are **aligned row-wise**, padded with NaNs where necessary to allow side-by-side comparison between models.  
- The CSVs **fully capture the information needed** to interpret model performance, eliminating the need for the PNG plots.  
- This design enables:
  - Reproducibility  
  - Quantitative analysis  
  - Statistical comparisons between models  
  - Integration with other analysis tools

#### Classification Plots
1. **ROC Curve (Receiver Operating Characteristic)**
**Purpose:**  
- Measures the model’s ability to discriminate between positive and negative classes across all probability thresholds.
**What it shows:**  
- **False Positive Rate (FPR)** vs **True Positive Rate (TPR)** curve.  
- AUC (Area Under Curve) quantifies overall discrimination: 1.0 = perfect, 0.5 = random.
**CSV Columns:**  
- `fpr_LightGBM`, `tpr_LightGBM` → FPR and TPR arrays for LightGBM  
- `fpr_TCN_refined`, `tpr_TCN_refined` → FPR and TPR arrays for TCN  
- `auc_LightGBM`, `auc_TCN_refined` → constant columns with overall AUC per model  
- `prevalence` → proportion of positive cases (base rate)
**Interpretability:**  
- Each row corresponds to a threshold in probability space.  
- You can reconstruct ROC curve and compare AUC values directly from the CSV.

2. **Precision–Recall Curve (PR Curve)**
**Purpose:**  
- Focuses on model performance for the positive class, especially useful for imbalanced datasets.
**What it shows:**  
- **Recall (Sensitivity)** vs **Precision (Positive Predictive Value)** across thresholds.  
- Average Precision (AP) summarizes curve as a single number.
**CSV Columns:**  
- `recall_LightGBM`, `precision_LightGBM` → Recall and Precision arrays for LightGBM  
- `recall_TCN_refined`, `precision_TCN_refined` → Recall and Precision arrays for TCN  
- `ap_LightGBM`, `ap_TCN_refined` → constant columns with Average Precision per model
**Interpretability:**  
- Each row corresponds to a probability threshold.  
- You can calculate or plot PR curve from CSV.  
- AP gives a single metric to rank model performance.

3. **Calibration Curve (Reliability Diagram)**
**Purpose:**  
- Evaluates how well predicted probabilities reflect actual outcome likelihoods.
- Helps determine if the model is **overconfident** or **underconfident** in its predictions.
**What it shows:**  
- Fraction of positives (`frac_pos`) vs mean predicted probability (`mean_pred`) within bins.
- Perfect calibration occurs when predicted probabilities match observed frequencies (points lie on the diagonal).
**Bins explained:**  
- The predicted probability range [0, 1] is split into a fixed number of intervals (bins), typically 10.  
- Example bins for 10 divisions: [0–0.1), [0.1–0.2), …, [0.9–1.0].  
- Each row in the CSV corresponds to **one bin**:  
  - `mean_pred` = average predicted probability of patients in that bin  
  - `frac_pos` = fraction of patients in that bin who actually have the event
**CSV Columns:**  
- `mean_pred_LightGBM`, `frac_pos_LightGBM` → calibration bins for LightGBM  
- `mean_pred_TCN_refined`, `frac_pos_TCN_refined` → calibration bins for TCN  
- `brier_LightGBM`, `brier_TCN_refined` → Brier score (mean squared error of probabilities)  
- `ece_LightGBM`, `ece_TCN_refined` → Expected Calibration Error  
- `n_samples_LightGBM`, `n_samples_TCN_refined` → number of patients per bin, assess reliability of the fraction observed.
**Interpretability:**  
- Each row = one bin of predicted probabilities.  
- You can evaluate calibration visually or compute metrics directly from CSV.

4. **Probability Histogram**
**Purpose:**  
- Describes distribution of predicted probabilities.  
- Helps assess whether model is confident (predictions near 0 or 1) or uncertain (predictions near 0.5).
**CSV Columns (LightGBM and TCN separately):**  
- `pred_prob_*` → predicted probability per patient  
- `mean_*` → average predicted probability  
- `std_*` → standard deviation  
- `min_*`, `max_*` → range of predicted probabilities  
- `skew_*` → asymmetry of distribution  
- `kurt_*` → tail heaviness of distribution (high kurtosis = more extreme values)
**Interpretability:**  
- These statistics allow you to quantify prediction spread without plotting.  
- Skew >0 → distribution has longer right tail; Skew <0 → longer left tail.  
- Kurtosis >3 → heavier tails than normal; Kurtosis <3 → lighter tails.  
- You can reconstruct a histogram or evaluate confidence and risk distribution directly from CSV.


#### Regression Plots
1. **Scatter Plot (True vs Predicted)** 
**Purpose:**  
- Compare predicted values against the true values for both models.  
- Assess model accuracy visually or programmatically.
**CSV Columns:**  
- `y_true_LightGBM` → ground-truth values for LightGBM  
- `y_pred_LightGBM` → predicted values by LightGBM  
- `y_true_TCN_refined` → ground-truth values for TCN  
- `y_pred_TCN_refined` → predicted values by TCN  
**Interpretability:**  
- Points close to the identity line (y_true = y_pred) indicate accurate predictions.  
- Outliers indicate under- or over-prediction for specific patients.

2. **Residuals**
**Purpose:**  
- Show errors (residuals) of predictions for each patient.  
- Residual = predicted − true value.
**CSV Columns:**  
- `residual_LightGBM` → errors for LightGBM  
- `residual_TCN_refined` → errors for TCN_refined  
- `mean_res_*` → mean residual, indicates **bias** (0 = unbiased)  
- `std_res_*` → standard deviation, indicates **variability**  
- `min_res_*` / `max_res_*` → extreme errors  
- `skew_res_*` → asymmetry of residual distribution (positive = long tail above zero)  
- `kurt_res_*` → tail heaviness (higher = more extreme outliers)
**Interpretability:**  
- Residuals centered around 0 indicate unbiased predictions.  
- Spread indicates prediction variability.  
- Positive residual = overestimation, negative residual = underestimation.

3. **Residual KDE**
**Purpose:**  
- Smooth representation of residual distributions (Kernel Density Estimate).  
- Helps understand the error distribution shape beyond simple histograms.
**CSV Columns:**  
- `grid_LightGBM` → x-axis points for KDE of LightGBM residuals  
- `kde_LightGBM` → density at each grid point  
- `grid_TCN_refined` → x-axis points for KDE of TCN residuals  
- `kde_TCN_refined` → density at each grid point  
**Interpretability:**  
- Peaks indicate where most residuals lie.  
- Wide distribution = more variability.  
- Skew indicates asymmetric error tendency.

4. **Error vs Truth**
**Purpose:**  
- Examine relationship between residuals and true values.  
- Detect systematic bias at different outcome magnitudes.
**CSV Columns:**  
- `y_true_LightGBM` → true values for LightGBM  
- `residual_LightGBM` → residuals for LightGBM  
- `mean_res_LightGBM`, `std_res_LightGBM`, `min_res_LightGBM`, `max_res_LightGBM`, `skew_res_LightGBM`, `kurt_res_LightGBM` → summary stats for LightGBM residuals  
- `y_true_TCN_refined` → true values for TCN  
- `residual_TCN_refined` → residuals for TCN  
- `mean_res_TCN_refined`, `std_res_TCN_refined`, `min_res_TCN_refined`, `max_res_TCN_refined`, `skew_res_TCN_refined`, `kurt_res_TCN_refined` → summary stats for TCN residuals  
**Interpretability:**  
- Residuals should be randomly scattered around zero.  
- Any trend (positive or negative slope) indicates bias: errors increase or decrease with true value magnitude.
- Summary statistics allow **full numeric interpretation without plots**, showing overall bias, spread, and distribution shape.

**Why Summary Statistics Are Included for Regression Residuals**
- Scatter CSV (`y_true` vs `y_pred`) shows individual predictions but **does not quantify overall error distribution**.  
- Residuals CSV (`residual = predicted − true`) and Error-vs-Truth CSV capture prediction errors, where summary stats are meaningful.  
- **Included metrics for residuals**:
  - **Mean residual** → indicates overall bias (closeness to zero = unbiased).  
  - **Std residual** → measures variability of prediction errors.  
  - **Min / Max residual** → identifies extreme under- or over-predictions.  
  - **Skew** → asymmetry of residuals (positive = tendency to over-predict, negative = under-predict).  
  - **Kurtosis** → tail heaviness of residuals (high = more extreme errors).  
- These summary statistics allow **full interpretation of regression performance directly from the CSVs**, without needing visual plots.  
- For the Scatter CSV, summary stats are **not included**, since the raw `y_true` and `y_pred` values are sufficient for interpretation of pointwise accuracy.

---

### Visual Analysis (Curve-Based and Distributional Diagnostics)
#### Overview
- This section interprets the graphical outputs underlying the quantitative performance metrics.  
- While the comparison table summarises global scalar metrics (AUC, F1, AP, etc.), the plots and their corresponding CSVs provide a deeper understanding of how each model achieves its results → including discrimination shape (ROC/PR), calibration reliability, and probability distribution characteristics.
- Analyses are organised by target variable and grouped by diagnostic type:
  - **ROC & PR curves:** Evaluate discrimination and class imbalance handling.
  - **Calibration curves:** Assess reliability of predicted probabilities.
  - **Probability histograms:** Examine confidence spread and prediction certainty.
- The following subsections provide detailed interpretations of these visual diagnostics for each target variable.

#### Classification (`max_risk`)
1. **ROC Curve (fpr–tpr data)**  
**CSV summary:**
- LightGBM AUC = **0.846**, TCN AUC = **0.923**.  
- TCN jumps from TPR 0.0769 → 0.923 while FPR remains 0.0 → **perfect early separation**.  
- LightGBM’s TPR increases more gradually, reaching 0.769 at FPR = 0.5.  
- Both models reach TPR = 1.0 at FPR = 1.0.
**Interpretation:**
- Both models show **strong discriminative performance**, but TCN demonstrates a much steeper early ROC rise.  
- This means TCN identifies high-risk patients **earlier and more confidently** with fewer false positives.  
- LightGBM requires higher thresholds (more relaxed decision boundaries) to reach equivalent recall.
**Conclusion (ROC):**  
- TCN achieves **superior early discrimination** and overall ranking ability for detecting patients who ever reach high risk during admission.

2. **Precision–Recall Curve**  
**CSV summary:**
- Average Precision (AP): LightGBM = **0.9774**, TCN = **0.9897**.  
- Both maintain high precision (≥0.846) across all recall points.  
- Precision = 1.0 for both at recall between 0.769 → 0.0.  
- Curves nearly identical, with TCN slightly higher area.
**Interpretation:**
- Both models sustain **exceptionally high precision and recall**, meaning nearly all patients predicted as “ever high risk” truly were.  
- TCN’s small AP gain (+0.0123) reflects marginally better recall coverage without sacrificing precision.
**Conclusion (PR):**  
- Performance parity overall; TCN shows **minor improvement in sensitivity** to high-risk events without additional false alarms.

3. **Calibration Curve (mean predicted probability vs. fraction of positives)**  
**CSV summary:**
- **LightGBM:** mean_pred = 0.5087–0.9744, frac_pos = 0.0–1.0, Brier = 0.0973, ECE = 0.1160.  
- **TCN:** mean_pred = 0.7704–0.8619, frac_pos = 0.6–1.0, Brier = 0.1010, ECE = 0.1488.  
- Some missing bins (NaNs) due to sparse samples in upper range.
**Interpretation:**
- Both models are **overconfident**, predicting probabilities higher than the true positive rates.  
- LightGBM shows **wider confidence variability** (predictions spanning 0.5–0.97).  
- TCN produces **tightly grouped high probabilities** (0.77–0.86), suggesting consistent but inflated confidence.  
- LightGBM’s slightly lower ECE (0.116 vs 0.1488) indicates **modestly better calibration**.
**Conclusion (Calibration):**  
- LightGBM’s outputs are more dispersed and somewhat better calibrated.  
- TCN produces consistently confident scores, slightly overestimating actual positive frequency → acceptable if ranking is the clinical priority but less so for probabilistic interpretability.

4. **Probability Histogram**  
**CSV summary:**
- **LightGBM:** mean = 0.88299, std = 0.14358, min = 0.5087, max = 0.9957, skew = –1.2669, kurt = 0.5767.  
- **TCN:** mean = 0.83144, std = 0.04578, min = 0.7575, max = 0.8814, skew = –0.4921, kurt = –1.3832.
**Interpretation:**
- LightGBM produces a **broader probability range** (0.51–1.00) with heavier left skew → some patients predicted confidently low and others extremely high.  
- TCN’s probabilities are **tightly clustered** (0.76–0.88), suggesting the model is confident that nearly all patients are at elevated risk at some point.  
- TCN thus offers **less granularity** in risk differentiation but stronger uniform conviction in positive predictions.
**Conclusion (Histogram):**  
- LightGBM provides greater spread and differentiation between stable vs deteriorating patients.  
- TCN yields **more consistent but compressed confidence**, reflecting a bias towards detecting any possible deterioration.


**Overall Summary (`max_risk`)**
| Dimension | LightGBM | TCN (refined) | Interpretation |
|------------|-----------|---------------|----------------|
| **ROC** | AUC = 0.846 | AUC = 0.923 | TCN AUC is **9.3% higher** ((0.923–0.846)/0.846). Early ROC region shows TCN achieves **92.3% TPR at FPR = 0** vs LightGBM’s 7.7% → **~12× higher early sensitivity**, meaning it detects deteriorating patients earlier and with fewer false positives. |
| **Precision–Recall** | AP = 0.9774 | AP = 0.9897 | TCN AP **1.25% higher**, maintaining near-perfect precision while slightly improving recall → better detection of all patients who ever reached high deterioration risk. |
| **Calibration** | mean_pred 0.5087–0.9744, frac_pos 0.0–1.0, Brier 0.0973, ECE 0.1160 | mean_pred 0.7704–0.8619, frac_pos 0.6–1.0, Brier 0.1010, ECE 0.1488 | Both overconfident. LightGBM slightly better calibrated (ECE 28% lower). TCN more consistent but inflates risk probabilities. |
| **Probability Histogram** | mean 0.88299, std 0.14358, skew –1.2669, kurt 0.5767, min 0.5087, max 0.9957 | mean 0.83144, std 0.04578, skew –0.4921, kurt –1.3832, min 0.7575, max 0.8814 | LightGBM: wide confidence spread → finer patient separation. TCN: compressed high-confidence band → uniform conviction of deterioration risk. |

**Final Interpretation (`max_risk`)**  
- Both models achieve **strong discriminative performance** for identifying patients who ever reached high deterioration risk during admission.  
- **TCN_refined** shows **clear superiority in discrimination**, with **AUC +0.077 (~9.3% relative gain)** and **AP +0.0123 (~1.25% relative gain)** over LightGBM.  
- Most notably, TCN achieves **~12× higher early sensitivity (TPR)** at zero false positives → detecting high-risk patients earlier and more confidently.  
- LightGBM, though slightly behind in AUC, exhibits **more stable calibration** (ECE = 0.116 vs 0.1488; ~28% lower), indicating **more reliable absolute probability estimates**.  
- Probability distributions further distinguish the two:  
  - **LightGBM** spans a broad range (0.51–0.99), producing more probabilistic diversity and clearer separation between stable and deteriorating patients.  
  - **TCN** produces tightly clustered outputs (0.76–0.88), reflecting strong uniform conviction that patients were high risk at some point, but **reduced probability granularity**.  
- Clinically, this implies that TCN is **more aggressive and sensitive** → ideal for early detection and flagging any patient who may deteriorate; while LightGBM offers **greater interpretability** and nuanced risk scaling.  
- Both models are **overconfident** (predicted risk > observed rate), suggesting the need for **post-hoc calibration** (e.g., Platt scaling or isotonic regression) before clinical deployment.

**Conclusion:**  
- For the `max_risk` outcome — indicating whether a patient ever reached high deterioration risk — **TCN_refined is the superior model for detection sensitivity and early risk identification**.  
- It provides higher discriminative power (+9.3% AUC) and substantially earlier true-positive recognition with minimal false positives, making it highly effective for **early-warning or alert-based systems**.  
- However, if **probability reliability** or **gradual risk differentiation** is clinically important (e.g., risk scoring or triage thresholds), **LightGBM remains preferable** due to better calibration and more interpretable probability distributions.  
- Overall, **TCN_refined is best suited for binary high-risk detection**, while **LightGBM excels when probabilistic confidence and calibration are required** → both valuable for complementary roles in deterioration monitoring frameworks.

---

#### Classification (`median_risk`)
1. **ROC Curve (fpr–tpr data)**  
**CSV summary:**  
- LightGBM achieves **AUC = 0.9722**, TCN achieves **AUC = 0.8333**.  
- LightGBM reaches TPR = 1.0 at FPR = 0.0833, while TCN reaches TPR = 1.0 only at FPR = 1.0.  
- Early thresholds show that LightGBM has a steep TPR ascent while maintaining very low FPR.  
- TCN’s TPR rises more slowly relative to its FPR, indicating less early discrimination.  
**Interpretation:**  
- LightGBM has **stronger early discrimination**, capturing patients with elevated median risk while keeping false positives minimal.  
- TCN’s weaker early separation suggests difficulty in ranking moderate-risk vs low-risk patients.  
**Conclusion (ROC):**  
- LightGBM demonstrates superior ranking and early discrimination compared to TCN for median_risk → critical for accurately identifying patients who sustain elevated risk over time.

2. **Precision–Recall Curve**  
**CSV summary:**  
- LightGBM’s AP = **0.9167**; TCN’s AP = **0.6333**.  
- LightGBM maintains precision from 0.2 → 1.0 as recall decreases from 1.0 → 0.0.  
- TCN shows lower precision at all recall levels (0.2 → 0.333) and a smaller PR area.  
**Interpretation:**  
- LightGBM consistently identifies true high-risk cases without excessive false positives, even under class imbalance (prevalence = 0.2).  
- TCN struggles to maintain precision, reflecting lower confidence and poorer positive class separation.  
**Conclusion (PR):**  
- LightGBM significantly outperforms TCN on precision–recall performance, confirming stronger identification of patients with sustained moderate-to-high deterioration risk.

3. **Calibration Curve (mean predicted probability vs. fraction of positives)**  
**CSV summary:**  
- **LightGBM:** mean_pred = 0.0113 → 0.9674, frac_pos = 0.0 → 1.0, Brier = 0.0647, ECE = 0.0931.  
- **TCN:** mean_pred = 0.2979 → 0.6403, frac_pos = 0.0 → 0.5, Brier = 0.2007, ECE = 0.2512.  
- LightGBM spans almost the full probability range; TCN predictions cluster within lower-mid regions.  
**Interpretation:**  
- LightGBM shows **better calibration** and meaningful spread across the entire probability spectrum, giving interpretable confidence scores for patient-level risk.  
- TCN’s restricted probability range (0.30–0.64) and higher calibration errors indicate **compression of predicted risk**, limiting interpretability.  
**Conclusion (Calibration):**  
- LightGBM produces **probabilities more consistent with observed outcomes** (Brier ≈3× lower, ECE ≈2.7× lower).  
- TCN’s mid-range bias implies underconfidence for genuinely stable patients and overconfidence for borderline-risk cases.

4. **Probability Histogram**  
**CSV summary:**  
- **LightGBM:** mean = 0.2438, std = 0.3933, min = 0.0031, max = 0.9952, skew = 1.1668, kurt = -0.5172.  
- **TCN:** mean = 0.4512, std = 0.1160, min = 0.2979, max = 0.6416, skew = 0.2293, kurt = -1.2057.  
- LightGBM’s predictions are broadly distributed, capturing both confident low-risk and high-risk probabilities.  
- TCN’s predictions are narrowly concentrated around the midrange.  
**Interpretation:**  
- LightGBM provides **granular stratification** → distinguishing persistently low-, medium-, and high-risk profiles.  
- TCN’s compressed distribution suggests **reduced differentiation** between typical and atypical patients, treating most patinets as having similar moderate risk, not distinctly low or high.
**Conclusion (Histogram):**  
- LightGBM enables clearer patient risk separation over the admission period, while TCN’s output uniformity limits its practical utility for long-term monitoring, where the goal is detecting sustained high-risk patterns.

**Overall Summary (`median_risk`)**
| Dimension | LightGBM | TCN (refined) | Interpretation |
|------------|-----------|---------------|----------------|
| **ROC** | AUC = 0.9722 | AUC = 0.8333 | LightGBM achieves higher sensitivity at lower FPR (≈17% relative improvement), indicating stronger early discrimination. |
| **Precision–Recall** | AP = 0.9167 | AP = 0.6333 | LightGBM maintains high precision and recall; TCN underperforms (~31% lower AP). |
| **Calibration** | mean_pred 0.0113–0.9674, frac_pos 0–1.0, Brier = 0.0647, ECE = 0.0931 | mean_pred 0.2979–0.6403, frac_pos 0–0.5, Brier = 0.2007, ECE = 0.2512 | LightGBM better calibrated (Brier ≈3× lower, ECE ≈2.7× lower); TCN overconfident in midrange. |
| **Probability histogram** | mean 0.2438, std 0.3933, skew 1.1668, kurt -0.5172 | mean 0.4512, std 0.1160, skew 0.2293, kurt -1.2057 | LightGBM spans full probability range (better granularity); TCN’s predictions compressed around midrange. |

**Final Interpretation (`median_risk`)**
- LightGBM **clearly outperforms** TCN_refined across all major metrics for the `median_risk` target.  
- It achieves **AUC +0.1389 (~17% relative gain)** and **AP +0.2834 (~45% relative gain)**, showing superior discrimination and positive class precision.  
- Calibration performance reinforces this advantage: **Brier score ≈3× lower** and **ECE ≈2.7× lower**, confirming more reliable probability estimates.  
- LightGBM’s probability distribution covers the full confidence spectrum (0.00–1.00), supporting **fine-grained patient stratification** → a crucial feature when estimating a patient’s typical deterioration risk across their stay.  
- TCN’s predictions, constrained between 0.30–0.64, fail to reflect true heterogeneity in patient stability, yielding **compressed mid-range risk estimates**, making it less clinically informative for monitoring patients over time. 
- Clinically, this means LightGBM can better distinguish patients with consistently high-risk trajectories from those generally stable, offering more interpretable and actionable probability outputs.  

**Conclusion:**  
- For modelling `median_risk` — the central risk tendency over admission — **LightGBM is unequivocally superior**, offering better discrimination, calibration, and interpretability.  
- Its broader probability spread allows clear patient stratification and supports practical clinical decision-making, whereas TCN’s compressed and miscalibrated outputs limit reliability for sustained-risk prediction.

---

#### Regression (`pct_time_high`)
1. **Scatter Plot (`y_true` vs `y_pred`)**
**CSV summary:**  
- **LightGBM** predictions cluster tightly along the perfect `y=x` line, with residuals ranging from **-0.0645 → 0.0619**.  
- **TCN_refined** predictions have a broader spread, with residuals **0.00055 → 0.2177**, frequently overestimating high-risk duration.  
- Patients with **low true `pct_time_high`** (mostly stable) are predicted accurately by LightGBM but are overestimated by TCN.  
- Residuals’ **mean absolute error**: LightGBM ≈ 0.0382, TCN ≈ 0.0659 → TCN overestimates **~73% more** on average.  
- LightGBM’s residual skew = -0.332 vs TCN skew = -0.178, indicating LightGBM slightly underpredicts extreme high-risk cases but overall maintains better balance.  
**Interpretation (clinical):**  
- LightGBM provides **faithful estimates of sustained high-risk exposure**, allowing clinicians to identify patients with minimal or moderate risk periods accurately.  
- TCN’s overestimation inflates perceived deterioration time, potentially leading to **unnecessary escalations, monitoring, or interventions**.  
- LightGBM’s tighter residual distribution preserves both low- and high-risk extremes, maintaining actionable stratification of patients over their stay.  
**Conclusion (Scatter):**  
- LightGBM preserves clinical fidelity of high-risk duration across patients.  
- TCN_refined tends to **exaggerate risk exposure**, overestimating the percentage of time spent in high-risk states by ~70% relative to LightGBM.  
- **Clinical relevance:** LightGBM enables more reliable identification of patients requiring intervention and reduces false positives from inflated risk predictions.

2. **Residuals Distribution**  
**CSV summary**
| Model | Mean | Std | Min | Max | Skew | Kurtosis |
|-------|------|-----|-----|-----|------|----------|
| LightGBM | 0.00130 | 0.0382 | -0.0645 | 0.0619 | -0.332 | -1.141 |
| TCN_refined | 0.1106 | 0.0659 | 0.00055 | 0.2177 | -0.178 | -1.252 |
**Interpretation (clinical):**  
- LightGBM residuals centered near zero indicate **unbiased estimates of high-risk exposure**, accurately capturing the percentage of time patients spend in high-risk states.  
- TCN_refined residual mean of +0.111 indicates **systematic overestimation of high-risk duration by ~11%**, exaggerating patient deterioration.  
- LightGBM’s tighter residual SD (~0.0382 vs 0.0659 → 42% lower) ensures **more consistent patient stratification** and better detection of both low- and high-risk patients.  
- Negative skew in LightGBM (-0.332) suggests slight underprediction for extreme high-risk patients, but overall residuals remain tightly clustered; TCN’s smaller negative skew (-0.178) reflects less extreme underprediction but broader error distribution.  
**Comparison Statistics:**  
- **Mean absolute residual (LightGBM vs TCN):** 0.0382 vs 0.0659 → LightGBM reduces average error by **~42%**, reflecting substantially more accurate risk duration predictions.  
- **Max residual:** LightGBM = 0.0619 vs TCN = 0.2177 → TCN overestimates extreme high-risk periods by **>3×** compared to LightGBM.  
- **Clinical implication:** LightGBM’s residual profile allows clinicians to **trust predicted high-risk time**, supporting targeted interventions, whereas TCN may **inflate risk exposure**, potentially triggering unnecessary monitoring or interventions.  
**Conclusion (Residuals):**  
- LightGBM provides **more reliable, unbiased, and clinically interpretable predictions** of percentage time spent in high-risk states.  
- TCN_refined systematically overestimates sustained high-risk duration, reducing clinical fidelity and practical utility for patient monitoring.

3. **Residual KDE / Distribution**  
**CSV summary / KDE plot:**  
| Model | KDE Peak Residual | Approx. Residual Spread (±1 SD) |
|-------|-----------------|--------------------------------|
| LightGBM | 0.0 | ~0.038 |
| TCN_refined | +0.111 | ~0.066 |
**Observations from KDE CSV:**  
- LightGBM residuals are tightly concentrated around 0 (KDE peak ≈ 4.05 in density units), consistent with the numeric residual mean (~0.0013) and SD (~0.038).  
- TCN_refined residuals peak near +0.11 (KDE peak ≈ 2.40 in density units), with a broader spread (~0.066), confirming the numeric residual SD (~0.0659) and positive bias.  
**Interpretation (clinical):**  
- The KDE quantitatively corroborates the residual metrics: LightGBM errors are small and tightly clustered, meaning clinicians can **trust predicted high-risk durations to reflect actual patient experience**.  
- TCN’s wider, positively skewed residual distribution indicates moderate-risk patients are often **overestimated as high-risk**, potentially triggering unnecessary interventions or over-monitoring.  
- Approximate residual spreads from the KDE (±1 SD) match the numeric residual SDs, reinforcing the consistency between numeric and distributional analyses.  
**Conclusion (KDE / Distribution):**  
- The KDE visually and quantitatively supports the residual statistics: LightGBM provides **accurate, consistent predictions of sustained high-risk exposure**, while TCN systematically exaggerates risk, highlighting its **reduced reliability for patient stratification over the admission period**.

4. **Error vs True Values**
**CSV Summary:**  
| Aspect | LightGBM | TCN_refined | Comparative Insight |
|---------|-----------|-------------|----------------------|
| Mean residual | +0.0013 | +0.1106 | TCN overestimates overall high-risk exposure by **~11%** of the admission duration. |
| Variance trend (vs truth) | Flat (corr = −0.16) | Increasing with truth (corr = −0.41) | LightGBM maintains stable error regardless of true exposure; TCN errors grow larger and more variable for high-risk patients. |
| Bias pattern | Slight underprediction near extremes | Positive bias at low–mid true values, mild underprediction at high true values | TCN transitions from **overestimating** short-risk to **underestimating** prolonged-risk patients. |
| Residual–truth slope | ≈ 0 | −0.41 correlation | Confirms systematic bias reversal in TCN (over → under) as true risk exposure increases. |
| Range | −0.0645 → +0.0619 | +0.0005 → +0.2177 | TCN’s maximum error is **>3× larger**, indicating weaker calibration at both extremes. |
**Interpretation (clinical):**  
- **LightGBM:**  
  - Residuals remain tightly distributed around zero across all `pct_time_high` values.  
  - Indicates **strong calibration and uniform accuracy** — both brief and prolonged high-risk exposures are modelled reliably.  
  - The flat variance trend (corr ≈ −0.16) supports clinical interpretability: predicted high-risk time mirrors true deterioration exposure consistently.  
- **TCN_refined:**  
  - Residuals show strong positive bias at low true values (`pct_time_high` < 0.15), meaning **stable patients are overestimated** as being in high risk for longer durations.  
  - As `pct_time_high` increases beyond 0.2, residuals decline toward zero or slightly negative, reflecting **underestimation** for patients who truly spend longer in high-risk states.  
  - The negative correlation (−0.41) demonstrates a **regression-to-the-mean effect**: predictions are compressed toward the average, losing fidelity at the extremes.
**Clinical Implications:**  
- For **short-risk patients**, TCN’s positive residuals inflate deterioration time → **unnecessary escalation or monitoring**.  
- For **prolonged-risk patients**, underestimation may cause **delayed escalation**, as sustained deterioration is underrepresented.  
- LightGBM’s near-zero mean bias and consistent variance make it **trustworthy across all patient profiles**, ensuring fair and accurate triage across severity levels.
**Conclusion (Error vs True):**  
- The **Error–Truth analysis** confirms that **LightGBM** maintains stable, unbiased residuals across the full range of true deterioration durations.  
- **TCN_refined** displays heteroscedastic and directionally biased errors → systematically **overestimating low-risk** patients and **underestimating high-risk** ones.  
- These findings complement the **Residuals Distribution** section by pinpointing **where** TCN’s bias manifests and **how** its calibration degrades with patient severity, rather than just summarizing overall error magnitude.

**Overall Summary (`pct_time_high`)**

| Dimension | LightGBM | TCN_refined | Clinical Comparative Interpretation |
|-----------|-----------|-------------|-----------------------------------|
| **Scatter alignment** | Residual range −0.0645 → 0.0619 | 0.00055 → 0.2177 | LightGBM residuals ~3.5× tighter; predictions closely follow `y=x`, reflecting true proportion of stay in high-risk state. |
| **Mean error** | 0.0013 | 0.1106 | LightGBM effectively unbiased; TCN systematically overestimates high-risk duration by ~11% of the admission. |
| **Residual Std** | 0.0382 | 0.0659 | LightGBM SD ~42% lower → more consistent patient-level risk stratification; TCN shows greater variability, especially at high true values. |
| **Residual max** | 0.0619 | 0.2177 | TCN occasionally predicts excessively long high-risk periods (~3.5× LightGBM max), indicating weaker calibration at extremes. |
| **KDE peak** | 0 | 0.11 | LightGBM errors tightly concentrated at zero → high clinical reliability; TCN biased toward overestimation for low-to-mid pct_time_high patients. |
| **Error vs True trend** | Minimal systematic bias | Positive bias for low-to-mid values, slight underestimation at high values | LightGBM maintains proportionality across all risk levels; TCN displays regression-to-mean bias, compressing extremes. |


**Final Interpretation (`pct_time_high`)**  
- **LightGBM:**  
  - Residuals are consistently centered around zero across the full spectrum of true high-risk duration (`pct_time_high`).  
  - Flat variance and tight residual distribution support **strong calibration**, allowing clinicians to trust predicted high-risk time for both brief and prolonged deterioration episodes.  
  - KDE peak at zero confirms high prediction fidelity; extreme patients are not systematically over- or under-predicted.  
- **TCN_refined:**  
  - Positive residuals dominate at low-to-mid true values, indicating **systematic overestimation** for patients with short high-risk periods.  
  - At high true values, residuals slightly decrease or invert, showing **underestimation** of prolonged high-risk exposure.  
  - Error variance increases with higher true values, highlighting **heteroscedasticity** and reduced reliability for patients at the extremes.  
  - Overall, TCN exhibits a **regression-to-the-mean effect**, flattening true extremes and compressing predicted risk durations toward the population average.
- **Clinical Implications:**  
  - LightGBM enables precise **triage and prioritisation**, minimizing false escalation for stable patients while correctly highlighting patients with sustained high-risk states.  
  - TCN’s overestimation of low-risk patients may trigger unnecessary monitoring or interventions, while underestimation of high-risk patients could delay critical escalation.

**Conclusion (`pct_time_high`)**
- **LightGBM consistently outperforms TCN_refined** for predicting the proportion of admission spent in high-risk states.  
- Quantitative advantages:  
  - Max residual ~3.5× lower,  
  - Standard deviation ~42% lower,  
  - Mean bias near zero versus TCN +11% overestimation.  
- LightGBM’s **tight, symmetric, and unbiased residuals** ensure clinically actionable predictions, reliable stratification, and preservation of patient-level deterioration dynamics.  
- TCN_refined shows **broader, positively biased, and heteroscedastic errors**, reducing interpretability and practical utility for monitoring sustained high-risk exposure.  
- **Clinical takeaway:** For `pct_time_high`, LightGBM delivers the **most reliable, calibrated, and actionable predictions**, while TCN’s systematic biases and error variability limit its clinical applicability.

---

### Final Integrated Analysis and Conclusion
#### Overall Comparison Across All Targets
| Target | Best Model | Key Quantitative Advantages | Key Interpretive Insights |
|--------|------------|----------------------------|--------------------------|
| **max_risk** | TCN_refined | ROC AUC +0.077 vs LightGBM, AP +0.0123, ~12× higher early TPR at FPR=0 | Excels at detecting transient deterioration events; early warning sensitivity superior; probability spread compressed → aggressive detection but reduced calibration. |
| **median_risk** | LightGBM | ROC AUC +0.1389 (~17% relative), AP +0.2834 (~45% relative), Brier ≈3× lower, ECE ≈2.7× lower | Superior at identifying sustained physiological instability; well-calibrated, interpretable probabilities; TCN underperforms due to temporal-target mismatch. |
| **pct_time_high** | LightGBM | RMSE 0.038 vs 0.056 (~48% lower), R² 0.793 vs 0.548, mean residual ~0 vs 0.111, residual SD ~42% lower | Predicts proportion of high-risk exposure accurately; minimal bias; residuals tightly centered → reliable patient-level stratification; TCN overestimates low-risk and underestimates prolonged-risk patients. |

**Integrated Insights**
1. **Task–Model Alignment**
   - Temporal models (TCN) excel at **dynamic, short-term event detection** (`max_risk`), capturing sharp transient spikes.
   - Classical tabular models (LightGBM) excel at **aggregate, patient-level predictions** (`median_risk`, `pct_time_high`), leveraging summary statistics for stable and calibrated outputs.
2. **Calibration vs Discrimination**
   - LightGBM demonstrates **better calibration** across all targets (Brier, ECE), supporting probabilistic interpretability for clinical decision-making.
   - TCN provides **higher discrimination** for acute events but suffers from **heteroscedastic errors** and systematic bias in aggregated measures.
3. **Residual Patterns**
   - LightGBM residuals: tightly centered, symmetric, minimal heteroscedasticity → consistent clinical reliability.
   - TCN residuals: overestimation of low-risk, underestimation of high-risk, broader spread → regression-to-mean bias.
4. **Practical Clinical Implications**
   - **LightGBM:** reliable for patient-level triage, monitoring cumulative high-risk exposure, and assigning persistent risk scores.
   - **TCN_refined:** valuable for early-warning systems and detecting transient deterioration events, but requires careful calibration for aggregate or long-term predictions.
   - Combining models could leverage strengths: TCN for early alerts, LightGBM for stable risk stratification.

#### Final Conclusion
- **Best overall performer:** **LightGBM** for multi-target ICU deterioration prediction.  
- **Strengths:** Calibrated, interpretable, robust for small-sample, patient-level predictions; outperforms TCN on sustained and cumulative risk targets (`median_risk`, `pct_time_high`).  
- **TCN_refined:** Outperforms LightGBM in **early detection of acute deterioration events** (`max_risk`) due to temporal sensitivity but is less reliable for aggregate or long-term outcomes.  
- **Real-world context:** Both models have complementary roles; **LightGBM** ensures dependable daily patient-level monitoring, while **TCN** can provide additional alerting for sudden risk spikes.  
- **CV/Publication-ready insight:** This comparative study quantifies model performance with full metrics (AUC, AP, RMSE, R², residual distribution, calibration scores), demonstrates actionable differences in temporal vs static modelling, and highlights practical deployment considerations in ICU deterioration prediction.

---

### Performance Analysis Outputs (`performance_analysis.py`)
1. **Output Files Table**
**Summary Metrics + Plot Numericals**
| File Name | Type | Folder | Description / Purpose |
|-----------|------|--------|---------------------|
| `comparison_table.csv` | CSV | `comparison_metrics/` | Aggregated metrics for both models and all targets; includes ROC AUC, F1, RMSE, R², Brier, ECE. |
| `roc_max_risk.csv` | CSV | `comparison_metrics/` | ROC data for `max_risk`; includes FPR/TPR arrays, AUC, prevalence for both models. |
| `roc_median_risk.csv` | CSV | `comparison_metrics/` | ROC data for `median_risk`. |
| `pr_max_risk.csv` | CSV | `comparison_metrics/` | Precision–Recall data for `max_risk`; includes precision/recall arrays, AP for both models. |
| `pr_median_risk.csv` | CSV | `comparison_metrics/` | PR data for `median_risk`. |
| `calibration_max_risk.csv` | CSV | `comparison_metrics/` | Calibration data for `max_risk`; includes mean predicted probability, fraction of positives, Brier, ECE, and bin counts per model. |
| `calibration_median_risk.csv` | CSV | `comparison_metrics/` | Calibration data for `median_risk` (same structure as above). |
| `prob_hist_max_risk.csv` | CSV | `comparison_metrics/` | Predicted probability histogram for `max_risk`; includes distribution stats (mean, std, min, max, skew, kurtosis). |
| `prob_hist_median_risk.csv` | CSV | `comparison_metrics/` | Histogram data for `median_risk` (same structure). |
| `scatter_pct_time_high.csv` | CSV | `comparison_metrics/` | Regression scatter data (`y_true` vs `y_pred` for both models). |
| `residuals_pct_time_high.csv` | CSV | `comparison_metrics/` | Raw residuals + summary stats (mean, std, min, max, skew, kurtosis) for each model. |
| `residuals_kde_pct_time_high.csv` | CSV | `comparison_metrics/` | Kernel density of residuals (`grid`, `kde` arrays per model). |
| `error_vs_truth_pct_time_high.csv` | CSV | `comparison_metrics/` | Residual vs true values + residual statistics for both models. |

**Visualisations**
| PNG File | Folder | Description |
|----------|--------|------------|
| `roc_max_risk.png` | `comparison_plots/` | ROC curve overlay for `max_risk`. |
| `roc_median_risk.png` | `comparison_plots/` | ROC curve overlay for `median_risk`. |
| `pr_max_risk.png` | `comparison_plots/` | Precision–Recall curve for `max_risk`. |
| `pr_median_risk.png` | `comparison_plots/` | Precision–Recall curve for `median_risk`. |
| `calibration_max_risk.png` | `comparison_plots/` | Calibration plot (reliability diagram) for `max_risk`. |
| `calibration_median_risk.png` | `comparison_plots/` | Calibration plot for `median_risk`. |
| `prob_hist_max_risk.png` | `comparison_plots/` | Predicted probability histograms (LightGBM vs TCN). |
| `prob_hist_median_risk.png` | `comparison_plots/` | Probability histogram for `median_risk`. |
| `scatter_pct_time_high.png` | `comparison_plots/` | Regression true vs predicted overlay. |
| `residuals_pct_time_high.png` | `comparison_plots/` | Residual distributions with KDE overlay (both models). |
| `error_vs_truth_pct_time_high.png` | `comparison_plots/` | Residual vs truth scatter plot for regression. |
| `metrics_comparison_max_risk.png` | `comparison_plots/` | Grouped bar chart comparing ROC AUC, F1, Brier, ECE for `max_risk`. |
| `metrics_comparison_median_risk.png` | `comparison_plots/` | Metric comparison chart for `median_risk`. |
| `metrics_comparison_pct_time_high.png` | `comparison_plots/` | Regression metric comparison (RMSE, R²) for `pct_time_high`. |



2. **Folder Structure Diagram**
```text
src/
└── results_finalisation/
    ├── performance_analysis.py
    ├── comparison_metrics/
    │   ├── comparison_table.csv
    │   ├── roc_max_risk.csv
    │   ├── roc_median_risk.csv
    │   ├── pr_max_risk.csv
    │   ├── pr_median_risk.csv
    │   ├── calibration_max_risk.csv
    │   ├── calibration_median_risk.csv
    │   ├── prob_hist_max_risk.csv
    │   ├── prob_hist_median_risk.csv
    │   ├── scatter_pct_time_high.csv
    │   ├── residuals_pct_time_high.csv
    │   ├── residuals_kde_pct_time_high.csv
    │   └── error_vs_truth_pct_time_high.csv
    └── comparison_plots/
        ├── roc_max_risk.png
        ├── roc_median_risk.png
        ├── pr_max_risk.png
        ├── pr_median_risk.png
        ├── calibration_max_risk.png
        ├── calibration_median_risk.png
        ├── prob_hist_max_risk.png
        ├── prob_hist_median_risk.png
        ├── scatter_pct_time_high.png
        ├── residuals_pct_time_high.png
        ├── error_vs_truth_pct_time_high.png
        ├── metrics_comparison_max_risk.png
        ├── metrics_comparison_median_risk.png
        └── metrics_comparison_pct_time_high.png
```

**Rationale**
- **Purpose of SHAP (Phase 6):** 
  - This phase provides the definitive interpretability layer for the LightGBM models trained and validated in earlier phases. 
  - It quantifies **per-patient**, **per-feature** contributions to each model’s prediction, completing the interpretability pipeline.
- **Relation to Prior Phases:**  
  - **Phase 3 (Feature Importance):** Used cross-validated LightGBM split importances as an early sanity check for signal strength and feature relevance.  
  - **Phase 5 (Model Performance):** Provided the final metrics (AUROC, RMSE, etc.) establishing predictive validity.  
  - **Phase 6 (SHAP Analysis):** Builds directly on these by explaining *why* each model performs as it does — clarifying which features drive correct or incorrect predictions and how risk patterns emerge.
- **Clinical Interpretability:** 
  - SHAP allows direct mapping between model behaviour and clinical reasoning (e.g., showing how respiratory rate or SpO₂ influences high-risk classifications).
- **Classification Handling:**  
  - For binary LightGBM classifiers, SHAP returns two arrays (class 0 and class 1).  
  - The analysis correctly extracts the **class 1 (positive / high-risk)** SHAP array to interpret model behaviour in terms of clinically meaningful outcomes.
- **Design Priorities:**  
  - Dependence plots and patient-level SHAP summaries can be added later for deeper interpretation.  
  - For this phase, the focus was ensuring:
    - Numeric SHAP stability  
    - Per-patient (70 × 40) validity  
    - Transparent diagnostic logging  
    - Reliable mean absolute SHAP rankings and plots

---

### Model Interpretability Rationale
**Overview**
- Model interpretability provides the final analytical layer in the ICU deterioration prediction study.  
  - After Steps 1–2 quantified how LightGBM and TCN_refined differ in predictive performance,  
  - Step 3 focuses on why those differences exist → by analysing how each model internally processes information and which clinical variables most strongly drive predictions.
- Interpretability complements comparative analysis: it transforms performance numbers into mechanistic insight.  
- Where comparative metrics measure **what happened**, interpretability explains **why it happened**.

**Conceptual Relationship**
| Aspect | **Comparative Analysis (Steps 1–2)** | **Interpretability Analysis (Step 3)** |
|--------|--------------------------------------|---------------------------------------|
| **Core Goal** | Quantify and compare external performance outcomes between models. | Understand internal model reasoning → which features or time patterns drive predictions. |
| **Analytical Focus** | Output-level behaviour: AUC, F1, Brier, ECE, RMSE, R², calibration and residual patterns. | Input-level mechanisms: feature attributions (LightGBM SHAP values) and temporal relevance (TCN saliency maps). |
| **Primary Question** | “Which model performs better, and how do their results differ?” | “Why does each model behave this way, and what clinical factors influence its output?” |
| **Outputs** | Comparison tables, residual distributions, ROC/PR and calibration plots, numeric CSVs. | SHAP feature importance plots, contribution statistics, and temporal saliency visualisations. |
| **Analytical Depth** | Quantitative and diagnostic → reveals statistical patterns and numerical differences. | Explanatory and mechanistic → reveals causal structure behind those patterns. |
| **Interpretive Level** | External validation (outcomes). | Internal reasoning (model logic). |
| **Timing** | Conducted first, immediately after evaluation. | Conducted after comparative results are known. |
| **Role in Phase 6** | Establishes empirical benchmark and identifies observed strengths/weaknesses. | Explains the underlying drivers of those strengths and weaknesses. |

**Rationale and Integration**
1. **Sequential Logic**
  - Comparative analysis defines how well each model performs and in what ways they differ.
  - Interpretability builds on that foundation, exploring why those differences arise.
  - Together they form a complete analytical pipeline: **Performance → Behaviour → Reasoning**
2. **Complementary Role**
  - Steps 1–2 showed LightGBM’s stronger calibration and discrimination on small ICU datasets,  
     and TCN_refined’s weaker but temporally aware behaviour.
  - Step 3 investigates the mechanistic causes:  
    - Which features (e.g., SpO₂, respiratory rate, NEWS2 components) dominate LightGBM’s output.  
    - Whether TCN captured short-term physiological trends or suffered from data scarcity and target imbalance.
3. **Scientific Purpose**
  - Moves beyond model ranking into **model understanding**.  
  - Provides **clinical interpretability** → validating that learned relationships make physiological sense.  
  - Provides **technical interpretability** → confirming that performance differences stem from model architecture and data representation, not random noise.
4. **Outcome**
  - Establishes transparent reasoning for observed performance gaps.  
  - Informs how both models could be deployed together:  
    - LightGBM for stable, interpretable probability estimation.  
    - TCN for temporal anomaly detection in richer datasets.

**Analytical Progression and Fit**
- Interpretability is the **third and final analytical layer** in the evaluation pipeline.  
- It directly builds on the foundations established by Steps 1–2:

| Analytical Stage | Step | Focus | Question Answered | Output Type | Analytical Role |
|------------------|------|--------|-------------------|--------------|-----------------|
| **Quantitative Comparison** | Step 1 | Summary metrics (AUC, F1, Brier, ECE, RMSE, R², Accuracy, Precision, Recall) | *How well do the models perform overall?* | Unified comparison table | Establishes statistical and calibration baselines |
| **Behavioural Diagnostics** | Step 2 | Plot numeric CSVs (ROC, PR, Calibration, Residuals, KDE, Error–Truth) | *How do models behave across different risk levels or data distributions?* | Numerical diagnostics + visual plots | Explores detailed trends, variance, and reliability |
| **Interpretability** | Step 3 | SHAP (LightGBM) + Saliency (TCN) | *Why do models behave differently? What drives their predictions?* | Feature attributions + temporal relevance maps | Explains underlying causal structure |

- This staged design ensures a logical analytical flow:
  - **Metrics → Behaviour → Mechanism**  
  - **Performance → Comparison → Explanation**

**Summary**
- Model interpretability in Phase 6 is not a separate exercise but a logical continuation of comparative analysis which finalises the framework.  
- It transitions from performance benchmarking (Step 1) and behavioural diagnostics (Step 2) to mechanistic explanation (Step 3).  
- Together, these stages form a complete analytical progression:
  - **Step 1: Quantify — How well do models perform?**  
  - **Step 2: Diagnose — How do their errors and trends behave?**  
  - **Step 3: Explain — Why do they behave this way?**
- By integrating interpretability into Phase 6, the analysis moves beyond numerical comparison to causal understanding → transforming the project from a performance report into a scientifically reasoned, transparent explanation of ICU deterioration prediction models.
- Together, these stages ensure that model comparison is not only statistically valid but also scientifically and clinically explainable → an essential requirement for transparent deployment in critical-care prediction systems.

---

### Feature Importance (Phase 3) vs SHAP Interpretability (Phase 6)
1. **Purpose and Analytical Context**
  - Interpretability is a critical component of any clinical ML project, it explains why a model makes its predictions and whether those reasons are clinically valid.  
  - In this project, interpretability occurs at **two distinct stages**:

  | Stage | Goal | Timing | Analytical Role |
  |--------|------|--------|-----------------|
  | **Phase 3 – Feature Importance** | Early-stage interpretability for model validation and feature selection. | During model tuning and cross-validation. | Ensured the LightGBM model was learning physiologically coherent signals and informed later model refinement. |
  | **Phase 6 – SHAP Analysis** | Final interpretability for explanatory insight. | After final comparative evaluation (LightGBM vs TCN). | Provides definitive, model-aligned explanations linking features to final performance differences. |

  - These two stages together form a **progressive interpretability pipeline**:
    1. Feature importance confirms that the model learns meaningful features.  
    2. SHAP explains how those features influence final predictions and comparative behaviour.

2. **Conceptual Overview**
  **Feature Importance**
  - Feature importance in tree-based models (like LightGBM) quantifies how much each feature contributes to reducing prediction error across all decision splits in the model.  
  - For each tree, LightGBM measures:
    - **Split gain:** how much a feature reduces the loss function when used to split a node.  
    - **Frequency:** how often a feature is used for splitting.
  - The overall importance score is the **sum or average of these gains** across all trees.  
  - Formally, the importance of a feature fᵢ is calculated as: `Importance(fᵢ) = Σₜ₌₁ᵀ Σₛ∈Sₜ(fᵢ) ΔLₛ`
    - **ΔLₛ** = reduction in the loss function achieved by split *s* that uses feature *fᵢ*  
    - **Sₜ(fᵢ)** = the set of all splits in tree *t* that use feature *fᵢ*  
    - **T** = total number of trees in the model  
  - **Interpretation:**
    - Higher values mean the feature contributed more to reducing model error.
    - It provides **global**, direction-agnostic insight (it doesn’t tell you whether higher or lower values of the feature increased risk).

  **SHAP (SHapley Additive exPlanations)**
  - SHAP is a **game-theoretic framework** that decomposes each individual prediction into additive feature contributions.  
  - It answers: “How much did each feature push this prediction away from the average baseline?”
  - For each feature fᵢ, the SHAP value φᵢ represents its **marginal contribution** averaged over all possible combinations of other features:
  `φᵢ = Σₛ⊆F\{i}  [ |S|! (|F|-|S|-1)! / |F|! ] × [ f(S ∪ {i}) - f(S) ] `
  - where:  
    - **f(S)** = model’s output when only features in subset S are used  
    - **F** = set of all features  
    - **S** = subset of features not including i
    - The factorial weights **(|S|! (|F|-|S|-1)! / |F|!)** ensure fair averaging over all possible feature orderings  
  - **Key properties:**
    - **Additivity:** The sum of all feature SHAP values equals the prediction difference from the baseline.
    - **Consistency:** If a feature contributes more to the model in an updated version, its SHAP value will not decrease.
    - **Local + Global:** Aggregating SHAP values across all samples yields global feature importance, while inspecting single samples gives local explanations.

  **Why Both Matter**
  | Method | Focus | Directionality | Scope | Reliability |
  |---------|--------|----------------|--------|-------------|
  | **Feature Importance** | Counts or gain from tree splits | Non-directional | Global only | Fast, approximate |
  | **SHAP** | Shapley-value decomposition of prediction output | Directional (positive/negative) | Local + Global | Theoretically grounded, precise |
  - In summary:
    - **Feature importance** shows which features matter most overall.  
    - **SHAP** shows how and why they matter for each prediction.  
  - That’s why SHAP replaces feature importance for the **final interpretability step**, it adds the causal, directional insight needed to justify the model’s behaviour clinically.

3. **Why We Performed Feature Importance in Phase 3**
  **Purpose:**  
  - To validate the model’s learning behaviour during development and ensure no spurious or artefactual predictors dominated performance.
  **Rationale:**  
  - At this stage, the LightGBM model was being trained using **5-fold cross-validation** and **best hyperparameters** (`best_params.json`).  
  - By averaging feature importances across folds, we could:
    - Assess feature stability and consistency.  
    - Identify which variables (NEWS2, vitals, demographics) drove predictions most strongly.  
    - Verify that key physiological indicators (SpO₂, HR, RR, etc.) ranked high, confirming clinical plausibility.  
  **Outputs:**  
  - CSVs: `{target}_feature_importance.csv` (average split-count per feature).  
  - PNGs: Top 10 feature bar plots for `max_risk`, `median_risk`, and `pct_time_high`.
  | What It Measured | How It Worked | What It Showed |
  |------------------|---------------|----------------|
  | **Feature usage frequency** | Counted how often each feature was used to split decision trees. | Relative influence of each variable on prediction strength. |
  | **Global interpretability** | Aggregated across folds. | General insight into model focus and physiologic consistency. |

  **Summary:**  
  - Phase 3 feature importance acted as an exploratory interpretability checkpoint.
  - Confirming LightGBM’s internal logic before deeper temporal modelling and ensuring feature engineering choices were justified.

4. **Why We Are Now Performing SHAP in Phase 6**
  **Purpose:**  
  - To provide **final, model-specific interpretability** for the LightGBM model retrained in Phase 5 and compared against the TCN.  
  - The SHAP analysis directly explains why the final model achieved its observed discrimination, calibration, and regression patterns.
  **Rationale:**
  - The **Phase 6 LightGBM** is **not the same model** as Phase 3, it was retrained on a **70/15 train–test split** (to align with the TCN) and evaluated under different data conditions.  
  - Therefore, the old feature importance values are **no longer representative** of the final model’s behaviour.
  - SHAP provides both **global** and **local** interpretability, capturing not just which features matter, but *how* they affect predictions.

  | Feature Importance (Phase 3) | SHAP (Phase 6) |
  |-------------------------------|----------------|
  | Aggregated split counts across CV folds. | Shapley-based additive attributions per feature per sample. |
  | Directionless (only magnitude of importance). | Directional (positive/negative influence on prediction). |
  | Global ranking of key predictors. | Local + global interpretability → explains individual patient predictions. |
  | Simpler, computationally light. | More precise, computationally heavy. |
  | Suitable for early model validation. | Required for final mechanistic explanation. |

  **Why We’re Not Repeating Feature Importance Now:**  
  - If re-run on the Phase 6 LightGBM, feature importance would produce **broadly similar rankings** (SpO₂, RR, HR still top) because the model architecture and features are unchanged.  
  - However, the **exact numeric importances** would shift due to the different data distribution and smaller training size.  
  - These differences would add little new insight compared to SHAP, which already captures all relative and directional effects.  
  - Thus, repeating feature importance would be **redundant** → SHAP subsumes its value and provides far richer interpretive resolution.

5. **Relationship Between the Two Analyses**

| Aspect | **Phase 3: Feature Importance** | **Phase 6: SHAP Interpretability** |
|--------|---------------------------------|------------------------------------|
| **Analytical timing** | During model development and cross-validation. | During final comparative evaluation. |
| **Primary function** | Feature sanity check, early validation. | Definitive model explanation. |
| **Granularity** | Aggregate, non-directional. | Sample-level, directional, additive. |
| **Outcome link** | Aligned with model tuning performance. | Aligned with final performance metrics and comparisons to TCN. |
| **Interpretive value** | Confirmed physiologic relevance of model learning. | Explained why LightGBM generalised better and calibrated more accurately. |
| **Redundancy check** | Could be repeated, but offers little new insight. | Supersedes feature importance, final interpretive authority. |

6. **Why Doing Both Still Matters**
  - Performing both analyses — **feature importance (Phase 3)** and **SHAP (Phase 6)** — ensures interpretability at *two levels of the project lifecycle*:

  | Stage | Purpose | Value |
  |--------|----------|-------|
  | **Developmental (Phase 3)** | Confirm model coherence and detect spurious predictors early. | Guarantees sound model design and feature validity before final retraining. |
  | **Evaluative (Phase 6)** | Explain final model decisions and connect them to observed comparative trends. | Anchors performance differences (e.g., calibration, discrimination) in physiological reasoning. |

  - Together they provide **temporal interpretability continuity**:
    - Phase 3 established that the model learns meaningful patterns.  
    - Phase 6 demonstrates how those patterns manifest in the final model’s predictions.  
  - This ensures the interpretability narrative evolves alongside model maturity → from exploratory to definitive.

7. **Summary**
  - **Feature importance (Phase 3):** Early-stage, cross-validated interpretability verifying model learning stability and physiological plausibility.  
  - **SHAP (Phase 6):** Final-stage, model-aligned interpretability explaining why the final LightGBM performs as it does in the comparative analysis.  
  - **Both combined:** Provide a full interpretability trajectory from model validation to mechanistic explanation, ensuring that every stage of model development and evaluation is both quantitatively verified and qualitatively understood.

### SHAP (SHapley Additive exPlanations) Explained 
**What SHAP is**
- **SHAP (SHapley Additive exPlanations)** is a game-theoretic approach to explain how each input feature contributes to a model’s output.  
- It assigns each feature a *Shapley value*, derived from cooperative game theory, representing the **average marginal contribution** of that feature across all possible feature combinations.  
- In other words, SHAP answers: “If this feature were removed or changed, how much would the model’s prediction change on average?”
**How It Works**
  1. **Model Prediction as a Game:** Each input feature is treated as a “player” in a coalition whose goal is to achieve the model’s output (prediction).  
  2. **Shapley Value Calculation:** SHAP computes each feature’s contribution by averaging its marginal effect across all feature subsets, ensuring fair, additive, and consistent attributions.
  3. **Additivity Property:**  
    - For any model `f(x)`, the prediction can be expressed as: `f(x) = φ₀ + Σ φᵢ`
    - Where:
      - `φ₀` = baseline model output (the expected prediction when no features are present)  
      - `φᵢ` = contribution of feature `i` to the final prediction  
  4. **Interpretation:**  
    - Positive SHAP values → feature pushes prediction up (toward positive class or higher risk).  
    - Negative SHAP values → feature pushes prediction down (toward negative class or lower risk).  
    - Magnitude = strength of influence.
**Why SHAP Is Needed**
- Model performance metrics (AUROC, RMSE, etc.) tell **how well** a model works but not **why** it makes its decisions.  
- SHAP provides:
  - **Transparency:** Explains each prediction in terms of measurable feature contributions.  
  - **Trust and validation:** Clinicians can verify that model reasoning aligns with medical logic.  
  - **Debugging insight:** Detects overfitting or spurious correlations.  
  - **Scientific interpretability:** Enables reasoning about causal or clinical relevance of features.  
- In this project, SHAP (Phase 6) completes the interpretability chain → linking **model performance (Phase 5)** to **clinical understanding.**
**What Must Be Decided in a SHAP Analysis**
| **Decision** | **Explanation** | **Our Choice** | **Rationale** |
|---------------|----------------|----------------|----------------|
| **Explainer Type** | Determines which SHAP algorithm variant to use. Options include `TreeExplainer` (for tree models), `KernelExplainer` (model-agnostic), and `DeepExplainer` (for neural networks). | `TreeExplainer` | LightGBM is a tree-based model; `TreeExplainer` provides exact SHAP values efficiently without sampling noise. |
| **Model Output Type** | Defines what SHAP explains → either raw decision values (“logits”) or post-activation probabilities. For classification models, SHAP can explain the contribution to each class probability separately. | Default `raw` output for regression, **positive class (class-1)** output for classification. | For LightGBM, using `model_output="raw"` avoids unsupported configurations and numerical instability. Focusing on class-1 (the “high-risk” class) makes SHAP values clinically interpretable as feature contributions toward higher risk. |
| **Aggregation Method** | Specifies how SHAP values are summarised across all patients. Options include mean, mean absolute, or variance-based aggregation. | Mean absolute SHAP values | The mean of absolute SHAP magnitudes captures overall feature influence regardless of sign (direction). This is standard in global feature-importance reporting. |
| **Scope of Analysis** | Defines whether SHAP analysis is performed per individual prediction (local interpretability) or summarised across the entire dataset (global interpretability). | **Global (dataset-level)** | Our aim was reproducibility and model sanity checking. Local SHAP visualisations can be added later for patient-specific insights, but global analysis was prioritised to ensure stable feature ranking across the cohort. |
| **Data subset for SHAP** | Which samples to compute SHAP on (training vs test vs combined). | **Training set (70 patients)** | Reflects what the model actually learned, provides stable global importance estimates, and aligns with Phase 6’s goal of explaining the trained model rather than test-set noise. |
| **Visualisation** | Determines how SHAP values are presented. Can include summary plots, dependence plots, or ranked bar charts. | Top-10 feature bar plots | Focused on interpretability and concise communication. Highlighting the ten most influential features per target makes patterns immediately interpretable and publication-ready. |

**Why We Chose `TreeExplainer`**
- **Purpose-built for tree-based models:** 
  - `TreeExplainer` is designed specifically for gradient-boosted trees such as LightGBM, XGBoost, and CatBoost. 
  - It directly leverages the internal tree structure to compute **exact SHAP values** rather than approximations.  
- **Computationally efficient:** 
  - It achieves polynomial-time complexity, scaling efficiently even for large patient-level datasets.  
- **Avoids approximation noise:** 
  - Unlike model-agnostic explainers such as `KernelExplainer` or `SamplingExplainer`, it does **not** rely on random perturbations or surrogate models, ensuring stable, reproducible attributions.  
- **Faithful to LightGBM’s logic:** 
  - By tracing each decision path within the trained LightGBM ensemble, it produces SHAP values that exactly match how the model partitions feature space and assigns risk.  
- **Compared alternatives:**
  - `KernelExplainer`: works for any model, but slow and approximate.
  - `LinearExplainer`: fast and exact, but only valid for linear models.
  - `DeepExplainer`: suitable for neural networks, not gradient-boosted trees.
- **Conclusion:**  
  - `TreeExplainer` provides the most **accurate**, **efficient**, and **semantically aligned** interpretability method for LightGBM models, making it the clear choice for this phase.

**Limitations & pragmatic trade-offs**
- **We prioritised global stability over directional nuance:** mean |SHAP| gives robust rankings; directional analyses (positive vs negative effects, dependence plots) are useful but secondary for the pipeline’s primary goal (definitive feature ranking for comparative reporting).
- **Model-output modes:** We avoided forcing `model_output="probability"` because TreeExplainer with certain feature-perturbation settings can be unsupported or unstable across SHAP/LightGBM versions; using the default TreeExplainer + correct class extraction gives stable, reproducible results.
- **Local explanations / dependence plots are optional extras:** They add richer interpretation for individual patients or conditional effects, but are not required for the core Phase 6 comparative interpretability deliverable.

**Summary**
- SHAP bridges the gap between **performance** and **interpretation**.  
- It turns opaque LightGBM predictions into clear, quantitative explanations → showing exactly which patient features drove high or low risk classifications.  
- In this project, using `TreeExplainer` ensures that interpretability is not just qualitative, but **numerically grounded, reproducible, and clinically verifiable.**

### Interpretation of SHAP Outputs
#### Overview
- This section interprets SHAP values for the three LightGBM targets:
  - **max_risk**: the highest deterioration risk the patient reached during their stay.
  - **median_risk**: the average risk level throughout the stay.
  - **pct_time_high**: the percentage of time spent in high-risk states.
- The goal is to identify the most influential features for each outcome and provide physiologically meaningful insights.

#### Interpretation of SHAP Outputs (`max_risk`)

| Rank | Feature | Mean |SHAP| Value | Interpretation |
|------|---------|----------------|----------------|
| 1 | `spo2_min` | 1.082 | Lowest SpO₂ is the dominant predictor of high-risk classification, consistent with respiratory deterioration driving escalation. |
| 2 | `supplemental_o2_mean` | 0.697 | Higher average O₂ supplementation increases predicted risk, aligning with oxygen support needs. |
| 3 | `respiratory_rate_max` | 0.533 | Elevated maximum respiratory rate reflects physiological stress contributing to high-risk predictions. |
| 4 | `temperature_missing_pct` | 0.406 | Missing temperature measurements influence predictions → likely proxying clinical instability or gaps in monitoring. |
| 5 | `heart_rate_mean` | 0.266 | Persistent tachycardia moderately increases predicted risk. |
| 6–10 | Temperature- and heart rate–related variables | 0.18–0.15 | Vital signs collectively contribute to model discrimination of maximum risk. |
**Interpretation Summary**
- Primary drivers are respiratory physiology (SpO₂, O₂ delivery).
- Secondary drivers include temperature and heart rate.
- Non-contributing features (systolic BP, CO₂ metrics) have minimal impact.
**Conclusion:**  
- For `max_risk`, SHAP confirms the model aligns with clinical expectations, emphasizing oxygenation and respiratory status.

#### Interpretation of SHAP Outputs (`median_risk`)

| Rank | Feature | Mean |SHAP| Value | Interpretation |
|------|---------|----------------|----------------|
| 1 | `respiratory_rate_mean` | 1.301 | Average respiratory rate is the dominant feature for median risk, reflecting ongoing respiratory instability. |
| 2 | `spo2_mean` | 0.901 | Low average SpO₂ strongly influences risk predictions, consistent with hypoxia. |
| 3 | `heart_rate_max` | 0.636 | Maximum heart rate signals physiological stress. |
| 4 | `systolic_bp_missing_pct` | 0.635 | Missing BP readings indicate unobserved instability or monitoring gaps. |
| 5 | `level_of_consciousness_missing_pct` | 0.536 | Missing consciousness measurements impact predictions, highlighting incomplete observations during high-risk periods. |
| 6–10 | Respiratory rate and temperature variables | 0.5–0.15 | Continuous contribution of respiratory patterns and thermoregulation in median-risk predictions. |
**Interpretation Summary**
- Median risk prediction continues to prioritize respiratory and oxygenation variables.
- Missingness metrics act as indirect markers of instability.
- Temperature and heart rate are secondary contributors.
- Zero-contribution features (CO₂ metrics, some supplemental O₂) are non-informative.
**Conclusion:**  
- For `median_risk`, respiratory dynamics and oxygenation dominate, with missingness features serving as a proxy for clinical instability.

#### Interpretation of SHAP Outputs (`pct_time_high`)

| Rank | Feature | Mean |SHAP| Value | Interpretation |
|------|---------|----------------|----------------|
| 1 | `respiratory_rate_mean` | 0.034 | Average respiratory rate drives cumulative high-risk duration, emphasizing sustained respiratory instability. |
| 2 | `heart_rate_max` | 0.014 | Maximum heart rate contributes moderately, reflecting physiological stress. |
| 3 | `supplemental_o2_mean` | 0.012 | Mean supplemental O₂ requirement affects predicted high-risk duration. |
| 4 | `spo2_mean` | 0.012 | Average SpO₂ influences risk duration, consistent with hypoxia prolonging high-risk periods. |
| 5 | `temperature_median` | 0.011 | Temperature reflects systemic stress or infection. |
| 6–10 | `spo2_min`, `heart_rate_mean`, missingness features | 0.010–0.007 | Minor contributions from missing data metrics and secondary physiological measures. |
**Interpretation Summary**
- Respiratory and oxygenation metrics dominate cumulative high-risk time predictions.
- Missingness features contribute slightly, highlighting data completeness as an indirect marker.
- Low-contributing features are physiologically less relevant for predicting high-risk duration.
**Conclusion:**  
- For `pct_time_high`, SHAP reveals that sustained respiratory dynamics and oxygenation are key determinants, with minor influence from missingness and secondary vital signs.

#### Missingness Features as Clinical Instability Indicators
- Some SHAP features represent the fraction of missing data → `temperature_missing_pct`, `systolic_bp_missing_pct`, and `level_of_consciousness_missing_pct`
- These “missingness features” can act as **proxies for clinical instability**.  
- When vital signs or observations are not recorded, it may indicate periods of high-risk or urgent clinical activity. 
- The model has learned that gaps in monitoring often correlate with deterioration, so these features appear important in SHAP analysis, even though they do not directly reflect physiology.

#### Overall Summary
- **Respiratory features (RR, SpO₂, O₂ support)** are the strongest predictors across all targets.
- **Heart rate and temperature** contribute moderately.
- **Missingness features** (BP, LOC, temperature) indicate real-world data capture gaps and correlate with risk. Act as indirect indicators of clinical instability, reflecting how incomplete observations often coincide with acute deterioration episodes in real-world ICU data. 
- **CO₂ metrics and some supplemental O₂ features** have negligible influence, suggesting these signals either lacked sufficient data quality or were redundant with stronger respiratory indicators.  
- Overall, the SHAP results confirm that LightGBM’s learned decision logic is **clinically interpretable**, **physiologically coherent**, strengthening confidence in the model’s validity and transparency.

### Theoretical Background: Saliency Mapping in Deep Learning
1. **Overview**
  - Saliency mapping is a gradient-based interpretability technique used to understand which input features most influence a model’s prediction.  
  - It was first introduced in computer vision (Simonyan et al., 2014) and has since been adapted to time-series and clinical models to provide transparency about why a model made a given prediction.  
  - Saliency maps show how sensitive the model’s output is to small changes in each input feature, helping identify what matters most and when it matters in a temporal sequence.
2. **Core Mathematical Concept**
  - Given a model output `y` (scalar prediction) and an input tensor `x` with dimensions `(T, F)` where:  
    - `T`: timesteps  
    - `F`: features  
  - The saliency for each element `x_{t,f}` is computed as `S_{t,f} = | (∂y / ∂x_{t,f}) × x_{t,f} |` where:
    - `(∂y / ∂x_{t,f})` measures how sensitive the output `y` is to small changes in feature `f` at time `t`.  
    - Multiplying by `x_{t,f}` weights this sensitivity by the actual input value → importance weighted by feature activation.  
    - Taking the absolute value removes directionality, leaving only the magnitude of influence.  
  - The resulting saliency tensor has shape `(T, F)` for a single sample and `(N, T, F)` across a dataset, where `N` = number of patients or sequences.
3. **Purpose and Rationale**
  - Saliency maps answer the question: Which features and time points most strongly influenced the model’s prediction?
  - They are especially valuable for:
    - **Clinical transparency:** explaining which physiological variables drive deterioration risk.  
    - **Temporal reasoning:** understanding how importance shifts through time (something static SHAP or permutation methods cannot show).  
    - **Model validation:** confirming whether learned attention aligns with known medical patterns (e.g., rising respiratory rate before deterioration).  
  - We selected this method for the TCN because it provides fine-grained, time-resolved feature attribution, aligning with our goal of understanding how temporal convolution layers integrate signals over time.
4. **Why Saliency Mapping for a TCN**
  - Temporal Convolutional Networks (TCNs) process sequential data, learning temporal dependencies across timesteps.  
  - Saliency mapping complements this architecture because it:
    - Attributes influence to both time and feature dimensions.
    - Can be directly derived from gradients without modifying model architecture.
    - Works on any differentiable model (unlike attention weights, which require explicit attention mechanisms).  
  - For our refined TCN:
    - The model produces multiple output heads (`max_risk`, `median_risk`, `pct_time_high`), each capturing different aspects of clinical deterioration.  
    - Saliency allows per-head interpretability, understanding which features drive each type of prediction.
5. **Typical Applications**
  | Domain | Model Type | Saliency Purpose |
  |---------|-------------|------------------|
  | **Computer Vision** | CNNs | Identify important pixels in an image. |
  | **Natural Language Processing** | RNNs / Transformers | Attribute importance to input tokens or words. |
  | **Time-Series / Clinical AI** | RNNs, LSTMs, TCNs | Identify critical physiological variables and time windows. |
  | **Reinforcement Learning** | Policy Networks | Reveal which states or features drive action decisions. |
6. **Common Saliency Outputs**
  | Output Type | Description | Interpretability |
  |--------------|-------------|------------------|
  | **Per-sample saliency map** | Feature importance matrix `(T × F)` for one patient or example. | Fine-grained, individual-level. |
  | **Feature-level mean CSV** | Average saliency per feature across patients and timesteps. | Global feature importance ranking. |
  | **Temporal profile CSV** | Mean saliency over features per timestep. | When the model is most sensitive. |
  | **Top-feature temporal CSV** | Saliency curves for top features through time. | Explains evolving patterns. |
  | **Mean heatmap PNG** | Visual summary of top 10 features over time (log-scaled). | Human-readable visualisation. |
7. **Alternative Interpretability Methods**
  | Method | Description | Pros | Cons | Reason for Exclusion |
  |---------|-------------|------|------|----------------------|
  | **Integrated Gradients** | Accumulates gradients along a path from baseline to input. | Reduces noise, better theoretical grounding. | Requires multiple forward passes per input, computationally heavy for long sequences. | Too computationally expensive for 96×171 input tensors. |
  | **DeepLIFT** | Tracks contribution of each neuron relative to reference activation. | Handles sign and saturation better than raw gradients. | Needs a reference baseline, less intuitive for continuous physiological data. | Baseline definition unclear for clinical signals. |
  | **Grad-CAM** | Localises regions of interest in convolutional maps. | Popular in vision tasks. | Not suitable for 1D temporal data or multi-head outputs. | Designed for image feature maps, not sequential data. |
  | **SHAP / KernelSHAP** | Model-agnostic, feature-level attribution. | Works for any model, provides global feature importance. | Ignores temporal structure, computationally expensive. | Already implemented for LightGBM, not suitable for per-timestep interpretation. |
  | **Attention Weights** | Use built-in attention mechanisms to infer feature importance. | Direct interpretability when model includes attention. | Not a true gradient-based sensitivity measure; depends on model design. | TCN has no attention layer; unsuitable. |
8. **Considerations and Design Choices**  
  - **(a) Gradient × Input over Plain Gradient:**  
    - A plain gradient (∂y/∂x) only measures how much the output changes if an input feature changes slightly → shows sensitivity only, not importance, thus can highlight inactive features.
    - But a feature can have a high gradient even when its actual value is near zero (i.e., it’s sensitive but not active in this patient).
	  - By multiplying the gradient with the input value (x), we weight each feature’s sensitivity by how much that feature was actually contributing at that moment.
    - This produces Gradient × Input, which reflects both how strongly and how actively each feature influenced the model’s prediction.
  - **(b) Absolute Values:**  
    - Removes directionality (positive or negative influence).  
    - Focuses on strength of effect rather than its polarity → appropriate for risk prediction tasks.
  - **(c) Log Transform in Heatmaps:**  
    - Raw saliency magnitudes can span several orders of magnitude.  
    - Applying `log(1+x)` enhances contrast among low-magnitude values without overemphasising outliers.
  - **(d) Percentile-Based Color Scaling (5th–95th):**  
    - Excludes extreme saliency values that distort colormap interpretation.  
    - Allows balanced visual representation across moderate intensity ranges.
  - **(e) Perceptually Uniform Colormap (`plasma`):**  
    - Maintains consistent perceptual brightness differences across intensity levels.  
    - Ensures that color changes correspond linearly to saliency magnitude → vital for interpretability by clinicians.
  - **(f) No Per-Patient PNGs:**  
    - Individual 171×96 matrices are not visually interpretable.  
    - Aggregated results (mean, top features) offer better readability and reproducibility.
9. **Strengths and Limitations**
  | Strengths | Limitations |
  |------------|-------------|
  | Simple, fast, and directly interpretable. | Sensitive to model noise; gradients can fluctuate sharply. |
  | Works for any differentiable model, including TCNs. | Only captures first-order effects (no feature interactions). |
  | Produces temporally and feature-resolved attributions. | Can produce sparse (near 0) or unstable maps for highly nonlinear models (which is why averaging gradients is needed). |
  | Complements global methods like SHAP by adding temporal dimension. | Not causal; high saliency ≠ causative importance. |
10. **Rationale for Our Final Implementation**
- We selected **gradient × input saliency mapping** for the TCN interpretability module because it provides:
  - Temporal and feature-level transparency for clinicians.  
  - Computational efficiency (single backward pass per patient).  
  - Compatibility with multi-head architectures.  
  - Reproducibility and simplicity without parameter tuning.  
- Enhancements like integrated gradients or DeepLIFT were excluded due to computational cost, unclear baselines, and minimal added interpretive value for our structured temporal dataset.
- Our final design choices → log-scaling, percentile normalization, and top-feature aggregation → ensure that outputs are stable, interpretable, and visually meaningful without sacrificing mathematical rigor. 

---
### Chosen Saliency Approach: Gradient × Input Saliency
**Overview**
- The **Gradient × Input (Grad×Input)** method was selected as the interpretability technique for the TCN due to its balance between conceptual clarity, computational efficiency, and theoretical consistency with SHAP-based feature attribution used in Step 3.
**Rationale for Selection**
1. **Direct and Simple**
  - Grad×Input directly computes how sensitive the model’s output is to each input feature by using standard backpropagation.
  - It requires no architectural modification or surrogate model, only access to gradients and input tensors.
  - Each saliency value represents the instantaneous influence of a feature at a given timestep on the final prediction.
2. **Model-Agnostic for Differentiable Networks**
  - Works with any neural network that supports automatic differentiation (e.g., TCNs, RNNs, MLPs).
  - Unlike methods such as Grad-CAM or LRP, it does not depend on a particular model layer or structure.
  - Suitable for 1D temporal models where spatial activation maps (e.g., in CNNs) are not relevant.
3. **Computationally Feasible**
  - Efficient to run → only one backward pass per prediction head is needed.
  - Avoids the high computational cost of integrated gradients or occlusion-based perturbation tests.
  - Ideal for large temporal datasets with long sequences and many features.
4. **Theoretical Alignment with SHAP**
  - Both SHAP and Grad×Input seek to attribute portions of a model’s prediction to input features.
    - SHAP uses **game theory** (Shapley values) to estimate each feature’s contribution to the final prediction.
    - Grad×Input uses **calculus** (gradients) to measure local sensitivity of the prediction to input changes.
  - This conceptual similarity allows both methods to be used in unison:
    - **SHAP** → Explains what features matter most (global, static view).
    - **Grad×Input** → Explains when and how those features matter (local, temporal view).
**Why This Approach Fits the TCN**
- The TCN processes time-dependent physiological signals; Grad×Input naturally extends to temporal attribution by computing sensitivity and activity across all timesteps.
- It captures temporal patterns of importance without adding interpretive complexity or retraining overhead.
- The resulting saliency maps provide an intuitive and clinically interpretable view of when specific features most strongly influenced the model’s risk predictions.
**Summary**
- Grad×Input was chosen because it is:
  - **Direct:** uses standard gradients, no special model hooks required.  
  - **General:** applicable to any differentiable neural network.  
  - **Efficient:** computationally lightweight for large sequential data.  
  - **Aligned:** conceptually consistent with SHAP’s feature attribution, extending it into the temporal domain for deep models.

---

### Step 3 vs Step 4 Interpretability Comparison — SHAP (Static) vs Saliency (Temporal)
**Purpose**
- Two complementary interpretability methods were used:
  - **SHAP (Step 3)** for the LightGBM model.  
  - **Saliency Mapping (Step 4)** for the Temporal Convolutional Network (TCN).
- Each method targets a distinct level of reasoning:
  - LightGBM → “What features matter most overall?”
  - TCN → “When and how do those features over time influence predictions?”
- Combined, they link **static feature contribution** with **temporal sequence sensitivity**.

**Conceptual Framework**
| Model | Data Representation | Interpretability Method | Answers | Output Type |
|--------|--------------------|-------------------------|----------|-------------|
| **LightGBM** | Aggregated patient-level and timestamp-summary features (e.g. mean, max, slope) | **SHAP (TreeExplainer)** | Which features drive risk overall | Mean feature importance values, summary plots |
| **TCN** | Raw temporal sequences of physiological signals (96 timesteps × 171 features) | **Saliency Mapping (|grad × input|)** | When and how features affect risk over time | Temporal saliency maps, feature–time profiles |

**Analytical Goals**
- **Feature importance context:** Compare static SHAP importance with temporal saliency patterns.  
- **Temporal attribution:** Identify which time windows most strongly influence predicted risk.  
- **Feature dynamics:** Observe how physiological trends (e.g. ↑ HR, ↓ SpO₂) alter risk estimates.  
- **Cross-model validation:** Test whether LightGBM’s key features remain salient in the TCN over time.

**Why Separate Methods Were Needed**

| Aspect | SHAP (LightGBM) | Saliency (TCN) |
|--------|------------------|----------------|
| Model architecture | Tree-based, non-differentiable | Neural network, differentiable |
| Interpretability basis | Shapley values (coalitional feature attribution) | Gradients (local sensitivity of output to inputs) |
| Type of insight | Global feature-level importance | Local and temporal, timestep-level importance |
| Input representation | Static aggregates and summary statistics | Continuous multivariate physiological time-series |
| Outputs | CSVs, bar plots, summary figures | Temporal CSVs, heatmaps, per-feature time trends |

**How They Complement Each Other**
- **SHAP:** Quantifies what variables most influence overall deterioration risk.  
- **Saliency:** Reveals when those variables exert the strongest effect within a patient’s timeline.  
- The two methods form a joint interpretability framework:  
  - Feature dimension → importance ranking across variables.  
  - Temporal dimension → importance evolution across time. 

**Methodological Integration — How Step 3 and Step 4 Complete Phase 6**
1. **Step 3 (SHAP Analysis):**
  - Applied to the LightGBM baseline trained on static summary features.
  - Provided global feature importance rankings and clinical interpretability at the aggregated level.
2. **Step 4 (Saliency Mapping):**
  - Applied to the refined TCN model trained on full temporal sequences.
  - Added temporal localisation of model attention.
3. **Integration Across Steps**
- Both analyses draw from the same clinical variable space (e.g., HR, RR, SpO₂, BP) but use different input representations:
  - **Step 3 (SHAP – LightGBM):** operates on static aggregate features (e.g., means, slopes) → matrix of shape `(n_patients, n_features)` → `(15, 40)`.
  - **Step 4 (Saliency – TCN):** operates on raw temporal sequences of vital signs and lab trends → tensor `(n_patients, timesteps, features)` → `(15, 96, 171)`.
- The difference in representation reflects the distinct design goals:
  - **LightGBM/SHAP:** interpret overall feature relevance across aggregated static summaries.  
  - **TCN/Saliency:** interpret temporal sensitivity → how the model reacts to changes in features over time.
- Therefore, Phase 6 interpretability is complete only when both static (Step 3) and dynamic (Step 4) insights are jointly interpreted:
  - **Validation:** do TCN’s temporal dependencies emphasise the same clinically important variables that SHAP highlighted?
  - **Interpretation:** do saliency peaks appear at clinically plausible times relative to static importance patterns?

**Comparative Analysis Strategy**
- Compare **mean feature rankings** (SHAP) with **mean saliency magnitudes** (TCN).
  - Note: This comparison is conceptual, not one-to-one
- Examine whether features with high static SHAP importance (e.g., SpO₂, RR) also show high average temporal saliency.  
- Inspect **temporal saliency heatmaps** for known deterioration trajectories (e.g., SpO₂ drop preceding RR rise).  
- Evaluate **alignment vs divergence**:
  - Alignment → validates TCN attention as clinically meaningful.  
  - Divergence → suggests temporal or interaction effects missed by LightGBM.

**Alternative Interpretability Methods Considered**
| Method | Description | Rationale for Exclusion |
|--------|--------------|--------------------------|
| **Integrated Gradients** | Path-integrated version of Grad×Input, smooths noise | Added computational cost with minimal interpretive gain for our dataset size |
| **Temporal Integrated Gradients** | Time-aware variant integrating gradients along the temporal path | Computationally expensive for long sequences and offers marginal improvement over Grad×Input in stability for this TCN |
| **Grad-CAM** | Visual localisation via gradient-weighted activations | Designed for CNN image models, less suitable for 1D TCN sequences |
| **Layer-wise Relevance Propagation (LRP)** | Backpropagation-based attribution | Requires architecture modification and custom backward hooks |
| **Occlusion / Perturbation tests** | Measure prediction change when masking inputs | Computationally expensive for long sequences |
| **DeepSHAP** | Hybrid SHAP + DeepLIFT | Overkill for this lightweight TCN and hard to calibrate on temporal inputs |

- **Chosen approach:** Grad×Input saliency was the most direct, model-agnostic, and computationally feasible for explaining a 1D temporal model while aligning conceptually with SHAP’s feature-level reasoning. 

**Strengths and Limitations**
- **Strengths**
  - Compatible with any differentiable neural network (e.g., TCN, RNN).  
  - Simple to compute, aligns with gradient-based interpretability theory.  
  - Produces temporal maps showing “when” the model attends to key signals.  
  - Complements SHAP’s static feature importance with dynamic insight.  
- **Limitations**
  - Sensitive to model nonlinearity and gradient noise → can produce locally unstable saliency maps where small input perturbations lead to sharp gradient variations; mitigated by averaging or smoothing. 
  - Lacks absolute interpretability → highlights relative importance, not causal effect.  
  - Requires log scaling and percentile clipping for visual stability.  
  - Cannot directly handle non-differentiable models (why SHAP used for LightGBM).

**Summary**
- **SHAP (Step 3)** → Explains what matters most overall (static feature contribution).  
- **Saliency (Step 4)** → Explains when and how it matters (temporal reasoning).  
- Both are essential components of Phase 6: Model Interpretability.  
- Together, they provide a complete multi-dimensional interpretability suite:
  1. **Feature-level attribution** (SHAP)  
  2. **Temporal attribution** (Saliency)  
  3. **Cross-model comparison** confirming that deep temporal patterns align with static clinical reasoning.
- This unified analysis enables clinicians to see not only what variables drive deterioration risk but also when those signals become critical → a full temporal-feature interpretability framework.

### Saliency Quantitative Analysis - Background
#### Overview
**Purpose**
- This section focuses on explaining how to interpret the four quantitative and visual outputs generated for each target head (`max_risk`, `median_risk`, `pct_time_high`).
- Each output provides a distinct perspective on how and when the TCN model attributes importance to features across time.
- The goal is to extract clinically meaningful insights from saliency results and understand the temporal reasoning of the network.
**Analysis Components**
1. **Feature-Level Interpretation (*_feature_saliency.csv)**
	- Quantifies overall feature importance by averaging saliency across all patients and timesteps.
	-	Answers: “Which physiological variables most influence the TCN’s predictions overall?”
2. **Temporal Sensitivity (*_temporal_saliency.csv)**
	-	Shows how model sensitivity varies over the sequence.
	-	Answers: “At which time periods does the model respond most strongly to inputs?”
3. **Top Feature Temporal Profiles (*_top_features_temporal.csv)**
	-	Tracks the evolution of the top 5 features’ saliency over time.
	-	Answers: “When do specific key features become influential?”
4. **Visualisation — Mean Heatmap (*_mean_heatmap.png)**
	-	Displays saliency intensity for the top 10 features across all timesteps (log-scaled).
	-	Answers: “How do the most important features interact and vary over time?”
#### Interpreting Saliency Statistics
1. **Feature-Level Mean & Standard Deviation (`*_feature_saliency.csv`)**
	-	Purpose: Quantifies overall feature importance across all patients and timesteps.
	- Data Structure:
    | Column | Description |
    |--------|-------------|
    | `feature` | Name of the input feature |
    | `mean_abs_saliency` | Average magnitude of absolute saliency across all patients and timesteps → “how strongly the model relies on this feature overall” |
    | `std_abs_saliency` | Standard deviation of absolute saliency → “how consistently that feature mattered across patients and timesteps” |
	-	Interpretation:
    - `Relative variability = std / mean` → how unstable a feature's importance is compared to its average contribution.
      - Ratios ≈ 1–1.5 → consistent importance.
      - Ratios ≈ 2–3 → moderate variation across samples.
      - Ratios > 3–4 → highly inconsistent across patients (context-dependent saliency).
    - 
    | **Mean** | **Std** | **Interpretation** |
    |-----------|----------|--------------------|
    | High | Low | Strong, stable driver → universally important feature |
    | High | High | Strong but variable driver → important only in certain subgroups or periods |
    | Low | Low | Weak, consistent baseline feature → minimal contribution |
    | Low | High | Noisy, inconsistent influence → minimal contribution |
 
2. **Temporal Mean Profile (`*_temporal_saliency.csv`)**
  - Purpose: Shows when in the patient timeline the model is most sensitive to its inputs by averaging absolute saliency values across all features and patients at each timestep.
  - Data Structure:
    | Column | Description |
    |---------|--------------|
    | `timestep` | Sequential time index (0 → max sequence length - 1) |
    | `mean_abs_saliency` | Mean absolute saliency across all features and patients at that timestep |
  - Interpretation:
    | Feature / Trend | Meaning |
    |-----------------|---------|
    | Individual value | Each row represents overall sensitivity at that timestep. Higher `mean_abs_saliency` → model relies more heavily; lower → less attention at that timestamp. |
    | Peaks | Moments where model attention or predictive sensitivity increases; may correspond to clinically meaningful changes such as deterioration onset. |
    | Stable / flat sections | Model treats timesteps roughly equally → limited temporal dependence. |
    | Rising toward later timesteps | Model focuses on recent observations → short-term dependencies dominate. |
    | Early peaks | High saliency early → model attention to initial conditions or early warning signals. |


3. **Top-Feature Temporal Profiles (`*_top_features_temporal.csv`)**
	-	Purpose: Tracks how saliency for the top 5 features evolves over time.
  - Data Structure:
    | Column | Description |
    |---------|--------------|
    | `timestep` | Sequential time index (0 → max sequence length - 1) |
    | `<feature>` | 5 feature columns; mean absolute saliency across all patients at that timestep. Note: The feature columns differ for each model output (`max_risk`, `median_risk`, `pct_time_high`) since top features are selected independently per target. |
  - Interpretation:
    | Feature / Trend | Meaning |
    |-----------------|---------|
    | Individual value | Each row represents overall sensitivity for a feature at that timestep. Higher `mean_abs_saliency` → model relies more heavily; lower → less attention at that timestamp. Absolute magnitude is less important than relative ranking between features. |
    | Peaks | Timesteps when a feature strongly influences the prediction |
    | Stable / flat sections | Feature contributes consistently at similar levels across the timeline. |
    | Shifting peaks | I ndicate the model attends to different features at different stages of a patient’s sequence.
  - Comparison between features:
    -	Allows you to see which features dominate at each point in time.
    -	Helps clinicians identify key signals driving model predictions and their temporal order.
 
4. **Mean Heatmap (*_mean_heatmap.png)**
	- Purpose: 
    - Provides a visual overview of how model attention (saliency) varies across the top 10 most influential features and timesteps.
    - Summarises the information from the temporal and feature-level saliency CSVs into a 2D representation.
  - Data Structure:
    | Axis / Element | Description |
    |----------------|--------------|
    | **x-axis** | Timesteps (0–96) representing the patient sequence over time |
    | **y-axis** | Top 10 features ranked by mean saliency |
    | **Color intensity** | Log-scaled mean saliency magnitude (brighter = higher importance) |
    | **Each cell** | Mean saliency value for a specific feature–timestep pair |
- Interpretation:
  - **Bright horizontal bands** → features that consistently dominate across most timesteps (stable, globally important signals).  
  - **Bright vertical bands** → short time windows where multiple features simultaneously become important (periods of model attention, e.g., deterioration onset).  
  - **Scattered or patchy regions** → transient, context-specific importance that may vary between patients.  
- Analytical Use:
  - Qualitative validation of the quantitative CSV findings.  
  - Identifies alignment between stable drivers (from feature-level saliency) and temporally focused attention (from temporal profiles).  
  - Supports clinical interpretation by revealing when and which features jointly drive high-risk predictions.

#### Saliency Output Summary
| **Output** | **Focus** | **Granularity** | **Used For** |
|-------------|------------|------------------|----------------|
| `*_feature_saliency.csv` | Overall feature importance | Feature-level (global) | Ranking and variability across features |
| `*_temporal_saliency.csv` | Global model sensitivity over time | Time-level (global) | Identifying periods of peak model attention |
| `*_top_features_temporal.csv` | Evolution of top features over time | Time–feature (focused) | Understanding dynamics of key predictors |
| `*_mean_heatmap.png` | Combined saliency visualisation | Time–feature (visual) | Validating clinical and temporal patterns |

---

### Full Saliency Quantitative Analysis 
#### Overview
**Objective**
- To systematically interpret the TCN’s behaviour across four complementary outputs across three different targets (`max_risk`, `median_risk`, `pct_time_high`).
- To identify consistent feature importance patterns, temporal dependencies, and clinically plausible risk trajectories.
- These analyses together form the quantitative interpretability core of Step 4, providing temporal explainability to complement SHAP’s static feature-level insights.
**Analysis Components per Target**
1. **Feature-Level Interpretation (*_feature_saliency.csv)**: Quantifies overall feature importance by averaging saliency across all patients and timesteps.
2. **Temporal Sensitivity (*_temporal_saliency.csv)**: Shows how model sensitivity varies over the sequence.
3. **Top Feature Temporal Profiles (*_top_features_temporal.csv)** Tracks the evolution of the top 5 features’ saliency over time.
4. **Visualisation — Mean Heatmap (*_mean_heatmap.png)** Displays saliency intensity for the top 10 features across all timesteps (log-scaled).
**Scope of Analysis**
- **Goal:** extract interpretable clinical or temporal trends, not micro-level numeric commentary.
- For 12 outputs, the analysis should stay at the trend and pattern level, not per-timestep detail:
  - **Feature-level CSVs:** summarise top 5–10 features by mean and variability; highlight broad stability or volatility patterns.
  - **Temporal saliency:** describe general regions of high vs low attention (early, middle, late sequence).
  - **Top feature temporal CSVs:** note recurring peaks or synchronized trends across key variables.
  - **Mean heatmaps:** interpret overall structure (broad horizontal/vertical patterns) rather than pixel-level variation.
- Only expand when a pattern directly supports or contradicts earlier SHAP findings.

#### Saliency Analysis (`max_risk`)
**Feature-Level Mean & Standard Deviation (`max_risk_feature_saliency.csv`)**
1. **Context**
  - This analysis corresponds to the maximum patient-level deterioration risk predicted during admission.  
  - It captures which features most strongly and consistently influence the model when estimating each patient’s peak risk episode.
2. **Key Findings (General-Trend Scope)**
  | **Rank** | **Feature** | **Mean** | **Std** | **Interpretation** |
  |-----------|--------------|----------|----------|--------------------|
  | 1 | `heart_rate_roll24h_min` | 7.69e-05 | 1.87e-04 | Highest mean with high variance → strong but inconsistent influence. Model relies heavily on prolonged heart rate suppression (possible bradycardic or hypodynamic states) in some patients but not all, indicating subgroup-dependent importance. |
  | 2 | `news2_score` | 5.01e-05 | 1.15e-04 | Moderate mean, moderately variable → reliable global marker capturing multi-parameter deterioration risk; model consistently uses it but not as dominant as specific vitals. |
  | 3 | `temperature_max` | 4.77e-05 | 9.36e-05 | Mid–high mean, moderate variance → model identifies episodic temperature spikes (fever responses) as moderately influential; variability suggests influence only in febrile cases. |
  | 4 | `level_of_consciousness_carried` | 4.61e-05 | 9.17e-05 | Moderate mean, moderate variance → carried-forward consciousness values preserve deterioration context; consistently important where altered mental state persists, less so otherwise. |
  | 5 | `respiratory_rate_roll4h_min` | 4.37e-05 | 1.11e-04 | Moderate mean with high variance → model detects respiratory instability patterns (acute dips or fatigue) variably across patients, aligning with short-term deterioration episodes. |
3. **Interpretation Summary**
  - **Overall summary:** Mean quantifies global importance; Std reflects stability. Here, heart rate and respiratory patterns show high mean + high Std (episodic importance), while NEWS2 and temperature are moderate mean + lower Std (steady baseline predictors).
  - **Dominant predictors:** Rolling-window minima of **heart rate** and **respiratory rate** indicate the model prioritises **sustained physiological depression** rather than transient abnormalities when estimating maximum deterioration risk.  
  - **Summary indicators:** Variables such as **NEWS2 score** and **risk_numeric** act as clinically validated proxies, confirming internal consistency between learned and rule-based signals.  
  - **Moderate variability (std):** Most top predictors have **mid–high standard deviations**, showing that while generally influential, their impact fluctuates across patient trajectories.  
  - **Low-mean features:** Inputs below roughly `2×10⁻⁵` contribute marginally, likely encoding contextual or redundancy signals (e.g., time gaps, missingness, auxiliary stats).  
  - **Zero-saliency variables:** Static or unused inputs (e.g., CO₂ retainer fields) indicate non-representation in this risk regime, consistent with limited relevance to acute deterioration.
4. **Overall Summary**
	-	The `max-risk` output is primarily influenced by minimum values of key vital signs (e.g., heart rate, respiratory rate) and aggregate early-warning scores (e.g., NEWS2).
	-	These features capture sustained physiological deterioration (rolling time windows) rather than transient abnormalities, indicating that the model focuses on periods where a patient’s condition is persistently abnormal.
	-	Clinically, this makes sense: longer-lasting deviations from normal physiology are more predictive of a patient reaching their peak risk than short-term fluctuations.
	-	The variability (high standard deviation) in saliency across patients and timestamps highlights episodic or context-specific importance (importance is not uniform), reflecting that different patients reach peak risk through different combinations of physiological changes.

**Temporal Mean Saliency (`max_risk_temporal_saliency.csv`)**
1. **Context**
  - This represents the **average absolute saliency** per timestep, aggregated across all features and patients.  
  - It identifies **when in the sequence** the model is most sensitive when predicting **maximum deterioration risk**, i.e., which parts of the patient timeline contribute most to peak-risk estimation.
2. **Key Findings (General-Trend Scope)**
  | **Pattern Region** | **Approx. Timesteps** | **Trend** | **Interpretation** |
  |--------------------|------------------------|------------|--------------------|
  | Early sequence | 0–10 | Rapid decline from ~3.8×10⁻⁵ to ~1.2×10⁻⁵ | Model shows minimal reliance on earliest observations, suggesting low predictive relevance of baseline vitals. |
  | Mid sequence | 10–40 | Stable low plateau (~1.5–1.6×10⁻⁵) | Indicates that mid-trajectory states provide steady but limited incremental information for determining max risk. |
  | Late sequence | 40–70 | Gradual rise to ~2.2×10⁻⁵ | Reflects increasing model sensitivity to recent physiological patterns as deterioration approaches. |
  | End of sequence | 70–95 | Fluctuating peaks then decline (max ≈2.3×10⁻⁵ → 0.6×10⁻⁵) | The saliency spike before final decline suggests model focus on **late-stage instability**, followed by tapering when inputs become less informative. |
3. **Interpretation Summary**
  - **Temporal focus:** The model is **most attentive between timesteps ~55–75**, aligning with periods that likely correspond to **late deterioration onset** in patient sequences.  
  - **Early low saliency:** Minimal early saliency implies that initial stable conditions carry little weight when estimating maximum risk → consistent with deterioration being a dynamic rather than baseline phenomenon.  
  - **Late rise and fall:** The mid-to-late escalation indicates that **progressive physiological stress** drives peak-risk prediction, with declining saliency near the end possibly due to reduced input signal (e.g., short remaining sequences).  
  - **Interpretive pattern:** The smooth progression (rather than abrupt peaks) suggests the model captures **gradual worsening** rather than isolated episodic spikes.
4. **Overall Summary**
	-	The model’s attention increases toward the end of each patient’s sequence, showing a clear recency bias — recent physiological changes have the greatest effect on predicting maximum risk.
	-	Clinically, this means the model recognises that peak deterioration is usually preceded by sustained worsening near the end of a patient’s trajectory, not by early or isolated abnormalities.
	-	The late rise in saliency indicates effective detection of emerging instability patterns, consistent with identifying moments of highest clinical risk.
   
**Top-Feature Temporal Profiles (`max_risk_top_features_temporal.csv`)**
1. **Context**
  - Tracks how the saliency of the **top 5 features** changes across time (`timestep` = 0–95).  
  - Each column represents the mean absolute saliency for a feature at a specific timestep, averaged across all patients.  
  - It reveals when each physiological signal most influences the model’s estimation of a patient’s **maximum deterioration risk**.
2. **Key Findings (General-Trend Scope)**
   | **Feature** | **Temporal Pattern** | **Interpretation** |
   |--------------|----------------------|--------------------|
   | `heart_rate_roll24h_min` | Moderate baseline, sharp rise after timestep ~55, peaking between 60–75 | Late-sequence dominance reflects sensitivity to sustained low heart rate before deterioration; the model identifies cardiovascular depression near the deterioration point. |
   | `news2_score` | Steady influence throughout, mild increase mid-to-late (40–70) | Reflects continuous integration of multi-parameter risk signals; maintains stable contextual weight. |
   | `temperature_max` | Mild early activity, then stable midsection with occasional late bumps | Suggests episodic temperature relevance → important in subsets with febrile response but not universal. |
   | `level_of_consciousness_carried` | Fluctuating mid-to-late sequence (40–75) | Indicates model focus on persistent altered consciousness during evolving deterioration episodes, not transient episodes. |
   | `respiratory_rate_roll4h_min` | Low baseline, rising steeply from timestep ~65 onward | Signals mounting respiratory instability or fatigue close to deterioration onset; strong late contribution. |
3. **Interpretation Summary**
  - **Overall dynamics:** Most top features show **increasing saliency toward later timesteps**, confirming that the model places more weight on recent physiological changes when estimating maximum risk.  
  - **Temporal differentiation:**  
    - Heart rate and respiratory rate minima show clear late peaks, implying attention to **sustained declines** rather than short-term spikes.  
    - NEWS2 provides a **steady baseline signal**, anchoring the model’s interpretation across the timeline.  
    - Consciousness and temperature show **intermittent importance**, supporting their role as conditional or secondary cues rather than continuous drivers.  
  - **Pattern interpretation:** Saliency peaks cluster in the **final third (timesteps 60–80)**, coinciding with typical pre-deterioration phases.
4. **Overall Summary**
  - The TCN’s temporal saliency pattern for `max_risk` shows a **progressive rise in importance across time**, culminating shortly before the end of the sequence.  
  - This indicates that the model recognises **accumulating instability** across vital signs and integrates multi-system deterioration cues as the patient approaches their highest predicted risk.  
  - Clinically, this mirrors how maximum deterioration tends to emerge from **gradual physiological decline**, where sustained abnormalities in **heart rate, respiratory effort, and consciousness** signal worsening condition more reliably than isolated or early anomalies.

**Mean Saliency Heatmap (`max_risk_mean_heatmap.png`)**
1. **Context**
  - Visualises **temporal top-10 feature importance** for the model predicting **maximum patient deterioration risk** during admission.  
  - Color intensity (log-scaled mean absolute saliency) reflects **average model sensitivity** to each feature at each timestep across all patients.  
  - Bright (yellow) regions = high influence; darker (blue) = less importance.  
  - Captures where and when the model focuses most strongly across the admission timeline.
2. **Key Patterns**
  - **Overall Temporal Pattern:**
    - Saliency begins noticabley rising from **~40 hours onward**, indicating the model starts detecting mid-stay deterioration patterns.  
    - Attention **builds progressively**, peaking around **55-85 hours**, followed by a moderate plateau to 90–96 hours with almost no activation in the last few hours.
    - Early hours (0–10 h) show minimal activation, suggesting low predictive relevance of admission values alone. However in a few top features, admission values were bright. 
  | **Pattern** | **Description** | **Interpretation** |
  |--------------|----------------|--------------------|
  | **Late brightening (≈55–90hrs)** | Broad increase in brightness across most features in the final 30 hours. | Indicates **recency bias** → the model relies most on **recent physiological signals** when estimating maximum risk. |
  | **Singular persistent band** | `heart_rate_roll24h_min` is the brightest and most persistent feature; sustained saliency from ~15 h onward, especially intense around 15-45 and 50-80 h | Indicates prolonged low or unstable heart rate is a major risk signal that is used throughout the entirety of stay. |
  | **Sustained horizontal bands** | Scattered bright regions seen in `news2_score`, and `respiratory_rate` from roughly **40–96 hrs**. | Reflects **persistently important predictors**, capturing **sustained physiological decline** rather than isolated events. |
  | **Moderate horizontal presence** | `temperature_max`,`level_of_consciousness_carried` and `supplemental_o2_max` show moderate but steady activation throughout, `respiratory_rate_max` show moderate steady activation between 60-90hrs. | Suggests that **temperature spikes**, **prolonged altered consciousness**, **O2 requirments** contribute moderately meaningfully throughout, with tachypnoea contributing later on. |
  | **Bright patches** | Features such as `respiratory_rate_roll4h_min` (40-45, 70-80h), `systolic_bp_roll1h_max` (55-75h) and `respiratory_rate` (55-65, 75-85h) show bright sections between **40-80 hrs**. | Reflects **event-specific activation**, likely transient interventions or brief physiological responses. |
3. **Interpretation Summary**
  - The model’s focus increases gradually over time, peaking around **55-85 hrs**, showing that it relies more on **recent trends** to predict a patient’s maximum deterioration risk.  
  - **Rolling heart rate trends (`heart_rate_roll24h_min`)** dominate throughout as the most important feature in determining maximum risk.
  - **Respiratory metrics (`respiratory_rate`, `respiratory_rate_roll4h_min`)**, and **NEWS2 score** dominate across the later stages, suggesting that **prolonged abnormalities** in these systems are central to identifying deterioration.  
  - Features with isolated saliency bursts contribute briefly during likely **acute escalation points**, but they are secondary to sustained vital trends.
4. **Overall Summary**
  - The heatmap demonstrates a **clear late-stage concentration of saliency**, where most top features show maximal importance between **55-85 hrs**.  
  - This indicates that the model captures **progressive deterioration trajectories**, relying on **persistent physiological changes** rather than transient fluctuations.  
  - Clinically, this aligns with the pattern of patients gradually deteriorating toward their highest risk period, rather than risk being driven by early isolated abnormalities.  
  - Core signals (heart rate, respiratory rate, temperature, and consciousness level) remain key determinants of maximum risk across time, confirming model consistency with known clinical indicators.

**Max-Risk Overall Saliency Summary**
- **Primary Drivers:** 
  - Across all analyses, the model consistently prioritises **rolling heart rate minima (`heart_rate_roll24h_min`)**, **respiratory rate metrics (`respiratory_rate`, `respiratory_rate_roll4h_min`)**, and **NEWS2 score**. 
  - These features dominate both feature-level and temporal importance.
- **Temporal Focus:** 
  - Saliency rises gradually from **~40 hours**, peaks **55–85 hours**, and tapers slightly toward the end of the sequence.  
  - Indicates **recency bias**: the model relies more on recent physiological changes when estimating a patient’s maximum risk.
- **Sustained vs Episodic Signals:**  
  - `heart_rate_roll24h_min` shows **persistent high importance** throughout early-to-late sequence, reflecting cumulative cardiovascular suppression.  
  - Respiratory metrics and NEWS2 show **moderate-to-high sustained influence**, occasionally punctuated by **short-lived spikes** (likely representing acute deterioration episodes).  
  - Secondary features (temperature, consciousness, supplemental O₂) contribute steadily but less strongly.
- **Interpretation of Variability:**  
  - Feature-level **high standard deviation** indicates that some features’ influence varies across patients or over time; the model does not treat all patients identically.  
  - Temporal patterns confirm that maximum risk is driven by **progressive physiological deterioration** rather than isolated early anomalies.
- **Clinical Alignment:**  
  - Core signals and their temporal evolution align with clinical expectations: prolonged deviations in heart rate, respiration, and composite early warning scores precede peak risk events.  
  - Model behaviour captures both persistent underlying decline and event-specific surges, consistent with real-world deterioration trajectories.

#### Saliency Analysis (`median_risk`)
**Feature-Level Mean & Standard Deviation (`median_risk_feature_saliency.csv`)**
1. **Context**
  - This analysis corresponds to the model predicting **median (average sustained) deterioration risk** during admission.  
  - It quantifies which features the model consistently relied upon across all patients and timestamps to estimate **ongoing baseline risk** rather than transient deterioration events.  
  - Mean saliency indicates **overall feature importance**, while standard deviation (std) reflects **stability or variability** of that importance across patients and timesteps.
2. **Key Findings (General-Trend Scope)**
  | **Rank** | **Feature** | **Mean** | **Std** | **Interpretation** |
  |-----------|--------------|----------|----------|--------------------|
  | 1 | `heart_rate_roll24h_min` | 8.67×10⁻⁵ | 2.43×10⁻⁴ | Most influential and variable feature → model relies heavily on sustained low heart rate over 24 h as a stable indicator of chronic physiological stress. High variance shows it’s dominant in some patients but not all. |
  | 2 | `news2_score` | 8.17×10⁻⁵ | 1.93×10⁻⁴ | Consistently strong signal across patients → global deterioration metric integrated continuously into risk estimation. Moderate variability reflects consistent baseline importance. |
  | 3 | `heart_rate_missing_pct` | 7.45×10⁻⁵ | 1.58×10⁻⁴ | High mean and moderate variance → model interprets monitoring gaps as proxy indicators of patient instability or data sparsity linked to sustained risk. |
  | 4 | `risk_numeric` | 7.43×10⁻⁵ | 1.65×10⁻⁴ | Derived risk value measure acts as a contextual anchor; moderate mean + variance imply steady contribution without dominance. |
  | 5 | `heart_rate_roll24h_mean` | 7.17×10⁻⁵ | 1.71×10⁻⁴ | Average rolling heart rate maintains high importance → indicates that long-term cardiovascular measure (not extremes) defines typical sustained risk level. |
3. **Interpretation Summary**
  - **Dominant cardiovascular weighting:** The model’s most influential features are all heart rate–derived (rolling mean/min, missingness), showing reliance on chronic heart rate trends to infer sustained physiological stability.  
  - **Consistent integrators:** `news2_score` and `risk_numeric` provide continuous, context-aware risk input, aligning with the model’s goal of predicting average condition rather than episodic events.  
  - **Moderate variability across features:** The relatively high standard deviations imply that while the same key predictors are used, their relative strength varies between patients, reflecting personalised weighting of sustained physiology.  
  - **Low-importance features:** Most lower-ranked vitals and derived features (< 5×10⁻⁵ mean) contribute marginally, suggesting redundancy or minor contextual effects.
4. **Overall Summary**
  - The `median_risk` output is shaped primarily by **persistent physiological indicators** → especially long-term heart rate behaviour, continuous global scores (NEWS2), and data completeness metrics (missingness).  
  - This combination reflects a focus on **baseline physiological tone and monitoring continuity**, fitting the definition of **average sustained risk** across a stay.  
  - Clinically, this pattern suggests the model captures **chronic physiological burden** and **ongoing stability**, rather than acute or transient deterioration.  
  - Variability in feature influence indicates patient-specific pathways to sustained risk but consistent dependence on cardiovascular and systemic signals across the cohort.

**Temporal Mean Saliency (`median_risk_temporal_saliency.csv`)**
1. **Context**
  - Represents the **average absolute saliency per timestep**, aggregated across all features and patients, for predicting median (sustained) deterioration risk.  
  - This identifies when in the patient timeline the model is most sensitive to input signals when estimating the overall (typical) risk state rather than a peak event.  
2. **Key Findings (General-Trend Scope)**
  | **Pattern Region** | **Approx. Timesteps** | **Trend** | **Interpretation** |
  |--------------------|------------------------|------------|--------------------|
  | **Initial window** | 0–5 | High early peak (~7.5×10⁻⁵) followed by a rapid drop to ~2.0×10⁻⁵ | Early strong activation shows that **initial physiological presentation** heavily influences the model’s baseline understanding of sustained risk. |
  | **Early–mid sequence** | 5–15 | Flat low saliency (~2.0–2.5×10⁻⁵) | The model briefly de-emphasises inputs after the initial baseline, suggesting stability or limited new information. |
  | **Mid sequence** | 15–50 | Gradual, low-to-moderate plateau (~2.4–2.7×10⁻⁵) | Reflects steady model sensitivity to ongoing physiology; consistent monitoring but no major signal spikes. |
  | **Late sequence** | 55–80 | Broad, sustained elevation (peak ≈4.0×10⁻⁵ at ~timestep 62) | Indicates the **most influential phase**, where the model integrates long-term physiological signals to refine average risk estimates. |
  | **Final window** | 80–96 | Gradual decline from ~3.0×10⁻⁵ to <1.0×10⁻⁵ | Suggests diminishing relevance of final hours, possibly due to end-of-stay observations contributing little new information. |
3. **Interpretation Summary**
  - The model exhibits **two main saliency peaks** → a **brief early activation** reflecting the impact of admission and baseline status, and a **broader late activation** between ~55–80 hrs driven by sustained physiological context.  
  - The **midsection (15–50 hrs)** shows continuous moderate attention, supporting a pattern of **stable monitoring** rather than episodic focus.  
  - The **absence of sharp spikes** implies that the model interprets risk as a function of long-term trends, not sudden deviations.  
  - The **late broad peak** aligns with gradual refinement of predicted median risk as physiological data accumulate over time.
4. **Overall Summary**
  - Temporal saliency for `median_risk` shows a **bi-phasic pattern**: strong sensitivity at the start (baseline definition) and again during the later mid-to-end window (~55–80 hrs).  
  - The stable, continuous trend between these peaks confirms the model’s design to capture **sustained physiological consistency** rather than short-lived instability.  
  - Clinically, this means the model defines a patient’s **typical risk state** through both **initial condition** and **prolonged stability or decline**, not acute deterioration episodes.  
  - The declining saliency after 85 hrs indicates the model has already integrated key risk information earlier in the trajectory, consistent with prediction of a steady-state risk measure.

**Top-Feature Temporal Profiles (`median_risk_top_features_temporal.csv`)**
1. **Context**
  - Tracks how the saliency of the **top 5 features** changes across time (`timestep` = 0–95).  
  - Each column represents the **mean absolute saliency** for a feature at a specific timestep, averaged across all patients.  
  - This reveals when each feature most influences the model’s estimation of a patient’s median (sustained) deterioration risk across the hospital stay.
2. **Key Findings (General-Trend Scope)**
   | **Feature** | **Temporal Pattern** | **Interpretation** |
   |--------------|----------------------|--------------------|
   | `heart_rate_roll24h_min` | Very high at timestep 0 (~0.00028), drops sharply after ~1–5, then moderate steady baseline; gradual increase from ~50, peaks 55–65, declines after 80 | Initial low heart rate strongly predicts overall sustained risk; later sustained low HR contributes to typical risk profile. |
   | `news2_score` | Moderate initial saliency (~0.00021), mild oscillations mid-sequence; continuous moderate from ~10–85 | Acts as a **persistent global risk indicator**, reflecting cumulative multi-parameter deterioration signals. |
   | `heart_rate_missing_pct` | Moderate early (~0.00019), fluctuates throughout; peaks mid-late sequence (~55–65) | Highlights that **data gaps** or missing heart rate measurements contribute to sustained risk assessment. |
   | `risk_numeric` | Moderate initial (~0.00019), generally stable with small mid-to-late increases; peaks 55–70 | Prior risk score contributes steadily, serving as a **baseline predictor** of typical risk. |
   | `heart_rate_roll24h_mean` | Low early (0–55), sharp rise 55–75, highest around 60–70 | Late-sequence average heart rate indicates **ongoing cardiovascular status** driving median risk prediction. |
3. **Interpretation Summary**
  - **Temporal dynamics:** Unlike `max_risk`, the top features show **broadly distributed attention** across the stay, with early importance (timestep 0) for HR minima, then a late-sequence emphasis for average and cumulative indicators.  
  - **Steady baseline signals:** `news2_score` and `risk_numeric` provide continuous saliency, reflecting the model’s reliance on **persistent global indicators** for sustained risk.  
  - **Heart rate patterns:** Both minima and rolling mean contribute, but in **different phases** → minima dominate early, mean dominates later, capturing **initial condition plus ongoing cardiovascular trends**.  
  - **Moderate variability:** Features such as `heart_rate_missing_pct` add context-dependent information, indicating that missing data or variability in measurement subtly informs median risk.  
  - **Pattern implication:** Peaks are less sharp and less synchronised than `max_risk`, consistent with a target representing **average rather than acute risk**.
4. **Overall Summary**
  - The TCN’s temporal saliency pattern for `median_risk` emphasises **persistent, cumulative features** rather than acute peaks.  
  - Early heart rate minima indicate **baseline risk susceptibility**, while mid-to-late average heart rate, NEWS2 score, and risk_numeric reflect **ongoing physiological stability or instability**.  
  - Clinically, this aligns with **typical risk states**, where sustained deviations and cumulative physiological trends define a patient’s overall deterioration exposure, rather than single crisis events.

**Mean Saliency Heatmap (`median_risk_mean_heatmap.png`)**
1. **Context**
  - Visualises **temporal top-10 feature importance** for predicting the median (typical) deterioration risk during admission.  
  - Color intensity (log-scaled mean absolute saliency) reflects how strongly the model relies on each feature at each timestep across patients.  
  - Bright (yellow) = higher influence; darker (blue) = lower importance.  
	-	Reflects how the model captures sustained baseline and ongoing risk rather than transient peaks.
2. **Key Patterns**
  - **Overall Temporal Pattern:**
    - Brightness is distributed relatively evenly across the full sequence, though the most concentrated activation occurs between ≈55–80 hrs, where multiple features display sustained high saliency.
    - Early timesteps (0–5 hrs) show marked brightness in almost all features, indicating that initial physiological state contributes meaningfully to overall (median) risk estimation.
    - 5-15 hrs has almost zero activation throughout, suggesting other than initial activation (0-5 hrs), early timesteps provided almost no insight.
    -	The mid-period (15–50 hrs) maintains steady moderate activation, reflecting the model’s attention to ongoing physiological stability or mild variation.
    -	The late window (55–80 hrs) displays broad horizontal bands across key features, suggesting that persistent physiological signals remain influential through the later stages of admission.
    -	After ≈85 hrs, sharp taper, implying the model’s reduced reliance on final-sequence data, possibly due to diminishing available observations.
  | **Pattern** | **Description** | **Interpretation** |
  |--------------|----------------|--------------------|
  | **Continuous global indicators** | `news2_score` and `risk_numeric` stay bright from **0–96 hrs**, also with peaks throughout. | Model anchors predictions on **continuous global risk indicators**. |
  | **Sustained vital-sign importance** | `heart_rate_roll24h_min`, `heart_rate_mean` and `heart_rate_missing_pct` remain bright from **15–85 hrs**, especially **55–80 hrs** | Reflects **stable reliance on cardiovascular parameters** as markers of chronic or baseline risk. |
  | **Persistent band** | `heart_rate_roll24h_min` is the brightest and most persistent feature; sustained saliency from ~15 h onward, especially intense from 55-80 h | Indicates prolonged low or unstable heart rate is a major risk signal that is used throughout the entirety of stay. |
  | **Early respiratory prominence** | `respiratory_rate_roll24h_std` bright near **0–20 hrs**, fading to almost no activation for the rest of the timestamps | Suggests that **early respiratory variability** signals baseline instability that defines typical risk exposure. |
  | **Late prominent activation** | `heart_rate_roll24h_mean` is the only feature to have almost no activation from 5-55 hrs before having the second most prominent saliency bands from 55-85 hrs. | Signals that average rolling heart rate was a key late stage predictor of average risk. |
  | **Moderate steady activity** | `heart_rate_missing_pct`, `heart_rate_mean`, `systolic_bp`, `respiratory_rate_median` show mild but continuous activation | Indicates contextual physiological signals supporting general risk state. |
  | **Minimal vertical spikes** | Few transient bright cells (15-40 hrs) | Confirms **absence of discrete deterioration moments**, model tracks consistent patterns, not crises. |
3. **Interpretation Summary**
	-	Saliency is evenly distributed across time, showing the model maintains consistent attention rather than focusing on sharp peaks.
	-	Composite scores (`news2_score`, `risk_numeric`) and cardiovascular/respiratory trends dominate, reflecting reliance on stable, cumulative indicators of overall condition rather than short-term spikes.
	-	Early activation highlights the influence of initial physiological state, shaping the patient’s sustained risk profile.
	-	The smooth, continuous saliency evolution matches a target representing median (sustained) risk, not episodic deterioration.
4. **Overall Summary**
	-	The `median_risk` heatmap demonstrates that predictions are driven by persistent, system-wide physiological stability patterns, not discrete deterioration events.
	-	The model consistently integrates global indicators (NEWS2 score, risk value) with sustained vital trends (heart rate, respiration) throughout the admission timeline.
	-	Although attention is broadly spread, saliency peaks greatly in the later phase (≈55–80 hrs), suggesting that recent values still refine overall risk estimation.
	-	Clinically, this implies that typical patient risk is shaped by baseline state and long-term physiological behaviour, aligning with how average stability reflects ongoing health trajectory.

**Median-Risk Overall Saliency Summary**
-	**Primary Drivers:**
	-	Across all outputs, the model consistently emphasises rolling heart rate minima and mean (`heart_rate_roll24h_min`, `heart_rate_roll24h_mean`), global risk indicators (`news2_score`, `risk_numeric`), and heart rate missingness (`heart_rate_missing_pct`).
	-	Early-time heart rate minima highlight baseline susceptibility, while late-time mean heart rate captures ongoing cardiovascular trends contributing to sustained risk.
	-	Composite and derived scores provide continuous contextual information, anchoring predictions of typical patient risk rather than acute spikes.
-	**Temporal Focus:**
	-	**Saliency shows a bi-phasic pattern:** a brief early peak (0–5 hrs) reflecting initial physiological presentation, followed by broad late-to-mid activation (~55–80 hrs).
	-	The mid-period (15–50 hrs) maintains moderate, steady attention, supporting continuous integration of physiological stability rather than episodic events.
	-	Saliency declines after ~85 hrs, indicating minimal influence of final observations on median risk predictions.
- **Early vs Late Contributions:**
	-	`heart_rate_roll24h_min` maintains persistent high saliency early, reflecting initial patient baseline risk.
	-	`heart_rate_roll24h_mean` becomes dominant later, showing sustained cardiovascular trends drive the ongoing average risk assessment.
	-	`news2_score` and `risk_numeric` remain consistently bright throughout, supporting cumulative assessment.
	-	Other vitals (respiratory rate variability, systolic BP, missingness indicators) contribute moderate background information, without discrete “crisis” spikes.
-	**Interpretation of Variability:**
	-	Feature-level moderate-to-high standard deviations reflect variability in how strongly the model weighs these predictors across patients.
	-	Temporal patterns indicate the model treats median risk as a stable, cumulative property, rather than responding to isolated deterioration events.
	-	**Early activation vs late sustained signals illustrates dual-phase reliance:** initial condition sets baseline susceptibility, and later trends refine the predicted average risk.
- **Clinical Alignment:**
	-	Early heart rate minima reflect baseline cardiovascular vulnerability; later rolling mean trends and global scores capture ongoing physiological stability.
	-	The model’s sustained attention to rolling vital signs and integrated risk indicators aligns with typical patient trajectories, where median risk is shaped by cumulative physiological patterns rather than isolated acute deterioration.
	-	Absence of strong vertical spikes supports that median risk captures overall exposure to physiological instability rather than crisis events.

#### Saliency Analysis (`pct_time_high`)
**Feature-Level Mean & Standard Deviation (`pct_time_high_feature_saliency.csv`)**
1. **Context**
  - Quantifies **average and variability of saliency** for each input feature when predicting the percentage of time spent in a high-risk state (`pct_time_high`).  
  - The **mean** represents the average contribution of each feature across all patients and timesteps.  
  - The **standard deviation (std)** reflects how variable that contribution is between cases, indicating whether the model’s reliance on a feature is consistent or patient-specific.  
2. **Key Findings (General-Trend Scope)**
  | **Rank** | **Feature** | **Mean** | **Std** | **Interpretation** |
  |-----------|--------------|----------|----------|--------------------|
  | 1 | `systolic_bp_roll24h_min` | 1.98×10⁻⁵ | 4.56×10⁻⁵ | Strongest predictor with highest variability. Chronic low blood pressure drives extended high-risk exposure in many patients, but influence is uneven, reflecting subgroup-specific cardiovascular instability. |
  | 2 | `level_of_consciousness` | 1.93×10⁻⁵ | 4.32×10⁻⁵ | Consistent neurological signal. Persistent altered consciousness reliably indicates prolonged physiological stress and sustained risk. |
  | 3 | `respiratory_rate_missing_pct` | 1.76×10⁻⁵ | 4.09×10⁻⁵ | Missing respiratory readings correlate with poor monitoring or unstable respiration, adding consistent weight to cumulative risk estimation. |
  | 4 | `heart_rate` | 1.68×10⁻⁵ | 4.11×10⁻⁵ | Variable influence. Episodic tachycardia and heart rate volatility contribute to prolonged instability but not uniformly across cases. |
  | 5 | `systolic_bp` | 1.63×10⁻⁵ | 3.67×10⁻⁵ | Baseline systolic BP maintains moderate, steady influence; lower variability suggests broad, background relevance across most patients. |
3. **Interpretation Summary**
  - **Dominant domains:** Neurological (`level_of_consciousness`) and cardiovascular (`systolic_bp_roll24h_min`, `systolic_bp`) features lead the prediction, representing persistent systemic compromise.  
  - **Secondary contributors:** Respiratory missingness and heart rate signals add to risk duration, marking periods of incomplete monitoring or intermittent instability.  
  - **Variability pattern:** Most top features show **high standard deviations (≈4×10⁻⁵)**, implying that while the model consistently considers them important, their relative impact varies by patient trajectory.  
  - **Cross-system weighting:** The mixture of neurological, cardiovascular, and respiratory features demonstrates that cumulative deterioration reflects **multi-system strain over time**, not dominance of a single vital parameter.  
  - **Low-impact features:** Metrics below ≈1.0×10⁻⁵ (e.g., derived rolling slopes, temperature) show weak or context-limited contributions, aligning with their limited temporal saliency in heatmaps.
4. **Overall Summary**
  - The `pct_time_high` output is driven by features that encode **sustained or recurrent physiological instability**, particularly in blood pressure and consciousness.  
  - Respiratory and heart rate variables contribute intermittently, capturing fluctuations that lengthen total time in a high-risk state.  
  - The combination of high mean and high variability among leading predictors suggests varying deterioration pathways, where different physiological systems dominate in different patients.  
  - Clinically, this reflects the model’s interpretation of prolonged risk as **cumulative instability** across neurological, cardiovascular, and respiratory axes; a pattern typical of patients who remain unwell for extended periods rather than experiencing discrete acute events.

**Temporal Mean Saliency (`pct_time_high_temporal_saliency.csv`)**
1. **Context**
  - Represents the **average absolute saliency per timestep**, aggregated across all features and patients, for predicting the **percentage of time spent in high-risk states** during the stay.
  - Identifies **when** in the timeline the model is most sensitive to patient data when estimating cumulative risk exposure.  
  - Unlike peak or median risk, this reflects **total burden of instability**, integrating both early and late physiological contributions.  
2. **Key Findings (General-Trend Scope)**
  | **Pattern Region** | **Approx. Timesteps** | **Trend** | **Interpretation** |
  |--------------------|------------------------|------------|--------------------|
  | **Initial window** | 0–5 | Sharp early peak (1.96×10⁻⁵ → ~9×10⁻⁶ by t=5) | Model heavily weights **initial physiological state** to define each patient’s baseline high-risk exposure potential. |
  | **Early–mid phase** | 5–20 | Rapid drop then stable low plateau (~7–8×10⁻⁶) | Reflects transition to steady monitoring; minimal change signals limited incremental information after baseline. |
  | **Mid sequence** | 20–55 | Fluctuating low–moderate band (~7–8.5×10⁻⁶) | Suggests **ongoing sensitivity to intermittent physiological changes**; captures moderate but persistent contributions. |
  | **Late sequence** | 55–80 | Broad rise peaking at ~9.15×10⁻⁶ (≈timestep 79) | Indicates **renewed saliency toward the end**, consistent with late deterioration phases influencing overall exposure. |
  | **Final window** | 85–96 | Progressive decline to <3.6×10⁻⁶ | Model attention fades as physiological signal becomes less informative or stabilises at discharge. |
3. **Interpretation Summary**
  - The model demonstrates a **bi-phasic saliency pattern** → strong early activation (baseline definition) followed by a sustained mid-level plateau, and a **broad late rise** (~55–80 hrs).  
  - Early saliency reflects that **initial blood pressure, consciousness, and baseline vitals** are key to estimating how long a patient remains unstable.  
  - The long middle period of moderate attention (20–55 hrs) indicates **continuous risk integration** rather than episodic responses.  
  - The late resurgence (55–80 hrs) corresponds to **renewed instability or cumulative deterioration**, showing that many patients who spent long periods in high risk tend to have another wave of deterioration later, not just early instability.
  - The lack of sharp spikes supports that `pct_time_high` captures extended instability periods, not isolated events.
4. **Overall Summary**
  - Temporal saliency for `pct_time_high` highlights a **dual emphasis**: early physiological presentation and later sustained instability both shape cumulative high-risk exposure.  
  - The mid-phase consistency suggests the model interprets ongoing physiological status as **continuously additive** to total risk time.  
  - Clinically, this indicates that **patients who start unstable and later redevelop instability** accumulate greater overall risk exposure.  
  - The early and late peaks mirror the **two major phases of deterioration burden** → initial vulnerability and subsequent relapse or persistence of instability.

**Top-Feature Temporal Profiles (`pct_time_high_top_features_temporal.csv`)**  
1. **Context**  
  - Tracks how the saliency of the **top 5 features** evolves over time (`timestep` = 0–95) when predicting **the percentage of time a patient spent in a high-risk state**.  
  - Each value represents the **mean absolute saliency** of a feature at a given timestep, averaged across all patients.  
  - This identifies when specific physiological domains most influence the model’s estimation of **cumulative high-risk exposure** over the hospital stay.  
2. **Key Findings (General-Trend Scope)**  
  | **Feature** | **Temporal Pattern** | **Interpretation** |
  |--------------|----------------------|--------------------|
  | `systolic_bp_roll24h_min` | Moderate early activation (0–10 h), stable mid-level saliency with **two strong peaks** around **30–50 h** and **60–80 h**, then gradual decline to end | Indicates that **chronic hypotension** and **recurrent low BP episodes** define cumulative exposure—both early and late phases signal prolonged systemic compromise. |
  | `level_of_consciousness` | Strong and persistent activation from **0–55 h**, then moderate taper until ~70 h before fading | Suggests that **sustained neurological impairment** remains one of the most consistent determinants of prolonged instability, marking patients with continuous altered consciousness as chronically high-risk. |
  | `respiratory_rate_missing_pct` | Low early signal, followed by a **steady mid-to-late rise** (30–90 h) | Reflects that **missing respiratory observations** or measurement gaps emerge during longer admissions, likely associated with progressive deterioration or reduced monitoring quality. |
  | `heart_rate` | Episodic peaks at **~16 h**, **45–55 h**, and **70–80 h** | Highlights **intermittent tachycardic surges** that contribute additively to cumulative instability—acute but transient stress periods layered on top of chronic risk. |
  | `systolic_bp` | Mild activation early (0–10 h), moderate mid (30–60 h), renewed elevation **after 65 h** | Reflects **recurrent BP instability**—a background driver of sustained physiological compromise through both early baseline and later relapse phases. |
3. **Interpretation Summary**  
  - **Temporal structure:** Saliency is **broadly distributed**, with two distinct phases of heightened activity—an **early period (0–50 h)** marked by neurological and circulatory instability, and a **late phase (60–80 h)** of multi-system reactivation.  
  - **Dominant contributors:** Persistent signals from `level_of_consciousness` and `systolic_bp_roll24h_min` indicate that **sustained hypotension and impaired consciousness** are central to predicting how long patients remain at high risk.  
  - **Dynamic components:** `heart_rate` and `respiratory_rate_missing_pct` add episodic and contextual information, representing transient physiological stress and deteriorating monitoring reliability.  
  - **Cumulative effect:** The asynchronous yet overlapping feature peaks indicate the model integrates **multiple overlapping physiological instabilities** rather than depending on a single deterioration event.  
4. **Overall Summary**  
  - The TCN’s temporal saliency for `pct_time_high` reflects a **dual-phase pattern**: early persistent hypotension and altered consciousness set the baseline for high-risk exposure, while later episodes of respiratory instability and cardiovascular reactivation extend total high-risk duration.  
  - This pattern aligns clinically with patients who remain unstable for prolonged periods—initial frailty or neurological suppression compounded by later systemic decompensation.  
  - The model thus captures **prolonged, evolving physiological instability**, distinguishing sustained high-risk exposure from isolated deterioration episodes.  

**Mean Saliency Heatmap (`pct_time_high_mean_heatmap.png`)**
1. **Context**
  - Visualises **temporal top-10 feature importance** for predicting the **percentage of time a patient spent in a high-risk state** (`pct_time_high`).  
  - Color intensity (log-scaled mean absolute saliency) represents the relative influence of each feature across 96 timesteps for all patients.  
  - Bright (yellow) = stronger contribution; dark (blue) = weaker influence.  
  - This target measures **cumulative exposure** to deterioration risk, reflecting how long a patient remained unstable, rather than the presence of a single deterioration event.
2. **Key Patterns**
  - **Overall Temporal Pattern:**
    - The heatmap shows **widespread bright activation** across the full sequence, far denser than in other targets.  
    - There is **no single dominant phase** — instead, activity clusters early–mid (0–50 hrs) and again late (60–85 hrs), with short gaps of low activation between.  
    - Early activation (0–5 hrs) across almost all features reflects the model’s use of **baseline physiological state** to set the foundation for cumulative high-risk exposure.  
    - Mid-period (10–50 hrs) shows alternating low and high saliency, suggesting intermittent instability rather than steady decline, though **consciousness** remains persistently bright throughout this phase.  
    - Late sequence (60–85 hrs) displays broad, intense activation across nearly all top features, indicating **system-wide recurrence or persistence of instability**.  
    - Activation drops sharply after ≈90 hrs, consistent with end-of-stay stabilisation or lack of further signal.

  | **Pattern** | **Description** | **Interpretation** |
  |--------------|----------------|--------------------|
  | **Early multi-feature activation** | Near-universal brightness at 0–5 hrs | The model anchors on baseline vital and consciousness levels to define each patient’s overall high-risk exposure. |
  | **Persistent neurological dominance** | `level_of_consciousness` remains bright through 0–55 hrs | Continuous altered consciousness signals prolonged physiological stress, a key marker of sustained risk. |
  | **Chronic cardiovascular weighting** | Systolic BP features (`roll24h_min`, `bp`, `bp_max`, `bp_missing_pct`) remain active across early and late windows | Ongoing hypotension and BP variability define both early and recurrent instability phases. |
  | **Intermittent respiratory influence** | Respiratory rate and missingness show scattered peaks 30–90 hrs | Periods of missing or unstable respiration reinforce high-risk duration, especially late in stay. |
  | **Late multi-system convergence** | Heart rate, SpO₂, and BP peaks coincide 60–80 hrs | Reflects a late, system-wide instability phase contributing to extended high-risk exposure. |

3. **Interpretation Summary**
  - The model distributes attention broadly across the timeline, reflecting the **continuous and cumulative nature** of the `pct_time_high` target.  
  - Unlike `max_risk` or `median_risk`, saliency remains high throughout, indicating that **risk exposure is shaped by long-term physiological burden**, not isolated deterioration periods.  
  - Persistent activation in `level_of_consciousness` and systolic BP features shows that **neurological state and hypotension** are consistent anchors of cumulative instability.  
  - Intermittent respiratory and oxygenation signals contribute episodically, aligning with **recurrent hypoxia or ventilatory compromise** as additive factors in prolonged high-risk states.  
  - The late clustering of brightness across multiple vital signs suggests that **recurring, multi-system strain** drives the length of time a patient remains at high risk rather than single-organ failure.
4. **Overall Summary**
  - The `pct_time_high` heatmap demonstrates a model that integrates **persistent and recurring physiological signals** over time to estimate cumulative high-risk exposure.  
  - Saliency is **densely and widely distributed**, showing the model treats prolonged deterioration as a **multi-phase process**—early vulnerability followed by sustained or re-emerging instability.  
  - Key drivers include **altered consciousness** and **chronic hypotension**, supported by episodic contributions from **respiratory and oxygenation markers**.  
  - Clinically, this suggests the model recognises that patients who remain in a high-risk state do so due to **enduring systemic compromise** with recurrent decompensation, rather than discrete acute events.

**Pct-Time-High Overall Saliency Summary**
- **Primary Drivers:**
    - The model consistently emphasises **sustained hypotension** (`systolic_bp_roll24h_min`, `systolic_bp`) and **neurological state** (`level_of_consciousness`) as dominant determinants of cumulative high-risk exposure.
    - Secondary contributors include **heart rate** and **respiratory missingness** (`respiratory_rate_missing_pct`), which introduce episodic or context-dependent signals extending total high-risk time.
    - High mean saliency combined with moderate-to-high standard deviation across these features indicates that **different patients accumulate high-risk time through varied physiological pathways**, reflecting subgroup-specific vulnerability patterns.
- **Temporal Focus:**
  - **Saliency exhibits a dual-phase pattern:** an early period (0–50 hrs) reflecting baseline vulnerability, and a late period (60–85 hrs) corresponding to recurrent or persistent physiological instability.
  - Early activation anchors the prediction in **initial BP, consciousness, and vital signs**, setting a baseline for expected cumulative high-risk duration.
  - The mid-phase (10–50 hrs) maintains moderate, fluctuating saliency, indicating **continuous integration of intermittent physiological changes** rather than acute events.
  - Late resurgence (55–80 hrs) aligns with **multi-system deterioration**, capturing patients who experience sustained or renewed instability.
  - Saliency declines sharply after ~90 hrs, consistent with **end-of-stay stabilisation** or completed integration of cumulative risk information.
- **Early vs Late Contributions:**
  - `systolic_bp_roll24h_min` and `level_of_consciousness` dominate early phases, marking patients with **baseline chronic hypotension and neurological compromise**.
  - `heart_rate` and late-phase BP features contribute mid-to-late, reflecting **episodic tachycardia and recurrent blood pressure instability**.
  - `respiratory_rate_missing_pct` provides context-dependent weight, showing that **monitoring gaps** or evolving respiratory compromise inform cumulative risk.
  - Other vitals (e.g., SpO₂, minor BP measures) add background information without discrete “crisis” spikes.
- **Interpretation of Variability:**
  - Feature-level high mean and moderate-to-high standard deviation reflect **both consistent importance and patient-specific variation** in how physiological features drive cumulative high-risk time.
  - Temporal patterns confirm that the model integrates **initial patient state plus ongoing and recurrent physiological instability**, rather than responding to single, isolated deterioration events.
  - The dual-phase reliance—early baseline plus late resurgence—illustrates **how initial vulnerability and later multi-system compromise combine to determine total high-risk exposure**.
- **Clinical Alignment:**
  - Early hypotension and impaired consciousness mark patients likely to remain at high risk for prolonged periods.
  - Late-phase increases in heart rate, BP, and respiratory missingness correspond to **renewed or persistent systemic instability**, clinically reflecting recurrent deterioration or protracted illness.
  - Overall, the model interprets `pct_time_high` as **cumulative physiological instability**, capturing both baseline vulnerability and later-stage multi-system compromise rather than isolated acute deterioration events.

#### Overall Saliency Summary
**Overall Outcomes**
- The analyses successfully identified the key physiological drivers, temporal patterns, and feature-level importance for each outcome, providing clinically interpretable insights into patient deterioration trajectories. 
- Overall, the saliency analysis demonstrates that the models:
  - Capture both **baseline vulnerability** and **dynamic trends** in patient vitals.
  - Highlight **target-specific primary drivers** (cardiovascular, respiratory, neurological, or composite scores).
  - Show **temporal dual-phase patterns** where early indicators anchor predictions and late trends refine risk estimates.
  - Provide **patient-specific variability**, reflecting heterogeneous physiological pathways leading to high-risk events or sustained instability.
**Comparative Saliency Table**

| **Aspect** | **`max_risk`** | **`median_risk`** | **`pct_time_high`** | **Clinical Alignment / Interpretation** |
|------------|--------------|----------------|-----------------|----------------------------------------|
| **Primary Drivers** | `heart_rate_roll24h_min`, respiratory rate metrics (`respiratory_rate`, `respiratory_rate_roll4h_min`), `news2_score` | `heart_rate_roll24h_min`, `heart_rate_roll24h_mean`, `news2_score`, `risk_numeric`, `heart_rate_missing_pct` | `systolic_bp_roll24h_min`, `systolic_bp`, `level_of_consciousness`; secondary: `heart_rate`, `respiratory_rate_missing_pct` | `max_risk` captures acute deterioration surges; `median_risk` reflects typical baseline + ongoing cardiovascular trends; `pct_time_high` captures cumulative, multi-system instability (neurological + cardiovascular + respiratory). |
| **Temporal Focus** | Gradual rise ~40h, peak 55–85h, slight taper | Bi-phasic: early peak 0–5h, moderate mid 15–50h, broad late 55–80h, decline after 85h | Dual-phase: early 0–50h (baseline vulnerability), mid 10–50h (moderate fluctuations), late 55–80h (recurrent multi-system instability), decline ~90h | All models incorporate temporal structure: `max_risk` emphasizes recent deterioration; `median_risk` and `pct_time_high` integrate both baseline and later trends, showing progressive accumulation of risk. |
| **Early vs Late Contributions** | Early: minor; Late: heart rate minima, respiratory rate, NEWS2 spikes | Early: `heart_rate_roll24h_min`; Late: `heart_rate_roll24h_mean`, `news2_score`, `risk_numeric` | Early: `systolic_bp_roll24h_min`, `level_of_consciousness`; Mid-to-late: `heart_rate`, BP, `respiratory_rate_missing_pct` | Early signals reflect baseline vulnerability; late signals reflect sustained or recurrent instability, aligning with clinical expectations of deterioration trajectories. |
| **Sustained vs Episodic Signals** | Persistent heart rate influence, intermittent respiratory/NEWS2 spikes | Persistent rolling heart rate and global risk scores, moderate background vitals | Persistent baseline features (BP, consciousness), episodic or context-dependent secondary features (HR, respiratory missingness) | `max_risk` emphasizes acute events; `median_risk` reflects cumulative typical trends; `pct_time_high` integrates continuous physiological burden with intermittent worsening events. |
| **Feature Variability** | High SD across patients; influence varies by case | Moderate-to-high SD; dual-phase reliance | High mean + moderate-to-high SD; patient-specific pathways influence cumulative high-risk time | All targets show patient-specific heterogeneity; `max_risk` captures peak-specific variation; `median_risk` and `pct_time_high` emphasize continuous and cumulative risk integration. |
| **Clinical Takeaways** | Prolonged deviations in HR, RR, and NEWS2 precede peak events; captures both chronic decline and acute surges | Baseline cardiovascular vulnerability + sustained trends; cumulative risk without acute spikes dominates | Early hypotension & impaired consciousness set baseline risk; late-phase HR, BP, and missing respiratory data reflect recurrent or persistent multi-system instability | Provides complementary perspectives: `max_risk` → acute peak risk; `median_risk` → typical ongoing risk; `pct_time_high` → total cumulative exposure. Supports real-world interpretation of patient deterioration trajectories. |

---

### SHAP vs Saliency Interpretability: Comparative Analysis
#### Objective
- This cross-model interpretability analysis evaluates whether **feature importance identified by SHAP (LightGBM)** aligns with **temporal saliency patterns (TCN)** across three clinical risk targets (`max_risk`, `median_risk`, `pct_time_high`).  
- **Note:** This is **not a one-to-one comparison**, as the two models are trained on different architectures and **distinct feature sets**, so exact feature-level correspondence cannot be assumed. The analysis focuses on convergence of insights rather than direct numerical alignment.
- Goals:
  - Confirm whether both interpretability methods converge on the same or similar **key predictors**.
  - Identify **temporal patterns, dual-phase dynamics, and patient-specific variability** that SHAP cannot reveal.
  - Integrate findings for **clinically meaningful insights** into patient deterioration trajectories.
- **Rationale:** Alignment across models strengthens confidence in feature relevance, while **temporal saliency adds context** about when and how predictors influence risk, providing richer interpretability than static feature importance alone.
#### Top Feature Overlap

| **Target** | **Top SHAP Features** | **Top Saliency Features** | **Alignment / Observations** |
|------------|---------------------|--------------------------|------------------------------|
| **Max Risk** | `spo2_min`, `supplemental_o2_mean`, `respiratory_rate_max`, `temperature_missing_pct`, `heart_rate_mean` | `heart_rate_roll24h_min`, `respiratory_rate`, `respiratory_rate_roll4h_min`, `news2_score`, `temperature_max`, `level_of_consciousness_carried` | Partial overlap: Both highlight **respiratory and cardiovascular metrics**. Saliency adds **temporal context**, showing heart rate minima and respiratory trends rising late in sequence, capturing cumulative deterioration leading to peak risk. SHAP confirms feature relevance without temporal resolution. |
| **Median Risk** | `respiratory_rate_mean`, `spo2_mean`, `heart_rate_max`, `systolic_bp_missing_pct`, `level_of_consciousness_missing_pct` | `heart_rate_roll24h_min`, `heart_rate_roll24h_mean`, `news2_score`, `risk_numeric`, `heart_rate_missing_pct` | Strong conceptual alignment: Both emphasize **heart rate and respiratory features**. Saliency reveals **bi-phasic temporal pattern** (early baseline, late sustained trends), unseen in SHAP. Missingness metrics appear in both, confirming clinical proxy for instability. |
| **Pct Time High** | `respiratory_rate_mean`, `heart_rate_max`, `supplemental_o2_mean`, `spo2_mean`, `temperature_median` | `systolic_bp_roll24h_min`, `systolic_bp`, `level_of_consciousness`; secondary: `heart_rate`, `respiratory_rate_missing_pct` | Partial alignment: SHAP identifies **respiratory and cardiovascular contributors**, while saliency highlights **cumulative, multi-system instability**. Dual-phase temporal pattern (early baseline + late resurgence) is unique to saliency, showing prolonged risk dynamics. |

#### Narrative Analysis
**Alignment**
- Across all targets, **major physiological domains** are consistently important in both methods:
  - **Max Risk:** Acute respiratory and cardiovascular indicators dominate.
  - **Median Risk:** Sustained cardiovascular trends and oxygenation.
  - **Pct Time High:** Multi-system cumulative burden (neurological, cardiovascular, respiratory).  
- **Missingness metrics** (temperature, BP, LOC) appear in both, reflecting the model’s recognition of data gaps as clinical instability proxies.
**Divergence**
- Saliency captures **temporal dynamics** that SHAP cannot:
  - **`max_risk`:** Late-sequence escalation in heart rate minima and respiratory rate indicates **dynamic detection of peak deterioration**.
  - **`median_risk`:** Bi-phasic pattern—early baseline importance followed by late sustained trends.
  - **`pct_time_high`:** Dual-phase emphasis; early baseline sets initial high-risk potential, late resurgence reflects **prolonged or recurrent instability**.  
- Saliency also highlights **episodic vs sustained contributions**, showing which features are transient vs persistent drivers of risk.

#### Integrated Clinical Interpretation
- **Static SHAP insights:** Provide feature-level importance, confirming which physiological measures (HR, RR, SpO₂, BP, consciousness) the models consider predictive.  
- **Temporal saliency insights:** Reveal what features matter and specifically when, capturing progressive deterioration, dual-phase effects (early and late), transient spikes and sustained periods of increased model sensitivity.  
- **Complementarity:**  
  - SHAP validates **feature relevance**.  
  - Saliency provides **dynamic understanding**, critical for clinical interpretation of risk evolution.  
- **Clinical takeaway:**  
  - `max_risk` → acute deterioration prediction, sensitive to **late-stage physiological trends**.  
  - `median_risk` → typical risk state, integrates **baseline, late-stage and sustained signals**.  
  - `pct_time_high` → total cumulative exposure, capturing **recurrent, persistent and widespread scattered instability**.  

#### Conclusion
1. **Alignment:** SHAP and saliency largely agree on top contributors, strengthening confidence in model reasoning.  
2. **Temporal enrichment:** Saliency adds insight into **timing, dual-phase patterns, and episodic vs sustained signals**.  
3. **Clinical coherence:** The combined SHAP and saliency analysis links model-predicted risk to physiologically meaningful events, showing which features drive risk and when, supporting transparent and interpretable predictions.
4. **Recommendation:** Both methods should be used jointly; SHAP for static feature confirmation, saliency for **dynamic temporal interpretability**.

## Phase 7A: Inference Demonstration (Deployment-Lite)
### Goal
- Deliver the first deployment-oriented version of the entire modelling pipeline.  
- Show that both trained models (LightGBM + TCN) can be reliably loaded, reconstructed, and used to generate predictions outside the training/evaluation environment.  
- Produce lightweight, deployment-ready outputs (predictions + top-features summaries) that mirror real-world model usage.  
- Provide an interactive per-patient lookup to demonstrate model behaviour in a clinically realistic way before full cloud deployment in Phase 7B.

### 1. Unified Inference Pipeline (`unified_inference.py`)
  - **Purpose**
    - Run batch inference for LightGBM and TCN models in one deployment-ready script.  
    - Save predictions and top-10 feature importance summaries.  
    - Provide optional interactive CLI for single-patient predictions.  
    - Ensure reproducible, dataset-agnostic outputs without requiring true labels. 
  - **Process (Summary)**
    0. **Initialise Imports & Paths**
      - Load standard, data, ML, and interpretability libraries (`json`, `pandas`, `numpy`, `torch`, `joblib`, `shap`).
      - Resolve project directories (`SCRIPT_DIR`, `SRC_DIR`, `PROJECT_ROOT`).
      - Load input paths:
        - Patient-level features (`news2_features_patient.csv`)
        - Test splits (`patient_splits.json`)
        - TCN: padding config (`padding_config.json`), prepared tensors (`prepared_datasets/`)
      - Load model paths:
        - LightGBM models (`lightgbm_results/`)
        - TCN weights and config (`tcn_best_refined.pt`, `config_refined.json`)
      - Create output folder: `deployment_lite_outputs/`.
    1. **Load Test Data for LightGBM**
      - Load test patient IDs → `test_ids`.
      - Subset patient-level features → `test_df`.
      - Define model input features (`feature_cols`) by excluding non-features (`subject_id`, `max_risk`, `median_risk`, `pct_time_high`).
      - **Rationale:** Only input features needed; binary targets and labels unnecessary for inference.
    2. **Compute LightGBM Inference**
      - Load and run each model (`max_risk`, `median_risk`, `pct_time_high`) on `X_test`.
      - Classification → positive-class probabilities; regression → continuous predictions.
      - Clip regression outputs at 0.
      - Save `lightgbm_inference_outputs.csv`.
    3. **Compute TCN Inference**
      - Load `TCNModel`, test tensors (`x_test`, `mask_test`), and configuration.
      - Reconstruct model architecture and load weights; set to evaluation mode.
      - Run forward pass with `torch.no_grad()` → deterministic outputs.
      - Extract outputs:
        - Convert logits → sigmoid probabilities
        - Inverse-transform regression (`expm1`), clip negatives at 0.
      - Build `df_tcn` and save `tcn_inference_outputs.csv`.
      - **Rationale:** No binary targets needed; architecture reconstruction required for loading weights; masks preserve sequence validity.
    4. **Compute LightGBM Interpretability (SHAP)**
      - Compute mean absolute SHAP values per feature for each target.
      - Keep **top-10 features** per target in dataframe.
      - Save numeric summary → `lightgbm_top10`.
      - **Rationale:** Lightweight, deployment-ready; mirrors Phase 6 top-10 summary; no plots to maintain lightweight outputs.
    5. **Compute TCN Interpretability (Gradient × Input Saliency)**
      - Load feature names from `padding_config.json`, map to TCN tensor features.
      - For each output head (`max_risk`, `median_risk`, `pct_time_high`):
        - Compute |gradient × input| saliency across patients and timesteps.
        - Aggregate to mean per feature (average across patients and timesteps), keep top-10 features.
      - Save numeric summary → `tcn_top10`.
      - **Rationale:** Matches Phase 6 methodology; numeric-only output keeps pipeline lightweight.
    6. **Merge Feature Summaries**
      - Concatenate LightGBM `lightgbm_top10` and TCN `tcn_top10` top-10 summaries → `combined_summary`.
        - Columns: `feature`, `mean_abs_shap`, `target`, `model`, `mean_abs_saliency`
        - Output: `top10_features_summary.csv`
      - Save as `top10_features_summary.csv` (60 rows: 2 models × 3 targets × 10 features).
      - **Rationale:** One consolidated, deployment-ready file; no plots or per-patient arrays; easy dashboard/reporting.
    7. **Interactive CLI: Single-Patient Inference**
      - Optional CLI interface post-batch inference.
      - Input patient ID → validate against `test_ids`.
      - Display that patient’s predictions for LightGBM (`lightgbm_preds`) and TCN (`prob_max`, `prob_median`, `y_pred_reg_raw`).
      - Loop until user exits.
      - **Rationale:** Optional, lightweight, reproducible CLI for quick inspection; uses precomputed outputs; supports deployment without extra artefacts.
  - **Outputs**
    - **Batch Predictions**
      - `lightgbm_inference_outputs.csv` → classification probabilities (`max_risk`, `median_risk`) + regression (`pct_time_high`) for all test patients.
      - `tcn_inference_outputs.csv` → probabilities and regression outputs for TCN model.
    - **Interpretability**
      - `top10_features_summary.csv` → combined top-10 features per target from LightGBM (SHAP) and TCN (Gradient×Input Saliency).
    - **Interactive CLI**
      - Optional terminal output for single-patient predictions using the same preprocessed inputs.
  - **Reasoning / Rationale**
    - **Reproducibility:** Batch inference ensures deterministic outputs and removes variation from looping or incremental processing.
    - **Unified pipeline:** Consolidates separate evaluation and interpretability scripts into a single workflow for both LightGBM and TCN.
    - **Interpretability tailored to model type:** LightGBM → SHAP; TCN → Gradient×Input Saliency. Only top-10 features retained to keep outputs lightweight, consistent with Phase 6 methodology.
    - **Binary targets omitted:** Inference does not compute metrics; outputs are generated from input features only, without label reconstruction or calibration.
    - **Regression clipping:** Ensures numeric predictions are valid (no negative percentages).
    - **Deployment-ready and dataset-agnostic:** Any dataset with the required feature columns / tensors can be passed directly into the script without outcome labels.
    - **Optional CLI:** Lightweight inspection of single-patient predictions without recomputation; aligns with batch outputs for consistency.
### End Products of Phase 7A
- **Fully functional deployment-lite inference pipeline**
  - Single, unified script (`unified_inference.py`) that handles both LightGBM (patient-level) and TCN (time-series) models.
  - Supports **batch inference** for all test patients and **interactive per-patient predictions** via CLI.
- **Reproducible, deployment-ready outputs**
  - Clean CSVs of predictions for both models:
    - LightGBM → classification probabilities + regression.
    - TCN → classification probabilities + regression.
  - Combined top-10 feature importance summary:
    - LightGBM SHAP + TCN Gradient×Input saliency.
    - Provides quick interpretability without heavy visualisation.
- **Lightweight, minimal, and consistent**
  - No extraneous evaluation steps or binary label recreation.
  - Regression outputs clipped to valid ranges.
  - Fully dataset-agnostic: any dataset with required features/tensors can be processed directly.
- **Optional interactive interface**
  - CLI allows rapid inspection of predictions for a single patient.
  - Uses the same preprocessed inputs and outputs as batch inference for consistency.
- **Canonical demonstration of end-to-end deployment**
  - Shows reproducible, production-ready inference for both model families.
  - Serves as a **foundation for full cloud deployment or dashboard integration in Phase 7B**.
### Summary
- Phase 7A finalises the project by delivering a **deployment-ready, unified inference pipeline** for both LightGBM (patient-level) and TCN (time-series) models. 
- It consolidates all preprocessing, model loading, prediction, and interpretability steps into a single, reproducible workflow. 
- This phase demonstrates **end-to-end usability**, allowing both batch inference on datasets and interactive per-patient inspection. 
- By stripping out evaluation-specific steps (binary target recreation, metric computation) and focusing solely on producing predictions and top feature summaries, Phase 7A produces a **lightweight, deterministic, and dataset-agnostic pipeline**. 
- It serves as the canonical end-point of the modeling workflow, ensuring that all previous phases—from model training to interpretability analysis—can be executed in a consistent and production-ready manner. 
- In essence, Phase 7A **completes the project** by transforming the experimental models and scripts into a cohesive, deployment-ready system that can be used directly for inference, reporting, or future cloud-based deployment.

**Rationale**
- **Batch inference ensures reproducibility** 
  - By running predictions on the entire dataset at once, eliminates variability introduced by looping, streaming, or per-record execution. 
  - This mirrors the stable behaviour required in deployment contexts.
- **A unified LightGBM + TCN inference pipeline**
  - Consolidates previously separate evaluation scripts into one consistent workflow. 
  - Both models now load inputs, preprocess, predict, and export outputs in a synchronised and deterministic way, preventing inconsistencies across model families.
- **Interpretability methods are matched to model architecture**:  
  - LightGBM uses **SHAP**, the most appropriate and widely validated method for tree-based models.  
  - TCN uses **Gradient×Input saliency**, the correct differentiable method for neural sequence models.  
  - Only **top-10 features** per target are retained to keep the outputs lightweight while still aligned with Phase 6 interpretability.
- **Binary target recreation is intentionally omitted** 
  - Inference does not evaluate performance.  
  - Predictions are generated directly from input features; true labels, metrics, or calibration computations are not part of deployment and would add unnecessary steps.
- **Regression outputs are clipped at 0** 
  - Prevent invalid negative predictions (e.g., negative percentage of high NEWS2 time). 
  - This ensures deployment-safe numeric outputs.
- **Single-patient inference** 
  - Uses the same processed inputs and feature mapping as batch inference, guaranteeing that CLI outputs match the CSV files exactly. 
  - The terminal interface is kept intentionally minimal to support real-time case inspection without clutter.
- **The final pipeline is deployment-ready and dataset-agnostic**: 
  - Any future dataset containing the required feature columns or valid tensors can be passed directly into the script, without requiring outcome labels or metric calculations.

  ### Folder Directories
```text
project_root/
├─ data/
│  └─ processed_data/
│     └─ news2_features_patient.csv        # Patient-level features for LightGBM
├─ src/
│  ├─ ml_models_tcn/
│  │  ├─ deployment_models/
│  │  │  └─ preprocessing/
│  │  │     ├─ patient_splits.json         # Test/train/val patient splits
│  │  │     └─ padding_config.json         # TCN feature names / tensor order
│  │  └─ prepared_datasets/
│  │     ├─ test.pt                        # TCN input tensor for test patients
│  │     └─ test_mask.pt                   # TCN mask tensor for padded sequences
│  ├─ prediction_diagnostics/
│  │  └─ trained_models_refined/
│  │     ├─ tcn_best_refined.pt            # TCN trained weights
│  │     └─ config_refined.json            # TCN architecture hyperparameters
│  ├─ prediction_evaluations/
│  │  └─ lightgbm_results/
│  │     └─ {target}_retrained_model.pkl   # LightGBM models per target
│  │ 
│  │ 
│  └─ scripts_inference/                   # Phase 7A Folder
│     ├─ unified_inference.py              # Phase 7A deployment-lite script
│     └─ deployment_lite_outputs/
│        ├─ lightgbm_inference_outputs.csv
│        ├─ tcn_inference_outputs.csv
│        └─ top10_features_summary.csv     # Combined SHAP + Saliency top-10
```

### Deployment Rationale, Standards, and Strategy
#### 1. Overview
- Deployment is the final stage of the machine learning lifecycle → converting a trained model from a research artifact into a usable, reproducible inference system.  
- It demonstrates **end-to-end capability**, bridging data science with software engineering and MLOps.  
- For this project, deployment serves as both a technical milestone and a portfolio signal of readiness for real-world AI development.

#### 2. Standard ML Pipeline and Deliverables
- A complete ML pipeline typically progresses through five phases, each producing defined outputs:

| **Phase** | **Goal** | **Core Deliverables** |
|------------|-----------|------------------------|
| **1. Data Engineering** | Collect, clean, preprocess, and structure input data. | Datasets, preprocessing scripts, normalisation configs (`scaler.pkl`, etc.) |
| **2. Model Development** | Build and train model architectures. | Training scripts, hyperparameter configs, trained weights (`model.pt`) |
| **3. Evaluation** | Assess model performance and generalisation. | Metrics reports, validation plots, comparison to baselines |
| **4. Deployment** | Serve the model for inference in a reproducible environment. | Inference API, containerised service, documentation |
| **5. Monitoring (MLOps Stage 5)** | Track model performance in production. | Logging, drift detection, retraining triggers |

- Deployment is **Phase 4** → transforming research code into an inference-ready system that other users or systems can interact with.

#### 3. Standard MLOps Deployment Practice
- MLOps (Machine Learning Operations) integrates ML workflows into reliable software systems.  
- A standard deployment stack (framework) includes:

| **Component** | **Standard Practice** | **Tools** |
|----------------|------------------------|------------|
| **Containerisation** | Freeze dependencies and environment | Docker |
| **Model Serving** | Provide inference endpoint | FastAPI, Flask, TorchServe |
| **Model Registry** | Store model artifacts | MLflow, Hugging Face Hub, AWS S3 |
| **CI/CD** | Automate build, test, and deploy | GitHub Actions, Jenkins |
| **Monitoring** | Track latency, drift, and usage | Prometheus, Grafana, ELK stack |
| **Cloud Hosting** | Deploy on scalable platform | AWS SageMaker, GCP Vertex, Azure ML |
- This is the **production-grade “gold standard”** setup used in large organisations handling live traffic or regulated environments.

#### 4. Why Full Deployment is Not Always Appropriate
- Full MLOps deployments are complex and resource-intensive. They require:
  - Persistent cloud infrastructure.
  - Continuous monitoring and retraining pipelines.
  - Secure authentication, CI/CD, and cost management.
- For research or academic projects, this level of infrastructure is excessive.  
- Instead, a **phased approach**, starting with a “deployment-lite” inference layer, achieves almost all learning and signalling benefits without cost or operational overhead.

#### 5. Deployment-Lite Rationale
- A **deployment-lite** pipeline focuses on reproducibility and usability rather than scalability.  
- It simulates how a production service works, but within a self-contained environment.
- A phased approach is **standard practice** in most professional ML teams → first prove inference works reliably (Phase 7A: deployment-lite) → then move to a scalable, monitored environment (Phase 7B: full deployment).

| **Aspect** | **Deployment-Lite** | **Full Production Deployment** |
|-------------|--------------------|-------------------------------|
| **Goal** | Demonstrate inference usability | Serve live traffic |
| **Infrastructure** | Local or single-container API | Distributed cloud system |
| **Stack** | FastAPI + Docker | FastAPI + CI/CD + Monitoring |
| **Hosting** | Render, Hugging Face Spaces, or local | AWS / GCP / Azure |
| **Scope** | Inference + basic logging | Full lifecycle management |
| **Use Case** | Research, education, portfolio | Production-scale applications |

#### 6. Our Project’s Deployment Plan
**Context**
- The project contains two validated models and their interpretability artefacts:  
  - **LightGBM (tabular / patient-level):** trained, evaluated, and explained with **SHAP** (global feature attributions saved as CSVs and PNGs).  
  - **TCN_refined (temporal / sequence-level):** trained, evaluated, and explained with **gradient×input saliency** (per-target temporal saliency CSVs and heatmaps).
- The next stage is **to demonstrate end-to-end usability**, allowing reproducible inference from raw inputs.

**Objectives**

| **Objective** | **Explanation** |
|----------------|-----------------|
| **Create a runnable inference pipeline** | Allow predictions from a patient input sequence. |
| **Demonstrate reproducibility** | Load trained model, preprocessing scaler, and padding config exactly as in training. |
| **Expose predictions via CLI or API** | Command-line (`python unified_inference.py`) or FastAPI endpoint (`/predict`). |
| **Prepare for cloud deployment** | Modularise code for future Render or Hugging Face hosting. |

**Phase Progression**
- This mirrors industry progression: start lightweight, then containerise and scale.

| **Phase** | **Goal** | **Deliverable** |
|-----------|----------|-----------------|
| Phase 7A – Deployment-Lite | Local inference demo | CLI + optional FastAPI endpoint |
| Phase 7B – Cloud Deployment | Public showcase | Render-hosted or Hugging Face app with same logic |

**Why Deployment-Lite first**  
- Low engineering cost, high scientific payoff.  
- Demonstrates reproducible inference and full pipeline understanding without full cloud infra.  
- Enables immediate demo and reviewer validation.  
- Serves as the canonical step before full cloud/CI deployment (Phase 7B).

#### 7. Why This Matters
**Skills Demonstrated**
- End-to-end ML fluency: data → model → inference → service.
- MLOps awareness: modular architecture, dependency control, and deployment flow.
- Software engineering principles: environment reproducibility, code versioning, modular inference logic.
- Communication & documentation: clear usage, API specification, and logging.
**Why Recruiters Value It**
- A deployment stage proves:
	-	You can operationalise machine learning, not just run experiments.
	-	You understand production constraints (inputs, outputs, reproducibility).
	-	You can communicate technical systems in a structured, maintainable way.
- It separates research-only candidates from production-ready engineers.

### Inference vs Evaluation: Key Piepline Differences
#### Overview
- When moving from model evaluation to real-world deployment, the pipeline must change.  
- Evaluation phases (5 and 6) are designed to judge the model → requiring ground-truth labels, threshold tuning, calibration, and additional post-processing to maximise metric quality.  
- Deployment inference (Phase 7A), however, is designed to use the model → taking any dataset, running it through the trained model, and returning predictions exactly as the model outputs them.  
- Because of this fundamental difference in purpose:
  - Certain steps used during evaluation are not appropriate for deployment.
  - Deployment must remain **model-faithful**, **data-agnostic**, and **free from metric-specific adjustments**.
  - Evaluation optimises metrics; inference preserves predictions.
#### Differences Between Evaluation and Inference
1. **Evaluation Pipelines (Phase 5 / Phase 6)**
  - **Purpose:** Measure model performance on a test set.
  - **Requirements:**
    - Ground truth labels (binary targets) to compute metrics like accuracy, F1, ROC-AUC.
    - Regression calibration or log-space transformations to reduce systematic bias.
    - Post-processing to ensure fair and comparable metrics across models.
  - **Goal:** Optimise metric evaluation, not modify the model’s raw predictions.
2. **Deployment / Inference Pipeline (Phase 7A – Deployment-Lite)**
  - **Purpose:** Generate predictions for any dataset, without computing metrics.
  - **Requirements:**
    - No ground truth needed; predictions are collected as-is.
    - Binary target recreation, threshold tuning, or log-space calibration is unnecessary.
    - Outputs are the **raw model predictions**, ready for downstream use (CSV, API, dashboards).
#### Summary 
1. **Key Point**
  - Previous evaluation “improvements” (binary target recreation, threshold tuning, log-space calibration) exist **only to optimise evaluation metrics**.
  - Including them in deployment would **alter raw predictions**, misrepresenting the model’s outputs.
  - Deployment should faithfully report **exact model outputs**, unmodified by evaluation-specific post-processing.
2. **Conclusion:**  
  - For deployment inference:
    - Skip metric-based post-processing.
    - Preserve raw outputs from the trained model.
    - Ensure outputs are consistent and reproducible for any input dataset.

### Final Inference Script Decisions 
#### Overview 
- This section explains why the final pipeline is structured the way it is, including the design decisions, model-specific considerations, and safeguards added during development. 
- It reflects all changes made during debugging and refinement of the inference pipeline.

#### 1. Inference Modes
**Batch Inference (Default)**
- **Design**
  - Primary mode of operation that runs immediately when the script is executed.  
  - Performs the full inference pipeline end-to-end:
    - Loads LightGBM models and generates predictions for **all test patients**.
    - Loads the TCN architecture + weights and computes full-batch predictions (entire testset evaluated as one tensor).
    - Saves outputs to CSV for both models.
    - Computes interpretability summaries for both models (LightGBM SHAP top-10, TCN saliency top-10).
    - Saves a unified interpretability summary CSV.
  - Functionally mirrors the evaluation (Phase 5) and interpretability (Phase 6) logic, but consolidated into a single deployment-ready script.
- **Rationale**
  - Ensures reproducibility and consistent data ordering for both models.
  - Guarantees interpretability is computed using the exact same predictions produced during batch inference.
  - Suitable for deployment-lite pipelines where the full dataset is evaluated at once.
**Single-Patient Inference (CLI)**  
- **Design**
  - A small CLI added after batch inference completes.
  - Does not modify tensors, data ordering, or batch outputs.
  - Workflow:
    1. User inputs a patient ID.
    2. Input is type-checked and validated against `test_ids`.
    3. Script prints LightGBM and TCN predictions for that patient cleanly in the terminal.
- **Rationale**
  - Mimics typical real-world inference workflows (e.g., single-patient lookup interfaces).
  - Useful for demonstration, debugging, and quick verification of individual patients.
  - Keeps deployment flexible while preserving the integrity of batch mode.

#### 2. Model Loading and Architecture Decisions
**LightGBM**
- **How it works**
  - Each LightGBM `.pkl` file contains the **entire model**: structure, hyperparameters, trained trees, and metadata.
  - Loading with `joblib.load()` restores a fully functional model without additional configuration.
  - Inference uses:
    - `predict_proba()` for classification targets.
    - `predict()` for regression targets.
  - Input features come directly from the test DataFrame using the exact `feature_cols` used during training.
- **Rationale**
  - LightGBM models are self-contained and require no manual architecture reconstruction.
  - Ensures perfect consistency with the training pipeline.
  - No activation or inverse transformation is needed, since LightGBM produces final output values directly.
  - Only requirement is correct DataFrame feature ordering.
**TCN (Temporal Convolutional Network)**
- **How it works**
  - PyTorch `.pt` files only contain **weights**, not the model architecture.
  - The architecture must be rebuilt using `model_architecture` parameters from the config JSON.
  - Temporal feature ordering is reconstructed using the `padding_config` to guarantee the tensor matches training order.
  - Post-processing is required:
    - **Classification:** logits → probabilities using `torch.sigmoid()`.
    - **Regression:** apply inverse transform using `np.expm1()` because the model was trained on log-transformed targets.
- **Rationale**
  - PyTorch requires explicit model reconstruction to match training architecture exactly.
  - Temporal models depend on strict, stable feature ordering → padding config must be loaded.
  - Manual activation and inverse transformations restore outputs to the correct human-interpretable scale.

#### 3. Binary Target Recreation — Not Needed for Inference
**Rationale**
- In this inference pipeline, we only generate predictions, not evaluation metrics.  
- True binary labels are only required for computing metrics (accuracy, F1, ROC-AUC).  
- Since no metric calculations are performed, recreating binary targets from the CSV is unnecessary.  
- This simplifies the pipeline and ensures it works with any input dataset without requiring outcome columns.
**For deployment-level inference**
- We never check whether predictions are right or wrong.  
- We only pass patient features into the model and collect outputs.  
- Binary target columns would be unused extra data.  
- Creating them would falsely imply that metrics or calibration steps follow.
**Conclusion**  
- Including binary target recreation in the inference pipeline adds unnecessary computation and complexity without any benefit.

#### 4. Negative Regression Outputs — Clipping
**Problem**  
- LightGBM produced a negative regression value for the first patient.  
- This output corresponds to **predicted % time high NEWS2**, which cannot logically be < 0.  
- Even if rare, negative values are unacceptable in any deployment or clinical-facing setting.
**Fix Implemented**  
- Clip regression values for LightGBM and TCN at 0 using: 
  - LightGBM:`df_lightgbm["y_pred_reg"] = df_lightgbm["y_pred_reg"].clip(lower=0)` 
  - TCN: `y_pred = np.clip(y_pred, a_min=0, a_max=None)`
- Both implementations guarantee that any negative prediction becomes 0, enforcing validity, preserving meaning and guaranteeing deployment-safe outputs.
**Why clipping is applied only to regression outputs**
-	Classification outputs are passed through sigmoid, so they are mathematically guaranteed to lie in the range 0–1.
-	Regression heads do not pass through a bounded activation; therefore negative values can appear after model prediction (LightGBM) or inverse transform (expm1) in TCN.
- Thus regression is the only output type requiring safety clipping.

#### 5. Interpretability Decisions
**LightGBM - SHAP**
- **How it works**
  - SHAP is the standard interpretability method for tree-based models.
  - It computes feature-level contributions for each prediction using the same feature ordering as the LightGBM model.
  - SHAP values for the 15 test patients are obtained via TreeExplainer.
- **Decision**
  - Compute **mean absolute SHAP values** for each target across all test patients.
  - Extract **top-10 features per target** only.
  - Omit bar-plot visualisations to keep the deployment pipeline lightweight.
  - Mirrors the Phase 6 interpretability workflow but strips down to essential numeric outputs.
- **Rationale**
  - SHAP provides rigorous, model-accurate attributions.
  - Restricting to top-10 features maintains interpretability while keeping inference fast and outputs compact.
  - Eliminating plots prevents clutter and unnecessary dependencies in deployment.
**TCN — Gradient × Input Saliency**
- **How it works**
  - Gradient × Input measures how sensitive the model output is to changes in each input timestep and feature
  - The multiplication by the input value incorporates the actual activation level of the feature.
- **Decision**
  - Compute **mean absolute saliency** across all patients, timesteps and features.
  - Extract **top-10 features per target head**.
  - Do not compute: full temporal heatmaps, per-patient saliency matrices, full saliency plots from Phase 6.
- **Rationale**
  - Matches the interpretability method used in the full pipeline while providing a streamlined, numeric-only summary.
  - Reduces computational overhead and output size.
  - Represents the most informative yet lightweight subset of Phase 6 interpretability, the aggregated top-10 features.

#### 5. Unified Results for Deployment
**Decision**  
- Final outputs:
  - `lightgbm_inference_outputs.csv`
  - `tcn_inference_outputs.csv`
  - `top10_features_summary.csv`
- One output file per model family to maintain separation and clarity.  
- One merged interpretability file to provide a concise, deployment-ready summary.  
- Avoid storage-heavy artefacts (heatmaps, large matrices, per-patient timestep arrays).
**Rationale**  
- Aligns with production expectations for pipelines, dashboards, and automated ingestion systems.  
- Ensures outputs remain lightweight, deterministic, and compact.  
- Simplifies downstream use without sacrificing model transparency or traceability.

---

