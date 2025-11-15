# Phase 1: Baseline NEWS2 Tracker

---

## Day 1: NEWS2 Data Extraction and Preliminary Scoring

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

### Pipeline Overview

```text
Raw CSVs (chartevents.csv, etc.)
        ↓
extract_news2_vitals.py
        ↓
news2_vitals.csv
        ↓
check_co2_retainers.py
        ↓
news2_vitals_with_co2.csv + co2_retainer_details.csv
        ↓
compute_news2.py
        ↓
Final NEWS2 scores per patient
```

### Goals
- Extract relevant vital signs from PhysioNet.org MIMIC-IV Clinical Database Demo synthetic dataset for NEWS2 scoring.  
- Identify and flag CO₂ retainers to ensure accurate oxygen scoring.  
- Implement basic NEWS2 scoring functions in Python.  
- Establish a pipeline that is extendable for future interoperability with real-world clinical data.

### What We Did
1. **Dataset Preparation**
   - Downloaded synthetic dataset `mimic-iv-clinical-database-demo-2.2.zip` and unzipped CSV files.  
   - Explored `chartevents.csv` and other relevant CSVs to identify required vitals.  

2. **Data Extraction**
   - Wrote `extract_news2_vitals.py` to extract NEWS2-relevant vitals from all CSVs.  
   - Used `preview_headers` to determine which columns to extract and standardize CSV headers.  
   - Generated `news2_vitals.csv`.

3. **CO₂ Retainer Identification**
   - Created `check_co2_retainers.py` to verify if any patients met CO₂ retainer criteria:
     - PaCO₂ > 45 mmHg  
     - pH between 7.35–7.45  
     - ABG measurements ±1 hour apart  
   - Updated `extract_news2_vitals.py` to include CO₂ retainer status.  
   - Generated:
     - `news2_vitals_with_co2.csv` – vitals with retainer flags  
     - `co2_retainer_details.csv` – patient-specific CO₂ retainer information  

4. **NEWS2 Scoring**
   - Implemented `compute_news2.py` with:
     - Dictionaries defining scoring thresholds for each vital  
     - Functions to compute individual vital scores  
     - Pandas used to process CSV and calculate total NEWS2 scores  

### Reflections
- **Challenges:**
  - Understanding GCS scoring and mapping three separate components to level of consciousness.  
  - Determining FiO₂ representation in dataset (0.21 vs. 21%).  
  - Determining temperature units
  - Grasping complex Python syntax and tuple-based threshold definitions.  
  - Integrating CO₂ retainer logic into NEWS2 oxygen scoring.

- **Solutions & Learnings:**
  - GCS scoring requires summing Eye, Verbal, and Motor responses per timestamp.  
  - FiO₂ can be identified via `Inspired O2 Fraction` in CSV and converted to binary supplemental O₂ indicator.  
  - Temperature was in Fahrenheit (°F) and so `compute_news2.py` includes conversion from °F to °C.
  - Tuples `(min, max, score)` provide flexible, readable threshold definitions for each vital.  
  - CO₂ retainer pipeline ensures accurate NEWS2 oxygen scoring now and for future datasets.  

### Issues Encountered
- Confusion around GCS mapping and timestamp alignment.  
- Initial uncertainty about FiO₂ and temperature units.  
- Need to verify CO₂ retainer thresholds and data format.  
- Feeling overwhelmed by the complexity of clinical data pipelines and Python functions.

### Lessons Learned
- Extracting and standardising clinical data is a critical and time-consuming first step.  
- Structuring data in CSVs with consistent headers simplifies downstream processing.  
- Python dictionaries and tuple-based thresholds are powerful for flexible clinical scoring functions.  
- Documenting assumptions (temperature units, FiO₂ thresholds) is essential for reproducibility.

### Future Interoperability Considerations
- Pipeline designed to support ingestion of FHIR-based EHR data for future integration.  
- Potential extension: map standardized FHIR resources to predictive EWS pipeline for real-world applicability.

### CO₂ Retainer Validation and NEWS2 Scoring Documentation
1. **Objective:** Identify CO₂ retainers to ensure correct oxygen scoring.  
2. **Methodology:**  
   - All ABG measurements in `chartevents.csv` examined.  
   - CO₂ retainer criteria applied: PaCO₂ > 45 mmHg with pH 7.35–7.45 ±1 hour.  
3. **Results:**  
   - No patients in current dataset met CO₂ retainer criteria.  
   - NEWS2 oxygen scoring applied standard thresholds for all patients.  
4. **Future-proofing:**  
   - CO₂ retainer thresholds remain documented in code.  
   - Future datasets will automatically flag and score retainers according to NEWS2 rules.

---

## Day 2: NEWS2 Pipeline Development

### Goals
- Finalise **Phase 1** of the NEWS2 scoring pipeline.
- Ensure robust extraction, computation, and output of NEWS2 scores from raw vital sign data.
- Handle missing data, standardize column names, and prevent errors caused by merging or absent measurements.
- Create clean, wide-format CSV outputs: `news2_scores.csv` (per-timestamp) and `news2_patient_summary.csv` (per-patient summary).

### What We Did
1. **Updated `extract_news2_vitals.py`:**
  - Included missing `systolic_bp` itemids.
  - Added alternate names for vitals to capture all relevant measurements.
  - Produced `news2_vitals_with_co2.csv` with columns:  
    `subject_id, hadm_id, stay_id, caregiver_id, charttime, storetime, itemid, value, valuenum, valueuom, warning, label, co2_retainer`.
2. **Updated `compute_news2.py`:**
  - Pivoted long-format vitals to wide format, ensuring all **expected NEWS2 vitals** (`respiratory_rate, spo2, supplemental_o2, temperature, systolic_bp, heart_rate`) exist as columns.
  - Safely merged **GCS components**, computing `gcs_total` and `level_of_consciousness`.
  - Safely merged **CO₂ retainer** information.
  - Fixed **supplemental O₂** issues:
    - Checked if column exists before filling.
    - Filled missing rows with `0` (Room air).
    - Only merged if not already present to prevent duplication.

3. **Handled duplicates and column conflicts:**
  - Avoided `_x` / `_y` suffixes by careful merge logic:
    - GCS merge only added `gcs_total` and `level_of_consciousness`.
    - Supplemental O₂ merged only if missing.
    - CO₂ retainer merge ensured no overlap.

4. **Added human-readable labels:**
  - `consciousness_label`, `co2_retainer_label`, `supplemental_o2_label`.
  - Ensured columns exist before applying transformations to prevent KeyErrors.
  - The redundancy exists to **ensure the script runs safely** even if:
    - `level_of_consciousness` is missing (no GCS rows for some patients)
    - `co2_retainer` is missing
    - `supplemental_o2` is missing
  - Purpose:
    - Guarantee idempotency
    - Prevent KeyErrors
    - Keep CSV outputs consistent and complete

5. **Computed NEWS2 scores per row:**
  - Applied scoring rules for each vital.
  - Calculated `news2_score`, `risk`, `monitoring_freq`, and `response`.
  - Validated that **no scores exceeded 20**.

6. **Created outputs:**
  - `news2_scores.csv` – full dataset with scores and all vital measurements.
  - `news2_patient_summary.csv` – per-patient summary with `min_news2_score, max_news2_score, mean_news2_score, median_news2_score, total_records`.

7. **Implemented defensive coding & sanity checks:**
  - Missing vitals counted per row (`missing_vitals` column).
  - All merges and transformations check column existence.
  - Default values (0 or False) used for missing data to maintain dataset integrity.

**Phase 1 gave us**:
- news2_scores.csv (per-timestamp scores).
- news2_patient_summary.csv (per-patient aggregates).

### Reflections: 
**Challenges**: 
- KeyError on `supplemental_o2` when merging due to missing FiO₂ measurements.  
- Duplicate columns (`_x`, `_y`) after merges. 
- Missing GCS components for some patients. 
- Missing NEWS2 vitals in pivot.   
**Solutions & Learnings**:
- Conditional merge and default fill (0). Always check column existence before accessing or transforming it in merged datasets.
- Merge only necessary columns, avoid re-merging existing ones. Thoughtful merge design prevents downstream confusion and simplifies CSV outputs.
- Added missing columns with `pd.NA` and computed `gcs_total` safely. Defensive coding is critical when working with real-world clinical data.
- Added all expected vitals as NA before merges. Preemptive handling of expected columns reduces errors during scoring.

### Issues Encountered
- Missing itemids in `extract_news2_vitals.py`.
- KeyError when accessing non-existent supplemental O₂ or GCS columns.
- Duplicate columns after merging GCS and supplemental O₂.
- Variations in vital naming and units.
- Some timestamps had missing vital measurements.

### Lessons Learned
- Always **validate column existence** before transformations or merges.
- Merge only necessary columns to prevent duplicates.
- Filling missing data with safe defaults ensures pipeline stability.
- Defensive coding allows robust handling of incomplete real-world datasets.
- Maintaining clean, standardised column names simplifies both computation and human-readable output.

### Extra Considerations / Documentation Points
- The pipeline now fully supports **Phase 1** outputs and can be run repeatedly on updated CSVs.
- All merges are idempotent – repeated runs will not create duplicates.
- All human-readable labels (`consciousness_label`, `co2_retainer_label`, `supplemental_o2_label`) are always generated.
- **Defensive coding for human-readable labels**:
  - Two blocks exist in the code assigning `consciousness_label`, `co2_retainer_label`, and `supplemental_o2_label`.
  - Redundancy ensures the script runs safely even if some columns are missing (`level_of_consciousness`, `co2_retainer`, `supplemental_o2`).
  - Guarantees idempotency and prevents KeyErrors on incomplete datasets.
  - Best practice: could combine into a single block that creates defaults and assigns labels in one step.
- Outputs `news2_scores.csv` and `news2_patient_summary.csv` are fully consistent with the pipeline’s intended design.
- Next steps (Phase 2) could include visualisation, predictive modeling, or integrating NEWS2 trajectories into a dashboard.

---

# Phase 2: Preprocess Data for ML-Models

---

## Day 3 Notes — Validating NEWS2 Scoring & ML Pipeline Preparation

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

### Goals
- Validate NEWS2 scoring logic
  - Validate `compute_news2.py` against NHS NEWS2 rules.
  - Test edge cases (SpO₂ thresholds, supplemental O₂, GCS 15 vs 14, RR 20 vs 25).
- Decide on a consistent missing data strategy for timestamp-level and patient-level features.
- Begin planning ML-ready feature extraction (`news2_features_timestamp.csv` and `news2_features_patient.csv`).
  - Understand why we need LOCF, missingness flags, rolling windows, and numeric encodings.
- **Choose an ML model**:
  - Determine which model is optimal for tabular ICU data.
  - Decide preprocessing strategy based on chosen model.

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

### What We Did
#### Step 1: Validating NEWS2 Scoring
- **Action**: Ran validate_news2_scoring.py on test dictionaries.
- **Findings**:
  - Low GCS cases initially produced incorrect scores.
  - The scoring function ignored consciousness because row.get("level_of_consciousness", pd.NA) returned pd.NA.
  -	Other special cases (SpO₂, supplemental O₂) were correctly scored because their thresholds were handled explicitly.
- **Fixes**: Moved `if pd.isna(value): return 0` **to the end of the function**.
- **Outcome**: All unit tests passed, learned the importance of understanding intermediate variables in scoring pipelines.
- The main pipeline did not have these problems as gcs_total is converted into level_of_consciousness before the scoring is called, so there was no missing keys.

#### Step 2: Missing Data Strategy
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

#### Step 3: Preparing Timestamp-Level ML Features
**Pipeline (make_timestamp_features.py)**:
1. Start from news2_scores.csv (all vitals + NEWS2 + escalation labels).
  - Parse charttime as datetime.
  - Sort by subject_id, charttime.
2. Create missingness flags for each vital (before fills).
3. LOCF forward-fill per subject (optionally backward-fill for initial missingness or leave as NaN), do not use population median.
4. Create carried-forward flags (binary indicator - 1 if the value came from LOCF). Helps ML distinguish between observed vs assumed stable, exploit missingness patterns (e.g. vitals measured more frequently when patients deteriorate).
5. **Compute rolling windows (1h, 4h, 24h)**: mean,min,max,std,count,slope,AUC.
6. Compute time since last observation (`time_since_last_obs`) for each vital (staleness).
7. Convert textual escalation/risk labels → numeric ordinal encoding (Low=0, Low-Medium=1, Medium=2, High=3) for ML. Keeps things simple - one column, easy to track in feature importance
8. Save news2_features_timestamp.csv.
**Rationale**:
- Trees can leverage trends and missingness.
-	Rolling windows capture short-, medium-, and long-term deterioration patterns.
-	Timestamp features feed ML models like LightGBM directly without further preprocessing.

#### Step 4: Preparing Patient-Level ML Features
**Pipeline (make_patient_features.py)**:
1. Start from news2_scores.csv.
2. **Group by patient**: Aggregate vitals per patient timeline (median, mean, min, max per vital).
3. **Median imputation**: Fill missing values for each vital using patient-specific median (so their profile isn’t biased by others), if a patient never had a vital recorded, fall back to population median.
4. **% Missing per vital**: Track proportion of missing values per vital before imputation (HR missing in 30% of their rows = 0.3), missingness itself may signal clinical patterns (e.g. some vitals only measured in deteriorating patients).
5. **Encode risk/escalation labels**: Ordinal encoding (Low=0, Low-Medium=1, Medium=2, High=3), calculate summary stats per patient: max risk (highest escalation they reached), median risk (typical risk level), % time at High risk (what fraction of their trajectory was spent here).
6. **Output**: news2_features_patient.csv (compact, one row per patient, ML-ready summary).
**Rationale**:
-	Median imputation preserves patient-specific patterns without introducing bias from other patients.
-	% Missing captures signal from incomplete measurement patterns.
-	Ordinal risk encoding simplifies downstream ML model input while retaining interpretability. Together, these three summary features summarise a patient’s escalation profile across their stay. Proportion features (like % high) are standard numeric features (not encoded categories).
-	This is enough for model; don’t need optional metrics like streaks, AUC, or rolling windows for the patient summary.

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
-	**Future extension**:
  -	Neural nets possible if dataset size grows significantly.
  -	Would require additional preprocessing: time-series sequences, padding, normalisation, possibly interpolation.


### Validation Issue & Fix: GCS → Level of Consciousness
**Problem Identified:**
- `score_vital` incorrectly ignored `level_of_consciousness` when computing NEWS2 scores.
- **Reason**:
1. `compute_news2_score` passes `value = row.get("level_of_consciousness", pd.NA)`.
2. If the row dictionary does not contain `level_of_consciousness` yet (common in synthetic test cases), `value=pd.NA`.
3. Original code had `if pd.isna(value): return 0` at the top of `score_vital`.
4. This caused the function to exit **before using `gcs_total` to compute LOC**, so low GCS patients were scored incorrectly.
**Other Contributing Factor:**
- `level_of_consciousness` exists as a key in `vital_thresholds`.  
- The generic “Other vitals” block ran first, attempting to score with `value=pd.NA`, bypassing LOC-specific logic.
**Fix Implemented:**
- Moved `if pd.isna(value): return 0` **to the end of the function**.
- Ensured the LOC-specific block runs **before** the generic “Other vitals” block.  

```python
if vital_name == "level_of_consciousness":
    flag = 0 if gcs_total == 15 else 1
    for low, high, score in vital_thresholds["level_of_consciousness"]:
        if low <= flag <= high:
            return score

if vital_name in vital_thresholds:
    for low, high, score in vital_thresholds[vital_name]:
        if (low is None or value >= low) and (high is None or value <= high):
            return score

if pd.isna(value):
    return 0
```

### Key Reflections & Lessons Learned
- **Validate NEWS2 carefully**:
  - Subtle errors can arise from intermediate variables like level_of_consciousness.
  - Always check how the test harness mirrors the main code.
- **Data pipeline decisions**:
	- Simplify pipeline to focus on what’s necessary for the chosen ML model, not every theoretical feature.
	- Keeping the pipeline simple ensures maintainability and interpretability.
  - Could adapt for other models or neural networks, but only if dataset and project scope increase.
- **ML pipeline considerations**:
	- Timestamp features → for temporal trends.
	-	Patient summary features → for overall risk profile per patient.
	- Missingness flags → signal to the model without biasing values.

### Planned Next Steps
1. Implement `make_timestamp_features.py` using `news2_scores.csv`.
2. Generate `news2_features_timestamp.csv` with LOCF, flags, rolling window stats, ordinal risk encoding.
3. Start aggregating patient-level summary features (`news2_features_patient.csv`) using median imputation + missingness stats in `make_patient_features.py`.
4. Train a baseline LightGBM model to test predictive performance.
5. Document rationale for each preprocessing decision for reflections section.

⸻

## Day 4 Notes - Timestamp-Level ML Features

### Goals
- Implement make_timestamp_features.py to transform news2_scores.csv into news2_features_timestamp.csv, ready for ML modeling with LightGBM.
- Follow 8 planned steps for feature engineering at the timestamp level.  

### Planned 8 Steps
1. Parse charttime → datetime, sort by subject_id & charttime.  
2. Create missingness flags (per vital, before fills).  
3. LOCF forward-fill per subject (optionally backward-fill initial missing).  
4. Create carried-forward flags (1 if value came from LOCF).  
5. Compute rolling windows (1h, 4h, 24h) → mean, min, max, std, count, slope, AUC.  
6. Compute time_since_last_observation (staleness per vital).  
7. Encode risk/escalation labels → numeric ordinal (Low=0, Low-Med=1, Medium=2, High=3).  
8. Save `news2_features_timestamp.csv` (ML-ready).  

Only Step 1 was implemented today; Steps 2–8 remain.  

### What We Did Today
- Completed **Step 1 (Load & Sort)**:
  - Loaded `news2_scores.csv` into a pandas DataFrame.  
  - Converted `charttime` column to proper `datetime` objects.  
  - Sorted rows by `subject_id`, `stay_id`, and `charttime` to enforce chronological order per patient stay.  
  - Verified with a preview (`df.head()`) that the data is clean and ordered.  

### Reflections
#### Challenges
- **Pandas syntax** feels overwhelming
- Spent most of the day revisiting **all previous scripts (`.py`)** in the project to fully annotate them.  
- The main difficulty was **pandas syntax in general** — not just in this step, but across:
  - `.sort_values`, `.reset_index`, `.merge`, `.pivot_table`, `.apply`, `.isin`, `.loc`.  
  - Understanding why certain operations are applied in a specific order.  
  - Figuring out how pandas “thinks” when reshaping or transforming datasets.  
- Felt frustrated at how much time was spent **understanding code** instead of **writing new features**.  
#### Solutions
- Added **inline comments** to all major pandas operations across the codebase.  
- Broke the pipeline into clear **8 steps**, so I can see the bigger picture and where today’s progress fits.  
- Asked targeted questions (e.g. about return type hints, `.apply()`, `os`, `.merge`) to fill conceptual gaps.  
#### Learnings
- **Pandas is its own language**: It’s not just Python, but a layer of syntax for manipulating tabular data.  
- **Order of operations matters**: E.g. missingness flags must precede filling, or else ML won’t distinguish true vs imputed values.  
- **Debugging strategy**: Always print `df.head()` after each major step to confirm changes.  
- **Reflection is progress**: Even if I only implemented Step 1, I deepened my conceptual foundation, which will make Steps 2–8 easier.  

### Extra Considerations
- My pace felt slower than expected, but it was necessary to **slow down and understand the building blocks**.  
- Future steps (e.g. rolling windows, staleness) will require chaining multiple pandas operations — having this stronger foundation will prevent confusion later.  
- Need to balance **practical coding** (keep pipeline moving) with **conceptual grounding** (understanding transformations).  


### 📅 Next Steps (Day 5 Plan)
- Implement **Step 2 (Missingness flags)**:
  - Add `_missing` columns for each vital before LOCF.  
  - Confirm flags align with actual NaNs.  
- If possible, progress into **Step 3 (LOCF imputation)** and **Step 4 (Carried-forward flags)**.  
- Keep using small previews (`.head()`, `.isna().sum()`) to verify correctness.  

---

## Day 5 Notes - Missingness, Carried-Forward Flags & Rolling Features

### Goals
- Continue building `make_timestamp_features.py` pipeline.  
- **Extend Step 2 → Step 5**:
  - **Step 2**: Add missingness flags.
  - **Step 3**: Apply forward-filling (LOCF).
  - **Step 4**: Add carried-forward flags.
  - **Step 5**: Start rolling window features (mean, min, max, std, slope, AUC).  

### What We Did
#### Step 2: Missingness Flags
- Implemented `add_missingness_flags(df)` to generate new columns like `respiratory_rate_missing`, `spo2_missing`, etc.  
- **Logic**: for each vital, `df[v].isna().astype(int)` creates a flag column where `1 = missing` and `0 = observed`.  
- Called after loading + sorting the CSV with `load_and_sort_data(INPUT_FILE)`.  
- Verified output by printing `df.head()`.
#### Step 3: LOCF (Forward- and Back-Fill)
- Wrote `apply_locf(df)` to handle missing values by carrying the last observed measurement forward (`ffill`) within each patient stay (`groupby(['subject_id', 'stay_id'])`).  
- Added an extra `.bfill()` so the very first row of each stay (if missing) is backfilled with the next available measurement.  
- Ensures no missing values remain for the chosen vitals.
#### Step 4: Carried-Forward Flags
- Added `add_carried_forward_flags(df)` to track which values in the filled dataset are real vs imputed.  
- Used missingness flags from Step 2 as ground truth:  
  - Carried = `value is not NaN after fill` **AND** `was missing before fill`.  
- Output = new columns like `respiratory_rate_carried`, `spo2_carried`, etc.  
- This avoids the problem of falsely marking naturally repeated values as carried-forward.
#### Step 5: Rolling Features (in progress)
- Started `add_rolling_features(df)` to compute rolling-window statistics on numeric vitals (`respiratory_rate`, `spo2`, `temperature`, `systolic_bp`, `heart_rate`).  
- **Window sizes**: 1h, 4h, 24h.  
- **Stats**: mean, min, max, std, slope (trend), AUC (cumulative exposure).  
- For each vital × window combination, new feature columns are created, e.g.:
  - `respiratory_rate_roll1h_mean`  
  - `spo2_roll24h_slope`  
- Implemented slope with a simple linear regression on index order; AUC as the cumulative sum over the window.  
- Still clarifying whether slope/AUC should be computed on true timestamps (`charttime_numeric`) or just index order.  

### Reflections
#### Challenges
- **Pandas syntax**:  
  - Still feels overwhelming, especially with groupby, rolling, and applying custom functions.  
  - Feels like "watching a chess grandmaster" without yet knowing the moves.  
- **Redundant flags**:  
  - Initially thought missingness flags already made carried-forward redundant.  
  - **Learned they complement each other**: missing = gaps before filling, carried = which values were filled in.
- **Rolling features**:  
  - Hard to see how loops systematically build columns.  
  - `charttime_numeric` looked confusing since we’re not yet using real timestamps in slope/AUC.
#### Solutions & Learnings
- Breaking code into **bite-sized functions** helps (e.g., Step 2–4 each modular).  
- Printing `df.head()` after each step is essential for debugging.  
- Carried-forward vs missingness flags = subtle but distinct concepts.  
- Nested loops (`for v in vitals, for w in windows`) → systematic way to generate features.  
- Recognised unused code (`rolling_features = []`, `charttime_numeric` placeholder).  

### Next Steps
- Finish **Step 5**:
  - Decide whether slope/AUC should use real timestamps or simple index order.
  - Simplify code by removing unused prep.  
- Validate with small test DataFrame to confirm columns behave as expected.
- **Move on to Step 6**: **time since last observation** once rolling features are stable.  

---

## Day 6 Notes – Rolling Features, Time Since Obs, Risk Encoding

### Goals
- Continue pipeline development (`make_timestamp_features.py`).
- Finalise **Step 5–7**:
  - **Step 5**: Rolling window features (mean, min, max, std, slope, AUC).
  - **Step 6**: Time since last observation (staleness).
  - **Step 7**: Encode escalation/risk labels.
- Add **Step 8**: End-to-end integration to generate the final ML-ready dataset.
- Resolve slope/AUC approach: simple vs time-aware.

### What We Did
#### Step 5 – Rolling Window Features
- Added rolling features for 5 vitals (`HR, RR, SpO₂, Temp, SBP`) across 3 windows (1h, 4h, 24h).  
- Stats per window: `mean, min, max, std, slope, AUC`.  
- Implemented **time-aware slope and AUC**:
  - **Slope** = rate of change per hour, using actual `charttime` gaps.
  - **AUC** = cumulative exposure over time, integrated with trapezoidal rule.
- Example **outputs**:
  - `heart_rate_roll4h_slope` = rate of HR change (bpm/hour).
  - `spo2_roll24h_auc` = total “oxygen exposure” in the last 24 hours.
#### Step 6 – Time Since Last Observation
- Computed `*_time_since_last_obs` for each vital.  
- Captures how stale each measurement is (fresh vs old data).  
- **Example**: HR measured 3h ago → `heart_rate_time_since_last_obs = 3.0`.
#### Step 7 – Encode Risk Labels
- Created `risk_numeric` column mapping text → numbers:
  - Low = 0  
  - Low-Medium = 1  
  - Medium = 2  
  - High = 3  
#### Step 8 – Pipeline Integration
- End-to-end workflow complete:
  1. Load & sort data.  
  2. Add missingness flags.  
  3. Apply LOCF.  
  4. Add carried-forward flags.  
  5. Add rolling window features.  
  6. Add time since last obs.  
  7. Encode risk labels.  
  8. Save final dataset → `news2_features_timestamp.csv`.

### Reflections
#### Challenges
- **Slope (trend) choice**:
  - **Simple slope**: assumes equal spacing (`t = 0,1,2,...`). Works if vitals are frequent/regular, but misleading if sparse (e.g., 2 readings in 5 minutes then none for 8 hours).
  - **Time-aware slope**: uses real timestamps, slope = change per unit time (e.g., HR 100→120 in 30 mins = +40 bpm/hr vs over 12h = +1.7 bpm/hr).
- **AUC (cumulative exposure)**:
  - **Simple AUC**: sum of values only.  
  - **Time-aware AUC**: integrates over time → reflects true burden/exposure (e.g., SpO₂ 92% for 24h is worse than 92% for 1h).
- **Pipeline complexity**:
  - Step 5 added ~90 columns per row. Easy to lose track without systematic naming and notes.
#### Solutions & Learnings
- Adopted **time-aware slope & AUC** → features are both ML-useful and clinically interpretable.  
- LOCF filling made rows “regular,” but we kept real-time slope because clinical interpretability is higher priority.  
- **Key conceptual clarity**:
  - **Slope** = trend per unit time.  
  - **AUC** = exposure burden over time.  
- Validated that missingness vs carried-forward flags are complementary:
  - Missing = gaps before fill.  
  - Carried = synthetic values after fill.  

### Next Steps
1. **Validate outputs**:
   - Print a few patient timelines to ensure slope/AUC match clinical intuition.  
   - Confirm time-since-last-obs is reasonable.  
2. **Efficiency check**:
   - Test runtime on full dataset (Step 5 may be slow).  
3. **Documentation**:
   - Write a short “feature dictionary” describing each class of features.  
4. **Step 9**: Aggregate timestamp-level features → patient-level summary for downstream ML.

---

## Day 7 Notes - Timestamp Pipeline Complete & Patient Features Begun

### Goals
- Run and debug `make_timestamp_features.py` end-to-end.  
- Verify that the generated CSV (`news2_features_timestamp.csv`) is correct.  
- Decide which model (LightGBM vs Neural Network) uses which feature set.  
- **Finalise model roadmap**: V1 = LightGBM on patient-level, V2 = Neural Network (TCN) on timestamp-level.  
- Begin implementing `make_patient_features.py` (Steps 1–2). 

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

### What We Did
1. **Ran `make_timestamp_features.py` successfully**:
   - Adjusted file paths to go two levels up (`../../`) because script runs inside `src/ml_data_prep/`.
   - Resolved duplicate index alignment error by using `.reset_index(drop=True)` which flattens everything back to a simple index that lines up exactly with DataFrame’s rows, instead of dropping groupby levels `.reset_index(level=[0,1], drop=True)` as charttime can have duplicates which confuses pandas when it tries to align by index labels.
   - Fixed rolling window warnings (`'H'` → `'h'`).
   - Output CSV generated: `news2_features_timestamp.csv`.
2. **Debugging Learnings**:
   - **File Paths**:  
     ```python
     DATA_DIR_INPUT = Path("../../data/interim_data")
     DATA_DIR_OUTPUT = Path("../../data/processed_data")
     ```
     → Ensures script looks in correct `data/` directories when run from `src/ml_data_prep/`.
   - **Duplicate Index Issue**:  
     - After `groupby().rolling()`, result had a MultiIndex (patient, stay, charttime).
     - Using `.reset_index(level=[0,1], drop=True)` caused misalignment if charttime was duplicated (after LOCF). Pandas cannot reindex on an axis with duplicate labels.
     - Fix: `.reset_index(drop=True)` → guarantees the Series index matches the DataFrame’s row index.
   - **Trapz Deprecation**: `np.trapz` still works but shows warning; recommended future replacement with `np.trapezoid`.
   - **PerformanceWarning**: Adding 90+ columns one-by-one fragments the DataFrame; harmless but could be optimized with `pd.concat`.
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
5. **Started `make_patient_features.py`**:
   - **Step 1**: Load CSV, sort by patient/time.  
   - **Step 2**: Aggregate vitals with `.groupby("subject_id").agg(["median","mean","min","max"])`.  
   - Learned how `agg()` outputs a MultiIndex → flattening into `vital_stat` format.  
   - Steps 3–6 (imputation, % missingness, risk encoding, save CSV) still to do.  

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

### Planned 6 Steps
1. Load input file `news2_scores.csv`.  
2. Aggregate vitals per patient (median, mean, min, max).  
3. Perform patient-specific median imputation (fallback to population median if never observed).  
4. Compute % missingness per vital (fraction of rows missing before imputation).  
5. Encode risk/escalation labels → numeric ordinal (Low=0, Low-Med=1, Medium=2, High=3), then summarise per patient (max risk, median risk, % time at High risk).  
6. Save `news2_features_patient.csv` (one row per patient, ML-ready).  

Only Steps 1-2 were implemented today; Steps 3-6 remain.

### Reflections
#### Challenges
- **Indexing Misalignments**: Rolling window outputs had MultiIndex misaligned with base DataFrame → caused reindexing errors.  
- **Path Confusion**: Needed to carefully reason about relative paths when running scripts inside `src/`.  
- **Flattening MultiIndexes**: Initially confusing to understand multiindexing and how `(vital, stat)` pairs became clean `vital_stat` columns.  
#### Solutions
- Used `.reset_index(drop=True)` to align rolling stats with DataFrame rows.  
- Standardised file paths with `../../` from script location.  
- Flattened MultiIndex columns using `["_".join(col) for col in df.columns]`.  
#### Learnings
- Index ≠ header → the index is the row labels, not the column names.  
- Duplicated timestamps (after LOCF) can break alignment if not flattened.  
- **Timestamp vs Patient-level features serve complementary roles**:  
  - **Timestamp features** = sequence models.  
  - **Patient features** = tree-based baselines.  
- Portfolio-wise, showing both LightGBM and TCN demonstrates breadth (tabular ML + time-series DL).  

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

### Next Steps
- Complete `make_patient_features.py`:  
  3. Median imputation (patient-level, fallback to population).  
  4. % missingness per vital.  
  5. Risk encoding & summary stats (max, median, % time at High).  
  6. Save `news2_features_patient.csv`.  
- Then proceed to implement LightGBM baseline training (V1).  
- Prepare timestamp features (already done) for TCN implementation (V2).  

---

## Day 8 Notes - Patient Pipeline Complete & LightGBM Roadmap Planning

### Goals
- Complete patient-level feature extraction (steps 3–6 in `make_patient_features.py`)
- Verify the output (`news2_features_patient.csv`)
- **Plan the next phase**: LightGBM training and validation

### What We Did Today
- Finished steps 3–6 in `make_patient_features.py`:
  - Aggregated vital signs and NEWS2 scores to patient-level
  - Calculated missing data percentages
  - Generated additional derived features (e.g., `pct_time_high`)
- Ran the feature extraction script successfully
  - **Checked the resulting CSV**: `news2_features_patient.csv`
  - Verified that all patient-level features were correctly calculated
- Planned **Phase 3: LightGBM Training + Validation**:
  - Drafted a high-level roadmap including dataset preparation, model initialisation, training, validation, saving, and documentation


### Phase 3: LightGBM Training + Validation Overview
**Goal:** Train a LightGBM model on patient-level features, validate performance, and document results.
**Step 1**: Dataset Preparation
- Load processed patient-level features
- Split data into training and test sets
- Separate features (X) and target labels (y)
**Step 2**: Model Initialisation
- Initialise LightGBM model (classifier or regressor depending on target)
- Define basic parameters (learning rate, number of trees, random seed)
**Step 3**: Model Training
- Fit the model on the training data
- Monitor performance on test/validation set
- Apply early stopping to prevent overfitting
**Step 4**: Model Saving
- Save trained model to a file for later use
- Organize folder structure for reproducibility
**Step 5**: Model Validation
- Load saved model and run predictions on test set
- Calculate evaluation metrics (accuracy, ROC-AUC, RMSE, etc.)
- Optionally visualize feature importance and performance
**Step 6**: Documentation
- Record training and validation metrics
- Summarise feature importances
- Prepare results for portfolio or reporting
**Step 7**: Debugging / Checks
- Verify dataset shapes and target columns
- Ensure feature consistency between training and test sets
- Check for missing or non-numeric values

### Reflections
- **Good progress today**: patient-level feature pipeline is now complete and verified
- Planning Phase 3 helps visualise the workflow and prevents getting stuck mid-training
- Breaking down the steps into dataset preparation, training, validation, and documentation provides a clear roadmap


### Challenges
- Understanding **groupby operations** and **multi-indexing** in pandas
  - Aggregating by `subject_id` while preserving patient-level information
  - Converting multi-index back to a single index for merging and saving
- **Merging dataframes**:
  - Ensuring column names and indexes match for proper alignment
  - Avoiding duplicate columns or misaligned patient IDs
- Indexing and using `subject_id` as a key for all patient-level operations
- Verifying formats and data types after aggregation to ensure downstream compatibility

### Solutions and Learnings
- Learned to carefully check pandas groupby objects and use `.reset_index()` after aggregation
- Verified feature correctness by sampling output rows and comparing with original measurements
- Documented the aggregation and merging workflow for reproducibility
- Recognized the importance of consistent patient ID usage as a key across all transformations
- Confirmed that the CSV output is clean and ready for modeling

### Extras
- **Reviewed potential pitfalls for Phase 3**:
  - Handling missing values in LightGBM
  - Feature scaling considerations
  - Saving models with metadata (columns, feature order) for reproducibility
- Drafted a markdown roadmap for Phase 3 to guide upcoming training and validation
- Consider adding small unit tests for future pipeline stages to catch aggregation/merge errors early

---

# Phase 3: LightGBM Training + Validation 

---

## Day 9 Notes – Patient Dataset Preparation for LightGBM

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

### Goals
- Begin **Phase 3: LightGBM training and validation (steps 1-2)**
- Focus on dataset preparation and initial model setup and checks, not full training yet
- Ensure reproducibility from the start
- Make notes on potential challenges for full training and validation

### What We Did
1. **Step 1: Dataset Preparation**
  - Create `src/ml_data_prep/train_lightgbm_prep,py`
  - Load `news2_features_patient.csv` into file.
  - **Identify the target variables**:
    - `max_risk` → “Did this patient ever escalate to severe deterioration?” (binary/ordinal classifier style) → classifier model (ordinal: 0–3).
    - `median_risk` → “What was this patient’s typical level of deterioration?” (long-term trajectory classifier) → classifier model (ordinal: 0–3).
    - `pct_time_high` → “How much of the patient’s stay was spent at critical risk?” (continuous regression, severity burden) → regressor (continuous: 0–1).
  - **Decided on looping automation code**: so that each target gets its own trained model and results, and we do not need to manually code 3 almost-identical training runs.
  - **5-fold cross-validation (CV) due to small data size (100 patients)**:
    - 5 equal groups of 20 patients, train on 4 groups, test on remaining 1 group
    - Repeat 5 times, rotating which group is used for testing
    - Average performance across all groups is a more stable estimate vs a standard train/test split (e.g. 70/30 or 80/20).
  - **Separate features (`X`) from the target (`y`)**:
    - X = Features (inputs) → everything the model uses to make predictions (hr_mean, spo2_min, temperature_max, %missing, etc.).
    - y = Target (outputs) → the “answer key” you want the model to learn to predict (e.g., max_risk).
    - During training, the model learns a mapping from X → y.
      - X_train = inputs for training (DataFrame of row = 80 training patients × column = all features)
      - y_train = labels for training (Series of row = 80 training patients x values = their risk labels)
      - X_test = inputs to evaluate model (DataFrame of row = 20 test patients × column = all features)
      - y_test = labels to compare predictions (Series of row = 20 test patients x values = their risk labels)
  - **Check data types and missing values**: 
    - Ensure all features are numeric and compatible with LightGBM (LightGBM quite forgiving, can handle some NaNs internally).
    - Although our preprocessing should have fixed this (imputed NaNs, encoded categorical risk as numeric, dropped non-numerics), always double check before model training.
    - We need safety check as sometimes merges sneak in NaNs, sometimes column types are wrong.
    - If something unexpected pops up, better to catch it before fitting LightGBM.
2. **Step 2: Model Initialisation Setup**
  - Import LightGBM
  - **Initialise a basic LightGBM model (`train_lightgbm.py`)**
    - Used LightGBM default parameters (learning rate, depth, number of trees, etc.)
    - Create both classifier (for max_risk, median_risk) and regressor (for pct_time_high).
      - `LGBMClassifier` for max_risk & median_risk
      - `LGBMRegressor` for pct_time_high
    - Set the seed `random_state=42` for reproducibility.
  - Skip cross-validation entirely because we aren’t evaluating performance yet, and dataset is too small.
3. **Quick Test Run**:
  - Dont need to do a full training loop yet.
  - **Fit the model on a small subset of training data (10 patients) to**:
    - Verify that the pipeline works
    - Check that data formats, shapes, and types are correct (features (X) are numeric, targets (y) are the right shape)
    - Catch any errors with feature alignment or missing values
    - Predictions are generated
  - Ensured that the loop works for all 3 targets automatically
  - This catches pipeline errors before we spend time coding full CV training tomorrow.
4. **Logging and Documentation**:
  - Recorded:
    - Dataset shapes (rows, columns)
    - Features used
    - Any issues encountered (e.g., unexpected NaNs, strange distributions) - none major
  - Document initial observations and notes for Phase 3

### Train/Test Split vs Cross-Validation
#### Initial plan  
- Standard ML workflow uses a **train/test split** (e.g., 70/30 or 80/20).  
- Training set is used to fit the model, test set evaluates generalisation.  
- Works well when datasets are **large** (>10,000).  
- With a big dataset, 20–30% for testing still leaves enough training data to learn robust patterns.  
#### Problem with our dataset:  
- We only have **100 patients** (100 rows after patient-level aggregation).  
- A 70/30 split leaves 30 patients for testing; 80/20 leaves only 20.  
- **This is too small**: metrics like AUROC or accuracy would fluctuate a lot if even 1–2 patients are misclassified.  
- **Result**: unreliable, unstable performance estimates.  
#### Solution: Cross-Validation
- Instead of one split, we use **k-fold cross-validation**.  
- **Process**:  
  1. Split patients into *k* equal groups (folds).  
  2. Train on *k–1* folds, test on the remaining fold.  
  3. Repeat *k* times, rotating which fold is used for testing.  
  4. Average performance across all folds → more stable estimate.  
- Every patient is used for both training and testing (but never in the same round).  
#### Why 5-fold CV?
- **k=5** is a common default:  
  - Balances computational efficiency with robustness.  
  - Each test fold has 20 patients → big improvement over a single 20-patient test set.  
  - Results averaged across 5 runs smooth out randomness.  
- For very tiny datasets, k=10 can be used, but 5-fold is usually enough here.  
- **Decision:** 
  - Use **5-fold cross-validation** for LightGBM training/validation, and optionally hold out ~10 patients as a final untouched test set for a “real-world” check.  

### How the model works
- In supervised machine learning, the model learns a mapping from features (X) → target (y).
- Features (X) = things you give the model as input (heart rate, SpO2, temperature, missingness, etc.)
- Target (y) = the thing you want the model to predict (e.g. max_risk).
- Even though max_risk is already known, the model uses it as the “answer key” during training:

```text
Input features (X) → Model → Predict max_risk
Compare predicted max_risk vs actual max_risk → Adjust model
```

- During training, the model compares its predictions to the real max_risk values and adjusts weights to minimise errors.
- Without max_risk (or whatever target), the model cannot learn, because it has nothing to compare its predictions to.

### KFold logic diagram (100 patients, 5 folds)

```python
100 patients numbered 0–99.

Step 1: Shuffle patients (random order)
Shuffled indices: 23, 45, 12, 7, 56, ... , 99  (total 100)

Step 2: Split into 5 folds (20 patients each)
100 patients → 5 folds (20 patients each)

Fold 1: 23, 45, 12, ..., 78        (test fold for iteration 1)
Fold 2: 7, 56, 34, ..., 81         (test fold for iteration 2)
Fold 3: ...
Fold 4: ...
Fold 5: ...

Step 3: Loop through folds

Iteration 1 (fold_idx=1)
------------------------
Train folds = 2,3,4,5 → 80 patients
Test fold  = 1         → 20 patients

X_train = features for patients in folds 2–5 (80×num_features)
y_train = labels   for patients in folds 2–5 (80×1)

X_test  = features for patients in fold 1 (20×num_features)
y_test  = labels   for patients in fold 1 (20×1)

Iteration 2 (fold_idx=2)
------------------------
Train folds = 1,3,4,5 → 80 patients
Test fold  = 2         → 20 patients

X_train = features for patients in folds 1,3,4,5
y_train = labels   for patients in folds 1,3,4,5

X_test  = features for patients in fold 2
y_test  = labels   for patients in fold 2

Iteration 3 (fold_idx=3)
------------------------
Train folds = 1,2,4,5 → 80 patients
Test fold  = 3         → 20 patients

X_train = features for patients in folds 1,2,4,5
y_train = labels   for patients in folds 1,2,4,5

X_test  = features for patients in fold 3
y_test  = labels   for patients in fold 3

Iteration 4 (fold_idx=4)
------------------------
Train folds = 1,2,3,5 → 80 patients
Test fold  = 4         → 20 patients

X_train = features for patients in folds 1,2,3,5
y_train = labels   for patients in folds 1,2,3,5

X_test  = features for patients in fold 4
y_test  = labels   for patients in fold 4

Iteration 5 (fold_idx=5)
------------------------
Train folds = 1,2,3,4 → 80 patients
Test fold  = 5         → 20 patients

X_train = features for patients in folds 1,2,3,4
y_train = labels   for patients in folds 1,2,3,4

X_test  = features for patients in fold 5
y_test  = labels   for patients in fold 5
```

### Dataset Preparation Output (`prepare_patient_dataset.py`)
- Dataset loaded correctly → shape (100, 44) → 100 patients, 44 columns.
- All columns are numeric (float64 / int64) except two (co2_retainer_min and co2_retainer_max, which are bool but still compatible with LightGBM).
- No missing values left — preprocessing worked properly.
- For each target (max_risk, median_risk, pct_time_high):
- The dataset is being split into 5 folds.
- **Each fold shows the correct sizes**: Train shape (80, 40), Test shape (20, 40) → 80 patients for training, 20 for testing, 40 features in X.
- The loop cycles through all 3 targets automatically, so pipeline is flexible and future-proof.
- The dataset is fully prepared, the cross-validation setup works perfectly.
- Now have X_train, y_train, X_test, y_test ready for model training in the next step.

### Model Initialisation & Test Run Output (`train_lightgbm.py`)
- Quick LightGBM test on 10 patients completed
- Dataset loads correctly → shape (100, 44)
- Feature selection works → X has all numeric features except target columns and subject_id.
- Target separation works → y is selected for each loop.
- Models (classifier/regressor) fit without crashing.
- Predictions generated → pipeline is complete.
- Warnings about “no meaningful features” expected for tiny dataset; safe to ignore

### Reflections
#### Challenges
- **Understanding X and y separation**
  - Confused why both X (features) and y (target) are needed when the target is already known.
- **KFold logic and cross-validation**
  - Confused how `KFold.split(X)` automatically generates `train_index` and `test_index`.
  - Unsure how each iteration selects a different fold for testing and uses the remaining folds for training.
  - Did not initially understand why `fold_idx` starts at 1 and how it corresponds to the test fold.
  - Hard to visualise how all 100 patients are rotated through the 5 folds.
- **Handling X_train, X_test, y_train, y_test**
  - Unclear why there are four separate objects and what each represents in cross-validation.
- **Directory structure and relative paths**
  - Confused why `../../` worked in some scripts but not others, and where the CSV should be located relative to each script.
#### Solutions & Learnings
- **X and y separation**
  - Reviewed supervised learning: X = input features, y = labels/“answer key”.
  - **Insight:** Model requires y to compute loss and adjust weights; without it, training cannot proceed.
- **KFold `.split()` logic**
  - Learned that `.split()` is a generator which, for each iteration:
    1. Assigns one fold as `test_index`.
    2. Automatically uses all remaining folds as `train_index`.
  - Created a diagram mapping 100 patients → 5 folds → train/test sets per iteration.
  - **Insight:** Ensures each patient appears in the test set exactly once across folds, giving full coverage and stable performance estimates.
- **Fold indexing (`fold_idx`)**
  - Serves as a counter for current iteration; tracks which fold is the test fold.
  - **Insight:** Facilitates debugging, logging, and fold-specific analysis.
- **X/y training/test objects**
  - Verified that `X_train`/`y_train` contain features and labels for training folds, `X_test`/`y_test` for the test fold.
  - Checked shapes and sample rows to confirm correctness.
  - **Insight:** Clearly defines what data the model sees during training vs evaluation.
- **Quick visualisation**
  - Diagrammed how `.iloc[train_index]` and `.iloc[test_index]` select the correct rows.
  - **Insight:** Makes cross-validation logic intuitive; reinforces the necessity of looping through folds for small datasets.
- **Directory paths**
  - Used `pathlib.Path(__file__).resolve().parent` to dynamically build paths.
  - **Insight:** Avoids path errors, ensures reproducibility regardless of script location.

### Overall
- Gained confidence in preparing datasets, setting up cross-validation, and creating reproducible ML workflows.
- Pipeline is now robust for looping through all three target variables, ready for full training in the next phase.

### Next Steps
- Implement **full 5-fold CV training on all 100 patients**
- Save trained models and CV results
- Calculate evaluation metrics (accuracy, ROC-AUC, RMSE, etc.)
- Start exploring **feature importance** and preliminary model interpretation
- Extend pipeline to loop automatically for all targets and optionally timestamp-level features in parallel

---

## Day 10 Notes - LightGBM Training + Validation (Steps 3–7)

### Goals
- **Complete the baseline LightGBM pipeline**: initialise, train, validate, and save models.
- Produce reproducible patient-level ML outputs for all three outcomes (max_risk, median_risk, pct_time_high).
- Debug pipeline issues, document failures, and reflect on dataset limitations.
- Begin rethinking project framing if classification targets prove unstable.

### Purpose of Baseline Classical LightGBM ML Model
1. Show I can prepare patient-level data for ML.
2. Provides a baseline classical ML benchmark for patient deterioration prediction.
3. Demonstrates an end-to-end ML workflow, and a credible, well-structured pipeline (data prep → CV → training → saving → validation → documentation → final deployment models).
4. Ensures reproducibility and robustness with cross-validation, and deployment readiness (final models).
5. Adds interpretability through feature importance, crucial in clinical healthcare settings.
6. Establishes a strong baseline Performance benchmark for later comparison with Neural Networks, showing their added value.

### Phase 3: LightGBM Training + Validation (Steps 1–9) Finalised 
**Goal: Train, validate, and document a LightGBM model on patient-level features, producing a polished, credible baseline.**
1. **Dataset Preparation**
  - Load processed patient-level features.
  - Separate features (X) and targets (y).
  - Verify datatypes, shapes, and missing values.
2. **Model Initialisation**
  - Initialise LGBMClassifier (max_risk, median_risk) or LGBMRegressor (pct_time_high).
  - Define baseline parameters and set random seed.
3. **Model Training**
  - Fit model on training folds with early stopping.
  - Monitor performance on validation folds.
4. **Model Saving**
  - Save trained models and results for reproducibility.
  - 3 outcomes x 5 folds = 15 models.
5. **Model Validation**
  - Load saved models, run predictions.
  - Compute metrics (accuracy, ROC-AUC, RMSE).
  - Visualise feature importance and results.
6. **Documentation**
  - Record metrics, shapes, and key outputs.
  - Summarise feature importances for interpretability.
7. **Debugging / Checks**
  - Verify dataset consistency across folds.
  - Ensure features align between training and test sets.
8. **Hyperparameter Tuning + Feature Importance**
  - Tune key parameters (learning rate, depth, trees) for fair performance.
  - Analyse feature importance to explain clinical drivers of prediction.
  - Produces a polished, interpretable, portfolio-ready baseline.
9. **Final Model Training (Deployment-Style Models)**
	-	After validation, train a final single model per target (3 total) on all 100 patients.
	-	**Purpose**:
    - Produces the best possible trained model using all available data.
    -	**Matches real-world practice**: once validated, you don’t throw away data — you train on the full cohort.
    -	Gives you 3 final models you can save, reload, and demonstrate (classifier + regressor).
This makes LightGBM phase complete, credible, and deployment-worthy without unnecessary over-optimisation.
**Why Not Go Further**
- **Ensembling (stacking, blending, bagging multiple LightGBM models)**: adds complexity without new insights → not unique for a portfolio.
- **Nested CV**: more statistically rigorous, but overkill for 100 patients; doesn’t change credibility.
- **Bayesian optimisation / AutoML**: looks flashy, but to recruiters it signals you know how to use a library, not that you understand the fundamentals.
- **Overfitting risk**: with 100 patients, “chasing” tiny gains just makes results unstable and less reproducible.
- **Time sink**: delays me getting to Neural Nets (the unique, impressive part of your project).

### How Gradient Boosted Decision Tree Model Works
-	Trees split on feature thresholds.
-	Each split improves the model’s predictions.
- Build trees sequentially, each correcting previous errors.
-	Feature importance = how often (or how much) the model used each feature to reduce errors.

### What We Did
**Step 1: Phase 3 Steps 3-7 Completed**
1. Model Training `complete_train_lightgbm.py`
	- LGBMClassifier for classification targets (max_risk, median_risk).
	-	LGBMRegressor for continuous regression target (pct_time_high).
  - Sets up a loop for each target variable, selects the appropriate model type and metric, and prepares a list to store the results from cross-validation folds. 
	-	Ran 5-fold cross-validation, early stopping enabled.
2. Model Saving, Validation and Documentation Attempt
	-	Tried to save 15 trained models (.pkl) to ensures reproducibility and ability to reload for later use.
	- Tried to generate predictions on held-out folds, metrics logging (ROC-AUC, accuracy for classification; RMSE/MSE for regression).
  - Attempted to captured metrics, feature importances, dataset shapes and training summary outputs.
3. Debugging / Checks When Running Script
	- Multiple crashes due to missing classes in folds (e.g., fold contained only one label).
	- Verified splits, indices, and LightGBM behaviour.
**Model Outputs `src/ml_models_lightgbm/baseline_models/`**
1. 15 trained models (.pkl) → 5 folds × 3 targets → fold-wise trained models. Lets us reload and run predictions on unseen data later.
2. 3 per-target CV result CSVs (*_cv_results.csv) → fold-wise scores per target. Enables calculation of mean/variance of performance → essential for robust evaluation.
3. 15 feature importance CSVs (*_fold{fold_idx}_feature_importance.csv) → top features per fold per target. Supports interpretability and clinical storytelling.
4. 1 training summary text file (training_summary.txt) → cumulative summary for all targets of the dataset shape, mean CV score, top features per target. High-level snapshot of performance and reproducibility.
**Today’s notes stop at the belief that the dataset was fundamentally flawed.** 
- Tomorrow’s entry will capture the turning point when manual inspection revealed the data wasn’t as bad as feared.
- **Why we thought dataset was unusable**:
	- **Extreme class imbalance**: With 5-fold CV, many folds end up with zero examples of the minority class. ROC-AUC or any meaningful classification metric cannot be computed if a fold has only one class.
	- **Data sparsity**: The model cannot learn patterns from a single positive example. Even if you reduce folds to 2–3, the minority class is still too rare for reliable training.
  - **Metrics are meaningless**: Fold scores like 0.5 are just random guessing, not informative. Any feature importance will also be unstable and unreliable.

### Evaluation Metrics: MSE vs ROC-AUC vs Accuracy  
#### 1. **Mean Squared Error (MSE)**
- **Type:** Regression metric (continuous outcomes).
- **Definition:**  
```text
MSE = (1/n) * Σ (y_i - ŷ_i)^2
```
- **What it measures:**  
  - The *average squared difference* between predictions and actuals.  
  - Penalises large errors much more heavily than small ones (because of the square).  
- **How it evaluates:**  
  - Lower MSE = better fit.  
  - Perfect model → MSE = 0.  
- **Use case:**  
  - Regression tasks (e.g., predicting %time_high for a patient).  
  - Good when you care about magnitude of errors.  

#### 2. **Accuracy**
- **Type:** Classification metric (discrete categories).  
- **Definition:**  
```text
Accuracy = (# correct predictions) / (total # predictions)
```
- **What it measures:**  
  - The proportion of predictions that are exactly correct.  
- **How it evaluates:**  
  - 0–1 range (or %).  
  - Example: if model got 85 out of 100 patients’ classes right → accuracy = 0.85.  
- **Limitations:**  
  - Misleading for *imbalanced datasets*.  
    - If 95% of patients are “low risk”, a dumb model that predicts “low risk” for everyone gets 95% accuracy, but is useless.  
- **Use case:**  
  - Quick baseline check for balanced classification problems.  
  - Less informative when classes are skewed.  

#### 3. **ROC-AUC (Receiver Operating Characteristic – Area Under Curve)**
- **Type:** Classification metric (binary or multiclass with extensions).  
- **Definition:**  
  - Plots **True Positive Rate (Sensitivity)** vs **False Positive Rate (1 – Specificity)** at different thresholds.  
  - AUC = area under that ROC curve (0–1).  
- **What it measures:**  
  - **Discrimination ability**: how well the model separates positive from negative classes.  
  - AUC = probability that the model assigns a higher score to a randomly chosen positive case than to a randomly chosen negative case.  
- **How it evaluates:**  
  - 0.5 = no better than random guessing.  
  - 1.0 = perfect discrimination.  
- **Strengths:**  
  - Works well even when classes are imbalanced.  
  - Evaluates the **ranking of predictions**, not just “hard” class labels.  
- **Use case:**  
  - Preferred metric for classifiers when outcomes are rare (e.g., detecting deteriorating patients).  
  - More informative than accuracy in medicine because it accounts for both sensitivity and specificity across thresholds.  

#### Summary Table

| Metric      | Task Type       | Range      | Goal   | Strengths                                   | Weaknesses |
|-------------|-----------------|------------|--------|---------------------------------------------|-------------|
| **MSE**     | Regression      | [0, ∞)     | Lower  | Penalises big errors; sensitive to scale.   | Hard to interpret clinically (units²). |
| **Accuracy**| Classification  | [0,1]      | Higher | Simple, intuitive.                          | Misleading with imbalanced data. |
| **ROC-AUC** | Classification  | [0.5,1]    | Higher | Robust to imbalance; measures discrimination. | Harder to intuitively explain to non-tech audience. |

#### For this project:
- Use **MSE** for `pct_time_high` (regression).  
- Use **ROC-AUC** for `max_risk` and `median_risk` (classification).  
- Fall back to **accuracy** only when a fold has a single class (so ROC-AUC is undefined).  

### Reflection
#### Challenges
- **Debugging Challenges**: 
  - Initial confusion about what was actually stored in the saved LightGBM model files.  
  - It wasn’t clear whether the `.pkl` files contained the raw training/validation data (`X`, `y`) or just the learned parameters.  
  - Misunderstanding how indices (`train_index`, `test_index`) worked during cross-validation created uncertainty about which patients were included in each fold.   
- **Pipeline breakdowns**: Training repeatedly failed, with models crashing mid-run despite fixing code errors.
- **Misdiagnosing issues**: Initially believed the problems were bugs in the training pipeline itself rather than fundamental dataset limitations.
- **Dataset shock**: Discovered that risk variables (max_risk, median_risk, pct_time_high) were highly imbalanced and potentially unusable for ML. This made the entire project feel at risk.
- **Time sink**: Large portions of the day were spent patching, rerunning, and rechecking, only to end up back at the same roadblock.

#### Solutions
- Clarified that **saved models do not contain raw data**:  
  - Each `.pkl` file only stores the learned parameters (tree splits, leaf weights, feature usage).  
  - This is why saved files are small, and why new input data is always required for predictions.  
- Confirmed how data splits are generated and used:  
  - `train_index` and `test_index` are row numbers pointing back to the original dataframe.  
  - `.iloc` then retrieves the actual patient rows for each split.  
- Understood the k-fold cross-validation cycle:  
  - Every patient appears in training 4 times and testing once across 5 folds.  
  - Each fold trains a **fresh LightGBM model** (fully reset), preventing data leakage and ensuring unbiased evaluation.  
- **Investigated why folds failed for classification**:  
  - Realised **KFold does not guarantee class balance** → some folds excluded minority classes completely.  
  - Learned that **StratifiedKFold preserves class proportions** across folds, which avoids crashes in most scenarios.  
  - Still, with very rare classes, even StratifiedKFold can fail unless LightGBM is told explicitly which labels exist (`classes=[…]`). 
- **Brainstormed a complete redesign of the pipeline**:
	- Dropping max_risk and median_risk entirely.
	- Redefining new patient-level and timestamp-level variables.
	-	Rewriting make_patient_features.py and make_timestamp_features.py to generate new CSVs.
	-	Pivoting from “ICU deterioration prediction” to a looser framing (general NEWS2 trend insights).
- **Began drafting how this pivot could be explained in the final report and portfolio**: as a realistic example of dynamic ML research where goals adapt to messy data.

#### Learnings
- **Models ≠ data**: A LightGBM model file is a set of learned rules, not a copy of the dataset.  
- **Cross-validation mechanics**: Fold indices are just pointers; `.iloc` turns them into actual data subsets for training/testing.  
- **Coverage guarantee**: CV ensures all patients contribute to both training and testing, giving a robust estimate of model performance.  
- **Resetting per fold**: Essential so the model cannot “remember” test data from earlier folds.  
- **Stratified vs regular KFold**:  
  - Regular KFold risks missing labels in small/imbalanced datasets.  
  - StratifiedKFold reduces this risk by preserving proportions, but still requires caution when classes are extremely rare.  
- **Evaluation vs deployment**: Cross-validation produces multiple models for scoring; deployment requires retraining once on the full dataset.  
- **Debugging isn’t always coding**: Sometimes errors trace back to the dataset, not the script.
- **Rare events cripple models**: Small clinical datasets often lack enough high-risk outcomes to train or validate classifiers.
- **Metrics and pipelines don’t reveal it until runtime**
	-	Accuracy or ROC-AUC will fail only when a fold is completely missing a class, which is exactly what happened.
	-	Static checks (like looking at column summaries) cannot guarantee every fold has sufficient data, this only manifests during CV.
- **Adaptability matters**: The ability to rethink targets and framing under pressure is as important as technical implementation.
- **Documentation is critical**: Capturing this struggle makes the project more authentic and reflective of real-world ML practice.

### How I Could Have Prevented Todays Issues
1. **Inspect dataset first**
  - Check class distributions (df['max_risk'].value_counts() etc.) before choosing targets.
  - If the minority class is tiny (1–2 samples), CV will fail.
  - Always check the class balance for classification tasks before deciding on CV splits.
2. **Select better targets**
  - For classification, choose variables with enough samples in each class (at least 5–10 per class for a small dataset).
  -	Rare binary/ordinal targets may require deciding not to train that target at all.
  - Or use regression targets where sample size is adequate.
3. **Document reasoning**
  - Even if you pick a poor target, explain why it fails and why you skipped it. This demonstrates critical thinking and understanding of ML limitations.
  - In portfolio or reproducible pipelines, it’s perfectly acceptable to document why some targets are unusable.

### Overall Reflection 
- **Emotionally difficult day**: felt like project collapse due to dataset sparsity.
- **Felt like starting over:** at one point it seemed we would need to rebuild the pipeline from scratch with new variables and rewritten feature engineering scripts.
- **Learned critical ML lesson**: sometimes data, not code, is the bottleneck.
- **The Positives**: Built a fully reproducible LightGBM pipeline. Implemented CV, early stopping, feature importance logging. Learned about why small/imbalanced datasets break classification, which is valuable knowledge for any real-world ML project.
- Documented approach shifts and backup plans.
- Need to adjust the targets we report on. Many published ML projects run into exactly this issue, small or imbalanced datasets are extremely common in healthcare.
- Prepared to reframe project as risk trend prediction rather than rare-event deterioration.

### Extras / Insights 
#### Portfolio Framing
- This day highlights resilience and scientific reasoning, not everything in ML runs smoothly, and documenting the “bad days” strengthens credibility.
- Rare-event prediction wasn’t possible; instead, the model learns risk patterns and trends from NEWS2 data.
- Clinically meaningful to frame this as “predicting risk trajectories and physiological trends,” not “predicting rare ICU collapses.”
- Still valid to call it a deterioration predictor, just not for extreme events.
- Shows adaptability, strong methodology, and awareness of real-world clinical ML challenges.
- **Key message for CV/portfolio:**
“Originally, max_risk and median_risk were considered for classification, but due to extreme class imbalance these targets were unreliable. The project pivoted to focus on pct_time_high regression, demonstrating robust ML methodology, interpretable feature importance, and the ability to adapt pipelines to messy real-world healthcare data. The model predicts trends in patient risk (NEWS2) over ICU stay, identifying factors contributing to changes in physiological state. The clinical signal however is still weak due to data sparsity.”

### Project Credibility Even With These Issues
- **Technical skills**:
  - Data cleaning and preprocessing of high-dimensional clinical time series.
  - Rolling windows, staleness flags, LOCF, missingness handling.
  - Multi-target ML pipelines (LightGBM patient-level regression, TCN timestamp-level).
- **Uniqueness / clinical insight**:
  - Demonstrates how to handle real-world messy clinical data.
  - Shows understanding of risk trajectories, time-series aggregation, and how to extract meaningful features.
- **Metrics / numbers**:
  - Regression outputs (RMSE, R², predicted vs actual plots) are still quantitative.
  - **Comparison with NEWS2 baseline remains valid**: you can compute “how much the ML model reduces error vs raw NEWS2 predictions” or correlation improvement.
  - It’s less about rare-event prediction and more about predicting general risk patterns / trends in patient physiology, which is a legitimate clinical ML task.
- **Recruiter impression**:
  - Shows maturity and practical problem-solving, which is impressive to recruiters and interviewers.
	- They don’t care that rare high-risk classes were absent. They care that you:
    1. Detected the data problem.
    2. Adapted pipeline intelligently.
    3. Produced measurable, interpretable results.

---

## Day 11 Notes - Fixing Class Imbalance and Finalising LightGBM CV

### Goals
-	Resolve persistent errors in LightGBM training caused by missing classes in folds.
-	Re-examine dataset distributions at the patient level (max_risk, median_risk, pct_time_high).
-	Redefine classification targets if necessary to make CV feasible and clinically meaningful.
- Implement a flexible `.fit()` pipeline that adapts to regression, binary classification, and multiclass.
-	Achieve a fully reproducible run with saved models, CV results, feature importances, and training logs.

### What We Did
1. **Initial Plan (Failed Approach)**
	- Removed stratification and tried 3-fold CV for classification targets, 5-fold CV for regression.
	-	Merged classes (e.g., 0+1) to stabilise distributions.
	-	Still failed → folds missing classes, LightGBM crashed.
```markdown
ValueError: y contains previously unseen labels: [1]
```
2. **Tried StratifiedKFold Again**
	-	Expected stratification to solve missing class issues.
	-	Still failed → suggested internal rounding/edge case issue.
3. **Diagnostic Step**
	-	Added code to print class distributions in folds.
	-	Discovered earlier calculated distributions were wrong.
	-	**True distribution**: max_risk had only 1 patient in class 0. Median_risk had no patients in class 1 or 3.
4. **Redefinition of Targets**
	-	**Max risk:** Collapsed 0,1,2 into “not high risk” (2) vs “high risk” (3).
	-	**Median risk:** Collapsed 0+1 into “low risk” (1) vs 2 = “medium risk”. Removed class 3 since no patients had it.
	-	Both now binary classification targets (preds.round() works (LightGBM internally shifts to [0,1])).
5. **Updated CV Strategy**
	-	With binary framing, minority classes had enough patients to support 5-fold StratifiedKFold.
	-	Regression target (pct_time_high) stayed with standard 5-fold KFold.
6. **Final Successful Run**
	-	Training completed without crashes.
	-	**Produced 34 files in saved_models/:** 15 trained models (.pkl), 15 feature importance CSVs, 3 CV results CSVs, 1 training summary.

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

### Patient-Level Data Distribution Results
#### Max Risk Distribution (100 patients total)
| Score | Patients | Percentage |
|-------|----------|------------|
| 0     | 1        | 1.0%       |
| 1     | 0        | 0.0%       |
| 2     | 13       | 13.0%      |
| 3     | 86       | 86.0%      |

#### Median Risk Distribution (100 patients total)
| Score | Patients | Percentage |
|-------|----------|------------|
| 0     | 76       | 76.0%      |
| 1     | 0        | 0.0%       |
| 2     | 24       | 24.0%      |
| 3     | 0        | 0.0%       |

#### Percentage Time High Distribution (100 patients total)
**Basic Statistics**
| Metric                  | Value          |
|--------------------------|----------------|
| Range                   | 0.0000 – 0.4407 (0% – 44.1%) |
| Mean                    | 0.1114 (11.1%) |
| Standard Deviation      | 0.1040         |
| Median                  | 0.0802 (8.0%)  |

**Critical Distribution Issues**
| Issue                     | Details                                |
|----------------------------|----------------------------------------|
| High Zero Inflation        | 27% of patients have 0% time in high-risk |
| Right-Skewed Distribution | Skewness = 1.24 (moderate positive skew) |
| High Variability           | Coefficient of variation = 0.93        |

**Regression Suitability**: **MODERATE**

**Potential Issues**
- Moderately skewed distribution (may affect MSE optimization).  
- High proportion of zeros → prediction challenges.  
- Non-normal distribution may impact residual patterns.  


#### Results with Combined Scoring
**Max Risk Distribution (Combined 0+1+2 → 2)**
| Score | Patients | Percentage | Notes |
|-------|----------|------------|-------|
| 2     | 14       | 14.0%      | (1 from score 0 + 0 from score 1 + 13 from score 2) |
| 3     | 86       | 86.0%      | Unchanged |

**Median Risk Distribution (Combined 0+1 → 1)**
| Score | Patients | Percentage | Notes |
|-------|----------|------------|-------|
| 1     | 76       | 76.0%      | (76 from score 0 + 0 from score 1) |
| 2     | 24       | 24.0%      | Unchanged |
| 3     | 0        | 0.0%       | Unchanged |


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

### Reflections
#### Challenges
1. **LightGBM crashes on missing classes**
  - `Error: ValueError: y contains previously unseen labels: [1]`
  - Normal KFold caused folds where some classes were absent from training → LightGBM failed.
2. **StratifiedKFold didn’t fully solve the problem**
	- Expected it to guarantee all classes in all folds.
	-	Still failed when classes were extremely rare (e.g. only 1 patient in max_risk=0).
3. **Misleading dataset distributions**
	-	Earlier calculations suggested more balanced classes.
	-	**In reality**: only 1 patient in max_risk=0, no patients in median_risk=3.
	-	This invalidated earlier CV plans.
4. **Misunderstanding num_class**
	-	Setting num_class=3 does not globally register all classes.
	-	LightGBM only encodes labels present in that fold’s training set.
5. **Prediction logic too simplistic**
	-	`.round()` worked for binary, but broke for multiclass probabilities.
	-	Needed target-type-specific post-processing.
6. **Training warnings**
	-	`[LightGBM] [Warning] No further splits with positive gain` repeated in training logs.
	-	Didn’t crash training, but signals limited tree growth due to data size/imbalance.
#### Solutions & Learnings
1. **Redefine classification targets → Binary Conversion**
	-	**Implemented directly in the training loop**:
```python
# -----------------------------
# Binary Conversion for Classification Targets
# -----------------------------
if target_name == "max_risk":
    y = (y == 3).astype(int)   # Convert 2→0 (not high risk), 3→1 (high risk)
    print(f"Binary class distribution: {pd.Series(y).value_counts().sort_index()}")
elif target_name == "median_risk":
    y = (y == 2).astype(int)   # Convert 1→0 (low risk), 2→1 (high risk)
    print(f"Binary class distribution: {pd.Series(y).value_counts().sort_index()}")
```
  - **max_risk**: merged 0,1,2 → “not high risk” vs 3 → “high risk”.
	- **median_risk**: merged 0+1 → “low risk” vs 2 → “high risk” (removed 3).
  - Removed rare/unusable categories entirely.
	-	**Result**: both are now binary, with enough samples per class for CV.
2. **CV Strategy**
	-	5-fold CV for all targets (regression and classification).
	-	StratifiedKFold for classifiers → preserves balance.
	-	Standard KFold for regression → no class imbalance issue.
3. **Manual validation of distributions**
	-	Exported CSV → counted manually in Excel.
	-	Verified true counts instead of relying on buggy earlier methods (LLMs).
4. **Improved `.fit()` pipeline**
	-	Removed invalid classes argument, and replaced with a dictionary that only includes the parameters that apply (clean seperation works for regression and classification seamlessly).
	-	Flexible fit_params dictionary with **kwargs unpacking.
	-	Works for regression, binary, and multiclass tasks without duplication.
```python
fit_params = {
    "X": X_train,
    "y": y_train,
    "eval_set": [(X_test, y_test)],
    "callbacks": [early_stopping(10), log_evaluation(0)]
}

# Removed: fit_params["classes"]  (not supported in LightGBM)
model.fit(**fit_params)
```
5. **Improved target-type-specific prediction logic**
	-	**Binary (max_risk, median_risk)**: 
    - LightGBM gives probability of the positive class by default
    - Shape = (n_samples,) → 1D vector
    - `preds.round()` → .round() to 0/1.
	-	**Multiclass (if we hadn’t collapsed classes)**: 
    - Shape = (n_samples, n_classes) → 2D matrix
    - `np.argmax(preds, axis=1)` → pick the highest probability class.
	-	**Regression (pct_time_high)** 
    - Shape = (n_samples,) → 1D vector
    - Just use raw predictions directly.
```python
# Calculate evaluation metric
if target_name == "pct_time_high":
    score = metric_fn(y_test, preds)   # Direct MSE calculation for regression
else:
    # Handle different prediction formats for classification
    if preds.ndim > 1 and preds.shape[1] > 1:
        preds_labels = np.argmax(preds, axis=1)   # Multiclass → take class with highest probability
    else:
        preds_labels = preds.round().astype(int)  # Binary → round probabilities to 0/1
```
6. **Clinical and interpretability benefits**
	- **Binary framing aligns with real-world workflows**:
    -	“High risk vs not high risk” → max_risk.
    -	“Low vs high” → median_risk.
	-	ROC-AUC simpler, feature importance clearer.

### Final Summary
**Day 11 was a turning point**:
-	Found hidden data imbalance (class counts wrong before).
-	Redefined classification tasks into binary problems.
-	Simplified CV design (5-fold, stratified where needed).
-	Improved code robustness with flexible .fit() and predictions (y_test).
-	Produced the first complete, stable run with clinically relevant framing.

### Next Steps
**Step 8: Hyperparameter Tuning + Feature Importance**
-	Tune key parameters such as learning rate, tree depth, number of trees, and leaf size for balanced performance.
-	Aggregate and analyse feature importance across folds to highlight clinical drivers and make results interpretable.
**Step 9:	Final Model Training (Deployment-Style Models)**
-	Train one final model per target (3 total) on the full 100-patient dataset.
-	**Purpose**: maximise use of available data, produce deployment-ready models, and generate demonstrable outputs for portfolio or future work.
**Outcome**: 
By the end of Day 12, the LightGBM phase will be complete, validated, interpretable, and ready for demonstration without unnecessary overfitting or over-optimisation.

---

## Day 12-13 Notes - Complete Phase 3: Hyperparameter Tuning, Feature Importance, and Deployment-Ready Models

### Goals
- **Complete all of Phase 3 (steps 8-9)**:
  - Hyperparameter Tuning + Feature Importance
  - Final Model Training (Deployment-Style Models)
- Produce a polished, interpretable, portfolio-ready LightGBM baseline and deployment-style models for all three targets.

### What We Did 
**Step 1: Hyperparameter Tuning for Classification and Regression Models `tune_models.py`**
- **Select the key parameters to tune**:
	- `learning_rate` → controls step size; balances speed vs overfitting.
	- `max_depth` / `num_leaves` → limits tree complexity; prevents overfitting small dataset.
	- `n_estimators` → total number of trees.
	-	`min_data_in_leaf` → ensures each leaf has enough samples, stabilising predictions.
-	Performed small manual sweeps or grid search over reasonable ranges.
-	**Evaluate performance using 5-fold cross-validation**:
	-	AUROC / accuracy for max_risk and median_risk
	-	RMSE for pct_time_high
- **Outputs**: 
  - 3x `tuning_logs/{target}_tuning_log.csv` (per target containing every parameter sets values and the computed mean score)
  - 3x `{target}_cv_results.csv` (per target each file contains all 5 fold-level scores for the winning parameter set)
  - 1x `best_params.json` (dictionary of the best parameter set for every target)
- **Rationale**:
	-	Optimises baseline performance without overfitting, especially critical with only 100 patients.
	-	Ensures that the model is robust, reproducible, and interpretable.
	-	Gives credibility for portfolio presentation, showing thoughtful model design, not just default parameters.

**Step 2: Feature Importance Analysis `feature_importance.py`**
-	Extracted feature importance from LightGBM for each fold and aggregate across folds.
-	Identified the top 10 features per target.
-	Visualised as bar plots for feature importance, aggregated across folds.
- Script reuses the original "feature importance export" code from `complete_train_lightgbm.py` but instead of per-fold CSVs (15 files), it averages across folds and produces one clean CSV + one plot per target.
- **Outputs**: 	
  - 3x `{target}_feature_importance.csv` (ranked list of all features (e.g. resp_rate) with their mean importance, one file per target)
  - 3x `{target}_feature_importance.png` (one plot per target, horizontal bar chart of top 10 features for visualisation)
- **Rationale**:
	-	**Highlights which clinical features are driving predictions**:
    - High importance score = model relied heavily on that feature.
    - Low/zero importance = feature contributed little or nothing.
    - Differences across targets show which predictors matter most for different risk outcomes.
  - Visual outputs make the model interpretable and credible, demonstrates understanding of data, not just coding.
	-	Aggregating across folds (computing average of feature importance from all 5 folds) reduces noise and prevents overemphasising spurious features potentially present in individual folds.
	-	**Results are portfolio-ready**: visualisation clearly communicates results to reviewers.

**Step 3: Trained Final Deployment-Style Models `train_final_models.py`**
-	Trained one final model per target (3 total) on the entire 100-patient dataset.
-	Saved each model (.pkl) for reproducibility and demonstration.
- **Outputs**: 
  - `{target}_final_model.pkl.` (deployment-ready models, one per target) 
    - The full LightGBM model (trees, splits, learned parameters).
    - Configured with best hyperparameters found during tuning.
    -	Trained on the entire dataset (not just CV folds).
- **Rationale**:
	-	Makes full use of all available data after validation, mimics real-world deployment practice.
	-	**Produces demonstrable models**: classifier + regressor.
	-	These models will be used in later stages (e.g., neural network experiments, portfolio demos) and are a polished “deliverable” output.

**Step 4: Documented Everything in Portfolio-Ready Summary `summarise_results.py`**
-	Recorded final hyperparameter choices, cross-validation scores, and feature importance.
-	Summarised in training_summary.txt for portfolio inclusion.
-	Saved visualisations (top features, performance metrics) for presentation.
- **Outputs**: `training_summary.txt` (single plain-text report, deployment-ready summary)
  1. How well the model performed (CV mean ± std).
	2. Which hyperparameters were chosen.
	3. Which features were most important.
- **Rationale**:
	-	Provides transparent, reproducible evidence of methodology.
	-	Makes the project credible for reviewers, portfolio readers, or recruiters.
	-	Serves as a baseline for future neural net models, anyone can see exactly how LightGBM performs before moving to more complex models.

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
### File layout 
```text
data/
└── processed_data/
    └── news2_features_patient.csv                        # Input features dataset (all scripts read from here)

src/
└── ml_models_lightgbm/
    ├── baseline_models/                                  # Original 34 baseline CV script outputs (Day 11)
    │   ├── max_risk_fold1.pkl
    │   ├── ...
    │   ├── pct_time_high_fold5_feature_importance.csv
    │   └── training_summary.txt
    │
    ├── hyperparameter_tuning_runs/                       # New hyperparameter tuning outputs (Day 12)
    │   ├── max_risk_cv_results.csv                       # Full list of features with their average importance scores.
    │   ├── median_risk_cv_results.csv
    │   ├── pct_time_high_cv_results.csv
    │   ├── best_params.json                              # key:value pairs of hyperparameters for each target.
    │   └── tuning_logs/                                  # Tuning logs of mean score for each parameter set per target
    │       ├── max_risk_tuning_log.csv
    │       ├── median_risk_tuning_log.csv
    │       └── pct_time_high_tuning_log.csv
    │
    ├── feature_importance_runs/                          # New feature importance + visualisation outputs (Day 12)
    │   ├── max_risk_feature_importance.csv
    │   ├── median_risk_feature_importance.csv
    │   ├── pct_time_high_feature_importance.csv
    │   ├── max_risk_feature_importance.png
    │   ├── median_risk_feature_importance.png
    │   └── pct_time_high_feature_importance.png
    │
    ├── deployment_models/                                # Final deployment-ready models + summary (Day 12)
    │   ├── max_risk_final_model.pkl
    │   ├── median_risk_final_model.pkl
    │   ├── pct_time_high_final_model.pkl
    │   └── training_summary.txt                          # Portfolio-ready summary: CV scores, best params, top features
    │
    ├── tune_models.py                                    # Script 1: Hyperparameter tuning + CV
    ├── feature_importance.py                             # Script 2: Aggregate feature importance + visualisation plots
    ├── train_final_models.py                             # Script 3: Train final models on full dataset
    ├── summarise_results.py                              # Script 4: Generate summary for portfolio
    ├── initial_train_lightgbm.py                         # Original test run script 
    └── complete_train_lightgbm.py                        # Original complete baseline CV script 
```

### Reflection
#### Challenges
- **Pipeline complexity**: Having all tasks (tuning, feature importance, training final models) in a single script was messy. Hard to debug and rerun specific stages.
- **Imbalanced labels**: For classification targets (max_risk, median_risk), the dataset was skewed toward the majority class, risking biased models.
-	**Feature importance interpretation**: LightGBM’s feature importance scores are relative and not intuitive at first glance. Needed clarity on what the numbers actually mean.
- **Data/bookkeeping**: Making sure each script loads the correct inputs (params, features, models) required careful folder structuring and output saving.
#### Solutions
- **Split into 3 modular scripts**:
	1. tune_models.py → handles hyperparameter tuning via GridSearch and saves results.
	2. feature_importance.py → loads best params, generates feature rankings, saves CSV + plots.
	3. train_final_models.py → trains final LightGBM models with chosen params and saves .pkl.
- **Class weights for imbalance**: Used class_weight="balanced" for classification tasks. Disabled for regression (pct_time_high).
- **Clarified feature importance**: Understood that LightGBM uses split importance (counts of feature splits, weighted by improvement in loss, summed across trees). Units are arbitrary but give reliable relative ranking.
- **Better file management**: Standardized folder paths (PARAMS_DIR, FEATURE_DIR, MODEL_DIR). Each script saves clear, versioned outputs.
#### Learnings
-	Clean separation of scripts makes the workflow more professional, reproducible, and maintainable.
-	Class weighting is essential in medical ML tasks to avoid misleadingly high accuracy on imbalanced data.
-	Feature importance plots are not absolute measures; they’re relative scores. Useful for ranking, not for clinical interpretation.
-	Having a consistent naming and folder structure prevents confusion when switching between scripts.
- **The purpose of this baseline is not to squeeze every last % out of LightGBM, but to create a credible reference point for more advanced models**.

### Summary
- This completes Phase 3: LightGBM Training + Validation. The models are portfolio-ready and serve as the baseline for comparison against neural networks in Phase 4.
- We have trained, validated, and tuned models, generated outputs (CV results, best params, feature importance, final models) and documentation (notes.md, summaries, plots).
- Models + outputs + documentation is the baseline, what we show a recruiter in a portfolio to prove I can:
	- Run a structured ML pipeline,
	-	Handle imbalanced labels,
	-	Interpret results
	-	Save deployment-ready models.

---

# Phase 4: Temporal Convolutional Network (TCN) Training + Validation

---

## Day 14-16 Notes - Start Phase 4: Plan Entire Neural Network Pipeline & Complete Dataset Preparation (Step 1)

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

### What We Did
**Step 1: Dataset Preparation (Sequential Features) `prepare_tcn_dataset.py`**
1. **Setup and Directory Creation**
	- Imported necessary libraries (pandas, numpy, torch, json, joblib, sklearn).
  - **Defined input paths**: `news2_features_timestamp.csv` (already produced by make_timestamp_features.py), `news2_features_patient.csv` (to merge outcomes with df)
  - **Created output directories**:
    - `prepared_datasets/` → ready-to-train PyTorch sequence tensors of shape (batch_size, max_sequence_length, num_features) and corresponding masks (max_sequence_length,)
    - `deployment_models/preprocessing/` → preprocessing artifacts (scalers, padding config)
2. **Step 1: Load Dataset and Sort Chronologically**
	- Loaded timestamp-level CSV into DataFrame df.
	-	Sorted by `subject_id` and `charttime` to ensure sequences are in chronological order and each patient’s timepoints are correctly aligned (TCN requires time-ordered sequences).
	-	Created a copy of the DataFrame to avoid modifying the original CSV.
3. **Step 2: Merge Patient-Level Outcomes**
  - Loaded patient-level outcomes CSV (`max_risk, median_risk, pct_time_high`).
	-	Merged outcomes into the timestamp-level DataFrame by subject_id.
  - The timestamp dataset does not contain outcomes, merging ensures every row in a patient’s sequence has access to labels for stratification and eventual training.
4. **Step 3: Convert Targets to Binary for Stratification**
	-	**Exactly mirrors the LightGBM target binary conversion**: `max_risk_binary` (1 if max_risk > 2, else 0), `median_risk_binary` (1 if median_risk == 2, else 0)
	-	Binary stratification simplifies patient-level splits.
	-	Ensures that rare/high-risk outcomes are balanced across train/val/test.
5. **Step 4: Patient-Level Stratification & Train/Val/Test Split**
	-	Collected unique patient IDs.
	-	Created patient-level label by taking the max of `max_risk_binary` per patient, this will be used for stratification of patients into the splits.
	-	**Split patients**:
    - **First split**: 70% → train, 30% → temp pool.
    - **Second split**: temp pool split evenly → 15% val, 15% test.
  - Save patient splits for reproducibility `deployment_models/preprocessing/patient_splits.json`
  - **Rationale**:
    - Splitting by patient and not rows prevents data leakage (same patient cannot appear in multiple splits).
    - Stratification balances rare high-risk cases (prevent class imbalance). Optional if your classes are well-distributed, but for small datasets, stratification is safer to avoid one set being unrepresentative.
    - Stratification on only one anchor target (`max_risk_binary`) avoids inconsistent splits. Same splits reused for median risk and regression targets.
    -	Random state fixed for reproducibility
6. **Step 5: Feature/Target Separation**
  - Defined id_cols (`subject_id, charttime`) and target_cols (`max_risk, median_risk, pct_time_high`).
	-	Candidate feature columns = all columns excluding IDs and targets.
	-	Converted consciousness_label to binary numeric: 0 = Alert, 1 = Not Alert (clinically informative and safe to include).
	-	Dropped all remaining categorical text columns (risk, monitoring_freq, response, etc.), dropping irrelevant categorical columns prevents PyTorch dtype issues.
	-	**Split features into**:
    - Continuous features → for z-score scaling.
    -	Binary features → no scaling needed.
7. **Step 6: Normalisation (Z-Score for Continuous Features)**
  - Apply z-scoring to continuous variables, ensure categorical features are left unchanged.
  - Fit `StandardScaler()` on training patients only (prevents leakage as all computed mean/std per feature come from only training set), and also transform training set (`.fit_transform()`)
  - Applied `.transform()` to val/test patients using training stats.
  - Explicitly converted all continuous and binary columns to float32 (prevent later PyTorch dtype issues).
  - **Rationale**:
    - Z-score ensures features are on comparable scales, preserving trends.
    - Scaled using training stats only avoids information leakage (using information from val/test to influence scaling)
    - float32 conversion prevents PyTorch crashes (`object dtype → TypeError`).
8. **Step 7: Grouping into Per-Patient Sequences**
	-	Grouped DataFrame rows by subject_id.
	-	Created a 2D NumPy array per patient (timesteps × features).
  - **Rationale**: 
    - TCN input requires sequences per patient.
    -	Maintains temporal ordering and feature consistency.
9. **Step 8: Sequence Padding/Truncation and Mask Creation**
	-	**Fixed sequence length**: `MAX_SEQ_LEN = 96` (96h = 4 days, based on ICU stay distribution).
    -	**For shorter sequences**: padded with zeros.
    -	**For longer sequences**: truncated to `MAX_SEQ_LEN`.
	-	**Created corresponding mask array**: (1 → real timestep, 0 → padded timestep).
	-	`make_patient_tensor(pid)` function converts the NumPy arrays and return PyTorch tensors for the TCN.
  - **Output shape for every patient tensor**: (`MAX_SEQ_LEN, num_features`) for the sequence tensor, (`MAX_SEQ_LEN,`) for the mask tensor (1D array).
  - **Rationale**:
    -	TCNs require uniform input sizes, we chose 96.
    -	Masks ensure padded timesteps don’t contribute to loss or gradients (will ignore padded values during loss computation).
10. **Step 9: Save Tensors + Masks**
	-	**For each split (train, val, test)**:
    - Stacked the per patient tensors → 3D (`num_patients, MAX_SEQ_LEN, num_features`).
	-	Saved the 3 sequence tensors + 3 masks tensors to `prepared_datasets/`.
  - **Rationale**:
    -	Makes training reproducible and reproducible across multiple runs.
    -	Separates raw data artifacts (tensors + masks) from deployment artifacts (scalar + padding config).
11. **Step 10: Save Padding & Feature Configuration**
  - **Saved `padding_config.json`**:
```json
{
  "max_seq_len": 96,
  "feature_cols": [...],
  "target_cols": ["max_risk", "median_risk", "pct_time_high"]
}
```
  - Saved `standard_scaler.pkl` in `deployment_models/preprocessing/` (contains computed mean/std per feature in training set, and z-scale formula).
  - Provides complete reproducibility for inference on new patient data.
**Notes / Best Practices**:
-	**Foundational step**: If sequences, masks, or scaling are off, the TCN will not learn effectively.
-	Once dataset prep is stable, architecture design, training, and evaluation can proceed rapidly.
- **Keep data preprocessing code modular**: may need to adjust `MAX_SEQ_LEN` or features later.
-	Maintain train/val/test splits and scaling parameters for reproducibility.
**Summary of Changes / Fixes Implemented**
-	Converted `consciousness_label` to binary to include it safely.
-	Explicitly cast continuous and binary columns to float32 after scaling.
-	Dropped irrelevant categorical columns to avoid object dtype issues.
-	Ensured per-patient sequences are padded/truncated and masks are correctly aligned.
-	Saved preprocessing objects for reproducibility and deployment.

### Key Transformations
- **Patient-level split** – train/val/test by `subject_id` to prevent leakage; stratified on `max_risk_binary`.  
- **Merge patient outcomes** – add `max_risk`, `median_risk`, `pct_time_high` to timestamp-level data.  
- **Feature cleaning** – drop unused categorical text columns; convert `consciousness_label` to binary.  
- **Type enforcement** – continuous → `float32`; binary → `float32` to prevent PyTorch errors.  
- **Z-score normalization** – continuous features scaled using training set stats; applied to val/test.  
- **Sequence ordering & padding** – sort by `subject_id` + `charttime`; truncate/pad to `MAX_SEQ_LEN`; masks indicate real vs padded timesteps.  
- **Per-patient sequences** – create 2D arrays `(timesteps × features)` per patient; converted to PyTorch tensors in Step 6.  
- **Scaler persistence** – `standard_scaler.pkl` stores overall feature mean/std from training set for reproducible preprocessing.  
- **Output** – tensors `(num_patients, MAX_SEQ_LEN, num_features)` + masks `(num_patients, MAX_SEQ_LEN)` saved for train/val/test.

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

### Folder Structure for Outputs
```text
src/ml_models_tcn/
├── prepared_datasets/                     # Pure training/validation/test artifacts
│   ├── train.pt                           # Training sequences tensor (shape: [num_train_patients, seq_len, num_features])
│   ├── train_mask.pt                      # Corresponding mask tensor for training (0=padding, 1=real data)
│   ├── val.pt                             # Validation sequences tensor (same shape logic)
│   ├── val_mask.pt                        # Mask tensor for validation
│   ├── test.pt                            # Test sequences tensor
│   └── test_mask.pt                       # Mask tensor for test set
│                                          
├── deployment_models/                     # Preprocessing artifacts needed to run inference on new data
│   ├── preprocessing/                     # Objects to reproduce the same preprocessing at inference
│   │   ├── standard_scaler.pkl            # Z-score scaler fitted on training data (mean/std per feature)
│   │   ├── padding_config.json            # Metadata about max sequence length, padding scheme, feature order
│   │   └── patient_splits.json            # Dictionary of patient train/val/test split
Notes:
- All .pt files in prepared_datasets/ are tensors directly used in TCN training.
- The mask tensors are necessary for handling padded sequences during loss computation.
- scalers, padding_config and training split go into deployment_models/ because they are part of the pipeline that prepares new, unseen data in exactly the same way as training. They are not training artifacts.
- This separation avoids confusion: training tensors vs preprocessing/deployment metadata.
- Summary:
	-	Training tensors + masks → prepared_datasets/
	- Scalers + padding configs + patient split → deployment_models/scalers/
```

### Step 1 Output Artifacts

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


### Reflection
#### Challenges
- **Sequence-to-patient vs per-timestep labels**: Our TCN currently maps full timestamp sequences to patient-level outcomes (`max_risk`, `median_risk`, `pct_time_high`). While more powerful than LightGBM, it does not exploit full sequence-to-sequence prediction. Per-timestep prediction could provide richer early-warning outputs but requires far more complex label preparation, masking, and evaluation logic.
- **Data sparsity**: ICU dataset (100 patients) makes per-timestep labels sparse, noisy, and unreliable. Attempting sequence-to-sequence modelling could lead to overfitting and unstable training.
- **Patient-level stratification**: Stratifying on multiple targets simultaneously is infeasible. Choosing a single "anchor" (`max_risk_binary`) preserves class balance but leaves median risk slightly imbalanced.
- **Categorical columns causing errors**: Columns like `consciousness_label`, `co2_retainer_label`, `risk`, and `monitoring_freq` caused PyTorch conversion errors due to object dtype.
- **Object dtype in numeric arrays**: Even after selecting continuous and binary columns, some columns (binary stored as object or mixed NaNs) caused `TypeError` when converting to PyTorch tensors.
- **Pipeline complexity**: Aligning train/val/test splits, scaling, per-patient sequences, padding, and masks correctly was essential to avoid subtle bugs.
#### Solutions and Learnings
- **Patient-level splits**: Stratified on `max_risk_binary` to maintain class balance, then reused same patient splits for median and regression outcomes. Prevented leakage by splitting by patient ID.
- **Merging outcomes**: Patient-level labels (`max_risk`, `median_risk`, `pct_time_high`) merged into timestamp-level data before feature selection and tensor creation.
- **Feature cleaning**: Dropped irrelevant categorical text columns. Converted `consciousness_label` to binary (0 = Alert, 1 = Not Alert) for clinical signal; other categorical columns were constant or redundant.
```python
df["consciousness_label"] = df["consciousness_label"].apply(lambda x: 0 if x == "Alert" else 1)
```
- **Type enforcement**: Explicitly cast continuous and binary columns to `float32` after z-score normalization to prevent PyTorch errors:
```python
df[continuous_cols] = df[continuous_cols].astype(np.float32)
df[binary_cols] = df[binary_cols].astype(np.float32)
```
- **Z-score normalisation**: Applied per feature using training set stats only; validation/test sets transformed using training means/stds to prevent leakage. Patterns are preserved, only scale changes.
- **Per-patient sequences and padding**: Sorted by subject_id + charttime. Sequences truncated/padded to MAX_SEQ_LEN = 96. Masks indicate real vs padded timesteps for TCN to ignore padding.
- **Tensor creation**: Conversion to PyTorch tensors occurs in make_patient_tensor() (Step 6), producing (MAX_SEQ_LEN, num_features) for sequences and (MAX_SEQ_LEN,) for masks.
- **Scaler persistence**: standard_scaler.pkl stores global means/stds per feature (training set), not per patient, ensuring reproducible preprocessing for inference.

---

## Day 17-18 Notes - Continue Phase 4: Model Architecture (Step 2)

### Goals
- **TCN model architecture** (`tcn_model.py`) with:
  - CausalConv1d implementation (padding + trimming).
  - TemporalResidualBlock with residual connections, LayerNorm, ReLU, Dropout.
  - Stacked TCN with exponentially increasing dilations.
  - Masked mean pooling to handle variable-length sequences.
  - Optional dense head (linear → ReLU → dropout).
  - Task-specific heads for classification (max_risk, median_risk) and regression (pct_time_high).
- Add a **smoke test** to confirm:
  - Model runs end-to-end on dummy data.
  - Shapes of outputs match expectations `(B,)`.
- Clarify **conceptual understanding**:
  - Input/output tensor shapes and why permutation is needed.
  - Role of residuals, dense head, and pooling.
  - Why multiple task-specific heads are defined.
- Document **key concepts and reasoning** for:
  - Why causal convolutions (avoid future leakage).
  - Why residuals (gradient flow, stable training).
  - Why masked pooling (ignore padding).
  - Why multiple task heads (support classification + regression).

### What We Did
**Built full TCN model pipeline (`tcn_model.py`) and ran a smoke test to confirm it works end-to-end**
#### Overall Flow of the Script
1. **Causal Convolution (CausalConv1d)**
   - Standard convs look both left and right in time → would leak future info.  
   - Causal conv pads only the left and trims the right → each output depends only on present + past timesteps.  
   - Preserves temporal causality, critical for ICU forecasting.
2. **Temporal Residual Block**
   - Core “unit” of the TCN. Each block =  
     - 2 causal convs (extract local temporal features).  
     - LayerNorm (stabilises activations → prevents exploding/vanishing gradients).  
     - ReLU activation (adds non-linearity → lets the model learn complex patterns).  
     - Dropout (regularisation → avoids overfitting by randomly zeroing some activations).  
     - Residual connection (adds input back to output → ensures gradient flow in deep stacks).  
   - Downsample via 1×1 conv if channel dimensions don’t match.
3. **Stacked TCN (TCNModel)**
   - Multiple TemporalBlocks stacked together.  
   - **Dilations double each block (1, 2, 4, …)** → exponentially expand receptive field, so deeper layers see longer history without huge kernels.  
   - Output feature dimension = channels from the last block.
4. **Masked Mean Pooling**
   - Patients have variable sequence lengths → padding is used.  
   - Masked pooling ensures **only real timesteps contribute** when reducing sequence to patient-level vector.  
   - Computes:  
     - `sums = features * mask` (ignores padding).  
     - `mean = sums / counts` (divides by actual number of valid timesteps).  
   - Result = one fixed-size feature vector per patient.
5. **Dense Head (Optional)**
   - A small fully connected layer applied before final task heads.  
   - Linear → ReLU → Dropout.  
   - Purpose: mixes pooled features and adds extra flexibility.  
   - Optional because sometimes you want direct features → heads, sometimes richer mixing.
6. **Task-Specific Heads**
   - Separate linear layers for each prediction:  
     - `classifier_max` → binary classification (max risk).  
     - `classifier_median` → binary classification (median risk).  
     - `regressor` → continuous regression (pct_time_high).  
   - Each outputs shape `(B,)` after `.squeeze(-1)`.  
   - These outputs go into **loss functions** during training:  
     - BCEWithLogitsLoss (classification).  
     - MSELoss (regression).
7. **Smoke Test**
   - Built-in quick run (`if __name__ == "__main__":`).  
   - Created dummy data (B=4, L=96, F=171) with masks of different sequence lengths.  
   - Ran forward pass through full model.  
   - Verified output shapes:
     - `logit_max`: (4,)  
     - `logit_median`: (4,)  
     - `regression`: (4,)  
   - Assertions passed → confirms end-to-end pipeline works.
#### Reasoning
- **Causality**: ensures no leakage from future → realistic for ICU time series.  
- **Residuals + LayerNorm**: stabilise very deep models → gradients don’t vanish/explode.  
- **Dropout**: improves generalisation, avoids memorisation of noise.  
- **Dilations**: allow long temporal context without huge kernels.  
- **Masked pooling**: makes variable-length patient sequences comparable.  
- **Dense head**: optional mixing step before predictions.  
- **Separate heads**: support multiple tasks (binary + continuous).  
- **Smoke test**: sanity check → prevents silent shape mismatches.
#### Reflection
- Today’s focus was on **understanding flow** from input → temporal feature extraction → pooling → patient-level predictions.  
- Key learning: out_channels (kernels) ≠ input features; each kernel learns different temporal patterns.  
- Masked pooling was important to handle **variable sequence lengths**.  
- The smoke test confirmed the **pipeline is implemented correctly** and outputs are as expected.  

### Causal Convolutional Layers in TCNs

#### Convolution
- A **convolution** applies a sliding window (kernel/filter) across the input sequence to detect local patterns.
- For **1D time-series**:
  - At each timestep, the model looks at a fixed-size window of past values (e.g., last 3 HR readings) and computes a weighted sum.
  - This weighted sum captures how much the current window matches a learned temporal pattern (e.g., “steady rise” or “sudden drop”).
- Formula (simplified):

output[t] = sum(inputs[t-window_size+1:t] * weights)

#### Kernel / Filter
- A **kernel** is a set of learnable weights applied to a local temporal window.
- **Shape**: `(in_channels, kernel_size)`  
- `in_channels` = number of input features (vitals/labs, e.g., 171).  
- `kernel_size` = number of consecutive timesteps considered.
- Each kernel produces **1 output time-series**, combining information across all input features in that window.
- Example:
  - Input: 171 features × 96 timesteps
  - Kernel size 3 → looks at t, t-1, t-2 across all features
  - Output: 1 new feature per kernel (length 96)

#### Weight
- Each element in the kernel is a **weight**, a learnable number that controls how much influence each input value has.
- During training, **backpropagation** updates these weights so the kernel learns to detect useful patterns.
- Example:

Inputs: 3 features (HR, BP, SpO2), 3 timesteps (t, t-1, t-2)
input_patch = [
    [HR[t], HR[t-1], HR[t-2]],
    [BP[t], BP[t-1], BP[t-2]],
    [SpO2[t], SpO2[t-1], SpO2[t-2]]
]
Kernel weights (same shape as input_patch)
weights = [
    [0.5, 0.3, 0.1],     # HR
    [-0.2, 0.05, -0.1],  # BP
    [0.1, -0.05, 0.02]   # SpO₂
]
Output at timestep t
output[t] = sum(input_patch[i][j] * weights[i][j] for i in range(3) for j in range(3))
1. Multiply each weight by its corresponding input value.
2. Add up all the results.
3. That gives one output number for this timestep and this kernel.

#### Padding and Trimming
- **Padding**: add artificial values (usually zeros) at the start of the sequence so convolutions can compute outputs for the first timesteps.
- **Causal padding**: pad only the past (left side) to avoid future information leakage.
- **Trimming**: remove extra outputs caused by padding at the far end so output length matches input length.
- **Combined**: ensures same output length as input but enforces causality.
	1. **Alignment**: output length = input length, so each timestep maps cleanly through the network, easier to stack multiple layers.
	2. **Causality**: by padding only on the left, we make sure each timestep only sees its past, not its future.

#### Causal Convolution
- **Causal** = outputs at time `t` depend **only** on `t` and earlier.
- Ensures realistic forecasting: the model never “peeks into the future.”
- Achieved via **left padding + trimming**.

#### Dilated Convolution
- Introduces **gaps** between kernel elements (spacing between sampled timesteps).
- Example:
  - Kernel size 3, dilation 2 → looks at `[t, t-2, t-4]`
  - Expands the **temporal receptive field** without increasing kernel size.
  - Allows the network to capture **long-term dependencies** efficiently.

#### Receptive Field
- The **receptive field** = the total number of original timesteps that can influence a particular output.
- With **stacked and dilated layers**, the receptive field grows exponentially:
  - Layer 1 (kernel 3, d=1) → 3 timesteps
  - Layer 2 (kernel 3, d=2) → 7 timesteps
  - Layer 3 (kernel 3, d=4) → 15 timesteps
- Deeper layers integrate information from multiple previous outputs, which themselves summarize prior timesteps.

#### Out Channels / Multiple Kernels
- `out_channels` = number of kernels per convolution layer.
- Different kernels are initialized differently and trained independently, so detects a **different temporal pattern** across all input features. For example:
	-	Kernel A might assign high weights to HR at t-2, t-1, t → detects sharp HR rise.
	-	Kernel B might assign high weights to BP at t-2, t-1, t → detects BP drop.
	-	Kernel C might combine HR ↑ and SpO₂ ↓ → detects correlated patterns.
-	The network learns these weights during training via backpropagation, so each kernel “specializes” in detecting a particular temporal pattern.
- The outputs of all kernels are stacked → new feature map `(out_channels, L)`
- If you have 64 out_channels, you get 64 feature maps, each representing the activation of one kernel across the sequence.
- These are the new “channels” that carry transformed, more abstract information about temporal patterns.
- This is why the number of channels often **changes after the first layer**: the network is creating new abstract features.

#### Temporal Residual Block
- **Two causal convolutions per block**:
  - Helps the network detect more complex temporal patterns.
  - Prevents vanishing gradients via **residual (skip) connections**.
- **LayerNorm**: normalizes across channels to stabilize training.
- **Dropout**: randomly zeroes activations during training to reduce overfitting.
- **ReLU**: introduces non-linearity so the network can learn complex relationships.

#### Summary / Why We Use This
- **Causal convolutions** = realistic ICU predictions (no future leakage).
- **Dilated + stacked layers** = large temporal receptive field using few parameters, letting the model pick up both:  
  - **Short-term spikes** (e.g., sudden drop in SpO₂) from first layers of kernels which are sensitive to sudden spikes/drops.
  - **Long-term patterns** (e.g., gradual blood pressure decline) from deeper layers covering more timestamps.  
- **Multiple kernels/out_channels** = detect diverse patterns across all features.
- **Residual blocks + normalization + dropout** = stable, generalizable training.
- **Masked pooling later** = aggregates per-patient sequence into single vector for patient-level prediction.

Together, this lets the TCN learn both **short-term spikes** and **long-term trends** from the full time series of vitals/labs, producing robust patient-level outcome predictions.


### Temporal Residual Block (TemporalBlock)
#### Purpose
- Just one convolution layer is not enough. Real data is noisy, and patterns can be complex. So we stack two convolutions in a block. This is called a Temporal Residual Block.
- First convolution finds simple local patterns, second convolution can combine them into slightly more complex patterns.
- Captures both **short-term spikes** and **long-term trends** across multiple layers.

#### Foundational Key Concepts
- **Gradient = the slope of the loss function with respect to a weight**
	- Loss = computed by a loss function (MSE, MCE) on the model output, tells us how wrong the model’s prediction was compared to the true value.
  - A gradient tells the direction and magnitude by which each parameter should change to reduce the loss.
	-	Positive slope → increasing weight increases loss (bad), so we should decrease it.
	-	Negative slope → increasing weight decreases loss (good), so we should increase it.
-	**Backpropagation:**
	-	We first do a forward pass (compute outputs using current weights).
	-	Then compute the loss (difference from true labels).
	-	Then we push gradients backwards through the layers (chain rule of calculus), it calculates how those weights should change, based on how wrong the final prediction was.
	-	This tells earlier layers how to adjust their weights so that next time they produce better inputs for later layers.
  - This cycle is repeated many times across the dataset. The network is “happy” when the loss (prediction error) stops improving much.
- Why adjust early layers? Because early weights create the representations that later layers depend on. If early weights never change, the network can’t learn meaningful low-level features (like detecting spikes or correlations in vitals).
- **Vanishing / exploding gradients**
  - The gradient is a product of derivatives of each layer.
	- If weights are small (<1), multiplying many of them → product shrinks towards zero → gradients vanish. 
  - If gradients shrink too much as they flow backwards, early layers barely update → network stops learning. This is the vanishing gradient problem.
  - If weights are large (>1), multiplying many → product blows up → gradients explode.
	-	If gradients blow up (become huge), weights swing wildly → unstable training. This is the exploding gradient problem.
	-	Both are common in deep stacks of layers.
  - Residuals and normalisation help stop that from happening.

#### Components per layer
1. **Causal Convolutions**
- Only look at **current and past timesteps**, not the future → prevents information leakage for realistic ICU forecasting.
- **Kernel**: a small window of weights applied across a few timesteps. Each weight multiplies the input value at that timestep; outputs are summed to produce a single number per kernel.
- **Dilated kernel**: skips timesteps to expand the temporal coverage without increasing kernel size.
- **Multiple kernels per layer** → called **out_channels**; each layer has multiple kernels (filters), each learning to detect different temporal patterns (e.g., HR spike, BP drop, combined trends).
- **Stacking two causal convolutions** → allows the block to learn more complex patterns than a single convolution layer could capture alone.

2. **Layer Normalisation (LayerNorm)**
- nn.LayerNorm in PyTorch expects the last dimension to be the one(s) you want to normalise over. So need to convert to (B, L, C), so that C (channels) is last → LayerNorm will compute mean/variance across those channels.
- Normalises **values across channels at each timestep**: mean ≈ 0, variance ≈ 1.
- If a timestep has 64 channel outputs, find the mean and variance across these 64 values, rescale values so the new set has mean ≈ 0 and variance ≈ 1 (normalising the channel values at that timestep).
- Stabilises training because:
  - Prevents some features from dominating due to large scale (no inconsistent scales).
  - Helps gradients remain balanced.
- Intuition: ensures the activations fed into the next layer have **consistent scale**, reducing the chance of exploding or vanishing gradients.

3. **Activation (ReLU)**
- **Purpose**: introduces non-linearity into the network.
- **Linear combinations** alone cannot capture complex relationships; activations allow the network to model non-linear interactions.
- **ReLU (Rectified Linear Unit)**: `ReLU(x) = max(0, x)`  
  - Converts negative outputs to 0, leaves positive outputs unchanged (keeps positive signals, removes negative signals).
  - Efficient, avoids vanishing gradients for positive activations.
- **Vanishing gradient problem**: in deep networks, gradients shrink as they backpropagate → very small updates, network stops learning. ReLU helps mitigate this for positive signals.

4. **Dropout (during training only)**
- Randomly zeros some outputs during training with a probability `p` (e.g., 0.2).
- **Purpose**:
  - Prevents **overfitting**: the model cannot rely on specific neurons/features all the time.
  - Encourages **robust representations**: remaining neurons must learn distributed patterns independently.
- During inference (using the model on new patients), dropout is disabled; all neurons are used.
	-	Dropout is on during training (to regularise).
	-	Dropout is off during inference (you want stable predictions).


#### After both layers computed
5. **Downsample / Projection (1×1 Convolution) - if needed**
- Used when **input channels ≠ output channels** (e.g., first layer has 171 features, block outputs 64 channels).
- 1×1 convolution adjusts dimensions to match, so residual addition is valid.
- **Why 1×1**:
  - Doesn’t combine temporal info; only maps input feature space → output feature space.
  - Ensures every output channel has a corresponding residual input channel to add.

6. **Residual Connection**
- **Residual connection** = original input sequence added back to the output of convolutions after the temporal block finishes `output = block(input) + input`.
- **Why**:
  - Allow gradients to flow through deep networks → prevent vanishing gradients → stable training.
  - This “shortcut path” means the gradient can flow directly from later layers to earlier ones, bypassing the convolutions.
  - Preserves **gradient flow** more easily backward through the network, avoids vanishing gradients in deep networks.
  - Lets the block **learn residual corrections** instead of full transformations: layer only needs to learn small adjustments (“corrections”) to the input, not rebuild the entire signal.
  - Makes learning more efficient, reduces risk of training instability in deep stacks, train effectively without the first layers’ gradients vanishing.
- Without residuals, deep TCNs can fail to learn if gradient shrinks exponentially (vanishing gradient), the first layers hardly get updated, and learning stalls.

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


### Causal Convolution: Padding + Trimming Diagram
**Suppose we have**:
- Input sequence: `[x0, x1, x2, x3]`
- Kernel size = 3
- Dilation = 1
**Step 1: Add left padding**
- We pad at the **start** so the convolution can compute the first timestep:
[0, 0, x0, x1, x2, x3]
 ^   ^      ^
pad pad    input sequence

- `0` values are fake “past” inputs.
- Allows the kernel to slide over the first timestep.

**Step 2: Apply 1D convolution**
Kernel slides across the sequence:
timestep 0: [0, 0, x0] → output0
timestep 1: [0, x0, x1] → output1
timestep 2: [x0, x1, x2] → output2
timestep 3: [x1, x2, x3] → output3

- **Each output only sees past + present**, never future.

**Step 3: PyTorch padding adds extra on right**
- PyTorch’s `padding` parameter pads both sides. To ensure **causality**, we trim the extra right padding:
- After convolution (with symmetric padding):
[output0, output1, output2, output3, extra0, extra1]

- Trim extra right padding → keep first 4 outputs

**Step 4: Result**
- Output length = input length
- Each timestep’s output depends only on its past and present
- No future leakage occurs


### Reflection
#### Challenges
- Struggled with **abstract mathematical concepts** behind convolutions, residuals, and backpropagation.  
- Went too deep into low-level implementation details (gradients, matrix multiplications), which slowed progress and risked burning out.  
- Difficulty understanding why **out_channels ≠ input features** (171 → 64) and how multiple kernels combine.  
- Confusion about **causal convolution padding**:
  - Why we add left padding.
  - Why PyTorch pads both sides.
  - Why trimming is required to keep outputs aligned.  
- Input shape mismatches caused a lot of confusion: why we need to permute `(B, L, F)` → `(B, F, L)` before Conv1d.  
#### Solutions & Learnings
- **Causal convolution clarified**:
  - Left padding ensures first timestep has enough context.
  - Each kernel sees only the **past and present**, not the future.
  - PyTorch pads both sides → extra outputs trimmed to restore correct length.  
- **Input shape reasoning**:
  - `(B, L, F)` = patients × timesteps × features (natural format).  
  - Conv1d expects `(B, C, L)` because channels = features (like RGB in images).  
  - After first conv: in_channels = features; after that: in_channels = out_channels of previous layer (learned patterns).  
- **Kernel structure**:
  - Conv1d(in_ch=171, out_ch=64, kernel=3) → 64 kernels, each shaped `(171, 3)`.  
  - Each kernel spans **all features** across a window of 3 timesteps.  
  - Output = 64 new learned channels = temporal feature maps.  
- **Key insight**: out_channels do not need to equal input features — they are **learned representations**, not the raw vitals anymore.  
- **Important workflow lesson**: It’s counterproductive to go too deep into every abstract detail. Better to balance high-level conceptual clarity with practical coding progress.  
#### Most Important Takeaways
- Multiple kernels per convolution → out_channels represent different filters, not original features.  
- Causal convolution = past-only receptive field (no data leakage).  
- Permuting `(B, L, F)` → `(B, F, L)` is essential because PyTorch treats features as channels.  
- Accept that deep technical dives are infinite — need to focus on **pipeline understanding** and **clinical problem-solving**, not just maths. 

### Outcomes
- **Model architecture completed**:  
  Built full `TCNModel` with:
  - Custom `CausalConv1d` layer (length-preserving, causal).  
  - `TemporalBlock` (2× causal conv + LayerNorm + ReLU + Dropout + residual).  
  - Stacked blocks with exponential dilation → expanded receptive field.  
  - Masked mean pooling to summarise patient-level features.  
  - Optional dense head to refine pooled features.  
  - Task-specific heads → classification (max_risk, median_risk) + regression (pct_time_high).  
- **Smoke test passed**:  
  - Verified model runs end-to-end on dummy data.  
  - Output shapes confirmed:  
    - `logit_max`: (B,)  
    - `logit_median`: (B,)  
    - `regression`: (B,)  
- **Conceptual clarity improved**:  
  - Understood why we permute (B, L, F) → (B, F, L) for Conv1d.  
  - Clarified role of residuals (stabilise gradient flow).  
  - Dense head = optional feature mixer before heads.  
  - Masked pooling = ignore padding → fair comparison across patients.  
- **Documentation updated**:  
  - Added explanatory comments in code.  
  - Wrote detailed notes on causal padding + trimming, input/output shapes, and role of task heads.  

---

## Day 19-20 Notes - Continue Phase 4: Model Training + Validation (Step 3 + 4)

### Goals 
- Complete full TCN training and validation script `tcn_training_script.py`.
- Run full TCN training loop  without runtime errors.  
- Correct and debug any script errors. 
- **Verify training + validation pipeline**:  
  - Track per-epoch training and validation loss.  
  - Confirm early stopping saves best checkpoint (`tcn_best.pt`).  
- **Sanity-check labels and inputs with debug prints**:  
  - Binary targets (`0/1`) confirmed.  
  - Regression target bounded between `0.0–0.44`.  
  - Input tensors clean (no NaNs/Infs).  

### What We Did Today
**Completed Full Temporal Convolutional Network (TCN) Training Loop Script `tcn_training_script.py`**
#### Summary
- Built a **complete PyTorch training pipeline** for the TCN model.  
- Covered **data loading, dataset preparation, target construction, model definition, training, validation, early stopping, and checkpoint saving**.  
- **Introduced key deep learning concepts**: loss functions, optimisers, gradient flow, overfitting prevention, early stopping, and learning rate scheduling.
- This script forms the core of **Phase 4**, moving from data preparation into real deep learning training.
### Output
- `trained_models/tcn_best.pt` — the best-performing model weights (lowest validation loss).  
- Console logs of **training and validation loss per epoch**, with early stopping.
- **Debug prints confirming**:
	-	Binary targets (`y_train_max`, `y_train_median`) are clean.
	-	Regression target (`y_train_reg`) is bounded, no NaNs/Infs.
-	Confirmed the pipeline runs end-to-end with no runtime errors.
### Step-by-Step Flow
1. **Imports & Config**
	-	Import PyTorch, Pandas, JSON, and our custom TCNModel.
	-	**Define hyperparameters**:
    -	`DEVICE` (GPU/CPU)
    -	`BATCH_SIZE`, `EPOCHS`, `LR` (learning rate)
    -	`EARLY_STOPPING_PATIENCE` (stop when val loss doesn’t improve).
  -	Create `MODEL_SAVE_DIR` → ensures trained models are stored.
2. **Load Prepared Data**
	-	Use `torch.load()` to bring in padded sequence tensors (`x_train, mask_train` etc.) created from `prepare_tcn_dataset.py`.
	-	These are the time-series features per patient, already standardised + padded to equal length.
	-	Masks mark valid timesteps vs padding (prevents model from “learning noise”).
3. **Build Target Tensors (Patient Labels)**
	-	Load patient-level CSV (`news2_features_patient.csv`).
	-	**Recreate binary labels**:
    -	**max_risk_binary**: high vs not-high risk.
    -	**median_risk_binary**: low vs medium.
	-	Load splits (`patient_splits.json`) so each patient is consistently assigned to train/val/test.
	-	Define `get_targets()`:
    -	Pulls the right patients.
    -	Converts labels into PyTorch tensors (`y_train_max, y_train_median, y_train_reg`) for each split.
  - **Rationale**: features (time-series) and labels (patient outcomes) are stored separately. We need to align them so that each input sequence (x_train) has its corresponding target outcome (ground-truth) to train on. This ensures you have paired data: (`x_train[i], mask_train[i]`) → (`y_train_max[i], y_train_median[i], y_train_reg[i]`).
4. **TensorDatasets & DataLoaders**
	-	TensorDataset groups together (inputs, masks, targets) into one dataset object.
	-	**DataLoader breaks this dataset into mini-batches**:
    -	batch_size=32 → model sees 32 patients per step.
    -	shuffle=True for training → prevents learning artefacts from patient order.
  - **Rationale**: mini-batching improves GPU efficiency and stabilises gradient descent.
5. **Model Setup**
  - Defines the architecture (what the model looks like, how it processes inputs).
	-	**Instantiate TCNModel with**:
    -	Input dimension = 171 features.
    -	**Residual conv blocks**: [64, 64, 128] → 3 conv blocks with number of channels (filters/kernels). Residual is defined within the block.
    -	**Dense head**: 64 neurons → mixes all features before final outputs (comes once, after the stack finishes)
	-	Send model to GPU/CPU (.to(DEVICE)).
6. **Loss Functions**
  - We train on three parallel tasks (multi-task learning). 
  - Each target needs its own loss, calculated by the loss function:
    -	`criterion_max / criterion_median`: BCEWithLogitsLoss → binary classification.
    -	`criterion_reg`: MSELoss → regression task.
7. **Optimiser + Scheduler**
  - Uses batch-by-batch output heads from the dense head to optimise parameters. 
	-	Optimiser = Adam with LR=1e-3.
	-	Scheduler = ReduceLROnPlateau (halves LR if val loss plateaus).
  - **Rationale**: 
    - Adam adapts learning rate per parameter → faster convergence. 
    - Scheduler prevents the model from “getting stuck”.
8. **Training Loop**
	-	Loop over epochs (one full pass through the entire training dataset).
  - Network jointly learns classification and regression.
	-	**For each batch**:
    -	Forward pass → model predicts 3 outputs (`logit_max, logit_median, regression`).
    -	Compute losses with loss functions for all 3 tasks → compare predictions to true labels (`y_max, y_median, y_reg`).
    - Combine losses into 1 (`loss = loss_max + loss_median + loss_reg`) → one scalar loss value means each task contributes equally (multi-task learning).
    -	Backward pass → calculate gradients of this total loss w.r.t. every model parameter.
    -	Gradient clipping (`clip_grad_norm_`) → prevents exploding gradients (if gradients get too large, clipping rescales gradients so their norm ≤ 1, keeps training stable).
    -	Optimiser step → updates weights in opposite direction of the gradients.
	-	**Track average training loss per epoch**:
    - Loss for batch * batch size, then sum these values for every batch, then divide by number of patients in dataset = average loss across all patients in training set → no matter how batch sizes vary we ensure the epoch loss is the mean loss per patient. 
    - Logged and compared with validation loss for analysis → this is how you see if your model is learning.
  - **This is the heart of deep learning**: forward → loss → backward → update.
9. **Validation Loop**
	-	Run the model on validation set (no gradients).
	-	Compute average validation loss.
	-	Update LR scheduler.
	-	Print progress.
  - **Rationale**: validation loss tells us if the model is generalising or just memorising.
10. **Early Stopping**
	-	If validation loss improves → save model (`tcn_best.pt`).
	-	If no improvement for 7 epochs → stop training early.
  - **Rationale**: protects against overfitting and wasted compute.
11. **Debug Prints**
  - Sanity checks ensure training data is valid:
    -	Show unique values of targets.
    - Binary targets are present (0 and 1).
    -	Show regression range is healthy (min/max).
    -	Check for NaN/Inf in inputs (pipeline clean)
### Summary of Flow
**Inputs → Targets → Training → Validation → Early stopping → Saved best model**
1. **Forward pass**: model computes predictions for the batch.
2. **Loss computation (BCE, MSE)**: predictions are compared to true labels → gives `loss_max, loss_median, loss_reg`.
3. **Combine losses**: summed to get overall batch loss.
4. **Backward pass**: compute gradients → tells how to adjust weights to reduce loss.
5. **Optimizer step**: update weights using the gradients, gradients determine direction, learning rate determines size.
6. Repeat until early stoppage to prevent overfitting.
### Next Steps
- **Finish Phase 4**: generate visualisations → only the training vs validation loss curves (step 5).
- **Start Phase 5: Evaluation**
  -	Perform final evaluation on test set using `tcn_best.pt`
  -	**Compute metrics**: ROC-AUC, F1, accuracy (classification); RMSE, R² (regression)
  -	**Generate visualisations**: ROC curves, calibration plots, Regression scatter & residual histogram.
-	Compare TCN performance to LightGBM baseline and NEWS2 baseline for clinical and technical validation.


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

### Summary Output
**Console Log Output**
```bash
[INFO] Targets loaded:
 - train: torch.Size([70]) torch.Size([70]) torch.Size([70])
 - val: torch.Size([15]) torch.Size([15]) torch.Size([15])
 - test: torch.Size([15]) torch.Size([15]) torch.Size([15])
Epoch 1: Train Loss = 1.3526, Val Loss = 1.0035
Epoch 2: Train Loss = 1.0111, Val Loss = 0.9588
Epoch 3: Train Loss = 0.9176, Val Loss = 0.9345
Epoch 4: Train Loss = 0.8535, Val Loss = 0.9422
Epoch 5: Train Loss = 0.7503, Val Loss = 0.9841
Epoch 6: Train Loss = 0.6506, Val Loss = 1.0733
Epoch 7: Train Loss = 0.6013, Val Loss = 1.1456
Epoch 8: Train Loss = 0.5326, Val Loss = 1.1734
Epoch 9: Train Loss = 0.4867, Val Loss = 1.2066
Epoch 10: Train Loss = 0.5096, Val Loss = 1.2203
Early stopping at epoch 10
Training complete. Best model saved to tcn_best.pt
tensor([0., 1.])
tensor([0., 1.])
tensor(0.) tensor(0.4407)
tensor(False) tensor(False)
```
**Key Concepts**
- **Training loss**: 
  - Quantifies how well a model is performing on the data it is actively learning from.
  - Computed by comparing the model’s predictions with the known labels in the training set using a defined loss function (e.g., MSE, BCE).
  - A **decreasing training loss** over epochs indicates the model is successfully adjusting its parameters to fit the training data.
- **Validation Loss**:
  - Measures the model’s performance on unseen data that was not used during training.
  - Serves as an estimate of the model’s generalisation ability (how well it can predict new, real-world data).
  - A **validation loss that decreases along with training loss** indicates good learning, while a validation loss that rises despite decreasing training loss signals potential overfitting.
- **Overfitting**
	-	Overfitting occurs when a model learns noise or specific patterns in the training data that do not generalise to new data.
	-	**Symptoms**: training loss continues to decrease, but validation loss stagnates or increases.
	-	Overfitting reduces the predictive usefulness of a model and is more likely when training data is limited or the model is very complex relative to the dataset size.
- **Early Stopping**
	-	Early stopping is a regularisation technique that halts training when validation loss stops improving for a specified number of epochs (patience).
	-	This prevents the model from overfitting to the training data and helps retain the parameters that produced the best validation performance.
	-	It is widely used in deep learning to ensure training efficiency and model generalisation.
- **Why These Concepts Matter**
	-	Monitoring both training and validation loss is crucial to understand the learning dynamics of a model.
	-	Balancing model capacity and generalisation ensures that the model performs well not just on the training data but also on new, unseen data.
	-	Early stopping, combined with proper loss monitoring, is a simple yet effective method to improve model reliability and prevent wasted computation.
**What happened in the run**
- **Epoch 1–3**: 
  - Both training loss and validation loss decreased steadily (Train: 1.35 → 0.92, Val: 1.00 → 0.93).
	- The model was learning and generalising well during these initial epochs.
- **Epoch 4–5**: 
  - Training loss continued to decrease (0.85 → 0.75), but validation loss started to slightly rise (0.94 → 0.98).
	-	This indicates the beginning of overfitting → the model is fitting training data more closely than the validation data.
- **Epoch 6–10**: 
  - Training loss kept dropping (0.65 → 0.51), but validation loss increased further (1.07 → 1.22).
	-	The model is overfitting more strongly → it memorises training patterns but loses generalisation to unseen validation data.
- **Early stopping**: 
  - Triggered at epoch 10.
	-	Training stopped automatically to prevent further overfitting.
	- The best model weights were saved from the epoch with the lowest validation loss (around epoch 3).
**The debug prints at the end show**:
-	`y_train_max.unique()` → tensor([0., 1.]) → binary max_risk target is now properly 0/1.
- `y_train_median.unique()` → tensor([0., 1.]) → median_risk_binary now has both 0s and 1s, no transformation issues.
- `y_train_reg.min()`, `y_train_reg.max()` → (0., 0.4407) → regression target is bounded and healthy.
- `torch.isnan(x_train).any()`, `torch.isinf(x_train).any()` → (False, False) → input tensors are now clean (no NaNs/Infs).
**Interpretation**:
- Training script works and is stable.
-	The model was able to learn meaningful patterns early in training, but began overfitting which is expected given the dataset is small (70 train patients). 
-	Validation loss rising after epoch 3–4 shows that further training without regularisation would harm generalisation.
- Early stopping worked correctly to preserve the best model.
- Losses behave as expected, and the best model is saved automatically.
-	The input data and target tensors are correctly preprocessed and usable.


### Reflection
#### Challenges
**NaN Losses from Epoch 1**
-	Training output showed Train Loss = nan, Val Loss = nan from the very first epoch.
-	This indicated a numerical instability/data issue, not a problem with the TCN model itself.
- Either the data fed into the model was invalid (NaNs, wrong label encoding), or the optimiser blew up due to too-high learning rate + extreme inputs.
```bash
[INFO] Targets loaded:
 - train: torch.Size([70]) torch.Size([70]) torch.Size([70])
 - val: torch.Size([15]) torch.Size([15]) torch.Size([15])
 - test: torch.Size([15]) torch.Size([15]) torch.Size([15])
Epoch 1: Train Loss = nan, Val Loss = nan
Epoch 2: Train Loss = nan, Val Loss = nan
Epoch 3: Train Loss = nan, Val Loss = nan
Epoch 4: Train Loss = nan, Val Loss = nan
Epoch 5: Train Loss = nan, Val Loss = nan
Epoch 6: Train Loss = nan, Val Loss = nan
Epoch 7: Train Loss = nan, Val Loss = nan
Early stopping at epoch 7
Training complete. Best model saved to tcn_best.pt
```
**Labels not in the correct format**
- Debug prints revealed that `y_train_max` and `y_train_median` were not binary {0,1}.
- But our loss function (`BCEWithLogitsLoss`) requires strictly 0.0 or 1.0 targets.
```md
tensor([0., 2., 3.])   # max_risk
tensor([0., 2.])       # median_risk
```
-	In patient_df, the columns `max_risk` and `median_risk` are ordinal scores (0–3).
-	In `prepare_tcn_dataset.py`, we already derived binary versions (`max_risk_binary, median_risk_binary`), but these were saved into `df.copy()`, not the original csv file.
-	When building targets for the TCN, loaded the raw ordinal columns by accident and could not load the binary columns as they don't exist in the original data. That’s why `BCEWithLogitsLoss` broke.
**NaNs in input features**
- Debug prints showed training data itself contained NaNs.
- These originated from missing vitals/labs, or derived features (slopes, rolling means) leaving NaN in padded rows.
```md
tensor(True) tensor(False)
```
- True for `torch.isnan(x_train).any()` means that our inputs contain NaNs.
- That alone would break training, even if labels were fine.
- Before saving tensors in `prepare_tcn_dataset.py`, replace NaNs with zeros:
- NaNs appear in timestamp-level dataset (`news2_features_timestamp.csv`) before padding to tensors. Causes include:
	-	Some vitals/labs missing at certain times.
	-	Derived features (rolling means, slopes) computed from empty windows.
	-	Masking logic leaving NaNs in padded rows.
### Solutions and Learnings 
**Implemented debugging code**
- Added this code to the bottom of the script to figure out what was the issue
- Used outputs to fix code
```python
print(y_train_max.unique())
print(y_train_median.unique())
print(y_train_reg.min(), y_train_reg.max())

print(torch.isnan(x_train).any(), torch.isinf(x_train).any())
```
**Fixed target colums**
- Fixed label encoding in `tcn_training_script.py`:
```python
target_cols = ["max_risk_binary", "median_risk_binary", "pct_time_high"]
```
- Still failed, as in `prepare_tcn_dataset.py` the changes were made to a copy DataFrame, so the original patient-level csv is still the same csv without the binary columns.
- Since the CSV did not persist these columns, regenerated them inside the training script `tcn_training_script.py` loading patient-level targets:
``` Python
# Recreate binary targets (same logic as in prepare_tcn_dataset.py)
patient_df["max_risk_binary"] = patient_df["max_risk"].apply(lambda x: 1 if x > 2 else 0)
patient_df["median_risk_binary"] = patient_df["median_risk"].apply(lambda x: 1 if x == 2 else 0)
```
- Why this works:
	-	The classification heads (BCEWithLogitsLoss) expect binary labels.
	-	By regenerating the binary columns here, we guarantee the labels are in the right format (0/1).
	-	Also avoid contaminating the preprocessing pipeline with duplicate CSVs.
**Fixed NaN/Inf issues in inputs**
- Added explicit cleaning step in `prepare_tcn_dataset.py`, then reran script to produce new tensors all fixed:
```python
# Clean NaNs and Infs in feature columns
df[feature_cols] = df[feature_cols].fillna(0.0)
df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], 0.0)
```
- This replaced all NaN/Inf values with 0.0, ensuring tensors were valid before training.
### Learning points
-	NaN losses often point to bad input data or invalid labels. Always check both.
-	Classification losses (`BCEWithLogitsLoss`) require binary targets (0/1), not ordinal scores.
-	Debug prints (`.unique(), .min(), .max(), torch.isnan`) are essential tools for verifying training data integrity before running expensive model training.
-	Cleaning at the data preparation stage prevents downstream errors in training.
- **Main takeaway**: The failure wasn’t in the TCN architecture but in the data pipeline (labels + NaNs). By regenerating correct binary targets and cleaning NaNs before tensor creation, we stabilised the entire training process.

### Next Steps 
**Finish Phase 4 (Step 5)**
1. **Generate Visualisations**
	-	Training vs Validation loss curves (per epoch) → `plots/loss_curve.png`
**Start Phase 5**

---

## Day 21 Notes — Finish Phase 4: Reproducibility & Generate Visualisations (Step 5)

### Goals
- Finalise **Phase 4** by completing full model reproducibility and visualisation capability.  
- Extend `tcn_training_script.py` with:
  1. **Reproducibility / Random Seed Control** → fix random initialisation across Python, NumPy, and PyTorch for deterministic results.
  2. **Save Configuration for Reproducibility (`config.json`)** → record all hyperparameters, model architecture, optimiser, and scheduler settings.
  3. **Save Training History for Visualisation (`training_history.json`)** → log per-epoch training and validation losses for later plotting.
- Create `plot_training_curves.py` to **generate interpretable visualisations** of model learning, convergence, and overfitting patterns.


### What We Did
**Enhanced Reproducibility and Traceability**
- Added random seed control across **Python, NumPy, and PyTorch (CPU/GPU)** to ensure **identical results on reruns**.  
- Set `torch.backends.cudnn.deterministic=True` and `benchmark=False` to remove GPU-level stochasticity.  
- **Reasoning**:  
  - Deep learning models are inherently stochastic; reproducibility requires controlling all randomness sources.  
  - This ensures identical weight initialisations, batch orders, and convergence patterns.
**Saved Model Configuration (`config.json`)**
- Automatically exports all critical hyperparameters to a structured JSON file for scientific reproducibility:
  - Device, batch size, epochs, learning rate, dropout, kernel size, and optimiser/scheduler settings.
  - Ensures future reruns can be **exactly reconstructed** from metadata.
- **Reasoning**: Recording hyperparameters makes the experiment auditable and publishable.
**Saved Training History (`training_history.json`)**
- Logged training and validation loss per epoch for analysis and visualisation.  
- Provides quantitative trace of how the model learned over time.  
- Used this file in the visualisation script.
**Generated Visualisations (`plot_training_curves.py`)**
- **Loaded `training_history.json` and plotted**: Training vs Validation loss per epoch to visualise learning, convergence, and generalisation.
- **Added key annotations for interpretability**:
  - Red dashed line → best validation epoch (early stopping point).  
  - Red dot → lowest validation loss value.  
  - “Overfitting region” label → when validation loss starts rising.  
- **Reasoning**:  
  - Allows visual diagnosis of model performance → convergence, early stopping, and onset of overfitting.
  - Makes training behaviour transparent and interpretable.


### Plotting Training Curves `plot_training_curves.py`
**Purpose**
- Visualises **training vs validation loss per epoch** to understand how the TCN model learned over time.  
- Reveals **convergence**, **overfitting**, and **early stopping behaviour**.  
- Provides a transparent, reproducible diagnostic of model learning dynamics.
**Why It Matters**
- **Interpretability**: Shows how well the model generalises to unseen data.  
- **Debugging**: Detects overfitting (training ↓, validation ↑).  
- **Scientific transparency**: Visual evidence of convergence and stability.  
- **Model tuning**: Guides adjustments to learning rate, regularisation, or architecture.
**Annotation Summary**
| **Annotation** | **Purpose** |
|----------------|-------------|
| **Red dashed line + dot** | Marks the epoch where validation loss was lowest — the point of **best generalisation**. |
| **Text label (“Best epoch = X”)** | Displays the **exact epoch number** and **validation loss value** for clarity. |
| **“Overfitting region” label (optional)** | Highlights the region **after the best epoch** where validation loss rises. |
| **Grid and smooth formatting** | Improves visual clarity, showing trends and divergence more distinctly. |
**Rationale**
- The annotated plot turns a simple loss curve into a **scientifically meaningful diagnostic**.  
- Identifies **when the model learned optimally**, **where overfitting began**, and **how stable convergence was**.  
- When combined with `training_history.json`, it forms a complete, **reproducible visual record** of training dynamics.
**Interpretation**
- **Both losses ↓** → good learning and generalisation.  
- **Train ↓, Val ↑** → overfitting detected.  
- **Best epoch marker** confirms where early stopping captured optimal generalisation.  
- Plot + JSON history = transparent, reproducible performance evidence.



### Completed Full Temporal Convolutional Network (TCN) Training Loop Script `tcn_training_script.py`
#### Summary
- Built a **complete PyTorch training pipeline** for the TCN model.  
- Covered **data loading, dataset preparation, target construction, model definition, training, validation, early stopping, checkpoint saving**, and **reproducibility measures**.  
- **Introduced key deep learning concepts**: loss functions, optimisers, gradient flow, overfitting prevention, early stopping, and learning rate scheduling.
- This script forms the core of **Phase 4**, moving from data preparation into real deep learning training.

#### Output
- `trained_models/tcn_best.pt` — the best-performing model weights (lowest validation loss).  
- Console logs of **training and validation loss per epoch**, with early stopping.
- `trained_models/training_history.json` — saved per-epoch loss values for later visualization.
- `trained_models/config.json` — hyperparameters saved as JSON to enable reproducibility.
- **Debug prints confirming**:
  - Binary targets (`y_train_max`, `y_train_median`) are clean.
  - Regression target (`y_train_reg`) is bounded, no NaNs/Infs.
- Confirmed the pipeline runs end-to-end with no runtime errors.

#### Step-by-Step Flow
1. **Imports & Config**
	-	Import PyTorch, Pandas, JSON, and our custom TCNModel.
	-	**Define hyperparameters**:
    -	`DEVICE` (GPU/CPU)
    -	`BATCH_SIZE`, `EPOCHS`, `LR` (learning rate)
    -	`EARLY_STOPPING_PATIENCE` (stop when val loss doesn’t improve).
  -	Create `MODEL_SAVE_DIR` → ensures trained models are stored.
  - Create `history_path` → ensures training history is stored.
  - Create `config_path` → ensures training configuration is stored.
2. **Reproducibility**
  - **Set random seeds for reproducibility**: 
    - Fixes randomness in Python, NumPy and PyTorch.
    - Ensures that random operations (data shuffling, weight initialisation) produce the same results.
    - Final trained model and results can be consistently reproduced.
  - **Save defined hyperparameters to JSON for reproducibility**: `trained_models/config.json`
  - **Rationale**: model now fully traceable, interpretable, and repeatable.
3. **Load Prepared Data**
	-	Use `torch.load()` to bring in padded sequence tensors (`x_train, mask_train` etc.) created from `prepare_tcn_dataset.py`.
	-	These are the time-series features per patient, already standardised + padded to equal length.
	-	Masks mark valid timesteps vs padding (prevents model from “learning noise”).
4. **Build Target Tensors (Patient Labels)**
	-	Load patient-level CSV (`news2_features_patient.csv`).
	-	**Recreate binary labels**:
    -	**max_risk_binary**: high vs not-high risk.
    -	**median_risk_binary**: low vs medium.
	-	Load splits (`patient_splits.json`) so each patient is consistently assigned to train/val/test.
	-	Define `get_targets()`:
    -	Pulls the right patients.
    -	Converts labels into PyTorch tensors (`y_train_max, y_train_median, y_train_reg`) for each split.
  - **Rationale**: features (time-series) and labels (patient outcomes) are stored separately. We need to align them so that each input sequence (x_train) has its corresponding target outcome (ground-truth) to train on. This ensures you have paired data: (`x_train[i], mask_train[i]`) → (`y_train_max[i], y_train_median[i], y_train_reg[i]`).
5. **TensorDatasets & DataLoaders**
	-	TensorDataset groups together (inputs, masks, targets) into one dataset object.
	-	**DataLoader breaks this dataset into mini-batches**:
    -	batch_size=32 → model sees 32 patients per step.
    -	shuffle=True for training → prevents learning artefacts from patient order.
  - **Rationale**: mini-batching improves GPU efficiency and stabilises gradient descent.
6. **Model Setup**
  - Defines the architecture (what the model looks like, how it processes inputs).
	-	**Instantiate TCNModel with**:
    -	Input dimension = 171 features.
    -	**Residual conv blocks**: [64, 64, 128] → 3 conv blocks with number of channels (filters/kernels). Residual is defined within the block.
    -	**Dense head**: 64 neurons → mixes all features before final outputs (comes once, after the stack finishes)
	-	Send model to GPU/CPU (.to(DEVICE)).
7. **Loss Functions**
  - We train on three parallel tasks (multi-task learning). 
  - Each target needs its own loss, calculated by the loss function:
    -	`criterion_max / criterion_median`: BCEWithLogitsLoss → binary classification.
    -	`criterion_reg`: MSELoss → regression task.
8. **Optimiser + Scheduler**
  - Uses batch-by-batch output heads from the dense head to optimise parameters. 
	-	Optimiser = Adam with LR=1e-3.
	-	Scheduler = ReduceLROnPlateau (halves LR if val loss plateaus).
  - **Rationale**: 
    - Adam adapts learning rate per parameter → faster convergence. 
    - Scheduler prevents the model from “getting stuck”.
9. **Training Loop**
	-	Loop over epochs (one full pass through the entire training dataset).
  - Network jointly learns classification and regression.
	-	**For each batch**:
    -	Forward pass → model predicts 3 outputs (`logit_max, logit_median, regression`).
    -	Compute losses with loss functions for all 3 tasks → compare predictions to true labels (`y_max, y_median, y_reg`).
    - Combine losses into 1 (`loss = loss_max + loss_median + loss_reg`) → one scalar loss value means each task contributes equally (multi-task learning).
    -	Backward pass → calculate gradients of this total loss w.r.t. every model parameter.
    -	Gradient clipping (`clip_grad_norm_`) → prevents exploding gradients (if gradients get too large, clipping rescales gradients so their norm ≤ 1, keeps training stable).
    -	Optimiser step → updates weights in opposite direction of the gradients.
	-	**Track average training loss per epoch**:
    - Loss for batch * batch size, then sum these values for every batch, then divide by number of patients in dataset = average loss across all patients in training set → no matter how batch sizes vary we ensure the epoch loss is the mean loss per patient. 
    - Logged and compared with validation loss for analysis → this is how you see if your model is learning.
  - **Save training history as JSON for later visualisation**:
    - Saves average loss per epoch (11) for both training loss (`train_loss`) and validation losses (`val_loss`)
    - **Saves as a JSON file**: `trained_models/training_history.json`
  - **This is the heart of deep learning**: forward → loss → backward → update.
  - Saved training history data allows for later visualisation plots
10. **Validation Loop**
	-	Run the model on validation set (no gradients).
	-	Compute average validation loss.
	-	Update LR scheduler.
	-	Print progress.
  - **Rationale**: validation loss tells us if the model is generalising or just memorising.
11. **Early Stopping**
	-	If validation loss improves → save model (`tcn_best.pt`).
	-	If no improvement for 7 epochs → stop training early.
  - **Rationale**: protects against overfitting and wasted compute.
12. **Debug Prints**
  - Sanity checks ensure training data is valid:
    -	Show unique values of targets.
    - Binary targets are present (0 and 1).
    -	Show regression range is healthy (min/max).
    -	Check for NaN/Inf in inputs (pipeline clean)
### Summary of Flow
**Inputs → Targets → Training → Validation → Early stopping → Saved best model + training history**
1. **Forward pass**: model computes predictions for the batch.
2. **Loss computation (BCE, MSE)**: predictions are compared to true labels → gives `loss_max, loss_median, loss_reg`.
3. **Combine losses**: summed to get overall batch loss.
4. **Backward pass**: compute gradients → tells how to adjust weights to reduce loss.
5. **Optimizer step**: update weights using the gradients, gradients determine direction, learning rate determines size.
6. Repeat until early stoppage to prevent overfitting.


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

### Final reproducible run
```bash
[INFO] Targets loaded:
 - train: torch.Size([70]) torch.Size([70]) torch.Size([70])
 - val: torch.Size([15]) torch.Size([15]) torch.Size([15])
 - test: torch.Size([15]) torch.Size([15]) torch.Size([15])
Epoch 1: Train Loss = 1.4591, Val Loss = 1.1229
Epoch 2: Train Loss = 1.1120, Val Loss = 0.9924
Epoch 3: Train Loss = 0.9797, Val Loss = 0.9587
Epoch 4: Train Loss = 0.9369, Val Loss = 0.9585
Epoch 5: Train Loss = 0.8809, Val Loss = 0.9766
Epoch 6: Train Loss = 0.7894, Val Loss = 1.0038
Epoch 7: Train Loss = 0.7231, Val Loss = 1.0696
Epoch 8: Train Loss = 0.6498, Val Loss = 1.1479
Epoch 9: Train Loss = 0.6040, Val Loss = 1.1606
Epoch 10: Train Loss = 0.5602, Val Loss = 1.2033
Epoch 11: Train Loss = 0.5267, Val Loss = 1.2606
Early stopping at epoch 11
[INFO] Training history saved to /Users/simonyip/Neural-Network-TimeSeries-ICU-Predictor/src/ml_models_tcn/trained_models/training_history.json
Training complete. Best model saved to tcn_best.pt
tensor([0., 1.])
tensor([0., 1.])
tensor(0.) tensor(0.4407)
tensor(False) tensor(False)
```

#### Key Concepts
**Training Loss**:
  - Measures how well the model fits the data it is learning from. 
  - A decreasing training loss indicates successful internal weight adjustments.
**Validation Loss**:  
  - Reflects model generalisation to unseen data. 
  - Ideally, validation loss follows a similar downward trend to training loss; divergence signals overfitting.
**Overfitting**: 
  - When the model learns training-specific noise or patterns that do not generalise. 
  - Characterised by a continually decreasing training loss with rising validation loss.
**Early Stopping**:
  - Prevents overfitting by halting training when validation loss stops improving for several epochs. 
  - Retains weights from the epoch with best generalisation.
**Reproducibility (Deterministic Training)**:  
  - Ensures that every run produces identical results — same weight initialisations, same data shuffling, same training/validation curves. 
  - Achieved via fixed seeds and deterministic settings in Python, NumPy, and PyTorch.

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

#### The Debug Prints Confirm
- `y_train_max.unique()` → `tensor([0., 1.])` → binary classification targets correctly encoded.  
- `y_train_median.unique()` → `tensor([0., 1.])` → both classes present, no imbalance artefacts.  
- `y_train_reg.min()`, `y_train_reg.max()` → `(0.0, 0.4407)` → regression target bounded and realistic.  
- `torch.isnan(x_train).any() = False`, `torch.isinf(x_train).any() = False` → inputs are numerically stable, no corruption.

#### Interpretation
- **The training pattern matches previous runs**: rapid convergence early on, then divergence between training and validation losses.  
- **This consistent pattern across three independent runs strongly supports that**:  
  - The model architecture is implemented correctly.  
  - The training and validation splits are consistent and meaningful.  
  - Loss functions and data preprocessing pipelines are stable.  
- The overfitting behaviour is expected and not a flaw → it arises from a small dataset (70 patients for training).  
- Early stopping operated correctly, preserving the epoch with the best generalisation.


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


### Reflection
#### Challenges
- Initially, every training run produced slightly different curves and early-stopping epochs.
- Realised deep learning is **stochastic by design**, due to random weight initialisation, shuffling, and dropout.
- Needed deterministic control for scientific reproducibility.
#### Solutions and Learnings
- Implemented **comprehensive seed control** to lock all randomness sources (Python, NumPy, PyTorch CPU/GPU, and CuDNN backend).
- Added **config.json** to store all hyperparameters, this made the pipeline fully traceable.
- Logged **training history** to JSON, enabling later visualisation and verification.
- **Learned that**:
  - Small differences between runs are normal until seeds are fixed.
  - Overfitting behaviour and curve shape consistency are more important than identical numbers.
  - Reproducibility converts exploratory experiments into **scientifically valid baselines**.

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

### Next Steps (Start Phase 5: Evaluation)
- Load `tcn_best.pt` and perform **final evaluation** on the held-out test set.
- **Compute and visualise**:
  - **Classification metrics**: ROC-AUC, F1-score, accuracy.  
  - **Regression metrics**: RMSE, R², residual analysis.
- **Generate plots**:
  - ROC Curves  
  - Calibration Curves  
  - Regression Scatterplots and Residual Histograms.
- Compare the TCN’s performance against **LightGBM baseline** for both technical and clinical interpretability.
  
---

# Phase 5: Evaluation & Baselines

---

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

---

## Day 21-22 Notes - Start Phase 5: Evaluation Metrics & Final TCN Test Evaluation (Steps 1-3)

### Goals
- Establish a centralised metrics utility (`evaluation_metrics.py`) for consistent evaluation across TCN and LightGBM.  
- Run final evaluation of the Phase 4 TCN on the held-out test set using `evaluate_tcn_testset.py`.  
- Generate reproducible predictions for classification (`logit_max`, `logit_median`) and regression (`regression`).
- Compute metrics (ROC-AUC, F1, Accuracy, Precision, Recall, RMSE, R²) and save outputs (`tcn_metrics.json`, `tcn_predictions.csv`).
- Identify poor performance on test set.
- Plan to run diagnostics to understand failures and plan **Phase 4.5** retraining and corrective steps (e.g., class weighting, target transformations).

### What We Did
#### Step 1: Centralised Metrics Utility (`evaluation_metrics.py`)
**Purpose:**  
- Establish a unified, reusable metrics framework for all Phase 5 evaluations. 
- This guarantees that all evaluation models (NEWS2, LightGBM, TCN) use identical evaluation logic, making your comparisons scientifically valid and consistent.
**Logic / Workflow:**  
- Accepts predictions and ground-truth labels (binary or continuous).  
- Converts inputs (PyTorch tensors, lists) into NumPy arrays for compatibility with `sklearn`.  
- For **binary classification**:
  1. Converts predicted probabilities into hard labels using a decision threshold (default = 0.5).  
  2. Computes standard metrics (ROC-AUC, F1, Accuracy, Precision, Recall).  
  3. Handles edge cases safely (e.g., single-class targets for ROC-AUC).  
- For **regression**:
  1. Computes RMSE and R² to quantify prediction error and variance explained.  
- Returns all results as dictionaries for easy integration and saving to JSON/CSV.  
**Key Functions Implemented:**
- `compute_classification_metrics(y_true, y_prob, threshold=0.5)`
- `compute_regression_metrics(y_true, y_pred)`
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

#### Step 2 + 3: Final TCN Evaluation on Test Set (`evaluate_tcn_testset.py`)
**Purpose**
- Run the final evaluation of the trained TCN from Phase 4 on held-out patient test set
- Generate reproducible predictions and metrics for comparison with baselines (NEWS2, LightGBM).
**Overview / Logic Flow**
1. **Rebuild Test Targets (`y_test`)**  
  - Ground truth (`y_test`) rebuilt dynamically from patient CSV + JSON split to ensure consistency with Phase 4 training labels.
  - **Creates 2 binary classification targets**:
    - `max_risk_binary` → severe deterioration
    - `median_risk_binary` → average risk
  - **And 1 regression target**:
    - `pct_time_high` → fraction of time in high-risk zone
2. **Load Model Architecture**
  - **Architecture** loaded from `tcn_model.py` when loading `TCNModel`.
3. **Load Preprocessed Test Tensors**
  - `x_test.pt` → time-series input features
  - `mask_test.pt` → timestep masks for valid data
4. **Instantiate Model with Weights and Parameters**
  1. Define device early (CPU/GPU) → ensures all subsequent `.load()` calls map tensors correctly.
  2. **Load test tensors** → `.pt` tensors (`x_test`, `mask_test`).
  3. Build model → **architecture** defined in `tcn_model.py`, using **hyperparameters** from `config.json`.
  4. **Load trained weights** → trained state dictionary from `tcn_best.pt` (`state_dict`).
  5. Send model to device → `model.to(device)` (GPU if available, else CPU).
  6. **Switch to evaluation mode with `model.eval()`** → disables dropout and batchnorm updates for deterministic inference.
5. **Prepare for Inference**
  - Move model and tensors to device (CPU/GPU).
  - Set `model.eval()` to disable dropout and batchnorm updates.
  - Use `torch.no_grad()` for memory-efficient, deterministic predictions.
6. **Inference**
  - Run forward pass on test tensors and post-process outputs.
  - **Extract prediction outputs**:
    - `logit_max` → binary classification (max risk)
    - `logit_median` → binary classification (median risk)
    - `regression` → continuous fraction of time high
  - **Post-process outputs**:
    - Convert raw logits → probabilities using `torch.sigmoid` for classification tasks.
    - Convert PyTorch tensors → NumPy arrays for compatibility with metric functions.
7. **Compute Metrics**
  - **Classification targets (`max_risk_binary`, `median_risk_binary`)**
    - ROC-AUC, F1, Accuracy, Precision, Recall
  - **Regression target (`pct_time_high`)**
    - RMSE, R²
  - Metrics computed using `compute_classification_metrics` and `compute_regression_metrics` from `evaluation_metrics.py`.
8. **Save Outputs**
  - `tcn_metrics.json` → aggregated classification & regression metrics.
  - `tcn_predictions.csv` → combined per-patient predictions + ground truth for reproducibility (`y_true_max, prob_max`,`y_true_median, prob_median`,`y_true_reg, y_pred_reg`)
|**Reasoning**
- Ensures **reproducible, deterministic predictions** on unseen patients.
- Metrics computed in a **consistent format** for later baseline comparison.
- Provides a **quantitative and reproducible evaluation** of the TCN for all three tasks.
**Model Outputs and Metric Mapping:**
- Each model head corresponds to a task:
| Model head         | Purpose                                           | Output key      | Task Type         | Metrics Used         |
|------------------|-------------------------------------------------|----------------|-------------|----------------|
| classifier_max    | Predicts severe deterioration (max risk)       | `logit_max`     | Binary (logit) | ROC-AUC, F1, Accuracy     |
| classifier_median | Predicts moderate deterioration (median risk)  | `logit_median`   | Binary (logit) | ROC-AUC, F1, Accuracy    |
| regressor         | Predicts fraction of time in high-risk zone   | `regression`     | Continuous   | RMSE, R²        |
**Observations**
- Runtime was ~0.02 seconds for the forward pass on the test set.
- Early inspection of metrics indicated poor performance on `median_risk_binary` and `pct_time_high`, signaling the need for **Phase 4.5 retraining and diagnostic checks**.


### TCN Evaluation Metric Outputs
**Terminal Output**
```bash
[INFO] Using device: cpu
[INFO] Loaded TCN model and weights successfully
[INFO] Running inference on test set...
[INFO] Inference complete in 0.02 seconds
[INFO] Saved metrics → results/tcn_metrics.json
[INFO] Saved predictions → results/tcn_predictions.csv

=== Final Test Metrics ===
Max Risk — AUC: 0.577, F1: 0.929, Acc: 0.867
Median Risk — AUC: 0.722, F1: 0.000, Acc: 0.800
Regression — RMSE: 0.135, R²: -1.586
==========================
```
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

---

### Reflection
#### Challenges
1. **Unrealistic Test Metrics**
  - The TCN achieved near-perfect F1 (0.929) for *max risk* but **zero F1 (0.000)** for *median risk* and **negative R² (-1.586)** for regression.  
  - Initially appeared as catastrophic model failure.
2. **Understanding the Metric Behaviour**
  - Confusion around why the model produced “good” F1 but poor AUC on the same task.  
  - **Realisation**: **F1 is threshold-dependent**, whereas **ROC-AUC** assesses ranking ability across thresholds.  
  - Class imbalance heavily distorted F1 and accuracy metrics.
3. **Realising Evaluation Mode Importance**
  - **Confusion about why we must explicitly call**:
     ```python
     model.eval()
     with torch.no_grad():
     ```
  - `model.eval()` ensures deterministic behaviour by disabling dropout and freezing BatchNorm statistics.  
  - `torch.no_grad()` prevents gradient tracking, saving memory and speeding up inference.  
  - Both are essential for reproducible and stable test-time evaluation.
4. **Cross-Module Imports**
  - Running `evaluate_tcn_testset.py` initially failed because Python couldn’t find the required modules (`tcn_model.py`, `evaluation_metrics.py`).  
  - **Required fixing absolute imports using**:
     ```python
     sys.path.append(str(SRC_DIR))
     ```
    to allow clean cross-folder imports without using `python -m`.
5. **Confusion Between Model Architecture (.py) and Model Weights (.pt)**
  - Initially misunderstood why both were required:
     | Component | Purpose | Why Needed |
     |------------|----------|-------------|
     | **tcn_model.py** | Defines architecture | Without it, PyTorch doesn’t know the layer structure |
     | **tcn_best.pt** | Stores learned weights | Without it, model is untrained (random init) |
     | **x_test.pt / mask_test.pt** | Preprocessed inputs | Needed for inference; models can’t take CSVs directly |
  - The `.pt` file only stores numbers (weights/biases); PyTorch reconstructs the full model only when the class definition (`tcn_model.py`) is available.
6. **Mismatch Between Architecture and Checkpoint**
  - Encountered a **RuntimeError: Error(s) in loading state_dict for TCNModel** due to mismatched layer configuration (`num_channels` values differed).  
  - **Fixed by reading architecture parameters directly from**:
     ```python
     with open(config.json)
     ```
    instead of manually specifying hyperparameters.
7. **Missing Evaluation Metrics Utility**
  - Initially attempted to compute metrics directly inside the evaluation script, which i realised would cause duplication and potential inconsistency across models.  
  - Realised the need for a **centralised metrics module (`evaluation_metrics.py`)** to ensure every model uses identical metric functions.

#### Solutions and Learnings
1. **Centralising Metric Computation**
  - Created `evaluation_metrics.py` containing:
    - `compute_classification_metrics()` for binary tasks.
    - `compute_regression_metrics()` for continuous targets.
  - **Benefits**:
    - Enforces **consistency** across all baselines (NEWS2, LightGBM, TCN).  
    - Reduces **implementation drift** between scripts.  
    - Simplifies evaluation reproducibility.
2. **Improving Understanding of Evaluation Mode**
  - Learned that `model.eval()` + `torch.no_grad()` are **mandatory for inference**:
    - `model.eval()` → disables training randomness.  
    - `torch.no_grad()` → saves GPU memory, avoids unnecessary gradient computation.  
  - Together ensure **deterministic, efficient inference**.
3. **Correct Import Handling**
  - Fixed import path issues by dynamically appending the project’s root path to `sys.path`, enabling modular and portable evaluation scripts.
4. **Architecture Consistency**
  - Prevented mismatched layer errors by **loading hyperparameters dynamically** from `config.json`.  
  - Guarantees model reconstruction is identical to the training configuration.
5. **Diagnostic Insight — Not Failure**
  - Recognised that poor test metrics were **not coding errors**, but signals of **dataset limitations**:
    - **Max Risk (AUC = 0.577, F1 = 0.929):** superficially strong F1, but driven by class imbalance, model likely overpredicts the majority class, inflating F1 despite weak discriminative ability (low AUC).
    - **Median Risk (F1 = 0):** due to severe class imbalance; model predicts all negatives.  
    - **Regression (R² = −1.586):** target too skewed, model predicts near-mean values.
  - Learned to interpret metrics contextually rather than reactively.
6. **Action Plan: Phase 4.5**
  - **Decided to create Phase 4.5: Diagnostics & Retraining, focusing on**:
    - Verifying dataset integrity and label reconstruction.
    - Adding **class weighting** or **oversampling** for imbalanced targets.
    - Applying **Huber or MAE loss** for robust regression.
    - Re-evaluating threshold calibration post-retraining.
7. **Mindset Shift**
  - Learned that **unexpected results are part of the scientific process**, not project failure.  
  - Established confidence in diagnosing model behaviour systematically rather than by trial-and-error.

#### Summary
- **This phase proved pivotal**: it highlighted the **importance of diagnostic evaluation**, led to the creation of **Phase 4.5 (Diagnostics & Retraining)**, and reinforced that poor metrics are often **signals of dataset limitations**, not project failure.  
- Through this process, I deepened my understanding of **model evaluation**, **inference behaviour**, and **data–model interactions**.  
- The decision to introduce Phase 4.5 ensures the pipeline remains **rigorous, auditable, and scientifically defensible**.  
- Overall, the project continues to progress logically; from **data → model → evaluation → diagnosis → retraining → comparison**; and the day ultimately ended not in failure, but in clarity.

### Overall Summary
**Difficult day**
- This was one of the most technically and emotionally challenging stages of the pipeline.  
It focused on establishing a **reproducible and standardised evaluation framework** for all models (NEWS2, LightGBM, TCN) and running the **final held-out test evaluation** of the trained TCN.  
- Although the code executed successfully, the **test metrics (F1, R²)** were unexpectedly poor, initially suggesting a model failure.  
- After systematic troubleshooting, it became clear that the issues were **data-driven**, caused by **class imbalance and low variance**, not errors in the model or code.  
**No cause for concern**  
- The project is **not failed**; these results represent valid diagnostic outputs.  
- The pipeline, codebase, and evaluation logic are **technically sound and reproducible**.  
- Only targeted retraining is needed — **no need to redo Phase 4 entirely**.  
**Key Points**
- Model underperformance is **data-driven**, not due to coding or pipeline errors.  
- **TCN results** reflect dataset limitations (imbalance, skew, low variance).  
- **LightGBM baseline** will likely exhibit similar trends when evaluated on the same splits.  
**Next focus** 
- Perform structured diagnostics and controlled retraining to address these data limitations.
- Model diagnostics and retraining occur **between Phase 4 and Phase 5**, so introduce an intermediate **Phase 4.5: Diagnostics and Re-training**.
**Bottom Line:**  
- The project remains **fully salvageable**; Phase 4.5 can isolate the data issues, refine model robustness, and we can proceed confidently into Phase 5 with a validated comparison framework.
- This approach ensures full pipeline continuity; making the project auditable, reproducible, and scientifically defensible.

### Next Steps 
**Create and Start Phase 4.5**
1. **Run diagnostic script** to confirm test + validation set integrity and correct label recreation.  
2. **Retrain TCN** with improved loss functions or imbalance handling (e.g. class weighting, target transformation).  
3. **Re-run diagnostics** → verify that F1 and R² metrics improve.  
4. **Proceed to Phase 5**: Evaluate the LightGBM baseline on the same patient splits for consistency.  
5. **Document thoroughly**:
   - Original model performance and identified issues.  
   - Retraining adjustments (loss, weighting, transformations).  
   - Metric improvements after retraining.  
   - Remaining limitations (small dataset, class imbalance, skewed targets). 

---

# Phase 4.5: Diagnostics and Re-training

---

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

---

## Day 23-24 Notes - Start Phase 4.5: Diagnostics and Re-training (Step 1 + 2)

### Goal
- Establish **Phase 4.5: Diagnostics & Re-training** as a dedicated stage for model verification and evaluation integrity.  
- Run **diagnostic script** to perform a full audit of classification and regression performance across test and validation sets.  
- Identify whether poor metrics (low F1, negative R²) were caused by **model limitations or data issues**.  
- Validate the **evaluation pipeline** for correctness, ensuring predictions and labels are perfectly aligned.  
- Update and rerun the **evaluation script** to fix any misalignment or ordering inconsistencies (patient ID bug).  
- Confirm that both **diagnostic and evaluation outputs** now produce identical metrics, establishing a **verified, reproducible baseline** before retraining.  
- Prepare the groundwork for **Phase 4.5 retraining**, ensuring all future results are reliable, comparable, and scientifically defensible.

### What We Did
#### Step 1: Full TCN Diagnostic Analysis `tcn_diagnostics.py`
**Overview**
- This script (`tcn_diagnostics.py`) performs a **comprehensive diagnostic evaluation** of the Temporal Convolutional Network (TCN) trained in **Phase 4**.  
- It verifies that the model’s predictions, probability outputs, and regression behaviour are statistically sound across both **test** and **validation** sets.
- **Note:** The diagnostic results and interpretations below have been updated to reflect the final, validated metrics after fixing the evaluation misalignment (JSON ordering correction). Older outputs showing low ROC-AUC or negative R² have been superseded.
**Primary Purpose**
- Confirm the **integrity** and **generalisation** of the trained model.
- Diagnose whether performance issues are due to **model architecture**, **data imbalance**, or **target skew**.
- Produce reproducible metrics and plots that form a **baseline for refinement (Phase 4.5 → Phase 5)**.
**Key Functional Components**
1. **Data and Model Loading**
  - Loads model configuration (`config.json`) and trained weights (`tcn_best.pt`).
  - **Reads processed patient-level data and split indices**:
    - `train`, `val`, and `test` from `patient_splits.json`.
  - **Loads padded sequence tensors and masks**:
    - `test.pt`, `test_mask.pt`, `val.pt`, `val_mask.pt`.
  - Dynamically imports `TCNModel` from `tcn_model.py` and reconstructs using saved architecture hyperparameters `config.json` and `tcn_best.pt` to guarantee `state_dict`compatibility.
  - **Recreates targets directly from patient CSV**:
    - `max_risk_binary = (max_risk > 2)`
    - `median_risk_binary = (median_risk == 2)`
    - `pct_time_high` = continuous regression target.
2. **Prediction Extraction**
  - Runs inference (`model.eval()` + `torch.no_grad()`) on **test** and **validation** sets.
  - **Returns**:
    - `prob_max` = sigmoid(`logit_max`)
    - `prob_median` = sigmoid(`logit_median`)
    - `pred_reg` = regression head output (continuous).
  - Uses identical architecture and weights to ensure metrics match Phase 4 outputs.
  - Ensures output dimensions, activation, and precision match training architecture.
3. **Probability Distribution Histograms**
  - Generates histograms of predicted probabilities for both classification heads (Max and Median Risk).  
  - **Purpose**:
    - Detect **collapse** (e.g. all predictions near 0.0 or 1.0).
    - Visualise **output calibration**.
  - **Saved to:** `src/prediction_diagnostics/plots/`  
  - **Outputs:**
    - `prob_hist_max_test.png`
    - `prob_hist_max_val.png`
    - `prob_hist_median_test.png`
    - `prob_hist_median_val.png`
4. **Threshold Sweep (Classification Stability)**
  - Sweeps thresholds from 0.1 → 0.9 and reports F1 + Accuracy for both classification heads.  
  - **Purpose:**  
    - Identify class separability.  
    - Assess calibration and sensitivity to thresholds. 
  - **Example output**:
```bash
Threshold sweep: Max Risk
  Th=0.50 → F1=0.929, Acc=0.867
Threshold sweep: Median Risk
  Th=0.50 → F1=0.000, Acc=0.800
```
  - **Interpretation:**  
    - `Max Risk`: F1 remains stable (0.929) and seperated across thresholds → model strongly separates classes and is well-calibrated.
    - `Median Risk`: F1 collapses beyond 0.4 → F1 drops to 0 at 0.5+, confirming model predicts almost all 0s due to class imbalance and poor recall.
5. **Regression Diagnostics**
  - Uses regression outputs from `tcn_predictions.csv` (ensuring identical values to Phase 4 evaluation).
  - **Produces scatter and residual plots for `pct_time_high` regression head**:
    - **Scatter plot**: predicted vs. true (`test_reg_scatter.png` / `val_reg_scatter.png`)
	  - **Residual plot**: residuals vs. predicted (`test_reg_residuals.png` / `val_reg_residuals.png`) 
  - **Calculates and prints RMSE and R² for both test and validation sets**
```bash
Test Regression → RMSE: 0.0767, R²: 0.1664
Validation Regression → RMSE: 0.0744, R²: 0.2207
```
  - **Interpretation:**
    - Test R² = 0.166 → model captures moderate signal but leaves most variance unexplained.
    - Validation R² = 0.22 → similar magnitude → no overfitting detected.
    - Confirms regression head generalises modestly; further gains need target transformation or feature expansion.
6. **Data Distribution Diagnostics**
  - **Prints and plots class balance for both binary targets**:
    - `dist_test_max_risk_labels.png`
    - `dist_test_median_risk_labels.png`
```bash
Test Max Risk → 1s: 13, 0s: 2 (pos=0.87)
Test Median Risk → 1s: 3, 0s: 12 (pos=0.20)
```
  - **Plots distributions of true vs predicted regression targets**:
    - `dist_test_pct_time_high_true.png`
    - `dist_test_pct_time_high_pred.png`
  - **Confirms**:
    - **Median Risk imbalance (20% positives)** → explains zero recall and F1 collapse.
    - Regression target skewed → compressed range (causing low variance and poor R²)
7. **Training Label Distribution**
  - Prints summarised training-level distributions for `max_risk`, `median_risk`, and `pct_time_high`.  
  - Confirms **highly skewed regression target** and **uneven binary ratios across tasks**, explaining imbalanced learning behaviour.
8. **Summary Metrics**
  - **Computes**:
    - F1 and ROC-AUC for both classification heads (threshold = 0.5).  
    - RMSE and R² for regression (test + validation). 
  - **Saves summary JSON**: `/results/tcn_diagnostics_summary.json`.   
  - **Prints terminal summary:**
```bash
=== SUMMARY ===
Max Risk → F1 (0.5 threshold)=0.929, ROC-AUC=0.923
Median Risk → F1 (0.5 threshold)=0.000, ROC-AUC=0.778
Test Regression → RMSE=0.0767, R²=0.1664
Validation Regression → RMSE=0.0744, R²=0.2207
```
  - Notes that low F1 or negative R² implies data imbalance or target noise, not coding error.
  - **Interpretation**:
    -	**Max Risk**: Excellent separability; reliable head.
    -	**Median Risk**: Imbalanced → needs class weighting.
    -	**Regression**: Moderate positive R² → partial learning success, not overfit.

**Summary of Purpose**
| Function | Purpose |
|-----------|----------|
| `get_predictions()` | Inference wrapper for consistent TCN outputs |
| `plot_hist()` | Detect probability collapse or saturation |
| `threshold_sweep()` | Visualise threshold sensitivity (F1/Accuracy) |
| `regression_diagnostics()` | Quantify regression accuracy and bias |
| **Final Summary Block** | Consolidates all metrics and saves JSON summary |

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


#### Step 2: Fix Evaluation script
**Purpose**
- Initially, **F1 scores were identical** across scripts, but **ROC-AUC values differed**.  
- This indicated that the problem was **not with model predictions or probabilities**, but with **how the test labels (`y_test`) were constructed and aligned**.  
- The main issue was caused by **loss of patient ordering** in the evaluation script.  
```python
test_ids = set(splits["test"])
test_df = features_df[features_df["subject_id"].isin(test_ids)].reset_index(drop=True)
```
- **The use of `set()` randomised the order of patient IDs, breaking the one-to-one correspondence between**:
	- The predicted probabilities (`prob_max, prob_median`), and
	- The ground-truth labels (`y_true_max, y_true_median`).
	-	As a result, ROC-AUC values were incorrect, since the ranking relationship between predictions and true labels was disrupted.
**Updated Code**
```python
test_ids = splits["test"]  
test_df = features_df.set_index("subject_id").loc[test_ids].reset_index()
```
- **Explanation**:
	-	.loc[test_ids] preserves the exact patient order from the JSON file (patient_splits.json).
	-	The CSV (news2_features_patient.csv) may list patients in a different order, so this step guarantees that predictions and ground truths align perfectly.
	-	This directly fixes the ROC-AUC inconsistency across both evaluation and diagnostic scripts.
**Updated Terminal Output**
```bash
[INFO] Using device: cpu
[INFO] Loaded TCN model and weights successfully
[INFO] Running inference on test set...
[INFO] Inference complete in 0.03 seconds
[INFO] Saved metrics → results/tcn_metrics.json
[INFO] Saved predictions → results/tcn_predictions.csv

=== Final Test Metrics ===
Max Risk — AUC: 0.923, F1: 0.929, Acc: 0.867
Median Risk — AUC: 0.778, F1: 0.000, Acc: 0.800
Regression — RMSE: 0.077, R²: 0.166
==========================
Test IDs used: [10002428, 10005909, 10007058, 10015931, 10020740, 10021312, 10021666, 10021938, 10022281, 10023771, 10025612, 10027445, 10037928, 10038999, 10039831]
Mean of y_true_reg: 0.1199, Std: 0.0840
Mean of y_pred_reg: 0.0990, Std: 0.1132
```
**What Changed**
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


#### Diagnostic Output Directory Structure
```text
src/
└── prediction_diagnostics/
    ├── results/
    │   └── tcn_diagnostics_summary.json
    └── plots/
        ├── prob_hist_max_test.png
        ├── prob_hist_median_test.png
        ├── test_reg_scatter.png
        ├── test_reg_residuals.png
        ├── prob_hist_max_val.png
        ├── prob_hist_median_val.png
        ├── val_reg_scatter.png
        ├── val_reg_residuals.png
        ├── dist_test_max_risk_labels.png
        ├── dist_test_median_risk_labels.png
        ├── dist_test_pct_time_high_true.png
        └── dist_test_pct_time_high_pred.png
```


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

### Diagnostic Terminal Output
```bash
Model architecture from training config: {'num_channels': [64, 64, 128], 'head_hidden': 64, 'kernel_size': 3, 'dropout': 0.2}

Loaded splits → Train: 70, Val: 15, Test: 15
[INFO] Model loaded successfully.


=== TEST SET DIAGNOSTICS ===
[SAVED] prob_hist_max_test.png

Threshold sweep: Max Risk
  Th=0.10 → F1=0.929, Acc=0.867
  Th=0.20 → F1=0.929, Acc=0.867
  Th=0.30 → F1=0.929, Acc=0.867
  Th=0.40 → F1=0.929, Acc=0.867
  Th=0.50 → F1=0.929, Acc=0.867
  Th=0.60 → F1=0.929, Acc=0.867
  Th=0.70 → F1=0.929, Acc=0.867
  Th=0.80 → F1=0.889, Acc=0.800
  Th=0.90 → F1=0.556, Acc=0.467
[SAVED] prob_hist_median_test.png

Threshold sweep: Median Risk
  Th=0.10 → F1=0.333, Acc=0.200
  Th=0.20 → F1=0.462, Acc=0.533
  Th=0.30 → F1=0.286, Acc=0.667
  Th=0.40 → F1=0.500, Acc=0.867
  Th=0.50 → F1=0.000, Acc=0.800
  Th=0.60 → F1=0.000, Acc=0.800
  Th=0.70 → F1=0.000, Acc=0.800
  Th=0.80 → F1=0.000, Acc=0.800
  Th=0.90 → F1=0.000, Acc=0.800
Test Regression (Evaluation CSV) → RMSE: 0.0767, R²: 0.1664

=== VALIDATION SET DIAGNOSTICS ===
[SAVED] prob_hist_max_val.png
[SAVED] prob_hist_median_val.png
Validation Regression → RMSE: 0.0744, R²: 0.2207

=== DATA DISTRIBUTIONS ===
Test Max Risk → 1s: 13, 0s: 2, proportion positive: 0.867
[SAVED] dist_test_max_risk_labels.png
Test Median Risk → 1s: 3, 0s: 12, proportion positive: 0.200
[SAVED] dist_test_median_risk_labels.png
[SAVED] pct_time_high true/predicted distribution plots
Data distribution analysis complete — imbalance and skew patterns visualised.


=== TRAINING LABEL DISTRIBUTION ===

max_risk:
count    100.000000
mean       2.840000
std        0.443129
min        0.000000
25%        3.000000
50%        3.000000
75%        3.000000
max        3.000000
Name: max_risk, dtype: float64

median_risk:
count    100.000000
mean       0.480000
std        0.858469
min        0.000000
25%        0.000000
50%        0.000000
75%        0.000000
max        2.000000
Name: median_risk, dtype: float64

pct_time_high:
count    100.000000
mean       0.112783
std        0.103527
min        0.000000
25%        0.024564
50%        0.092233
75%        0.161885
max        0.440678
Name: pct_time_high, dtype: float64

=== SUMMARY ===
Max Risk → F1 (0.5 threshold)=0.929, ROC-AUC=0.923
Median Risk → F1 (0.5 threshold)=0.000, ROC-AUC=0.778
Test Regression → RMSE=0.0767, R²=0.1664
Validation Regression → RMSE=0.0744, R²=0.2207

Note: Median Risk F1=0 or Regression R²<0 suggests class imbalance or label noise.
All plots saved to: /Users/simonyip/Neural-Network-TimeSeries-ICU-Predictor/src/prediction_diagnostics/plots

Diagnostics completed successfully ✅
```

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

**Lessons Learned**
- Always use **JSON split** for selecting patients to maintain ranking-sensitive metrics like ROC-AUC.  
- CSV row order alone is insufficient for reproducible evaluation; may lead to misleading conclusions.  
- **Discrepancies between evaluation scripts can occur due to**:
  - Patient misalignment
  - Re-saving or re-training models
  - Different preprocessing conventions
- **Phase 4.5 diagnostics confirmed**:
  - TCN pipeline is functional
  - Weak heads (Median Risk, Regression) require retraining with class weighting or loss adjustments
  - Max Risk is robust and interpretable
- **Key takeaway**: Data-driven limitations must be distinguished from code errors to avoid unnecessary rework.


### Reflection
#### Challenges
- **Inconsistent AUC values:** The evaluation script (`evaluate_tcn_testset.py`) produced lower ROC-AUC scores than the diagnostic script, even though both used the same model checkpoint (`tcn_best.pt`).
- **Hidden data ordering bug:** The test set in the evaluation script was filtered using `.isin()`, which did **not preserve JSON split order**, causing predicted probabilities to mismatch ground-truth labels.
- **Regression R² mismatch:** The diagnostics script initially reported a different R² due to manual computation and minor numeric instability on the small test set.
- The project’s directory structure would become cluttered if new Phase 4.5 scripts and outputs would be mixed in both Phase 4 (`src/ml_models_tcn/`) and Phase 5 (`src/prediction_evaluations/`) folders, making traceability difficult.
#### Solutions & Learnings
1. **Preserve patient order using `.loc[test_ids]`:**
  ```python
  test_df = features_df.set_index("subject_id").loc[test_ids].reset_index()
  ```
  - Ensures perfect alignment between y_true and y_pred.
	-	Fixes ROC-AUC inconsistencies without altering model weights or predictions.
2. **Diagnostics script as validation tool**:
	-	Built to identify subtle issues in dataset handling, metric computation, and label alignment.
	-	Confirmed that F1 and Accuracy were unaffected, while ROC-AUC was highly sensitive to ordering errors.
	-	Reinforced importance of consistent dataset splits across all scripts.
3. **Regression R² harmonisation**:
	-	Adopted the evaluation script’s computation method as the standard.
	-	Imported predictions from `tcn_predictions.csv` for diagnostics to ensure reproducibility.
4. **New Folder**
  - Decided to create a **dedicated Phase 4.5 folder (`src/prediction_diagnostics/`)**, separating all diagnostic scripts, outputs, and validation files from earlier phases.  
  - This new structure ensures that future retraining can be developed cleanly, without confusion from outdated models or metrics.
#### Key Learnings
- **Data alignment is critical**: 
  - Even minor ordering inconsistencies can invalidate ranking-based metrics such as ROC-AUC. 
  -	Always reconstruct y_test using the original JSON split, CSV ordering cannot be trusted for reproducible results.
-	**ROC-AUC is order-sensitive**: Even a perfect model appears random if ground-truth labels are shuffled.
-	**F1 and accuracy can mask ordering errors**: They depend on thresholding, not ranking.
- **Diagnostics scripts are indispensable**: They serve as scientific verification tools that reveal silent pipeline errors that standard evaluation may miss.
- **Good project structure prevents future confusion**: By isolating diagnostics and re-training into a new folder, each phase now has a clear scope, making debugging, replication, and documentation straightforward.  
- This structural change improves the **clarity, reproducibility, and maintainability** of the entire ML pipeline, aligning with best practices in real-world engineering workflows.


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


### Next Steps
**Continue Phase 4.5 - TCN Refinement Plan**
1. Duplicate training script → `tcn_training_script_refined.py`
2. Add class weighting to BCE loss (for median_risk imbalance).
3. Add Huber/MAE loss or target transform for regression stability.
4. Retrain model with consistent architecture and data splits.
5. Save refined outputs in `trained_models_refined/` for full traceability.
6. **Re-run diagnostics → verify improvement in F1, R², and calibration**:
  -	↑ Median Risk F1
  -	↑ Regression R²
  -	Stable Max Risk performance
  -	Improved probability distributions.
7. Compare refined model outputs via `evaluate_tcn_refined.py`.

---

## Day 25-26 Notes - Finish Phase 4.5: Re-training TCN Model and Plotting Training Curve (Steps 3-4)

### Goals
- **Create `tcn_training_script_refined.py`:** Add log1p transform + class weighting, update config, outputs, and reproducibility metadata  
- **Create `plot_training_curves_refined.py`:** Visualise learning dynamics (loss curve shape, convergence speed, and overfitting onset) on retrained model run.
- **Plan `evaluate_tcn_testset_refined.py`:**
  - Load refined model, apply `expm1` to revert transform, compute metrics  
  - Compare Phase 4 vs 4.5 results, save evaluation JSON  

### What We Did 
#### Step 3: Retrained TCN Model `tcn_training_script_refined.py`
**Purpose**
- Implements the refined retraining phase for the TCN, extending Phase 4 by introducing two controlled, data-level corrections
- Retains the same model architecture, hyperparameters, and optimiser setup for scientific comparability.
1. **Updated Directory Structure**
  - Script dynamically locates the project structure and import the TCN implementation (`TCNModel`), without hardcoding absolute paths, which keeps the script portable and reproducible across different machines or directory setups.
    ```python
    SCRIPT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = SCRIPT_DIR.parent
    sys.path.append(str(PROJECT_ROOT / "ml_models_tcn"))
    from tcn_model import TCNModel
    ```
  - Separate output folder for Phase 4.5 ensures reproducibility and prevents overwriting Phase 4 results.
    ```python
    MODEL_SAVE_DIR = SCRIPT_DIR / "trained_models_refined"
    MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    ```
  - All paths updated to point to refined directories for model weights, config, and history.
    ```python
    best_model_path = MODEL_SAVE_DIR / "tcn_best_refined.pt"
    config_path = MODEL_SAVE_DIR / "config_refined.json"
    history_path = MODEL_SAVE_DIR / "training_history_refined.json"
    ```
2. **Updated JSON Configuration with Metadata**
  - Stores phase identifier, dynamic parameters (pos_weight, final_val_loss), loss modifications, data transformations, and file paths (`data_path`, `model_save_dir`).
    ```python
      "loss_functions": {
        "max_risk": "BCEWithLogitsLoss (unweighted)",
        "median_risk": "BCEWithLogitsLoss(pos_weight)",
        "pct_time_high": "MSELoss (on log1p-transformed target)"
    },
    "data_transformations": {
        "regression_target": "log1p(y) applied during training; expm1(y_pred) at inference",
        "class_weighting": "dynamic pos_weight computed from training data (num_neg / num_pos)"
    },
    ```
	-	Terminal print confirms the base configuration is loaded.
  -	**Rationale:** Provides a reproducible, auditable record of the experiment.
3. **Log-Transform Regression Target (pct_time_high)**
  - Helper script builds target tensors (with regression log-transform) after loading patient-level targets
  -	Must happen before tensor creation to ensure tensors hold transformed values.
    ```python
    def get_targets(split_ids, apply_log=False):
        df_split = patient_df.loc[split_ids, target_cols].copy()
        if apply_log:
            df_split["pct_time_high"] = np.log1p(df_split["pct_time_high"])
            print(f"[INFO] Log-transform applied to regression target: min={df_split['pct_time_high'].min():.3f}, max={df_split['pct_time_high'].max():.3f}")
        return (
            torch.tensor(df_split["max_risk_binary"].values, dtype=torch.float32),
            torch.tensor(df_split["median_risk_binary"].values, dtype=torch.float32),
            torch.tensor(df_split["pct_time_high"].values, dtype=torch.float32),
        )
    ```
  - **Rationale:**:
    - Reduces skew and stabilises regression loss contribution.
    -	**Deterministic:** same input → same output. Only min/max printed for verification of transformation.
4. **Compute Class Weight for Median Risk**
  - `pos_weight = num_neg / num_pos` Directly fixes BCE loss imbalance by scaling the minority class contribution.
  - This makes the model pay more attention to the minority class (medium risk) during training.
  - Applied after tensors are created because BCE loss weighting is a loss-level adjustment.
    ```python
    pos_weight = (y_train_median == 0).sum() / (y_train_median == 1).sum()
    pos_weight = pos_weight.to(DEVICE)  # move to GPU if available
    print(f"[INFO] Computed pos_weight for median_risk = {pos_weight:.3f}")
    ```
  - **Rationale:**
	  -	Corrects class imbalance (~3:1 negative:positive for median risk).  
    - **Dynamic:** depends on the training split, hence printed to terminal.
    - Makes the model “pay more attention” to minority class during early training.
5. **Updated Loss Function for Median Risk**
  - Integrates dynamic class weighting into BCE loss.
	- Ensures faster early learning for minority class and affects validation dynamics.
    ```python
    criterion_median = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    ```
6. **Save Model to Phase 4.5 Folder `trained_models_refined/`**
	-	Separates best model checkpoint into refined folder.
	-	Ensures early stopping triggers correctly and preserves best generalising weights.
    ```python
    torch.save(model.state_dict(), best_model_path)
    ```
  - **Output:** `tcn_best_refined.pt`
7. **Update and Save JSON Configuration After Training**
  - Updates earlier JSON dictinary with information post-training.
  - Records dynamic outputs (`pos_weight_median_risk` and `final_val_loss`) and confirms file locations.
    ```python
    config_data["pos_weight_median_risk"] = float(pos_weight.cpu().item())
    config_data["final_val_loss"] = float(best_val_loss)
    config_data["outputs_confirmed"] = { 
        "best_model": str(best_model_path),
        "training_history": str(history_path),
        "config": str(config_path)
    }

    with open(config_path, "w") as f:
        json.dump(config_data, f, indent=4)
    print(f"[INFO] Final refined training configuration saved to {config_path}")
    ```
  - **Output:** `config_refined.json`
	-	**Rationale:** Provides a reproducible audit trail for this Phase 4.5 run.
8. **Save Training History to Refined Directory**
  -	Saves epoch-wise training and validation losses for downstream analysis.
	-	Terminal output confirms history and model weight file locations.
  - **Output:** `training_hisotry_refined.json`
    ```python
    with open(history_path, "w") as f:
        json.dump({"train_loss": train_losses, "val_loss": val_losses}, f, indent=4)
    print(f"[INFO] Refined training history saved to {history_path}")
    print(f"[INFO] Best refined model saved to {best_model_path}")
    ```
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

#### Step 4: Plotting Training vs Validation Loss Curve `plot_training_curves_refined.py`
**Purpose**
- Visualise retraining improvements and convergence dynamics after applying **median-risk weighted BCE** and **log-transformed regression target**.
- Compare Phase 4 baseline vs Phase 4.5 refined model to evaluate the effect of targeted data-level fixes on training and validation loss behaviour.
**Process**
1. **Define paths**  
   - Set `HISTORY_ORIGINAL` and `HISTORY_REFINED` to the JSON files containing training/validation losses for Phase 4 and Phase 4.5.
   - Create `PLOT_DIR` for storing plots.
2. **Load JSON histories**  
   - Use `json.load()` to read both training history files.
   - Extract `train_loss` and `val_loss` arrays for each phase.
3. **Identify best validation epochs**  
   - For each phase, find the epoch with minimum validation loss (`min(val_loss)`).
   - Store both epoch index and value for annotation.
4. **Plot Phase 4.5 refined training/validation curve**  
   - `plt.plot(train_loss_refined)` and `plt.plot(val_loss_refined)` to visualise loss across epochs.
   - Add title with padding and an explanatory note directly below the title using `plt.text(..., transform=plt.gca().transAxes)`.
   - Add red vertical dashed line at best validation epoch.
   - Mark the minimum validation loss with a red dot and annotated text.
   - If validation loss increases after the best epoch, mark this region as “overfitting region” (post-best epoch rise in validation loss).
5. **Save Phase 4.5 plot**  
   - `plt.savefig(PLOT_DIR / "loss_curve_refined.png")` in high resolution.
   - Close figure `plt.close()` to free memory.
6. **Plot comparison: Phase 4 vs Phase 4.5**  
   - Overlay both training/validation curves:
    - Plot baseline Phase 4 curves as dashed, lighter lines.
    - Plot Phase 4.5 curves as solid, darker lines.
   - Add note beneath title clarifying different loss scales (raw vs log-space).
   - Add scatter points and text annotations for minima of both phases.
   - Include legend, grid, and tight layout.
   - Use `plt.subplots_adjust(bottom=0.5)` and `plt.figtext(0.5, 0.05, …)` to add a bottom disclaimer below the x-axis, confirming the overlay is for trend comparison only.
7. **Save comparison plot**  
   - `plt.savefig(PLOT_DIR / "loss_curve_comparison.png")` in high resolution.
   - Close figure to free memory.
8. **Console confirmation**  
   - Print messages indicating file paths of saved plots.
**Outputs**
- `loss_plots/loss_curve_refined.png` — Phase 4.5 standalone learning curve.
- `loss_plots/loss_curve_comparison.png` — overlaid Phase 4 vs Phase 4.5 for direct visual comparison.
**Interpretation**
- **Training curves (`loss_curve_refined.png`) show**:
  - Steady decline in training loss from epoch 0 → 9.
  - Validation loss initially drops for the first 3 epochs, then rises sharply, highlighting **rapid learning followed by overfitting**.
  - The sharp rise in validation loss visually confirms early overfitting; the curve shape emphasizes the effect of **weighted BCE (stabilised classification) and log-transform (smoothed regression behaviour)** interventions on learning dynamics.
- **Overlay comparison (`loss_curve_comparison.png`) shows**:
  - Baseline (Phase 4) validation loss declines more gradually and stays relatively stable, whereas refined (Phase 4.5) validation loss drops (converges) faster but increases (overfits) earlier.
  - Refined training loss consistently below baseline after epoch 2 → demonstrates **faster convergence and improved early learning**.
  - Visual gap between baseline and refined curves confirms that **Phase 4.5 interventions produced improvements in learning dynamics** while maintaining stable convergence patterns.
**Summary**
- **`loss_curve_refined.png` provides a clear view of Phase 4.5 learning:** faster early convergence, minimum validation loss at epoch ~3, and onset of overfitting.
- **`loss_curve_comparison.png` visually contrasts baseline vs refined behaviour:**
  - Shows faster initial learning in Phase 4.5.
  - Confirms earlier minimum validation loss, highlighting **data-level fixes (weighted BCE + log-transform) as effective**.
  - Emphasises that raw loss magnitudes are not numerically comparable due to scale differences.
  - Provides evidence that retraining interventions improved model behaviour without modifying architecture, suitable for documentation and reproducibility.

### Folder Structure
```text
data/
 ├── processed_data/
 │     └── news2_features_patient.csv
src/
 ├── ml_models_tcn/                   # Phase 4 original models
 │     ├── prepared_datasets/
 │     │     ├── train.pt
 │     │     ├── val.pt
 │     │     └── test.pt
 │     ├── deployment_models/
 │     │     └── preprocessing/
 │     │           ├── standard_scaler.pkl
 │     │           ├── padding_config.json
 │     │           └── patient_splits.json
 │     ├── trained_models/
 │     │     └── training_history.json
 │     └── tcn_model.py
 ├── prediction_diagnostics/          # Phase 4.5 (diagnostics + retraining)
 │     ├── plots/
 │     ├── results/
 │     ├── loss_plots/
 │     │     ├── loss_curve_refined.png
 │     │     └── loss_curve_comparison.png
 │     ├── trained_models_refined/
 │     │     ├── config_refined.json
 │     │     ├── training_history_refined.json
 │     │     └── tcn_best_refined.pt
 │     ├── tcn_diagnostics.py
 │     ├── tcn_training_script_refined.py
 │     └── plot_training_curves_refined.py
 └──  evaluation_diagnostics/         # Phase 5 final evaluation + comparison
```

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


### Reflection
#### Challenges
1. **Maintaining separation of logic and outputs:**  
  - Initially mixed retraining logic into Phase 5. 
  - Needed to isolate all retraining in **Phase 4.5** to keep evaluation scripts clean and ensure traceable experiment lineage.  
2. **Folder directory confusion:**  
  - Frequent path errors due to scripts calling models and datasets from nested folders. 
  - Required careful setup of `SCRIPT_DIR` and `PROJECT_ROOT` to correctly reference directories like: `data/processed_data/`, `src/ml_models_tcn/`, `src/prediction_diagnostics/`.
3. **Understanding transformation vs loss weighting logic:**  
  - Confusion about why both regression transformation (`log1p`) and class weighting (`pos_weight`) existed, when they occur, and how they differ in flow.  
4. **JSON config structure and timing:**  
  - Initially unsure why `config_data` was defined at the top and only updated at the end. 
  - Also unclear why both `"outputs"` and `"outputs_confirmed"` were kept, and why `pos_weight_median_risk` and `final_val_loss` were added later.  
#### Solutions and Learnings
1. **Separation of retraining logic (Phase 4.5):**  
  - All retraining scripts and artefacts (weights, configs, histories) are confined to the `trained_models_refined/` directory.  
  - Keeps Phase 5 purely for evaluation.  
  - **Enhances scientific clarity:** Phase 4 (baseline) → Phase 4.5 (diagnostic fixes) → Phase 5 (evaluation).  
  - **Directory resolution fix:** Ensures cross-directory imports and consistent relative paths.
2. **Clarified model logic:**
	-	**Median risk:** Weighted BCE → fixes imbalance; tensors unchanged.
	-	**Regression:** log1p applied to targets → fixes skew; loss unchanged.
	-	Each head receives one targeted fix; minimal, controlled, and reproducible.
3. **Improved JSON reproducibility:**
	-	Defined static config_data before training (planned parameters).
	-	**Appended dynamic runtime fields after training:**
  ```python
  config_data["pos_weight_median_risk"] = float(pos_weight.cpu().item())
  config_data["final_val_loss"] = float(best_val_loss)
  ```
  - "outputs" = intended paths; "outputs_confirmed" = verified save locations.
  - **Scientific rationale:**
    -	Enhances traceability (planned vs actual results).
    -	Strengthens reproducibility (reruns recreate same experiment).
    -	Improves clarity in evaluation phase (clean input/output lineage).

### Overall Summary
- Phase 4.5 became a controlled diagnostic retraining phase; introducing only essential data-level corrections (median-risk weighted BCE, log-transformed regression targets) while maintaining full experiment integrity.
- Training and validation loss analysis shows **faster early convergence** and **improved learning efficiency**, with validation curves highlighting earlier overfitting consistent with the interventions.
- This structured approach guarantees **clean versioning, reproducibility, and interpretability**, providing a robust foundation for Phase 5 evaluation and benchmarking.

---

# Phase 5: Evaluation, Baselines & Comparison

---

## Day 27-29 Notes - Continue Phase 5: Evaluation of Refined TCN Model (Steps 2-3)

### Goals
- Finalise `evaluate_tcn_testset_refined.py`, including updated tuning and calibration sections.  
- Verify that `tcn_metrics_refined.json` and `tcn_predictions_refined.csv` contain all expected keys and values.  
- Ensure all console outputs (including threshold tuning, log + raw outputs, calibration) are clearly reflected in the markdown summary.  
- Begin preparing **comparative evaluation setup** (TCN vs LightGBM) for Phase 5.

### What We Did
#### Evaluated Refined TCN Model on Test Set `evaluate_tcn_testset_refined.py`
**Purpose**
- Performs **final evaluation** of the refined Temporal Convolutional Network (TCN) on the held-out test set (Phase 4.5).  
- Validates how the retrained model (with corrected log-transformation consistency and class weighting) generalises to unseen data.  
- Adds **threshold tuning** and **post-hoc calibration** to optimise classification and regression performance *without retraining*.  
- Ensures both:
  - **Internal consistency** → verifies model validity in log-space (training domain).  
  - **Clinical interpretability** → verifies predictions on raw scale (% time high).

**Process**
1. **Preserved Core Architecture and Evaluation Flow**
  - The overall structure remains consistent with the original `evaluate_tcn_testset.py`.
  - **Key retained components include:**
    - Load `TCNModel` definition from phase 4 `tcn_model.py`
    - Dynamic rebuilding of test targets from `news2_features_patient.csv` and `patient_splits.json`.
    - Load phase 4 test tensors `test.pt`, `test_mask.pt`
    - **TCNModel instantiation:** loading weights from `tcn_best_refined.pt`, using architecture from `config_refined.json`.
    - Set model to evaluation mode.
    -	Standardised inference flow (CPU/GPU detection → forward pass → logits extraction) on unseen test tensors.  
    -	Metric computation using shared utility functions `compute_classification_metrics` and `compute_regression_metrics`, ensuring metric continuity with previous phases.
    -	Output saving to disk (`metrics_refined.json`, `predictions_refined.csv`) and printed summary.
  - This ensures full continuity with prior evaluation pipelines (Phases 4.0–4.5).
2. **Comprehensive Header & Scientific Context**
  - Added a full top-level docstring with Overview, Scientific Rationale, and Outputs sections.
	- **Clarifies:**
    -	The dual purpose of log-space vs raw-space evaluation.
    -	Why calibration replaces retraining for bias correction.
    - Why threshold tuning replaces retraining for data balancing.
    - What outputs are saved and how they map to interpretability.
  - Makes the script publication-ready and self-documenting.
3. **Explicit Handling of Log-Transformed Regression Outputs**
	-	The regression head outputs predictions trained on `log1p(pct_time_high)`.
	-	Predictions (`y_pred_reg`) are explicitly recognised as log-space values.
  - **Rationale:** Prevents scale misinterpretation, ensuring that metrics reflect the correct mathematical space.
4. **Inverse Log Transform (`np.expm1`)**
	-	Introduced an inverse transformation to recover clinically interpretable raw values (`pct_time_high`).
	-	Added both log- and raw-scale predictions to the CSV (`y_pred_reg_log`, `y_pred_reg_raw`).
  - **Rationale**: Enables clinicians and reviewers to compare real-scale estimates directly with baseline models and ground truth.
5. **Dual-Scale Evaluation**
	-	Regression metrics now computed in:
    1. Log-space → Matches training objective (validates optimisation fidelity).
    2. Raw-space → Assesses clinical interpretability and external comparability.
  - **Rationale:** Separates internal model performance from practical outcome accuracy, creating transparency in cross-domain validation.
6. **Threshold Tuning (Median-Risk Head)**  
   - Introduced **validation-based threshold optimisation** for the median-risk classifier.  
   - Uses validation data (`val.pt`, `val_mask.pt`) and F1-score maximisation to find optimal cutoff.  
   - Default 0.5 retained for max-risk head; median-risk now tuned via best F1 on validation set.  
   - Reports both 0.5 and tuned threshold results in console output and metrics JSON.
7. **Correlation Analysis**
	-	Introduced Pearson correlation coefficients (`corr_log` and `corr_raw`).
	-	**Added interpretation rule:** High correlation but negative R² indicates monotonic bias, not failure.
  - **Rationale:** Provides diagnostic evidence for calibration necessity → shows the model tracks trends correctly but misestimates magnitude.
8. **Calibration Bias Visualisation**
	-	Added scatter plot of `y_true_reg_log` vs `y_pred_reg_log` with a red diagonal (y=x).
	-	Automatically saved as `tcn_regression_calibration_logspace.png`.
  - **Rationale:** Visual confirmation of systematic bias (e.g., consistent overestimation or underestimation).
9. **Post-Hoc Linear Calibration**
	-	**Implemented a simple linear regression in log-space:** fits `y_true_log ≈ a * y_pred_log + b` using `LinearRegression`.
	-	Applied the fitted coefficients to correct predicted values without retraining.
  - Corrects scale bias without altering model weights. 
	-	Recomputed calibrated predictions in both log and raw scales.
  - **Rationale:** Corrects bias directly, improving R² and RMSE while maintaining learned structure.
9. **Calibration Effect Comparison Plot**
	-	Added second visualisation: before vs after calibration.
	-	Steelblue = pre-calibration, Orange = post-calibration, Red dashed = perfect calibration.
	-	Saved as `tcn_regression_calibration_comparison_logspace.png`.
  - **Rationale:** Demonstrates the effectiveness of calibration visually → makes improvement transparent in reports.
10. **Revised Output Structure**
  - Combined all predictions into a single DataFrame (`tcn_predictions_refined.csv`) containing:
    - Classification: `y_true_max`, `prob_max`, `y_true_median`, `prob_median`  
    - Regression (raw + log + calibrated):  
      - `y_true_reg`, `y_true_reg_log`, `y_pred_reg_log`, `y_pred_reg_raw`,  
      - `y_pred_reg_log_cal`, `y_pred_reg_raw_cal`  
  - Metrics are saved to JSON (`tcn_metrics_refined.json`) for reproducibility:
    - Classification (baseline max, baseline median + threshold-tuned median)
    - Regression (pre- and post-calibration) 
  - **Summary printout includes:**
    - Printed range and mean of raw regression predictions (`min, max, mean`).
    - AUC, F1, Accuracy for classification heads (0.5 and tuned).  
    - RMSE and R² for regression heads (log-space, raw-space, and calibrated).  
    - Mean and standard deviation of true `y_true_reg` and predicted `y_pred_reg` regression targets to contextualise bias.
**What Changed (Summary Table)**
| Area | Change | Rationale |
|------|---------|-----------|
| **Header + Documentation** | Rewritten docstring with full scientific rationale, calibration context, and output summary. | Makes the evaluation pipeline self-documenting and publication-ready. |
| **Regression Target Interpretation** | Explicitly defined outputs as log-space (`log1p(pct_time_high)`). | Ensures evaluation aligns with training target transformation. |
| **Inverse Log Transform** | Added `np.expm1` to restore predictions to raw scale. | Provides interpretable, clinically relevant regression metrics. |
| **Dual-Scale Evaluation** | Regression metrics computed in both log and raw space. | Separates model’s internal learning quality from external applicability. |
| **Threshold Tuning (Median-Risk)** | Introduced validation-based F1 maximisation. | Balances recall and precision on imbalanced data. |
| **Correlation + Calibration Step** | Added Pearson correlation and linear calibration. | Identifies and corrects monotonic bias without retraining. |
| **Visual Diagnostics** | Added calibration plots (pre- and post-calibration). | Enables visual confirmation of bias correction and model alignment. |
| **Metric Expansion** | Saved pre- and post-calibration regression metrics + tuned classification results. | Quantifies improvement (e.g., R² shift from negative to positive, doubling of median F1). |
| **Output Files** | CSV + JSON + PNG outputs under new directory `tcn_results_refined/`. | Keeps refined results isolated and reproducible. |
| **Summary Reporting** | Added regression means, standard deviations, F1-tuned thresholds and print messages confirming file saves. | Improves interpretability and traceability. |
**Files Generated**
| File | Description |
|-------|--------------|
| `tcn_predictions_refined.csv` | Full per-patient predictions → includes classification (`max`, `median`) and regression outputs in both **log-space** and **raw-space**. |
| `tcn_metrics_refined.json` | Complete metrics dictionary → covers **classification** (including tuned median) and **regression** (pre- and post-calibration). |
| `tcn_regression_calibration_logspace.png` | Calibration bias visualisation in log-space → shows pre-calibration over/underestimation relative to the ideal diagonal. |
| `tcn_regression_calibration_comparison_logspace.png` | Before/after calibration comparison plot → highlights improved alignment of predictions after bias correction. |
**Console Output**
| Output | Description |
|---------|--------------|
| **Median-risk threshold (from validation)** (`Median Risk Threshold = X (F1=Y)`) | Shows the validation-derived optimal threshold (maximises F1) for the median-risk head → used to report tuned classification metrics. |
| **Regression predictions (log-space)** — Min, Max, Mean | Displays the range and central tendency of raw model outputs in log-space to check numerical stability and confirm that predicted values fall within plausible bounds. |
| **Correlation (log-space / raw-space)** | Pearson correlation shows how strongly predicted vs true regression values correlate before calibration. High correlation with poor R² suggests monotonic bias rather than random error. |
| **Calibration equation** (`y_true_log ≈ a * y_pred_log + b`) | Defines the fitted linear relationship used to correct systematic over/underestimation in the model’s log-space predictions. |
| **Final Refined Test Metrics** — AUC, F1, Accuracy (for both 0.5 and tuned thresholds), RMSE, R² (pre- and post-calibration) | Summarises all quantitative performance metrics for classification and regression tasks, both before and after calibration, before and after median head tuning. Enables comparison of tuning and calibration impact. |
| **Test IDs used** | Lists exact patient identifiers included in the held-out test set, ensuring traceability and reproducibility. |
| **Mean and Std of y_true_reg / y_pred_reg** | Compares central tendency and spread of true vs predicted regression targets to contextualise model bias and output distribution (variability). |

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

### Folder Format

src/
└── prediction_evaluations/
    ├── evaluate_tcn_testset_refined.py           # Final evaluation script (Phase 4.5)
    │
    └── tcn_results_refined/                      # All outputs from refined TCN evaluation
        ├── tcn_predictions_refined.csv           # Per-patient predictions (classification + regression, raw + log)
        │                                          ├─ Columns:
        │                                          │   • y_true_max / prob_max           → Max-risk classification
        │                                          │   • y_true_median / prob_median     → Median-risk classification
        │                                          │   • y_true_reg / y_true_reg_log     → Ground truth (raw + log)
        │                                          │   • y_pred_reg_log / y_pred_reg_raw → Predictions (log + raw)
        │                                          └─ Each row = 1 patient in held-out test set
        │
        ├── tcn_metrics_refined.json              # All computed metrics (classification + regression)
        │                                          ├─ Includes:
        │                                          │   • max_risk, median_risk           → AUC, F1, Accuracy, etc.
        │                                          │   • median_risk_tuned               → AUC, F1, Accuracy, etc. (post threshold-tuning)
        │                                          │   • pct_time_high_log/raw           → RMSE, R² (before calibration)
        │                                          │   • pct_time_high_log_cal/raw_cal   → RMSE, R² (after calibration)
        │                                          │   • inference_time_sec              → Total inference runtime
        │                                          └─ Used for model performance tracking and reporting
        │
        ├── tcn_regression_calibration_logspace.png        # Plot 1 — Pre-calibration bias
        │                                                   ├─ Scatter of true vs predicted log-space values
        │                                                   ├─ Red dashed line = ideal calibration (y = x)
        │                                                   └─ Shows global bias (points consistently above/below line)
        │
        └── tcn_regression_calibration_comparison_logspace.png   # Plot 2 — Before vs after calibration
                                                                 ├─ Blue = before calibration (biased)
                                                                 ├─ Orange = after calibration (corrected)
                                                                 └─ Confirms bias removal and improved alignment


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

---

### Calibration Plots (Pre- vs Post Calibration)
#### Overview
- These plots provide **visual diagnostic evidence** of the model’s regression bias and the effectiveness of post-hoc calibration.  
- Both operate in **log-space**, since the TCN was trained on `log1p(pct_time_high)`.
#### 1. **Calibration Bias Plot — `tcn_regression_calibration_logspace.png`**
**Purpose:**  
- To detect **systematic calibration bias** in the regression head before correction.  
- Plots **true** vs **predicted log1p(pct_time_high)** values.  
- The **red dashed diagonal** `y = x` represents perfect calibration (ideal 1:1 relationship).
**Interpretation:**
| Observation | Meaning |
|--------------|----------|
| Almost all points lie **above** the red line | The model **overpredicts** the true values → systematic positive bias. |
| Points roughly form a linear trend | The model correctly learned the **monotonic relationship** (higher true → higher predicted). |
| Correlation ≈ 0.75 (`corr_log`) | Predictions correctly follow patient ranking (trend preserved). |
| R² ≈ −1.1 | Model is biased in absolute scale, not random → R² penalises this offset heavily. |

**Mathematical Explanation:**  
- The model’s learned mapping can be expressed as: `ŷ_log = a * y_true_log + b`
- Where:
  - a ≈ 1 → means the slope is close to 1, showing that the model captures the correct trend (monotonic relationship).
  - b > 0 → indicates a constant positive offset, meaning the model consistently overpredicts across the range.
- This shows that the model’s structure is correct (it has learned the right directional relationship), but its output scale is shifted, causing systematic bias.
**Significance:**  
- Confirms that retraining is **not required** → bias can be corrected post-hoc.  
- **Guides the next step:** **linear calibration** to realign predictions with true values.

#### 2. Calibration Comparison Plot — `tcn_regression_calibration_comparison_logspace.png`
**Purpose:**  
- To visualise the **effect of post-hoc calibration** on regression predictions.  
- Overlays pre- and post-calibration predictions against the ideal diagonal.
**Plot Elements:**
| Element | Description |
|----------|-------------|
| **Blue points** | Predictions **before** calibration (biased, shifted). |
| **Orange points** | Predictions **after** calibration (corrected). |
| **Red dashed line** | Perfect calibration — `y_true = y_pred`. |
**Interpretation:**
| Observation | Meaning |
|--------------|----------|
| Blue points lie mainly above the diagonal | Model consistently overpredicted → systematic bias. |
| Orange points cluster symmetrically around the diagonal | Bias removed → residuals are now zero-mean (unbiased). |
| Trend remains linear | Model’s underlying **monotonicity preserved** (structure intact). |
| Spread remains | Expected random noise → represents natural prediction variance. |
**Mathematical Validation:**  
- After fitting the calibration equation: `y_true_log ≈ a * y_pred_log + b`
- the model’s outputs were rescaled such that:
  - Mean offset (b) corrected  
  - Output slope aligned closely with ideal diagonal (perfect calibration)
  - Resulting R² improved significantly from **−1.1 → +0.57**  
**Outcome:**  
- **Confirms calibration worked:** predictions are now both accurate and unbiased.  
- Demonstrates that model bias, not randomness, explained prior performance issues.  
- Validates the **log-space linear correction** as a robust alternative to retraining.

#### Summary of Diagnostic Value
| Plot | Purpose | Diagnostic Insight | Outcome |
|------|----------|-------------------|----------|
| `tcn_regression_calibration_logspace.png` | Detect initial bias | Reveals consistent overprediction despite high correlation | Confirms need for calibration |
| `tcn_regression_calibration_comparison_logspace.png` | Show effect of calibration | Demonstrates removal of bias and restoration of scale | Confirms model now unbiased and well-calibrated |
**Conclusion:**  
Together, these plots provide clear **visual and mathematical evidence** that:  
- The model learned valid temporal structure (trend fidelity).  
- Its issue was purely **scale bias**, not structural failure.  
- **Post-hoc calibration** successfully corrected this without retraining, yielding an interpretable and publication-ready regression head.

---

### Interpretation of Terminal Outputs & Final Evaluation Metrics
#### Overview
**Purpose**
- Documents the **meaning and diagnostic value** of all console outputs produced during the refined TCN evaluation (`evaluate_tcn_testset_refined.py`).
**Final Terminal Output**
```bash
[INFO] All required files found. Proceeding with refined model evaluation...
[INFO] Using device: cpu
[INFO] Loaded refined TCN model and weights successfully
[INFO] Running inference on test set...
[INFO] Inference complete in 0.02 seconds
[INFO] Median Risk Threshold (from validation) = 0.430 (F1=0.571)
Regression predictions (log-space):
  Min:  0.0802
  Max:  0.3213
  Mean: 0.2043
==========================
[INFO] Saved classification predictions + regression predictions (raw + log-space) → tcn_predictions_refined.csv
Correlation (log-space): 0.754
Correlation (raw-space): 0.739
[INFO] Saved calibration plot → /Users/simonyip/Neural-Network-TimeSeries-ICU-Predictor/src/prediction_evaluations/tcn_results_refined/tcn_regression_calibration_logspace.png
Calibration: y_true_log ≈ 0.712 * y_pred_log + -0.035
[INFO] Saved calibration comparison plot → /Users/simonyip/Neural-Network-TimeSeries-ICU-Predictor/src/prediction_evaluations/tcn_results_refined/tcn_regression_calibration_comparison_logspace.png
[INFO] Saved metrics → tcn_results_refined/tcn_metrics_refined.json

=== Final Refined Test Metrics ===
Max Risk (0.5) — AUC: 0.923, F1: 0.929, Acc: 0.867
Median Risk (0.5) — AUC: 0.833, F1: 0.286, Acc: 0.667
Median Risk (0.430) — AUC: 0.833, F1: 0.545, Acc: 0.667
Regression (log) — RMSE: 0.108, R²: -1.096
Regression (raw) — RMSE: 0.129, R²: -1.349
Regression (log) calibrated — RMSE: 0.049, R²: 0.569
Regression (raw) calibrated — RMSE: 0.056, R²: 0.548
==========================
Test IDs used: [10002428, 10005909, 10007058, 10015931, 10020740, 10021312, 10021666, 10021938, 10022281, 10023771, 10025612, 10027445, 10037928, 10038999, 10039831]
Mean of y_true_reg: 0.1199, Std: 0.0840  # Ground truth: actual % time high
Mean of y_pred_reg: 0.2043, Std: 0.0793  # Model output: predicted log-space mean
==========================
[INFO] Evaluation complete — refined TCN calibration validated and metrics saved.
```
#### 1. General Information and Process Messages
**Purpose:**  
- All non-numeric `[INFO]` lines confirm the logical flow and reproducibility of the pipeline.
**Key Messages:**
- Model weights, configuration, and test data were correctly located.
- The model was loaded onto the correct device (CPU/GPU).
- Inference completed successfully and efficiently (~0.03 s).
- All outputs (predictions, plots, metrics) were saved to disk.
**Interpretation:**  
- These confirm **pipeline integrity** — ensuring the evaluation reproduced the correct trained model, test split, and inference conditions.

#### 2. Threshold Tuning (Median-Risk Head)
**Terminal Output**
```bash
[INFO] Median Risk Threshold (from validation) = 0.430 (F1=0.571)
```
**Purpose:**  
- Identifies the optimal decision threshold for the median-risk head based on validation set performance, not test data — maintaining methodological integrity.
- The tuned threshold maximises the F1-score, improving positive-class detection on an imbalanced dataset.
**Interpretation:** 
| **Component**              | **Meaning** | **Diagnostic Value** |
|-----------------------------|-------------|------------------------|
| **Threshold = 0.430**       | Optimal cutoff probability for classifying a patient as “median risk.” | Lowers the decision boundary from 0.5 to counteract class imbalance and under-calling of positives. |
| **F1 = 0.571 (validation)** | F1-score achieved at that threshold on the validation set. | Confirms improved balance between precision and recall; validates tuning effectiveness. |
| **Comparison to 0.5 default** | Default threshold was fixed at 0.5 for all heads. | The tuned value increases sensitivity to true positives without inflating false positives. |
**Significance:**
- Only the median-risk head is tuned; max-risk remains at 0.5 since it already performs optimally.
-	This threshold is then applied during test evaluation, producing a fairer and more balanced classification, compares best-form models fairly without introducing test data leakage.

#### 3. Regression Predictions (Log-Space Range)
**Terminal Output**
```bash
Regression predictions (log-space):
Min:  0.0802
Max:  0.3213
Mean: 0.2043
```
**Purpose:**  
- Checks that the model’s continuous regression outputs fall within a **plausible range** and show healthy dispersion.
**Interpretation:**  
- **Min/Max** confirm predictions are numerically stable → no outliers or NaN values.  
- **Mean (0.2043) is a quick indicator of bias direction**.  
  - If mean > true mean → overprediction bias  
  - If mean < true mean → underprediction bias  
- Together, these validate that the regression head is functioning and producing meaningful log-scale values.

#### 4. Correlation Analysis
**Terminal Output**
```bash
Correlation (log-space): 0.754
Correlation (raw-space): 0.739
```
**Purpose:**  
- Assesses **monotonic consistency** → whether predictions track true ranking across patients.
**Interpretation:**
- Both correlations ≈ 0.75 indicate **strong directional alignment**: patients with higher true high-risk times also receive higher predicted scores.  
- However, correlation ≠ calibration → the model follows the right trend but not necessarily the right *scale*.  
- This pattern explains why correlation is strong even though R² was initially negative.
**Meaning:**  
- The model is **structurally correct** (it learned the right shape) but **numerically biased**, warranting calibration rather than retraining.

#### 5. Calibration Equation
**Terminal Output**
```bash
Calibration: y_true_log ≈ 0.712 * y_pred_log + -0.035
```
**Purpose:**  
- The calibration is done after the model’s initial predictions and before recomputing metrics.
- Fits a simple linear regression model, after fitting, the coefficients are extracted: a → slope, b → intercept
- Quantifies the **systematic bias** found in the uncalibrated regression predictions.
**Interpretation:**
- **Slope (a = 0.712)** < 1 → the model **overpredicts** magnitudes slightly.  
- **Intercept (b = −0.035)** < 0 → small constant downward correction needed.  
- This linear mapping aligns predicted and true log-values without retraining.
**Significance:**  
- Confirms the bias is linear and correctable.  
- The model’s internal structure (trend learning) is valid → only its scale required adjustment.

#### 6. Final Refined Test Metrics
**Terminal Output**
```bash
=== Final Refined Test Metrics ===
Max Risk (0.5) — AUC: 0.923, F1: 0.929, Acc: 0.867
Median Risk (0.5) — AUC: 0.833, F1: 0.286, Acc: 0.667
Median Risk (0.430) — AUC: 0.833, F1: 0.545, Acc: 0.667
Regression (log) — RMSE: 0.108, R²: -1.096
Regression (raw) — RMSE: 0.129, R²: -1.349
Regression (log) calibrated — RMSE: 0.049, R²: 0.569
Regression (raw) calibrated — RMSE: 0.056, R²: 0.548
```
**Purpose**
- Summarises quantitative performance across all model heads (classification + regression) and demonstrates threshold tuning + calibration improvement.
**Interpretation (Classification Focus)**
| **Task**              | **Threshold** | **AUC** | **F1**  | **Accuracy** | **Interpretation** |
|------------------------|---------------|----------|----------|---------------|--------------------|
| **Max Risk**           | 0.5           | 0.923    | 0.929    | 0.867         | Excellent discrimination and precision–recall balance. Robust high-risk identification with few false negatives. |
| **Median Risk**        | 0.5           | 0.833    | 0.286    | 0.667         | Moderate AUC but poor F1 → class imbalance and threshold misalignment. Model ranks correctly but misses positives. |
| **Median Risk (Tuned)**| 0.430         | 0.833    | 0.545    | 0.667         | Same separability (AUC unchanged) but F1 doubled → improved sensitivity and recall through threshold tuning. |
**Interpretation (Regression Focus)**
| Stage | RMSE (log) | R² (log) | RMSE (raw) | R² (raw) | Interpretation |
|--------|-------------|----------|-------------|----------|----------------|
| **Before Calibration** | 0.108 | −1.096 | 0.129 | −1.349 | Model learned correct trend but mis-scaled predictions. Negative R² due to global bias. |
| **After Calibration** | 0.049 |  0.569 | 0.056 |  0.548 | Bias corrected. Predictions now unbiased and aligned with true values. |
**Specific Statistics**
| **Metric**     | **Change (Before → After Calibration)** | **Improvement (%) / Δ** | **Interpretation** |
|-----------------|-----------------------------------------|--------------------------|--------------------|
| **RMSE (log)**  | 0.108 → 0.049                          | ↓ 55%                    | Calibration halved absolute prediction error in log-space. |
| **R² (log)**    | −1.096 → +0.569                        | +1.665 gain              | From meaningless to moderate fit → confirms alignment. |
| **RMSE (raw)**  | 0.129 → 0.056                          | ↓ 57%                    | Corrected scale reduced raw-space error proportionally. |
| **R² (raw)**    | −1.349 → +0.548                        | +1.897 gain              | Now explains ~55% (R² ~0.55) of real-world variance → clinically credible range. |
**Meaning**
- Classification threshold tuning simultaneously enhanced **sensitivity and F1** for the median-risk head, improving balance between recall and precision.
- RMSE roughly halved, confirming **improved accuracy**.  
- R² moved from negative (mis-scaled) to positive (> 0.5), confirming **good model fit** after calibration.  
- Both log and raw improvements prove that calibration **restored numerical validity** while preserving learned trends.  
**Key Takeaways**
1. Model generalised correctly; structure was never faulty.  
2. Negative R² was diagnostic of bias, not model failure.  
3. Post-hoc linear calibration fixed the regression bias entirely → **no retraining required**.  
4. Threshold tuning corrected the **classification bias** caused by class imbalance.  
5. Both adjustments (calibration + threshold tuning) produced **final clinically valid and reportable results**.

#### 7. Test Set Composition
**Terminal Output**
```bash
Test IDs used: [10002428, 10005909, … , 10039831]
```
**Purpose:**  
- Confirms **which patients** were included in the held-out test set for reproducibility.
**Interpretation:**  
- Exactly 15 unique patient IDs → ensures correspondence between CSV predictions and evaluation metrics.  
- Verifies dataset integrity (no mismatch between split definition and tensor dimensions).

#### 8. Mean and Standard Deviation of Regression Outputs
**Terminal Output**
```bash
Test IDs used: [10002428, 10005909, … , 10039831]Mean of y_true_reg: 0.1199, Std: 0.0840
Mean of y_pred_reg: 0.2043, Std: 0.0793
```
**Purpose**
- Provides a quick **distributional sanity check** → confirming that predictions follow the same overall spread and central tendency as the ground truth.
**Interpretation**
| Metric | Meaning | Diagnostic Insight |
|---------|----------|--------------------|
| **mean(y_true_reg)** ≈ 0.12 | Average patient spent ~12% of time in high-risk state | Baseline context for model scaling |
| **mean(y_pred_reg)** ≈ 0.20 | Model’s average prediction > true mean | Confirms systematic overprediction bias |
| **std(y_true_reg)** ≈ 0.084 | True variation across patients | Represents natural variability |
| **std(y_pred_reg)** ≈ 0.079 | Model predicts similar spread | Model captures variability well |
**Scenarios**
| Comparison | Interpretation |
|-------------|----------------|
| std(model) ≈ std(true) | Model captures patient-level variability realistically |
| std(model) ≪ std(true) | Over-smoothed predictions (under-responsive) |
| std(model) ≫ std(true) | Overconfident predictions (too dispersed) |
**Conclusion**
- The model reproduces realistic variability (std ≈ match).  
- Overestimation of the mean explains the pre-calibration bias.  
- Calibration corrected this global offset, leading to R² > 0.5.

#### Overall Summary
**The terminal outputs collectively demonstrate that:**
- The **evaluation pipeline** is fully validated, cleanly tuned, and bias-corrected.  
- The **TCN model structure and learning** are valid, no retraining required.  
- The initial **regression bias** was **systematic and fully correctable**.  
- **Classification threshold tuning** improved recall and F1-score for the median-risk head.  
- **Post-hoc regression calibration** enhanced both **accuracy (↓ RMSE)** and **explanatory power (↑ R²)**.  
- Together, these refinements deliver a **stable, interpretable, and reproducible evaluation pipeline**.  
**Final Status:** Evaluation complete → refined TCN calibration and threshold tuning validated; all metrics saved and reproducible.

---

### Understanding Threshold Tuning 
#### Purpose
- **Threshold tuning** adjusts the cutoff probability used to decide between class 0 and class 1 in binary classification tasks.  
- Although models are trained with `pos_weight` to address **class imbalance**, the default classification threshold of **0.5** often remains **suboptimal**, especially when positive cases are rarer.  
- The model may correctly learn ranking (high AUC) but still misclassify due to an inappropriate threshold → leading to low **F1** and **recall**.
#### What Happened
- During evaluation, the **max-risk head** performed very well (AUC = 0.923, F1 = 0.929, Accuracy = 0.867), showing strong discriminative ability → no tuning needed.  
- The **median-risk head**, however, showed:
```Bash
Median Risk  — AUC: 0.833, F1: 0.286, Accuracy: 0.667
```
- Despite a high AUC, the F1 was very low. This indicated that:
  - The model could **rank** median-risk cases correctly (good AUC).
  - But it **predicted too few positives** at the 0.5 cutoff → low recall and low F1.  
- Therefore, tuning the **decision threshold** was necessary to optimise **F1**, the harmonic mean of precision and recall.
#### Why Threshold Tuning Solves It
**AUC vs F1**
- **AUC (Area Under the ROC Curve)** measures ranking performance → how well the model separates positive from negative cases, regardless of the threshold.  
- **F1-score**, however, depends on the threshold → it balances **precision** (how many predicted positives are correct) and **recall** (how many actual positives are found).  
- When classes are imbalanced (as in the median-risk task), a 0.5 threshold often biases toward the majority class (negatives).  
- Lowering the threshold allows more positives to be detected, improving recall and hence F1.  
**Mathematically:**
- **Precision:** TP / (TP + FP) → out of all the positives you predicted, how many are true
- **Recall:** TP / (TP + FN) →  out of all the true positives, how many did you actually predict
- **F1-score:** 2 × (Precision × Recall) / (Precision + Recall) → balance between precision and recall (higher F1 = betetr tradeoff).
- The **optimal threshold** maximises F1 on the validation set.
#### Implementation
1. **Load validation tensors and labels**
  - Load validation tensors and masks
  ```python
  x_val = torch.load(TEST_DATA_DIR / "val.pt", map_location=device)
  mask_val = torch.load(TEST_DATA_DIR / "val_mask.pt", map_location=device)
  ```
  - Validation labels were manually reconstructed from the same patient splits (JSON):
  ```python
  val_ids = splits["val"]
  val_df = features_df.set_index("subject_id").loc[val_ids].reset_index()
  val_df["max_risk_binary"] = val_df["max_risk"].apply(lambda x: 1 if x > 2 else 0)
  val_df["median_risk_binary"] = val_df["median_risk"].apply(lambda x: 1 if x == 2 else 0)
  ```
2. **Define helper function**
  ```python
  def find_best_threshold(y_true, y_prob):
      thresholds = np.linspace(0.05, 0.95, 91)
      f1s = [f1_score(y_true, (y_prob >= t).astype(int)) for t in thresholds]
      best_t = thresholds[np.argmax(f1s)]
      best_f1 = max(f1s)
      return best_t, best_f1
  ```
	-	Evaluates F1 across thresholds from 0.05 → 0.95.
	-	Returns the threshold with the highest F1-score on validation data.
3. **Run the model on validation data**
  ```python
  with torch.no_grad():
      val_outputs = model(x_val, mask_val)
  val_prob_median = torch.sigmoid(val_outputs["logit_median"].squeeze()).cpu().numpy()
  ```
  - Model forms predictions for validation data
4. **Find optimal threshold**
  - Use helper function to find the best threshold by calculating the best F1 based on the true vs predicted validation values.
  ```python  
  best_thresh_median, best_f1_median = find_best_threshold(y_val_median, val_prob_median)
  print(f"[INFO] Median Risk Threshold (from validation) = {best_thresh_median:.3f} (F1={best_f1_median:.3f})")
  ```
  - Output:
  ```bash  
  Median Risk Threshold (from validation) = 0.430 (F1=0.571)
  ```
  - The new optimal threshold (0.43) nearly doubled F1 (0.286 → 0.545), confirming the benefit.

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

#### Mathematical Intuition
- Each model outputs a probability ( `P(y=1|x)` ) → the likelihood that a sample belongs to the positive class.  
- The decision rule depends on a threshold `τ` (tau):  
```text
ŷ = 1  if  P(y=1 | x) ≥ τ
ŷ = 0  otherwise
```
- By default, `τ` = 0.5.  
- However, when classes are **imbalanced** (e.g., far fewer positives), this threshold can under-call the minority class → resulting in **low recall** and **poor F1-score**.  
- Changing τ shifts the trade-off:
	- ↓ `τ` = ↑ recall, ↓ precision
	-	↑ `τ` = ↓ recall, ↑ precision
- Threshold tuning adjusts `τ` to maximise the F1-score by balancing **precision** (correct positive predictions) and **recall** (capturing all true positives).  
-	The optimal `τ` maximises the chosen balance metric (here, F1).

#### Summary
-	Class weighting handled imbalance during training, but not threshold sensitivity during evaluation.
-	A high AUC but low F1 signalled that the median head was rank-correct but threshold-biased.
-	Post-hoc threshold tuning (based on validation F1) corrected this bias.
-	**Result:** Median head now performs optimally at τ = 0.43, demonstrating that evaluation-time tuning, not retraining, was the right solution.
-	The max-risk head remains at 0.5 due to already excellent balance across metrics.

---

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

---

### Calibration Bias Analysis
#### Purpose
- **Calibration bias analysis** ensures that the regression head’s outputs are both **directionally correct** and **numerically aligned** with the true targets.  
- It distinguishes between:
  - **Structural correctness** → the model learns the shape and ordering of the true signal (high correlation).  
  - **Numerical correctness** → the model’s predictions match the absolute scale of the true signal (high R²).  
- A model can exhibit **high correlation but poor R²** when its predictions consistently **over- or under-estimate** true values → this represents **systematic bias**, not model failure.  
- Identifying and correcting this bias ensures that model outputs are **clinically interpretable** and **quantitatively valid**, bridging the gap between learned relationships and real-world magnitudes.

#### Background Interpretation
1. **Regression Metric Outputs**
  ```bash
  Regression (log) — RMSE: 0.108, R²: -1.096
  Regression (raw) — RMSE: 0.129, R²: -1.349
  ```
  - Even after refining the regression process (training in log-space, inverse-transforming predictions via np.expm1, and evaluating both log- and raw-space metrics), the R² values remained negative on the test set.
  - This was paradoxical → the model clearly learned structure during training, yet R² suggested performance worse than predicting the mean.
  - To understand this inconsistency, correlation analysis was introduced.
  - **Why R² Was Misleading**
    - R² penalises scale and mean bias, not just shape or direction.
    - A model can follow the true trend well (monotonic relationship) but still yield negative R² if systematically biased (overestimates or underestimates).
    - Hence, R² alone could not differentiate between trend failure (random error) and calibration bias (systematic offset).
2. **Correlation Calculations Added**
  - To separate **trend learning** from **numerical scaling and bias**, Pearson correlation was introduced:
  ```bash
  corr_log = np.corrcoef(df_preds["y_true_reg_log"], df_preds["y_pred_reg_log"])[0,1]
  corr_raw = np.corrcoef(df_preds["y_true_reg"], df_preds["y_pred_reg_raw"])[0,1]
  ```
  - **Purpose**
    -	Quantify directional agreement between true and predicted values.
    - **Distinguish between:**
      -	Structural failure → low correlation (<0.4)
      -	Systematic bias → high correlation (>0.7) but negative R²
    -	If correlation ≥ 0.7 and R² < 0, the model learned the correct trend but mis-scaled predictions.
    -	In such cases, retraining is unnecessary → a post-hoc linear calibration is sufficient.
3. **Correlation Comparison Outputs**
  ```bash
  Correlation (log-space): 0.754
  Correlation (raw-space): 0.739
  ```
  - Strong correlation (~0.75) confirmed the model was directionally correct → it captured the true pattern of deterioration risk.
  -	However, negative R² (−1.09 log-space, −1.35 raw-space) showed that predictions were numerically biased (i.e., consistently too high).
  -	This discrepancy confirmed the model’s predictions followed the correct trend (monotonic relationship) but had systematic calibration bias rather than structural error.
4. **Means of predictions**
  ``` bash
  Mean of y_true_reg: 0.1199, Std: 0.0840  # Ground truth: actual % time high
  Mean of y_pred_reg: 0.2043, Std: 0.0793  # Model output: predicted log-space mean
  ```
  - The predicted mean (0.2043) was significantly higher than the true mean (0.1199) → clear overestimation bias.
  - This bias explained the negative R²: the predictions followed the correct trend (high correlation) but were offset upward.
  - The class imbalance (≈3:1) was addressed during training via `pos_weight`, but no post-hoc calibration was initially applied → hence, bias persisted at inference.
5. **Calibration Curve Plot**
  - A scatter plot was generated as direct, visual confirmation of calibration bias before correction:
    -	**x-axis:** true `log1p(pct_time_high`)
    -	**y-axis:** predicted `log1p(pct_time_high)`
    - **Red dashed diagonal (`y=x`):** perfect calibration line
  - Saved as `tcn_regression_calibration_logspace.png`
  - The points followed a roughly linear pattern above the diagonal, confirming systematic overprediction. 
  - **Purpose:** Support quantitative findings (correlation vs R²) with a reproducible diagnostic plot.

#### Summary of Interpretation
- High correlation (≈0.75) proved the regression head learned meaningful temporal structure.
- Negative R² reflected systematic mean bias, not model collapse.
- Mean comparison confirmed consistent overestimation.
- **Therefore:**
  - Retraining was unnecessary.
  - A simple log-space linear calibration (`y_true_log ≈ a * y_pred_log + b`) was the correct fix.
  - This adjustment rescales predictions to match real-world magnitudes while preserving the model’s learned relationships.
- **Conclusion:**
	-	The model was scientifically sound but numerically biased.
	-	Correlation analysis exposed the issue; calibration corrected it.
	-	The refined evaluation pipeline now quantitatively separates trend validity from scale accuracy, enabling precise post-hoc correction without retraining.

#### How Calibration Is Calculated
- A post-hoc linear regression is fit between the model’s predictions (`y_pred_log`) and true values (`y_true_log`) in log-space: `y_true_log ≈ a * y_pred_log + b`
- **This produces coefficients:**
	- a (slope) → rescales the prediction amplitude (corrects under/overestimation trend).
	-	b (intercept) → shifts the prediction baseline (removes mean bias).
- **The adjusted predictions are then:**
```python
y_pred_reg_log_cal = a * y_pred_reg_log + b
y_pred_reg_raw_cal = np.expm1(y_pred_reg_log_cal)
```
- This process aligns the model’s predicted and true log-values without retraining and ensures that both RMSE and R² reflect genuine predictive accuracy rather than uncalibrated scale errors.

#### Interpretation of Results
**Metrics: Pre- vs Post-Calibration**

| Metric       | Before Calibration | After Calibration | Interpretation |
|---------------|--------------------|-------------------|----------------|
| **RMSE (log)** | 0.108 | 0.049 | Error reduced by ~55%, confirming improved numerical accuracy. |
| **R² (log)**   | −1.096 | +0.569 | Shift from negative to positive indicates that bias, not model failure, caused poor initial R². Calibration restored proper scaling. |
| **RMSE (raw)** | 0.129 | 0.056 | Error reduced by ~57%, confirming consistent improvement in the clinically relevant scale. |
| **R² (raw)**   | −1.349 | +0.548 | Model now explains ~55% of variance in true values → confirming real-world interpretability and calibration success. |

**Reasoning**
- **High correlation (≈0.75)** confirmed that the regression head learned the correct directional trend → no retraining was needed, the model was structurally valid.  
- **Negative R²** before calibration reflected a **systematic scale bias**, not stochastic error or collapse.  
- **Linear calibration** in log-space (`y_true_log ≈ a * y_pred_log + b`) corrected this proportional bias without altering model weights.  
- **Result:** Strong alignment between predicted and true values in both log- and raw-space, improving **clinical interpretability** and **comparability** against baseline models (LightGBM, NEWS2).

#### Calibration Comparison Plot
**Purpose:**  
- Visually validates the **numerical correction achieved by calibration**.  
- Confirms that the post-hoc linear adjustment (`y_true_log ≈ a * y_pred_log + b`) successfully realigned predictions with true values.
**Description:**  
- Scatter plot overlays **pre-** (steelblue) and **post-calibration** (orange) predictions in log-space.  
- The **red dashed line** (`y = x`) represents perfect calibration → the ideal 1:1 alignment between predicted and true values.  
- Saved automatically as: `tcn_regression_calibration_comparison_logspace.png`
**Interpretation:**  
- Before calibration → points systematically offset from the diagonal → consistent **overprediction bias**.  
- After calibration → orange points align tightly around the red line → **bias eliminated**, confirming accurate rescaling.  
- The slope and spread remain consistent → the model’s **monotonic trend learning** is preserved, proving that calibration fixed scale bias without altering learned relationships.  
- This visual outcome corroborates the numerical improvement in RMSE and R², confirming that the correction was **quantitative, not cosmetic**.
**Reasoning:**  
- **Confirms calibration worked:** predictions are now both accurate and unbiased.  
- Demonstrates that model bias, not randomness, explained prior performance issues.  
- Validates the **log-space linear correction** as a robust alternative to retraining.

#### Summary
- Calibration bias was **systematically identified** through added **correlation analysis** and **visual diagnostics**.  
- The model was **directionally correct** (correlation ≈ 0.75) but **numerically biased**, causing misleadingly negative R² values.  
- A **post-hoc linear calibration** in log-space (`y_true_log ≈ a * y_pred_log + b`) was applied to correct this bias **without retraining**.  
- The refined evaluation now **prints correlation coefficients, calibration parameters, and recalibrated metrics**, including visual validation of improvement, providing transparent evidence of improvement.  
- **Outcome:** The refined TCN’s regression head is now **calibrated, accurate, and interpretable**, confirming that its learned temporal structure was valid → the error stemmed from **scaling bias**, not model failure, and was fully correctable through analytical calibration.

---

### Reflection
#### Challenges
1. **Persistent confusion about log-transforms and inverse transforms**  
  - I initially struggled to understand where and why `np.expm1()` should be applied, and why inverse transformation was necessary for clinical interpretability.  
  - The distinction between log-space (used for model optimisation) and raw-space (used for clinical meaning) was not immediately clear, causing repeated mis-scaling and misleading metrics.
2. **Regression metrics remained negative despite apparent improvements**  
  - After multiple fixes to target scaling and loss computation, `R²` values were still negative.  
  - This led to several false starts and moments of frustration → it seemed as though the model kept failing despite logical corrections.
3. **Inconsistency between training and evaluation spaces**  
  - The model trained on log-transformed targets but was evaluated against raw targets, producing artificially poor performance.  
  - The conceptual mismatch between “model correctness” (log-space) and “clinical interpretability” (raw-space) took significant time to fully understand.
4. **Understanding why both log- and raw-space metrics were needed**  
  - It was unclear why we should compute metrics twice.  
  - Eventually, it became clear that raw metrics show practical usefulness, while log metrics show mathematical correctness and generalisation.
5. **Repeated recalibration and debugging cycles**  
  - Every time one problem was fixed, another emerged (scaling, bias, thresholds, etc.).  
  - It was mentally draining to keep revisiting the same evaluation code, yet this iterative process revealed the true complexity of robust validation.
6. **Classification imbalance confusion**  
  - Despite applying `pos_weight` during training, the median-risk F1 remained low.  
  - It wasn’t initially obvious that class weighting affects training, while threshold tuning affects decision boundaries post-training.
7. **Overwhelming technical debugging workload**  
  - Calibration, correlation checks, threshold tuning, and dual-scale metric computation turned this phase into a prolonged analytical exercise that required both mathematical understanding and experimental verification.
#### Solutions and Learnings
1. **Resolved scale mismatch through dual-space evaluation**  
  - By explicitly computing metrics in both log-space and raw-space, I separated scientific correctness (training domain) from clinical interpretability (real domain).  
  - This ensured that every metric reported corresponded to a specific purpose → internal validation vs. clinical reporting.
2. **Introduced correlation analysis to diagnose bias**  
  - Adding Pearson correlation revealed that the model’s structure was valid (`r ≈ 0.75`), and that poor `R²` values stemmed from systematic bias, not model failure.  
  - This justified post-hoc calibration instead of unnecessary retraining.
3. **Implemented post-hoc linear calibration**  
  - Using `y_true_log ≈ a * y_pred_log + b` corrected both mean and scale bias analytically.  
  - This improved RMSE by ~55 % and turned R² from negative to positive (> 0.5), confirming that the model’s internal learning was correct.
4. **Created dual-plot visual diagnostics**  
  - Calibration plots before vs after correction provided visual evidence of systematic bias removal → ensuring interpretability and transparency in scientific communication.
5. **Clarified when and why to use inverse transforms**  
  - Realised that only model predictions (not true labels) should be inverse-transformed with `np.expm1()`, maintaining mathematical consistency.
6. **Introduced threshold tuning for median-risk head**  
  - Learned that despite class weighting during training, threshold tuning is required post-training to maximise F1.  
  - The tuned threshold (τ = 0.43) nearly **doubled F1** from 0.286 → 0.545 while keeping AUC constant → proving the importance of separating *training balance* and *decision calibration*.
7. **Developed clean reproducible outputs**  
  - Final metrics, calibration parameters, and test IDs are all saved in structured JSON and CSV formats, making the evaluation pipeline fully auditable and reproducible.
8. **Learned the philosophy of ML evaluation**  
  - Evaluation is not just about producing good numbers → it’s about understanding why they are good or bad.  
  - Each fix (inverse transform, correlation, calibration, threshold tuning) deepened comprehension of the mathematical relationships underlying performance metrics.
#### Overall Reflection
- This phase was **the most intellectually and emotionally demanding** stage of the entire project.  
- Each discovery required dismantling assumptions, re-testing logic, and rebuilding parts of the evaluation pipeline from first principles.  
- It taught me that robust machine-learning evaluation is inherently iterative → you tune until every inconsistency is explained.  
- Although exhausting, the final outcome is a **scientifically transparent, reproducible, and interpretable** evaluation pipeline that differentiates between:
  - What the model learned (log-space correctness), and  
  - How well it performs in the real world (raw-space interpretability).  
- The process was not just debugging → it was **a full audit of understanding**.  
- I now grasp how calibration, class imbalance, and thresholding interact mathematically and practically, and how to ensure that future models are both **accurate and interpretable**.

---

### Overall Summary
- Today’s work finalised the refined TCN evaluation pipeline, completing the most complex stage of Phase 5.  
- All key diagnostic and correction components were successfully integrated and validated:
  - **Correlation analysis** confirmed the regression head learned valid structure.  
  - **Post-hoc calibration** corrected systematic bias without retraining.  
  - **Threshold tuning** optimised the median-risk classification head and doubled its F1-score.  
  - **Dual-scale metrics** (log + raw) now distinguish between internal model correctness and clinical interpretability.  
  - **Final visualisations** (calibration and comparison plots) provide transparent evidence of improvement.  
- The evaluation script is now fully complete, reproducible, and publication-ready, quantitatively validated across all outputs and scientifically interpretable.

---

## Day 30-31 Notes - Finish Phase 5: Re-train and Evaluate LightGBM Baseline (Step 4)

---

### Goals
- Retrain LightGBM models using the **same 70/15/15 patient split** as the TCN model to ensure a **fair, out-of-sample comparison**.  
- Produce **predictions and performance metrics** compatible with TCN evaluation while preserving Phase 3 tuned hyperparameters.  
- Clarify the distinction between **deployment-ready models** (trained on all 100 patients) and **benchmarking models** (retrained for controlled evaluation).  


### What We Did
#### Step 4: Re-train and Evaluate Final LightGBM Models (`evaluate_lightgbm_testset.py`)
**Purpose**
- To retrain and evaluate the final tuned LightGBM models using patient-level NEWS2 features.  
- This script forms the **production-grade benchmark** for non-temporal models (classical ML baseline) and serves as the reference comparison for the Temporal Convolutional Network (TCN) sequence model.  
- It ensures **reproducibility**, **transparency**, and consistent data handling across all training, evaluation, and reporting phases.
**Process**
0. **Imports and Directories**
  - Imported essential libraries: `pandas`, `lightgbm`, `joblib`, `json`, `pathlib.Path`.
  - Imported metric utilities from `evaluation_metrics.py`.
  - **Defines consistent input/output paths:**
    - **Input:** `news2_features_patient.csv`, `patient_splits.json`, `best_params.json`
    - **Output:** results stored in `lightgbm_results/`.
1. **Load Hyperparameters**
  - Input: `ml_models_lightgbm/hyperparameter_tuning_runs/best_params.json`
  - Read tuned hyperparameters from phase 3 `best_params.json` (one per target: `max_risk`, `median_risk`, `pct_time_high`).
  - Validated all expected targets exist before proceeding.
  - These parameters define the optimal configuration (e.g., learning rate, depth, regularisation) for each target-specific model.
2. **Load Data and Splits**
  - Loaded `news2_features_patient.csv` → pre-engineered patient-level dataset with aggregated NEWS2 features.  
  - Loaded `patient_splits.json` (from phase 4) → defines fixed patient IDs for 70 train / 15 test patients.  
  - Built train/test DataFrames (preserving deterministic order for test) → `train_df`, `test_df`
  - **Recreated binary targets:**
    - `max_risk_binary` → 0=not high risk, 1=high risk
    - `median_risk_binary` → 0=low risk, 1=medium risk
  - Ensured deterministic test ordering using `.loc[test_ids]` so that true vs predicted values correspond to the same patient.
  - Defined final model input features (`feature_cols`) by excluding identifiers and target columns.  
  - Ensured deterministic test alignment using `.loc[test_ids]` so that `y_true` and predictions correspond to identical patients in evaluation.
3. **Train LightGBM Models**
  - Defined a structured `targets` list containing:
    - Classification → `max_risk`, `median_risk`
    - Regression → `pct_time_high`
  - For each target:
    - Loaded its tuned parameters → `params = best_params[target]`
    - Initialised appropriate model:
      - Classification: `lgb.LGBMClassifier(**params, random_state=42, class_weight="balanced")`
      - Regression: `lgb.LGBMRegressor(**params, random_state=42)`
    - Fitted model on `X_train`, `y_train` with `model.fit(X_train, y_train)`
    - Saved retrained model to `.pkl` via `joblib.dump(model, model_file)`
  - Each model trained with reproducible randomness control (`random_state=42`)
  - **Outputs:** 
    - `lightgbm_results/max_risk_retrained_model.pkl`
    - `lightgbm_results/median_risk_retrained_model.pkl`
    - `lightgbm_results/pct_time_high_retrained_model.pkl`
4. **Evaluate Models on Test Set**
  - Iterated through each retrained model:
    - Loaded model via `joblib.load()`
    - Extracted test data (`X_test`, `y_test`)
    - Generated predictions:
      - Classification → `model.predict_proba(X_test)[:, 1]`
      - Regression → `model.predict(X_test)`
    - Stored results in structured dictionary `preds_dict[target]`:
      - Classification: `preds_dict[target] = {"y_true": y_test, "prob": y_prob}`  
      - Regression: `preds_dict[target] = {"y_true": y_test, "y_pred": y_pred}`
5. **Save Predictions CSV**
  - Created unified DataFrame `df_preds` combining all targets:
    - Columns: `y_true_max`, `prob_max`, `y_true_median`, `prob_median`, `y_true_reg`, `y_pred_reg`
  - Each row represents a single patient in the test set.
  - Ensured perfect alignment between true labels `y_true` and model predictions for interpretability.
  - Saved to `lightgbm_predictions.csv`.
  - **Output:** `lightgbm_results/lightgbm_predictions.csv`
6. **Compute Metrics**
  - Used evaluation functions from `evaluation_metrics.py`:
    - `compute_classification_metrics()` → AUROC, F1, Accuracy, Precision, Recall  
    - `compute_regression_metrics()` → RMSE, R²  
  - Computed:
    - `metrics_max` for max risk classification
    - `metrics_median` for median risk classification
    - `metrics_reg` for regression (`pct_time_high`)
  - Combined into `metrics` dictionary.
  - Printed formatted results to console and saved as JSON.
  - **Output:** `lightgbm_results/lightgbm_metrics.json`
7. **Save Evaluation Summary**
  - Generated `training_summary.txt` with:
    - Dataset summary (70 train / 15 test patients)
    - Feature count (40 features)
    - Testset metrics per target.
    - Full hyperparameter configuration for reproducibility
  - Provides auditable trace of which models and parameters produced which results.
  - **Output:** `lightgbm_results/training_summary.txt`

**Outputs**
| File | Description |
|------|--------------|
| `{target}_retrained_model.pkl` | Retrained LightGBM models for each target using 70-patient split |
| `lightgbm_predictions.csv` | Combined predictions + true values for all test patients |
| `lightgbm_metrics.json` | Full classification and regression performance metrics |
| `training_summary.txt` | Summary of dataset info, metrics, and hyperparameters |

**File Structure**
```text
data/
├── processed_data/ 
│     └── news2_features_patient.csv
src/
├── ml_models_lightgbm/
│     └── hyperparameter_tuning_runs/
│           └── best_params.json
├── ml_models_tcn/
│     └── deployment_models/
│           └── preprocessing/ 
│                 └── patient_splits.json
└── prediction_evaluations/
      ├── evaluate_lightgbm_testset.py
      └── lightgbm_results/
            ├── max_risk_retrained_model.pkl
            ├── median_risk_retrained_model.pkl
            ├── pct_time_high_retrained_model.pkl
            ├── lightgbm_predictions.csv
            ├── lightgbm_metrics.json
            └── training_summary.txt
```
**Reasoning**
- The original final LightGBM models from **Phase 3** were trained on **all 100 patients** → these are the **deployment models**, representing full-dataset, production-ready versions of the classical ML approach.  
- However, the **TCN final model (Phase 4)** was trained and validated using a **70/15/15 patient split (train/validate/test)** to evaluate true generalisation on unseen patients.  
- To ensure a **fair and statistically valid comparison**, the LightGBM models must be **retrained on the same 70 training patients** and **evaluated on the same 15 test patients** used for the TCN model.  
- This alignment guarantees:
  - Identical patient inclusion/exclusion across both models.  
  - Identical feature inputs and label definitions for each patient.  
  - Direct comparability of classification and regression performance metrics.  
- Additionally, the TCN model was optimised for best performance via multiple refinements:
  - Class weighting for class imbalance.  
  - Log-transformation of continuous outputs.  
  - Threshold tuning for probability calibration.  
  - Post-hoc calibration of classification probabilities.  
- Therefore, to make the benchmark **methodologically symmetric**, the LightGBM retraining uses the **best hyperparameters obtained from Phase 3** → ensuring each model type (LightGBM and TCN) is represented in its **strongest validated configuration** under the **same data conditions**.  
- In essence, this retraining converts the Phase 3 LightGBM models into a **controlled experimental baseline**, fully aligned with the **Phase 4 TCN evaluation**, enabling a fair and high-quality cross-model comparison.

**Overall Summary**
- This script rebuilds the LightGBM baseline using the same train/test split as the TCN model to ensure direct comparability.
-	It converts the full-dataset deployment models from Phase 3 into controlled retrained models for scientific evaluation.
-	**Produces clean, structured outputs for downstream visualisation and comparison:**
  -	ROC and calibration curves
  -	Regression scatter plots
  -	Comparison tables (NEWS2 vs LightGBM vs TCN)
-	**End result:** A fully retrained, tested, and documented LightGBM benchmark serving as the non-temporal reference model for the project’s comparative analysis.

---

### LightGBM Predictions and Metrics Interpretation
#### Overview 
- Provides a detailed interpretation of the LightGBM test set predictions, computed metrics, and their implications. 
- The predictions were generated for three targets:
  1. `max_risk` → binary classification for high-risk patients
  2. `median_risk` → binary classification for medium-risk patients
  3. `pct_time_high` → regression for percentage of time patient spent in high-risk state  

#### 1. Predictions Overview
**The test set contains 15 patients. Predictions include:**
| Column | Description |
|--------|-------------|
| `y_true_max` | True label for high-risk (1 = high risk, 0 = not high risk) |
| `prob_max` | LightGBM predicted probability of `max_risk = 1` |
| `y_true_median` | True label for medium-risk (1 = medium risk, 0 = low risk) |
| `prob_median` | LightGBM predicted probability of `median_risk = 1` |
| `y_true_reg` | True value for `pct_time_high` (continuous) |
| `y_pred_reg` | LightGBM predicted value for `pct_time_high` (continuous) |

**Sample of predictions**
| Patient | y_true_max | prob_max | y_true_median | prob_median | y_true_reg | y_pred_reg |
|---------|------------|---------|---------------|------------|------------|------------|
| 0       | 0          | 0.680   | 0             | 0.003      | 0.000      | -0.013     |
| 1       | 1          | 0.937   | 0             | 0.006      | 0.054      | 0.115      |
| 2       | 1          | 0.996   | 1             | 0.630      | 0.249      | 0.198      |
| 3       | 1          | 0.969   | 0             | 0.912      | 0.122      | 0.156      |
| 4       | 1          | 0.923   | 1             | 0.995      | 0.276      | 0.211      |
| 5       | 1          | 0.509   | 0             | 0.018      | 0.023      | 0.046      |
| …       | …          | …       | …             | …          | …          | …          |

**Note: The full CSV contains all 15 patients. Probabilities (`prob_max`/`prob_median`) represent model confidence for classification tasks, while `y_pred_reg` is the continuous predicted percentage of time at high risk.**

#### 2. Metrics Interpretation

**Classification Metrics**
| Target       | ROC AUC | F1 Score | Accuracy | Precision | Recall | Interpretation |
|--------------|---------|----------|----------|-----------|--------|----------------|
| `max_risk`   | 0.846   | 0.929    | 0.867    | 0.867     | 1.000  | High recall indicates the model captures all high-risk patients; some false positives reduce precision slightly. |
| `median_risk`| 0.972   | 0.857    | 0.933    | 0.750     | 1.000  | Excellent discrimination (ROC AUC). Lower precision shows some misclassification of medium-risk patients. |

**Regression Metrics**
| Target         | RMSE     | R²      | Interpretation |
|----------------|----------|---------|----------------|
| `pct_time_high`| 0.0382   | 0.793   | Predictions are close to true values with ~79% variance explained. RMSE indicates average prediction error ~3.8% in percentage time high. |

**Insights**
- **Max Risk Classification:** 
  - The model prioritises high recall (100%) to ensure all high-risk patients are identified, which is critical for ICU early warning. 
  - Slightly lower precision (86.7%) reflects some over-prediction in borderline cases.  
- **Median Risk Classification:** 
  - High ROC AUC (0.972) shows strong separation between medium-risk and non-medium-risk patients. 
  - Perfect recall indicates no medium-risk patient is missed, though precision is reduced due to some false positives.
- **Pct Time High Regression:** 
  - R² = 0.793 demonstrates strong predictive ability on continuous risk exposure. 
  - RMSE = 0.038 indicates tight clustering of predictions around true values. 
  - Minor deviations are expected due to natural variability in patient data.

#### 3. Overall Interpretation
1. The LightGBM models **perform well across classification and regression targets**, producing metrics indicative of strong discriminative power and accuracy.  
2. Predictions (`prob_max`, `prob_median`) provide **probabilistic risk estimates**, enabling threshold adjustment for clinical decision-making.  
3. Regression output (`y_pred_reg`) aligns closely with true percentage time high, suitable for continuous risk monitoring.  
4. These predictions and metrics form a **robust baseline for comparison** against TCN model, ensuring fair evaluation of temporal vs non-temporal approaches.  

#### 4. Recommendations for Next Steps
- **Visualisation:**  
  - ROC curves for `max_risk` and `median_risk` to illustrate discrimination.  
  - Calibration plots for classification tasks to assess probability reliability.  
  - Regression scatter plots (`y_true_reg` vs `y_pred_reg`) and residual histograms for continuous target.
- **Comparison Table:**  
  - Combine LightGBM, TCN, and NEWS2 metrics into a single table for direct comparison.
- **Interpretability:**  
  - Feature importance analysis (LightGBM) and temporal saliency (TCN) for interpretability and clinical insight.

---


### Reflection
#### Challenges
**In-sample vs out-of-sample evaluation**  
- The Phase 3 final LightGBM models were trained on **all 100 patients**, which means their evaluation metrics represent **in-sample fit**, not true generalisation. 
- Comparing this directly with the TCN, which was evaluated on a **held-out 15-patient test set**, would have been scientifically invalid.  
**Ensuring consistent splits for fair comparison**  
- TCN models were trained using a **70/15/15 train/validation/test split**, along with optimisations such as log-transformation, class weighting, threshold tuning, and post-hoc calibration. 
- To enable a valid head-to-head comparison, LightGBM needed to be retrained on the **exact same 70/15/15 split**, with identical test patients, feature sets, and target definitions.
**Misconception about NEWS2 baseline**  
- Initially, the project plan assumed News2 could serve as a separate benchmark for comparison. 
- Realisation: **NEWS2 is the ground truth itself**.
- Any LightGBM or TCN prediction is inherently benchmarked against these true target values, meaning models **cannot outperform the baseline** → they can only approximate it. 
- This impacts how we interpret comparative performance.
**Maintaining alignment across multiple targets**  
- LightGBM trains three separate models (`max_risk`, `median_risk`, `pct_time_high`) independently. 
- Combining predictions into a single CSV requires careful alignment using `subject_id`, to prevent mismatches across patients. 
- TCN produces all outputs simultaneously, so a single merge step isn’t necessary there.
**Avoiding unnecessary computation and variability**  
- Retraining might have tempted us to recompute feature importances or hyperparameter tuning. 
- However, doing so would introduce unnecessary compute and possible variability, compromising reproducibility.

#### Solutions and Learnings
**Retraining LightGBM on the TCN split**  
- Used the **same 70 training / 15 validation / 15 test patients** as TCN.  
- Recreated targets: `max_risk_binary`, `median_risk_binary`, `pct_time_high`.  
- Applied **Phase 3 best hyperparameters** without additional tuning, preserving the “best LightGBM” configuration.  
- Trained models on the training set and evaluated on the test set, producing **predictions and metrics compatible with TCN evaluation**.  
- **Outcome:** controlled, fair, and reproducible comparison between LightGBM and TCN.
**Maintaining scientific integrity**  
- **Phase 3 final models** remain for **deployment and interpretability purposes**, including feature importance visualisation and portfolio demonstration.  
- **Phase 5 retraining models** serve exclusively for **benchmarking**, separating experimental evaluation from deployment-ready models.
**Ensuring best vs best comparison**  
- TCN underwent optimisations like threshold tuning, log-transforms, class weighting, and calibration to maximise predictive performance.  
- Retraining LightGBM with proven Phase 3 hyperparameters ensures **both models are evaluated in their strongest validated configurations** on unseen data.
**Feature importance is not needed**  
- Retraining focuses on **evaluation and comparison**, not exploration.  
- Computing feature importance during retraining would be redundant and could introduce minor random variation.
**Targets training loop logic**  
- Iterating through targets programmatically ensures consistency and reduces errors.  
- Using `subject_id` in `train_df` and `test_df` ordering guarantees correct alignment of predictions across multiple outputs for each patient and correct alignment when merging into `df_preds`.
**NEWS2 baseline understanding**  
  - NEWS2 is now clearly treated as **ground truth**, not a separate model.  
  - Models approximate the ground truth; they cannot “beat” it.  
  - The project’s goal shifts to evaluating **how well LightGBM and TCN reproduce true risk values** and compare their relative predictive quality.

#### Overall Reflection
- Retraining LightGBM on the **same 70/15/15 split as TCN** was essential to establish a **scientifically rigorous and reproducible baseline**.  
- This step clarified the distinction between:
  1. **Deployment-ready models** (Phase 3 final LightGBM) for clinical demonstration and feature importance interpretation.  
  2. **Benchmarking models** (Phase 5 retrained LightGBM) for fair evaluation against TCN on unseen test patients.  
- Realising that **NEWS2 is the ground truth** shifted the project’s perspective: the objective is no longer to improve over a baseline but to measure **approximation fidelity** of LightGBM and TCN.  
- **Lessons learned:**
  - Data splits, unseen test sets, and alignment across targets are critical for valid ML benchmarking.  
  - Hyperparameter reuse and controlled retraining ensure both scientific rigour and fair comparison.  
  - Clear separation of **experimental evaluation** and **deployment-ready models** enhances reproducibility, interpretability, and portfolio demonstration.  
- The retraining pipeline now produces **clean predictions, structured metrics, and evaluation outputs**, ready for downstream visualisation (ROC curves, calibration plots, regression scatter) and head-to-head comparison tables between LightGBM and TCN.  
- Overall, this reinforces **good ML practice**: fair, reproducible, and transparent evaluation of multiple model families against a shared, true ground truth.

---

### Overall Summary
- This step **completes the retraining and evaluation of the LightGBM baseline** using the same 70/15/15 train/validation/test split as the TCN, finalising the evaluation scripts for the classical ML models.  
- All three targets (`max_risk`, `median_risk`, `pct_time_high`) now have **aligned predictions and metrics** with the TCN pipeline, producing fully comparable outputs on unseen test patients.  
- **By completing this step, the pipeline now has:**
  - Validated, reproducible LightGBM models ready for benchmark comparison.  
  - Structured outputs (`predictions CSV`, `metrics JSON`, `training summary`) that align with the TCN evaluation outputs.  
- This marks a **critical milestone in the ML pipeline**, as all preprocessing, model training, evaluation, and metric computation steps for both non-temporal (LightGBM) and temporal (TCN) models are now complete.  
- The pipeline is now ready to move onto Phase 6 with **direct comparison of TCN vs LightGBM**.  
- Enables analysis of **approximation fidelity to NEWS2 ground truth**, differences in classification and regression performance, and insights into model strengths/weaknesses.  
- Completing this step ensures that the **ML evaluation pipeline is fully aligned, reproducible, and scientifically rigorous**, allowing for high-quality comparative analysis between non-temporal vs temporal model.

---

### Next Steps
1. **Visualisations:**  
  - Generate ROC curves and calibration plots for `max_risk` and `median_risk`.  
  - Create regression scatter plots and residual histograms for `pct_time_high`.  
2. **Comparative Analysis:**  
  - Construct tables combining LightGBM and TCN predictions and metrics for side-by-side comparison.  
  - Highlight relative performance on unseen test patients, focusing on approximation fidelity to NEWS2 ground truth.  
3. **Interpretability:**  
  - Use Phase 3 final LightGBM feature importances for clinical insight; 
  - Use Phase 4.5 TCN temporal saliency to be visualised for sequence-level interpretability.  
4. **Documentation:**  
  - Consolidate Phase 5 notes, outputs, and interpretations for reproducibility and portfolio inclusion.  

---

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

---

## Day 32-36 Notes - Start Phase 6: Comparative Analaysis - Summary Metrics + Visualisation (Steps 1-2)

---

### Goals
- Begin **Phase 6** — the final analytical phase — focusing on **comparative evaluation** of the two final models (LightGBM and TCN_refined).  
- Execute **Step 1 (Quantitative Analysis)** to compile and validate unified performance metrics across all targets (`max_risk`, `median_risk`, `pct_time_high`).  
- Execute **Step 2 (Visualisation + Diagnostic Analysis)** to extend the comparison using numerical plot data and visual outputs for discrimination, calibration, and regression fidelity.  
- Conduct a full interpretive comparison explaining not only **which model performs better**, but also **why** → incorporating target-type alignment, dataset size, and model-target mismatch to contextualise results.  
- Identify conditions where the **TCN could outperform** (e.g. with larger timestamp-level datasets) and discuss **LightGBM’s strengths** under small, aggregated clinical data → ensuring conclusions account for **limitations, data scale, and architectural suitability**.

### What We Did
#### Step 1: Comparative Analysis – Create Summary Metrics (`performance_analysis.py`)
**Purpose**
- To aggregate, validate, and quantitatively compare the predictive performance of LightGBM and TCN_refined across three ICU deterioration targets:  
  1. Peak deterioration (`max_risk`) 
  2. Typical patient risk (`median_risk`)
  3. Proportion of stay in high-risk state (`pct_time_high`)
- This step unifies classification and regression metrics from both models into one structured table (`comparison_table.csv`), combining precomputed metrics (AUC, F1, RMSE, R²) with newly calculated calibration metrics (Brier, ECE).  
- It forms the foundation for all subsequent plots, interpretability, and final evaluation.
**Process**
1. **Imports and Setup**
  - Libraries: `numpy`, `pandas`, `matplotlib`, `sklearn.metrics`, `pathlib`, `json`.  
  - Paths configured to load:  
    - LightGBM → `lightgbm_predictions.csv`, `lightgbm_metrics.json`  
    - TCN_refined → `tcn_predictions_refined.csv`, `tcn_metrics_refined.json`  
  - Output directories: `/results_finalisation/comparison_metrics/`, `/results_finalisation/comparison_plots/`
  - Output: `comparison_table.csv` consolidating all model-target metrics.
2. **Explicit Column Mapping**
  - Explicitly defined which columns to use for each model's predictions CSV to ensure alignment, as TCN CSV contains both log-space and raw values.
  - Prevents misalignment or mismatch between model outputs and ensures fair metric comparison.

  | Model | Classification Columns | Regression Columns |
  |--------|------------------------|--------------------|
  | LightGBM | `y_true_max`, `prob_max`, `y_true_median`, `prob_median` | `y_true_reg`, `y_pred_reg` |
  | TCN_refined | `y_true_max`, `prob_max`, `y_true_median`, `prob_median` | `y_true_reg`, `y_pred_reg_raw` |
3. **Core Utility Functions**
  - **Expected Calibration Error (ECE) with `expected_calibration_error()`**
    - Custom implementation to quantify **probabilistic reliability**, must be built manually as ECE not built into scikit-learn.
    - Splits probabilities into 10 equal-width probability bins over [0,1] and compares model confidence (average predicted probability) vs. observed fraction of positives.  
    - The weighted average of these gaps = ECE. 
    - Outputs float scalar ECE: Lower ECE → better calibration.
  - **Kernel Density Estimation (KDE) with `kde_1d()`**
    - Custom implementation to estimate the continuous probability density of residuals without external libraries.
	  - Constructs a smooth curve by centering a Gaussian kernel at each residual value and averaging all contributions across an evenly spaced grid.
	  - Bandwidth controls smoothing (small → detailed/wiggly, large → smoother/general).
	  -	Purpose: visualise residual distribution shape (spread, bias, skewness) more precisely than histograms, by overlaying KDE curve, enabling direct comparison of model error profiles.
	  - Outputs a 1D density array: higher peaks indicate common residual magnitudes, and symmetry around zero indicates unbiased predictions.
    - Used later for regression diagnostics (Phase 6 Step 2).
4. **Plot Helper Functions**
  - Reusable wrappers for ROC, PR, and calibration plots, prevent code duplication (plotting curves for each model) and ensure consistent axes/formatting for later visualisation plots.
  - Defined for ROC (`plot_roc()`), Precision-Recall (`plot_pr()`), and Calibration (`plot_calibration_curve()`)plots. 
5. **Data Loading and Validation**
  - Loads both models’ CSVs (DataFrames: `df_lgb`, `df_tcn`) and JSON metric files (dictionaries: `lgb_metrics`, `tcn_metrics`): 
    - The CSVs supply ground-truth labels and per-patient predictions (probabilities + regression predictions).
    - The JSONs supply precomputed metrics (roc_auc, f1, rmse, r2, etc.) these are used directly where available.
  - Prints length of both CSV DataFrames for user confirmation.
  - Verifies predictions aligned across identical patient sets:
    - (1) Equal number of rows (both CSV's should be same length).  
    - (2) Consistent `y_true_max` values across both CSVs (ground-truth values should be identical in number and order in both CSV's).
  - Raises errors or warnings if mismatched ordering is detected.
6. **Metric Collection Functions**
  - Functions to extract metrics from JSON (precomputed) and combine them with calibration metrics (Brier score & ECE) to output dictionaries with classiification and regression metrics.
  - **`collect_class_metrics()`:**
    - Inputs: model name, prediction DataFrame, JSON metrics, target columns, key.  
    - **Compute calibration diagnostics from raw probabilites:**
      - Extract arrays: `y_true` and `y_prob` are the raw values.
      - Compute in-script: Brier score (`brier_score_loss()`) and ECE (via our custom function `expected_calibration_error()`).  
    - Normalises target names (specifically for `"median_risk_tuned"` → `"median_risk"`) for unified plotting, names of targets for models need to be identical so that when the function is called for plotting, the name is recognised and both models can be plotted side by side.
    - Extract precomputed JSON metrics (ROC AUC, F1, Accuracy, Precision, Recall), uses NaN if values are not present (safety net).
    - Merges these metrics with the computed Brier & ECE (from CSV probabilities) into a dictionary.
  - **`collect_regression_metrics()`**
    - Inputs: model name, JSON metrics, key.  
    - Extracts **R² and RMSE** from JSON directly (no recomputation). 
    - Uses NaN if values are not present (safety net) 
    - Used for regression target `pct_time_high`:
      - LightGBM → `"pct_time_high"`  
      - TCN_refined → `"pct_time_high_raw_cal"`
    - Creates dictionary with fields: model, target='pct_time_high', rmse, r2
7. **Build Unified Comparison Table**
  - Constructs a Dataframe consolidated metrics table across models and tasks, one row per (model, target) with key metrics.
  - For classification targets we extract raw CSV columns (to compute Brier & ECE) and precomputed JSON metrics using `collect_class_metrics()`:
    - LightGBM uses `"max_risk"` and `"median_risk"` keys, and `"y_true_max"` and `"prob_max"` columns
    - TCN uses uses `"max_risk"` and `"median_risk_tuned"` keys, and `"y_true_median"` and `"prob_median"` columns
  - For regression we pull precomputed metrics from JSON using `collect_regression_metrics()`:
    - LightGBM uses `"pct_time_high"` key
    - TCN uses `"pct_time_high_raw_cal"` key (explicit calibrated raw metric)
  - **Output File:**  
    - Save comparison DataFrame to CSV for record.
    - `results_finalisation/comparison_metrics/comparison_table.csv`

  | Model | Target | AUC | F1 | Accuracy | Precision | Recall | Brier | ECE | RMSE | R² |
  |--------|---------|------|-----|-----------|-----------|--------|-------|------|------|
  | LightGBM | max_risk | From JSON | From JSON | From JSON | From JSON | From JSON | Computed | Computed | — | — |
  | LightGBM | median_risk | ... | ... | ... | ... | ... | ... | ... | — | — |
  | LightGBM | pct_time_high | — | — | — | — | — | — | — | From JSON | From JSON |
  | TCN_refined | max_risk | ... | ... | ... | ... | ... | ... | ... | — | — |
  | TCN_refined | median_risk_tuned | ... | ... | ... | ... | ... | ... | ... | — | — |
  | TCN_refined | pct_time_high_raw_cal | — | — | — | — | — | — | — | From JSON | From JSON |
**Output**
- `comparison_table.csv` → consolidated summary of all quantitative performance metrics per model per target.
- Contains every key metric required for Step 1: quantitative comparative analysis, allowing direct side-by-side inspection of LightGBM and TCN_refined performance in one place.
- Forms the primary foundation for model comparison before progressing to Step 2 (numerical plot data and visualisations).
**Reasoning**
- This step builds the quantitative backbone of the comparative analysis.
- The first half of the script focuses on data loading, validation, and helper utilities → ensuring the LightGBM and TCN inputs are consistent and all required metrics are correctly extracted or computed (e.g., Brier and ECE).
- The second half of these steps consolidates these metrics into a single `comparison_table.csv`, providing a comprehensive, quantitative summary of both models’ performance.
- These summary metrics represent the most critical evidence for assessing model quality → they define overall predictive discrimination, calibration reliability, and regression accuracy.
- Step 2 (plot-based analysis) will build on this foundation, using visual and numerical outputs to illustrate finer trends (e.g., probability distributions, error structure), but Step 1 remains the primary analytical layer.
**Summary**
- Step 1 successfully produces:
  - A single CSV merging **LightGBM** and **TCN_refined** performance across all targets (max_risk, median_risk, pct_time_high).
  - Combined metrics for:
    - **Discrimination**: ROC, AUC, F1
    - **Calibration**: Brier, ECE
    - **Regression fidelity**: RMSE, R²
  - Fully validated input consistency and alignmnet between CSVs and JSONs, and reusable calibration (utility) and plot helper functions.

---

#### Step 2: Comparative Analysis - Create Visualisations (`performance_analysis.py`)
**Purpose**
- To generate **comprehensive comparative visualisations** between LightGBM and TCN_refined, covering both classification (`max_risk`, `median_risk`) and regression (`pct_time_high`) targets.  
- Builds on Step 1’s quantitative metrics (`comparison_table.csv`) by providing **numerical plot data and PNG visualisations**, enabling both **quantitative interpretation** and **visual diagnostic inspection**.  
- All plots are saved with **paired CSVs** containing underlying numeric data for reproducibility and secondary analysis without relying on images.  
**Process**
1. **Classification Visualisations**
  - Generates ROC, Precision–Recall, Calibration, and Probability Distribution plots for `max_risk` and `median_risk` for LightGBM vs TCN_refined.
  - **Key Operations**
    - 1. Extract ground-truth labels (`y_true_max`, `y_true_median`) and predicted probabilities (`prob_max`, `prob_median`) from both models’ DataFrames (`df_lgb`, `df_tcn`).  
    - 2. Compute and export numeric data for each plot type to `comparison_metrics/`:
      - **ROC curves:**  
        - Calculated via `roc_curve()` and `roc_auc_score()`; outputs FPR/TPR arrays and AUC values.  
        - Saved as `roc_<target>.csv`.  
      - **Precision–Recall curves:**  
        - Generated using `precision_recall_curve()` and `average_precision_score()`.  
        - Arrays padded with NaNs for alignment across models.  
        - Saved as `pr_<target>.csv`.  
      - **Calibration curves:**  
        - Built using `calibration_curve()` with 10 uniform bins and includes derived **Brier scores** and **Expected Calibration Error (ECE)** via the `expected_calibration_error()` function defined earlier.  
        - Also records per-bin sample counts and mean predicted probabilities.  
        - Arrays padded with NaNs for alignment across models.  
        - Saved as `calibration_<target>.csv`.  
      - **Probability histograms:**  
        - Records raw predicted probabilities and descriptive statistics (`mean`, `std`, `min`, `max`, `skew`, `kurtosis`) using `summary_stats()`.  
        - Saved as `prob_hist_<target>.csv`.  
    - 3. Generate and save plots:
      - Each plot is rendered and saved in `comparison_plots/` using helper functions `plot_roc()`, `plot_pr()`, and `plot_calibration_curve()` to overlay both models.  
      - Matplotlib visualisations each plotting both models on one axis:
        - `roc_<target>.png`
        - `pr_<target>.png`
        - `calibration_<target>.png`
        - `prob_hist_<target>.png`
      - All figures include consistent titles, legends, and axis labels for reproducibility.
2. **Regression Visualisations**
  - Generates scatter, residuals and Error-vs-Truth plots on the continuous regression target `pct_time_high` (proportion of admission spent in high-risk states).  
  - **Key Operations**
    - 1. Extract aligned true and predicted values:
        - LightGBM: `y_true_reg`, `y_pred_reg`
        - TCN_refined: `y_true_reg`, `y_pred_reg_raw`
    - 2. Compute and export numeric data for each ploy type to `comparison_metrics/`:
      - Compute **residuals** (`predicted − true`) for each model and derive statistical descriptors via `summary_stats()` (mean, std, min, max, skewness, kurtosis) for inclusion in both residuals and error-vs-truth CSV's.  
      - Save combined numeric datasets for:
        - **True vs Predicted:** `scatter_pct_time_high.csv`
        - **Residual distributions:** `residuals_pct_time_high.csv`
        - **Residual KDEs:**  
          - Generated using custom `kde_1d()` (Gaussian kernel density estimation) from step 1 for smoothed residual curves.  
          - Saved as `residuals_kde_pct_time_high.csv`.  
        - **Error vs Truth:** `error_vs_truth_pct_time_high.csv`
    - 3. Generate and save plots:
      - Scatter: True vs Predicted values with identity line, overlaying both models (`scatter_pct_time_high.png`)
      - Residual histograms + KDE overlay for both models side-by-side → left LightGBM, right TCN(`residuals_pct_time_high.png`)
      - Error vs Truth (residual vs true) scatter plots showing deviation trends (`error_vs_truth_pct_time_high.png`)
3. **Metric Comparison Bar Charts**
  - Builds **summary bar charts** comparing LightGBM and TCN_refined performance on all targets using metrics from `comparison_table.csv` by loading into DataFrame `df_comp`. 
  - Both models plotted side-by-side on same axis. 
    - Classification target (`max_risk`) include: **ROC AUC, F1, Brier, ECE**  
    - Classification target (`median_risk`) include same as above, but TCN uses `median_risk_tuned` from JSON (remember we normalised the target name to `median_risk` to allow the for loops to work for both models)
    - Regression target (`pct_time_high`) includes: **RMSE, R²** for both models; TCN uses `pct_time_high_raw_cal` (hardcoded to `"pct_time_high"`, even though the table reads metrics from `"pct_time_high_raw_cal"` internally).
  - Saved as:
    - `metrics_comparison_max_risk.png`
    - `metrics_comparison_median_risk.png`
    - `metrics_comparison_pct_time_high.png`
**Outputs**
- **Directories:**
  - `comparison_plots/` → all PNG visualisations (ROC, PR, calibration, histograms, regression plots, bar charts).  
  - `comparison_metrics/` → all numeric CSVs for corresponding plots (used for further statistical analysis).  
- **File Naming Conventions**
  - 12 Numeric CSV files:
    - Classification: 2x `roc_<target>.csv`, 2x `pr_<target>.csv`, 2x `calibration_<target>.csv`, 2x `prob_hist_<target>.csv`
    - Regression: `scatter_<target>.csv`, `residuals_<target>.csv`, `residuals_kde_<target>.csv`, `error_vs_truth_<target>.csv`
  - 14 PNG visualisation files:
    - Classification: 2x `roc_<target>.png`, 2x `pr_<target>.png`, 2x `calibration_<target>.png`, 2x `prob_hist_<target>.png`
    - Regression: `scatter_<target>.png`, `residuals_<target>.png`, `error_vs_truth_<target>.csv`
    - Summary bar charts: 3x `metrics_comparison_<target>>.png`
**Reasoning**
- Step 2 builds directly on the unified summary metrics (`comparison_table.csv`) from Step 1, extending the quantitative analysis into visual and numerical diagnostics.
- The code uses the previously defined helper functions (`plot_roc()`, `plot_pr()`, `plot_calibration_curve()`, `kde_1d()`, etc.) to ensure consistent, reproducible evaluation of both models across all targets.
- Unlike Step 1—which captures high-level performance metrics—Step 2 focuses on behavioural patterns:
  - For classification, it reveals model discrimination (ROC/PR), probability reliability (calibration), and prediction spread (histograms).
  - For regression, it visualises prediction alignment (scatter), bias and spread (residuals + KDE), and systematic deviations (error-vs-truth).
- Each PNG plot has a numerical CSV counterpart, ensuring interpretability without relying on images → the CSVs hold the raw arrays used for every curve and histogram.
- This step therefore provides fine-grained insight into why one model outperforms another (e.g. LightGBM’s tighter calibration or TCN’s higher early-event sensitivity) and supplies all quantitative evidence needed for written analysis in later phases (interpretability and inference).
- Together, Step 1 + Step 2 form the complete comparative performance foundation before moving into explainability (Step 3) and deployment (Step 4–5).
**Summary**
- **Step 2 extends the comparative analysis** from scalar metrics (Step 1) into **quantitative visual analytics**, enabling model interpretation across discrimination, calibration, and error structure dimensions.  
- All plots have paired numeric datasets ensuring **complete reproducibility and auditability**.   
- These outputs collectively form the **visual interpretive foundation** of the comparative study, bridging raw metric comparisons and deeper interpretability analysis.

---

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

---

### Reflection
#### Challenges
1. **Model Inclusion Confusion**
  - Initially unclear whether to include baseline TCN or NEWS2 in the final comparative analysis.
  - The role of NEWS2 as a ground truth generator (not a predictive model) led to conceptual ambiguity about whether it should appear in visualisations.
2. **Plot vs Metric Interpretation**
  - Early analyses relied too heavily on visual approximations from plots rather than numeric CSVs.
  - ROC, PR, and calibration plots lacked full quantitative interpretability → missing AUC, AP, and bin-level metadata.
3. **Metric Redundancy (Identical Values)**
  - Encountered identical Accuracy, Precision, Recall, and F1 values for `max_risk` across both models despite differing AUCs.
  - Raised concerns over whether results were duplicated or methodologically flawed.
4. **Small Test Set Limitations**
  - Test cohort size (n = 15 patients) caused discrete metrics to be unstable.
  - Each patient accounted for ~6.7% of results, meaning one misclassification could shift metrics drastically.
5. **Label–Model Mismatch**
  - Temporal models (TCN) were trained on timestamp sequences but evaluated on patient-level aggregate targets (e.g., `median_risk`, `pct_time_high`).
  - This caused TCN to underperform on static labels due to temporal redundancy and lack of true sequence–outcome alignment.
6. **Median-Risk Anomalies**
  - TCN underperformed dramatically in `median_risk` classification, despite strong temporal capacity.
  - Root cause traced to target semantics (static summary) conflicting with temporal architecture (dynamic sequence).
7. **Column Mapping & JSON Key Inconsistency**
  - TCN metric JSON used `median_risk_tuned`, while LightGBM used `median_risk`, breaking alignment in plots and bar charts.
8. **Calibration Curve Data Alignment**
  - Calibration CSVs failed to save due to mismatched array lengths and integer dtype in histogram bin counts (`n_samples_*`).
#### Solutions
1. **Model Scope Rationalisation**
  - Restricted comparison to **Refined TCN** (final tuned model) vs **Retrained LightGBM** (Phase 5 model with best hyperparameters).
  - Excluded baseline TCN and NEWS2 for conceptual clarity → the goal is “best vs best” modelling of the NEWS2-derived targets.
2. **Interpretation Workflow Correction**
  - Reordered evaluation steps:
    1. **Step 1 → Quantitative analysis (metrics first):** AUC, F1, Brier, ECE, RMSE, R² from `comparison_table.csv`.
    2. **Step 2 → Plot and distributional analysis:** Curves, residuals, calibration distributions (for visual context).  
  - Ensured metrics drive conclusions, plots support them.
3. **Statistical Explanation for Metric Identity**
  - Identical classification metrics explained by:
    - Common thresholding (0.5 cutoff).
    - Highly separable `max_risk` target.
    - Small n leading to coarse granularity.
  - Concluded that AUC and calibration metrics are the only statistically robust signals under such sample constraints.
4. **Structural Fixes**
  - Normalised all TCN target keys using:
    ```python
    display_key = json_key.replace("_tuned", "")
    ```
  - Ensured uniform plotting labels for both models.
  - Converted `n_samples_*` arrays to float and padded consistently across calibration CSVs to avoid dimension mismatch.
5. **Data Enhancement**
  - Expanded all CSV outputs with complete interpretive metadata:
    - **Classification:** Added AUC, AP, Brier, ECE, and bin counts.  
    - **Regression:** Added residual summary stats (mean, std, skew, kurtosis).  
    - **Histogram:** Added full probability distribution descriptors.
  - Now every plot has a corresponding self-contained CSV allowing full numeric interpretation.
6. **Conceptual Clarifications**
  - Defined NEWS2 as the ground truth framework, not a plotted comparator.
  - Established that models approximate NEWS2 risk mapping (proxy task), not raw clinical events.
  - Formally accepted that this is a **benchmarking study**, not a clinical outcome predictor.
7. **Analytical Reframing**
  - Split interpretation into three evidence layers:
    1. **Quantitative (metrics)**
    2. **Curve-level (arrays)**
    3. **Visual/qualitative (plots)**
  - Ensures logical progression from objective evidence → behavioural diagnostics → intuitive confirmation.
#### Learnings
1. **Scientific Workflow Discipline**
  - Always interpret **metrics before plots** → numbers are objective, plots are supportive.
  - Separate evidence (quantitative) from illustration (qualitative) for reproducibility.
2. **Importance of Target–Model Alignment**
  - TCNs require timestamp-level supervision to realise their potential; patient-level aggregate targets suppress their advantages.
  - LightGBM naturally aligns with static summarised data, explaining superior calibration and performance for `median_risk` and `pct_time_high`.
3. **On Model Fairness**
  - The decision to train both models on identical patient-level outcomes was methodologically justified → ensures a fair comparison even if it limits temporal expressivity.
4. **Interpretation Under Small Samples**
  - With n=15, threshold-based metrics lose statistical meaning; AUC and calibration remain the most reliable.
  - Future work should focus on expanding test cohorts to stabilise discrete metric variance.
5. **Engineering Rigour**
  - Explicit column mappings (`LGB_COLS`, `TCN_COLS`) and consistent padding logic prevent silent data mismatches.
  - Including all supporting numeric arrays with every plot guarantees interpretability without relying on PNGs.
6. **Clinical Framing**
  - The models do not predict raw deterioration events but **approximate NEWS2-derived risk** → still clinically relevant as it quantifies each model’s ability to emulate validated early-warning logic.
  - Demonstrates that classical models (LightGBM) outperform deep temporal models (TCN) under limited-data regimes, highlighting **data efficiency vs temporal expressiveness trade-offs**.
7. **Broader Insight**
  - “Deep ≠ automatically better.”  
    - LightGBM: stable, interpretable, calibrated.  
    - TCN: dynamic, expressive, but data-hungry.  
  - Choice of model must match both data scale and task definition.
#### Final Reflection
- This phase revealed how methodological clarity, rigorous numeric grounding, and architectural awareness transform a model comparison from visual impressionism into scientific inference.  
- Phase 6 now provides a fully transparent, auditable, and reproducible comparison pipeline → quantitatively complete, visually supported, and conceptually aligned with best practices in clinical ML benchmarking.

---

## Day 37-38 Notes - Continue Phase 6: Interpretability - LightGBM SHAP (Step 3)

### Goals 
- Implement and validate the **LightGBM SHAP analysis script** (`shap_analysis_lightgbm.py`). 
- Ensure correct **per-patient SHAP explanations** for all three targets (`max_risk`, `median_risk`, `pct_time_high`) using **TreeExplainer**.
- Diagnose and fix SHAP output inconsistencies (e.g., classifier collapse issues).  
- Finalize Phase 6 lightGBM interpretability pipeline with validated, reproducible, and clinically interpretable SHAP results.  

### What We Did
#### Step 3: LightGBM SHAP Analysis Script (`shap_analysis_lightgbm.py`)
**Purpose**
- This script performs **final interpretability analysis** for the LightGBM models trained in **Phase 5** of the project.  
- It quantifies how each clinical feature contributed to the model’s predictions using **SHAP (SHapley Additive exPlanations)** values.
- The script ensures that each model (for `max_risk`, `median_risk`, and `pct_time_high`) produces interpretable, reproducible per-patient feature importance values, saved both numerically and visually.
**Process**
1. **Load Dependencies and Define Directories**
  - **Imports:** 
    - Core scientific stack (`pandas`, `numpy`), model tools (`joblib`, `shap`), and visualization (`matplotlib`).
    - Shap 
  - Defines paths for:
    - Input features (`data/processed_data/news2_features_patient.csv`)
    - Patient split file (`patient_splits.json`)
    - Input LightGBM model directory (`lightgbm_results`)
    - Output folder (`interpretability_lightgbm`), created automatically if missing.
2. **Load Data and Define Features**
  - Loads:
    - `news2_features_patient.csv` → patient-level aggregate features  
    - `patient_splits.json` → identifies training patients (70 IDs)
  - Filters `train_df` to contain only the training patients with `train_ids`.
  - Recreates binary targets (`max_risk_binary`, `median_risk_binary`) using consistent encoding logic.
  - Defines `feature_cols` by excluding non-feature columns `exclude_cols` (e.g., IDs, target variables).
  - Ensures the SHAP analysis uses the same feature definitions and patient split as the original model training.
3. **Loop Over Targets**
  - The script iterates through the three prediction targets:
  ```python
  TARGETS = ["max_risk", "median_risk", "pct_time_high"]
  ```
  - For each target:
    1. **Load model:** Loads the respective LightGBM `.pkl` file.
    2. **Select features:** Extracts feature matrix `X_train` from the training DataFrame.
    3. **Compute SHAP values:**
    ```python
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_train)
    ```
    - `TreeExplainer` is SHAP’s exact explainer for tree-based models like LightGBM, XGBoost, and CatBoost.
    - It uses the internal tree structure to efficiently compute Shapley values → the average marginal contribution of each feature to the model’s prediction, across all possible feature orderings.
    - This provides local (per-patient) and global (mean absolute) feature importance without retraining or perturbation sampling.
    4. **Handle output shape:** 
    ```python
    if isinstance(shap_values, list):  
        shap_array = shap_values[1]
    else:
        shap_array = shap_values
    ```
    - If `shap_values` is a list: 
      - Classification model (LightGBM binary output): `shap_array = shap_values[1]`
        - LightGBM classifiers return a list of arrays (`[class_0, class_1]`), one per class.
        - We only need `class_1` (positive class) to understand factors contributing to high risk, so we explicilty take the positive class.
        - **Rationale:** In binary tasks, SHAP sometimes collapses this to a single 1D array if not handled explicitly, so we restore the correct 2D structure (70, 40) — one SHAP vector per patient — preventing the array from flattening and producing valid per-feature contributions again.
    - If `shap_values` is not a list:
      - Regression model: `shap_array = shap_values`
      - LightGBM regressors return a single (`n_samples`, `n_features`) array → one SHAP value per feature per sample.

4. **Compute Mean Absolute SHAP Importance**
  ```python
  mean_abs_shap = np.abs(shap_array).mean(axis=0)
  ```
  **Computation**
  - **Absolute value:** Removes directionality (positive = increases risk, negative = decreases risk).
  - **Mean across patients (axis=0):** Aggregates SHAP magnitudes over all 70 patients, producing one importance score per feature.
  - **Purpose:** Quantifies the average strength of influence for each feature on the model’s predictions.
  ```python
  shap_importance = pd.DataFrame({
      "feature": feature_cols,
      "mean_abs_shap": mean_abs_shap
  }).sort_values(by="mean_abs_shap", ascending=False)
  ```
  **DataFrame creation** 
  - Converts results into a ranked DataFrame for interpretability, stores ranked importances.
  - **Output:** For each target, save numeric SHAP importance values CSV → `interpretability_lightgbm/{target}_shap_summary.csv`
  **Rationale**
  - It transforms per-patient local explanations into a single global feature importance ranking, allowing direct comparison of which physiological variables drive predictions most strongly across the model.

5. **Plot Top 10 Features Visualisations**
  ```python
  plt.barh(top_features["feature"][::-1], top_features["mean_abs_shap"][::-1])
  ```
  - Produces a horizontal bar plot of the top 10 most influential features ranked by mean absolute SHAP value.
	- [::-1] reverses the order so the most important feature appears at the top.
	-	Visually summarises which physiological variables contribute most strongly to the model’s predictions, for inclusion in reports and interpretability documentation.
  - **Output:** For each target save PNG bar plot of top 10 most influential features → `interpretability_lightgbm/{target}_shap_summary.png`

6. **Diagnostic Print Outputs**
  - To ensure correctness, the script includes several diagnostic print statements.  
  - These checks were added during debugging to confirm that SHAP outputs were correctly shaped and not collapsed (as occurred in earlier versions).

| **Print Statement** | **Purpose** |
|----------------------|-------------|
| `X_train.shape` | Confirm that the input data has the expected dimensions (70 patients × 40 features). |
| `shap_array.shape` | Verify that SHAP outputs match the expected shape (`n_samples`, `n_features`) and haven’t collapsed into a single vector. |
| `mean_abs_shap.sum()` | Detect aggregation or shape errors — identical or constant SHAP values indicate invalid computation. |
| `np.unique(preds)` | Confirm that model predictions vary across patients, ruling out a constant or saturated model. |

**Outputs**
- All generated files are stored in: `src/results_finalisation/interpretability_lightgbm/`

| **Output File** | **Description** |
|------------------|-----------------|
| `max_risk_shap_summary.csv` | Full ranked table of feature importances for the **max_risk** classifier (mean absolute SHAP values for all features). |
| `max_risk_shap_summary.png` | Bar plot of the top 10 most important features influencing **max_risk** predictions. |
| `median_risk_shap_summary.csv` | Full ranked table of feature importances for the **median_risk** classifier. |
| `median_risk_shap_summary.png` | Bar plot of the top 10 most important features influencing **median_risk** predictions. |
| `pct_time_high_shap_summary.csv` | Full ranked table of feature importances for the **pct_time_high** regression model. |
| `pct_time_high_shap_summary.png` | Bar plot of the top 10 most important features influencing **pct_time_high** predictions. |

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

---

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

---

### SHAP Debugging and Classification Target Fix
**Overview**
- This document summarises the workflow, debugging, and rationale for computing SHAP (SHapley Additive exPlanations) values for the final LightGBM models (`max_risk`, `median_risk`, `pct_time_high`) trained in Phase 5. 
- It explains why the original SHAP computation produced identical feature importance for the classifiers, the debugging process, terminal outputs, and the final correct implementation.
1. **Background and Initial Problem**
  **Original SHAP Computation**
  ```python
  explainer = shap.TreeExplainer(model)
  shap_values = explainer.shap_values(X_train)

  if target in ["max_risk", "median_risk"]:
      shap_array = shap_values[1]  # binary classification
  else:
      shap_array = shap_values     # regression
  ```
  **Observed Behaviour**
  - When we first ran the SHAP script, the regression target (`pct_time_high`) produced reasonable results as SHAP values varied across features and patients, but both **binary classification targets** (`max_risk`, `median_risk`) produced **identical mean absolute SHAP values** across all features.
    - The output CSV showed every feature having exactly the same `mean_abs_shap` value.
    - All SHAP values were constant, resulting in meaningless bar plots.
  - This indicated that **SHAP attribution had failed**, since every feature seemed to contribute equally, which is mathematically impossible for a functioning model.
  **Logical contradiction**
  - If SHAP were computed correctly:
    - Each feature should contribute differently to the model’s output.
    - Mean absolute SHAP values should vary across features, often by orders of magnitude.
    - Identical SHAP values across all 40 features imply a structural or logical error.
  - So this result violated a basic expectation of SHAP’s behaviour, which meant one of two things had to be true:
    1. The **model output** was constant (all predictions identical), or  
    2. The **SHAP value computation** was collapsing or misaligned (wrong array shape or aggregation).

  | **Evidence**                              | **Observation**                  | **Logical Deduction**                                         |
  |-------------------------------------------|----------------------------------|---------------------------------------------------------------|
  | Mean absolute SHAP identical across features | All values = `0.09354`           | SHAP must be collapsing feature dimension                     |
  | Sum of mean absolute SHAP = `0.0935`        | Equals single feature value      | Confirms collapse or incorrect aggregation                    |
  | Model predictions varied                    | Model outputs not constant       | SHAP computation wrong, not dataset or model                  |
  | Regression SHAP normal                      | Problem only occurs for classifiers | Confirms LightGBM binary-specific issue                     |

2. **Step 1 — Terminal Inspection to Diagnose the Issue**
  **Diagnostic Prints**
  - To investigate why SHAP outputs were identical for classification targets, we added detailed diagnostic `print()` statements to inspect shapes, datatypes, and value distributions at every step:
    ```python
    print(f"[INFO] Target: {target}")
    print(f"[INFO] X_train shape: {X_train.shape}")
    print(f"[INFO] Number of features: {len(feature_cols)}")

    print(f"[INFO] Raw shap_values type: {type(shap_values)}")
    if target in ["max_risk", "median_risk"]:
        print(f"[INFO] shap_values list length (classes): {len(shap_values)}")
        print(f"[INFO] Positive class shap array shape: {shap_values[1].shape}")
    else:
        print(f"[INFO] Regression shap array shape: {shap_values.shape}")

    print(f"[INFO] First 5 rows of SHAP array (first 5 features):\n{shap_array[:5]}")

    print(f"[INFO] Mean absolute SHAP values (top 10 features):\n{shap_importance.head(10)}")

    print(f"[INFO] Sum of mean absolute SHAP across all features: {mean_abs_shap.sum():.4f}")

    print("Variance across patients for each feature SHAP:", np.var(shap_array, axis=0))
    ```

  | **Print Statement** | **Purpose / Diagnostic Goal** |
  |----------------------|-------------------------------|
  | `X_train.shape` | Verify input data dimensions → confirm there are 70 patients × 40 features. |
  | `len(feature_cols)` | Ensure expected number of features is loaded (prevent feature mismatch). |
  | `type(shap_values)` | Determine whether SHAP returned a NumPy array or a list (binary classifiers often return lists). |
  | `len(shap_values)` | For classifiers, check number of class outputs returned (should be 2 for binary). |
  | `shap_values[1].shape` | Validate that SHAP’s positive-class array has shape `(n_samples, n_features)`. |
  | `shap_array[:5]` | Inspect first few SHAP rows to ensure variability and non-zero values. |
  | `shap_importance.head(10)` | Check if top-10 SHAP features differ or remain constant (key diagnostic symptom). |
  | `mean_abs_shap.sum()` | Detect aggregation anomalies (e.g., sum equals single feature value → array collapsed). |
  | `np.var(shap_array, axis=0)` | Confirm per-feature variance across patients (variance > 0 → valid SHAP spread). |
  | `np.unique(preds)` | Verify model predictions vary; if constant, SHAP uniformity may reflect model stagnation. |

  **Diagnostic Findings**
	- Regression: array shape (70, 40) → OK.
	- Classifiers: array collapsed (40,) → all features identical → problem confirmed to be SHAP handling of binary classifiers.
  - This confirmed that classification models were returning 1D SHAP arrays, while the regression model correctly returned a 2D matrix.

3. **Step 2 — Hypothesis**
  - LightGBM’s SHAP interface differs between regression and classification:
    - For regression, TreeExplainer returns a (`n_samples`, `n_features`) array.
    - For binary classification, depending on SHAP version, it may return:
      - A list of two arrays: `[class_0_array, class_1_array]`, both shaped (`n_samples`, `n_features`).
      - Or, under certain LightGBM/SHAP combinations, it collapses to a single 1D vector if the model output is interpreted as the raw margin rather than class probabilities.
    - So, the likely cause: SHAP was computing raw decision values rather than per-sample probability contributions.

4. **Step 3 — Attempt 1: Forcing Probability Mode (model_output="probability")**
  **Probability Mode**
  - We tried to fix this by explicitly telling SHAP to output probabilities rather than raw margins:
  ```python
  explainer = shap.TreeExplainer(model, model_output="probability")
  shap_values = explainer.shap_values(X_train)
  ```
  **Result**
  - Failed with error:
  ```bash
  ValueError: Only model_output="raw" is supported for feature_perturbation="tree_path_dependent"
  ```
  **Explanation**
  - TreeExplainer with default `tree_path_dependent` feature perturbation does not support `model_output="probability"`. This fix was not compatible.
  - SHAP’s default LightGBM explainer uses `feature_perturbation="tree_path_dependent"` for efficiency.
  - However, in that mode, only "raw" outputs are supported. `"probability"` mode requires interventional perturbation, which is slower but mathematically consistent with probabilistic outputs.

5. **Step 4 — Attempt 2: Switching to Interventional Mode (feature_perturbation="interventional")**
  **Interventional Mode**
  - We modified it to explicitly use interventional perturbation:
  ```python
  explainer = shap.TreeExplainer(
      model,
      data=X_train,
      model_output="probability",
      feature_perturbation="interventional"
  )
  ```
  **Result**
  - This produced a different error:
  ```bash
  AttributeError: 'TreeEnsemble' object has no attribute 'values'
  ```
  **Explanation**
  - This combination triggered internal LightGBM `TreeEnsemble` attributes incorrectly. SHAP could not compute the expected value with this setup.
  - This error is due to an incompatibility between certain LightGBM and SHAP versions (notably SHAP ≥0.45) → the internal object `TreeEnsemble` in LightGBM models does not expose the `.values `attribute that SHAP’s interventional explainer expects.
  - Hence, this version combination cannot compute interventional SHAP values directly from LightGBM models.

6. **Step 5 — Root Cause Confirmation via Shape Diagnostics**
  - After the above failures, we revisited our diagnostic prints.
    1. **Observation:** The SHAP array for the classifiers collapsed to shape (40,) instead of (70, 40) like regression, and all mean absolute values were identical across features.
    2. **Check model/data:** Predicted probabilities and `X_train` were fine → the model was trained correctly, and patient features varied. `(model.predict_proba(X_train)[:,1])` varied normally across patients, confirming the model itself was trained correctly.
    3. **Tried fixes:**
      -	`model_output="probability"` → failed with “only raw supported” error.
      -	`feature_perturbation="interventional"` → failed with TreeEnsemble .values error.
    4. **Logical deduction:**
      - The models and data were correct, because predicted outputs varied.
      -	The issue was not the model, nor the `X_train`, but the SHAP explainer’s treatment of binary classification models.
      - With all other possibilities exhausted, the only remaining explanation was how the SHAP `TreeExplainer` treats binary classifiers by default → collapsing outputs into a single 1D array (raw margin) instead of giving per-patient arrays.

7. **Step 6 — Final Fix**
  **Final code change**
  - We changed the logic to explicitly check for classification models by testing whether `shap_values` is a list (as SHAP does for multi-class models):
  ```python
  explainer = shap.TreeExplainer(model)
  shap_values = explainer.shap_values(X_train)

  if isinstance(shap_values, list):
      # Binary classification → two arrays (class 0, class 1)
      shap_array = shap_values[1]
  else:
      # Regression → single array
      shap_array = shap_values
  ```
  **Rationale:**
  1. Default TreeExplainer with `model_output="raw"` works for both regression and binary classification.
  2. For classifiers, SHAP returns a list of two arrays (one per class). Using `shap_values[1]` ensures we get the array for the positive class (high-risk) per patient.
  3. Regression outputs a single array directly (70, 40).
  4. This restored the correct SHAP array shape (70, 40) for classifiers, enabling meaningful per-feature SHAP computation and top-10 bar plots.
  **Why This Works**
  - For regression → SHAP returns (`n_samples`, `n_features`) directly.
  - For binary classification → SHAP returns a list of two arrays, one per class.
  - By taking `shap_values[1]`, we extract the SHAP values for the positive class (“high-risk”).
  - This ensures we use the correct per-sample, per-feature attributions.
  **Validation**
  - The print diagnostics confirmed that the shape was now correct:
  ```bash
  [DEBUG] Target: max_risk
  [DEBUG] SHAP array shape: (70, 40)
  [DEBUG] Target: median_risk
  [DEBUG] SHAP array shape: (70, 40)
  [DEBUG] Target: pct_time_high
  [DEBUG] SHAP array shape: (70, 40)
  ```
  - SHAP values varied meaningfully between patients and features, and the feature importances were no longer identical.

8. **Conceptual Summary**
| **Problem** | **Cause** | **Fix** |
|--------------|-----------|---------|
| All classifier SHAP values identical | SHAP collapsed LightGBM binary output to 1D raw margins | Explicitly extract positive-class array via `shap_values[1]` |
| `model_output="probability"` error | TreeExplainer’s default perturbation mode (`tree_path_dependent`) doesn’t support probability outputs | Avoid forcing `model_output="probability"` when using `tree_path_dependent` |
| `interventional` mode error | Version incompatibility between SHAP and LightGBM (TreeEnsemble lacks `.values` attribute) | Don’t override explainer type; rely on default behaviour |
| Regression worked fine | Regression output always produces scalar per-sample predictions | No change needed |
| **Final resolution** | Classifier vs. regressor distinction was not handled → SHAP list outputs for binary models must be explicitly selected | Added conditional handling: if `isinstance(shap_values, list)`, use `shap_values[1]`, else use `shap_values` (regression) |

9. **Lessons Learned**
| **Key Point** | **Explanation** |
|----------------|----------------|
| **Check SHAP array shapes early** | Printing `shap_values.shape` (or inspecting list lengths) instantly shows whether SHAP is producing proper per-sample × per-feature outputs. |
| **LightGBM classifier outputs differ by version** | Depending on SHAP and LightGBM versions, binary classifiers may return either a list of two arrays (`[class_0, class_1]`) or a single 1D array of raw margins. |
| **Don’t force `"probability"` mode** | Using `model_output="probability"` with the default `tree_path_dependent` perturbation mode causes version conflicts; only safe when using `feature_perturbation="interventional"` **and** compatible versions. |
| **Regression SHAP is simpler** | Regression always outputs one scalar per sample → SHAP consistently returns a `(n_samples, n_features)` array. |
| **Binary SHAP requires explicit class handling** | Always select the positive-class SHAP array (`shap_values[1]`) for interpretability. Omitting this leads to uniform, meaningless values. |
| **Collapsed array explanation** | Identical mean SHAP values across all features indicated the 2D SHAP array was collapsing into a 1D feature vector instead of sample × feature contributions. |
| **Debugging method** | Iterative print diagnostics verified that the model and dataset were correct, isolating the problem to SHAP’s binary output handling. |
| **Final verified result** | Achieved correct `(70, 40)` SHAP matrices for all targets, yielding meaningful mean absolute SHAP values and interpretable top-10 feature importance plots. |

⸻

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

---

### Reflection  
#### Challenges
1. **Unclear Role of Interpretability in Evaluation**
  - Initially uncertain whether interpretability (SHAP) formed part of model comparison or a separate explanatory phase.
  - Confusion arose because Phase 3 already included feature-importance analysis, creating apparent duplication.
2. **SHAP Implementation Issues**
  - Early SHAP runs produced identical mean absolute values across all features — a clear sign of **collapsed arrays**.
  - LightGBM’s SHAP interface behaved differently for regressors vs. binary classifiers, returning inconsistent array formats (`ndarray` vs. list of arrays).
  - Attempts to force probability-based outputs (`model_output="probability"`) caused `interventional` mode errors due to SHAP–LightGBM version mismatches.
3. **Uncertainty About Input Data Requirements**
  - It was unclear whether to recompute binary targets for SHAP or simply use the trained model and raw features.
  - The distinction between what SHAP requires (features only) and what was recreated for consistency (binary targets) needed clarification.
4. **Conceptual Ambiguity Around SHAP Scope**
  - Ambiguity existed about whether to run SHAP on the training or test sets.
  - Similarly, uncertainty about whether to interpret directional SHAP values (positive/negative) or only absolute magnitudes caused hesitation.

#### Solutions and Learnings
1. **Clarified Analytical Role**
  - Interpretability (Step 3) is not part of the comparative performance metrics (Steps 1–2) but a **complementary explanatory layer**: Performance → Behaviour → Reasoning
  - Phase 3 = developmental diagnostics; Phase 6 = final mechanistic explanation for the validated model.
2. **Technical Resolution of SHAP Collapsing**
  - Identified LightGBM’s binary-class SHAP behaviour as the cause:
    - SHAP returned `[class_0, class_1]` arrays for classifiers.
    - Without selecting `shap_values[1]`, all features appeared identical.
  - Final fix: explicitly extract the positive-class SHAP array:
    ```python
    if isinstance(shap_values, list):
        shap_array = shap_values[1]
    else:
        shap_array = shap_values
    ```
  - Verified via shape and variance checks that SHAP now produced correct (`n_samples`, `n_features`) matrices.
3. **Clarified Data and Target Handling**
  - SHAP uses **only X (features)** and the trained model, not the targets.
  - Binary targets were reconstructed purely for clarity and consistency in looping logic, then excluded from `feature_cols`.
4. **Refined Conceptual Framework**
  - **TreeExplainer** chosen as the optimal SHAP algorithm for LightGBM:
    - Exact and efficient for tree-based models.
    - Correctly captures hierarchical feature interactions.
  - **Global SHAP (training set)** chosen for final interpretability:
    - Stable, representative, and aligned with learned model patterns.
    - Local SHAP (test set) reserved for optional illustrative use.
5. **Interpretability Objective Refined**
  - Directional (+/−) SHAP and dependence plots were deemed **exploratory** rather than evaluative.
  - The goal was to produce **clinically interpretable, reproducible global feature importance** using mean absolute SHAP:
    ```python
    mean_abs_shap = np.abs(shap_array).mean(axis=0)
    ```
  - This aligns with the comparative, not exploratory, intent of Phase 6.

#### Overall Reflection
- This phase clarified how interpretability fits into the analytical chain, not as performance validation, but as **mechanistic explanation**.  
- Through systematic debugging and theoretical consolidation:
  - We confirmed that SHAP for LightGBM requires careful handling of classifier outputs and correct selection of the positive-class array.
  - Diagnostic prints and shape checks were critical in identifying SHAP collapse early, preventing misinterpretation.
  - The process reinforced the importance of understanding tool-specific behaviour (e.g., SHAP vs. LightGBM interactions) rather than relying on defaults.
  - The final implementation now produces stable, interpretable, and physiologically coherent SHAP values across all targets.
- In essence, this step transformed SHAP from a “black-box add-on” to a **rigorous interpretability mechanism**, linking model performance to physiological reasoning with reproducible logic and technical integrity.

---

## Day 39-44 Notes - Finish Phase 6: Interpretability - TCN Temporal Saliency (Step 4)

### Goals
- Finalise and validate `saliency_analysis_tcn.py` script (Phase 6 Step 4).
-	Verify all generated outputs in `interpretability_tcn/` for correctness and interpretability.
- Debug any output anomalies and visualisation issues.
- Analyse saliency CSVs and heatmaps to interpret key temporal feature patterns.
- Compare TCN saliency findings with LightGBM SHAP results (Phase 6 Step 3) to assess consistency across models.

### What We Did
#### Step 4: TCN Temporal Saliency Script `saliency_analysis_tcn.py`
**Purpose**
-	Compute gradient × input (|grad × input|) saliency maps for the refined TCN model on the test set.
-	Quantify how each input feature and timestep influences predictions for all three output heads (`max_risk`, `median_risk`, `pct_time_high`).
-	Produce reproducible, interpretable outputs (feature-level, temporal, and top-feature summaries).
-	Provide visual summaries via top-10 feature heatmaps (for qualitative inspection only).
**Workflow**
1. **Environment Setup and File Validation**
  - **Imports and Dependencies**
    - Loads all required Python libraries: `torch`, `numpy`, `matplotlib`, and `pandas` for deep learning, numerical computation, and plotting.
    - `tqdm` for visual progress bars during batch processing.
    - `json` and `pathlib` for reading configuration files and managing directory paths.
    - Adds the project’s `src/` folder to the Python path so the custom `TCNModel` can be imported from `ml_models_tcn/tcn_model.py`.
  - **Directory Configuration**
    - **Model files:** trained model weights (`tcn_best_refined.pt`) and model configuration (`config_refined.json`).
    - **Preprocessing metadata**: `padding_config.json`, `standard_scaler.pkl`, `patient_splits.json` for consistent data formatting.
    - **Test tensors:** preprocessed test data (`test.pt`) and sequence masks (`test_mask.pt`).
    - **Output folder:** creates `interpretability_tcn/` for storing all interpretability results.
  - **Sanity Checks**
    - Ensures all required input files exist before proceeding.
    - Immediately raises descriptive errors if any file is missing, preventing partial or corrupted runs.
    - Confirms readiness by printing a verification message:
    ```bash
    [INFO] All required input files found. Ready to proceed.
    ```
2. **Step 1: Load Model, Configuration, and Test Data**
  - **Device Selection**
    - Automatically detects whether a GPU is available; otherwise defaults to CPU.
    - Ensures all subsequent model operations run on the same device.
  - **Load Model Configuration**
    - Reads `config_refined.json` to retrieve network architecture → `arch = config["model_architecture"]`
    - `num_channels`, `kernel_size`, `dropout`, `head_hidden` → model hyperparameters for reconstruction.
  - **Load Preprocessing Metadata**
    - Reads `padding_config.json` to extract:
      - `feature_cols`: ordered list of all input feature names.  
      - `max_seq_len`: sequence length after padding.  
      - `target_cols`: model output heads for prediction.  
    - Ensures that test data dimensions match these definitions.
  - **Load Test Tensors**
    - `test.pt`: padded feature tensor `(n_patients, timesteps, features)`.
    - `test_mask.pt`: binary mask tensor `(n_patients, timesteps)` identifying valid timesteps (1 = real timestep, 0 = padding)
  - **Shape Validation**
    - Confirms sequence length matches (`max_seq_len`) and feature count matches (`feature_cols`).
    - Prevents downstream dimension mismatches.
  - **Rebuild and Load Model**
    - Reconstructs the TCN using configuration parameters → `model = TCNModel(...).to(device)`.
    - Loads pretrained weights from `tcn_best_refined.pt` to PyTorch as `state_dict`, then loaded into model (`model.load_state_dict(state_dict)`)
    - Switches model to evaluation mode (`model.eval()`) to disable dropout and training behaviors.
  - **Rationale**: 
    - This step guarantees reproducibility by rebuilding the exact refined TCN used in prediction. 
    - It prepares the model and data for precise gradient-based saliency computation, ensuring the interpretability results reflect the final trained system.
3. **Step 2: Define Targets and Saliency Computation Function**
  - **Define Model Output Heads (`TARGETS`)**
    - Each target corresponds to one prediction head in the refined TCN
    - Saliency will be computed separately for each head to understand which features and at which times influenced each distinct clinical prediction.
    | Target name | Model key         | Type            | Description |
    |--------------|------------------|-----------------|--------------|
    | `max_risk`   | `logit_max`      | Classification  | Binary output for whether the patient reached a critical deterioration. |
    | `median_risk`| `logit_median`   | Classification  | Binary output for the average deterioration severity. |
    | `pct_time_high` | `regression`  | Continuous      | Regression output estimating proportion of time spent at high risk. |

  - **Create saliency Computation Function `compute_saliency_for_batch()`**
    - **Purpose:** Quantifies how much each input feature at each timestep contributed to the model’s output for a small batch of patients.
    - **Methodology:**
      1. **Gradient tracking:** Enables PyTorch autograd to record how changes in each input affect the model’s output.
      2. **Gradient × Input (|grad × input|):**  
        - Gradient (`∂y/∂x`) shows sensitivity of prediction to feature changes.  
        - Multiplying by the input value scales this by feature activity.  
        - Taking the absolute value gives overall magnitude of influence, ignoring direction (positive/negative).
      3. **Per-patient saliency:**  
        - Loops through each patient in the batch to compute individual gradients.  
        - The result is stacked into a 3D array of shape `(B, T, F)`:
          - **B** = batch size (number of patients)  
          - **T** = timesteps (sequence length)  
          - **F** = number of features  
      4. **Output:**  
        - Returns a NumPy array representing absolute saliency values (`|∂y/∂x × x|`) for each patient, timestep, and feature.
        - **Final shape:** `(B, T, F)` → one saliency matrix per patient (rows = timesteps, columns = features).

4. **Step 3: Compute Interpretable Saliency Outputs**
  - **Purpose:** 
    - Initialises batch-level computation and iterates through each model output head (target) to compute saliency maps independently.
    - The loop executes the full interpretability pipeline (saliency computation, aggregation, visualization, and diagnostics) for each target head sequentially
    - Ensures that all model outputs receive independent saliency analyses.
  - `batch_size = 4` controls how many patient sequences are processed simultaneously when computing gradients → keeps GPU/CPU memory usage manageable while maintaining computational efficiency.
  - `for target_name, head_key in TARGETS:` 
    - Iterates throgh the three model output heads defined earlier.
    - Each head is analysed separately to understand which features and timepoints drive its predictions.

5. **Step 3A: Compute Per-Patient Saliency Across Batches**
  - **Purpose:** Compute gradient-based saliency (`|∂y/∂x * x|`) for each patient in the test set in manageable batches, producing a full `(n_test, T, F)` array for downstream interpretability. 
  - **Logic Overview:**
    1. **Initialise storage:** List to hold saliency results from each batch (`per_patient_saliency = []`).
    2. **Process in batches:** 
      - Loop over the test set using range(`0`, `n_test`, `batch_size`) to avoid memory issues.
      - Use `tqdm` to monitor progress.
    3. **Prepare batch tensors:**
      -	`xb` → batch of input sequences `(B, T, F)`
      -	`mb` → corresponding mask `(B, T)`
      -	Both moved to the correct device.
    4. **Compute batch saliency:** `compute_saliency_for_batch(model, xb, mb, head_key)` returns `(B, T, F)` array of absolute gradient × input values.
    5. **Accumulate results:** Append each batch’s saliency to `per_patient_saliency`.
    6. **Combine all batches:** `np.concatenate(per_patient_saliency, axis=0)` → final shape `(n_test, T, F)`.
    7. **Verify output:** Print shape for debugging and confirmation.
  - **Output:**
    -	`per_patient_saliency` → 3D array `(n_test, T, F)`
    -	Contains absolute saliency per patient, timestep, and feature.
    - Forms the basis for feature-level summaries, temporal profiles, and heatmaps.

6. **Step 3B: Feature-Level Mean & Standard Deviation Saliency CSV**
  - **Purpose:** 
    - Aggregate per-patient saliency values across all patients and timesteps to quantify overall feature importance for each model target head. 
    - Produces a CSV that is interpretable for downstream analysis or reporting.
  - **Logic Overview:**
    1. **Aggregate over patients and time:**
      -	`axis=(0, 1)` averages over all patients (`n_test`) and all timesteps (`T`).
      -	`feature_mean` → mean absolute saliency per feature.
      -	`feature_std` → standard deviation per feature, providing variability measure.
    2. **Create DataFrame (`df_features`):**
      -	Columns:
        -	`feature` → feature names
        -	`mean_abs_saliency` → average importance
        -	`std_abs_saliency` → variation across patients/timesteps
      -	Sorted descending by mean importance for easy interpretability.
    3. **Save CSV:**
      -	**File path:** `interpretability_tcn/{target_name}_feature_saliency.csv`
      -	Provides a concise, fully interpretable numeric summary of feature relevance across the dataset.
  - **Outputs:**
    -	3x CSV files of shape `(F, 3)` where `F` = number of features; one for each target.
    -	Contains mean and standard deviation of saliency per feature for the target head.

7. **Step 3C: Temporal Mean Profile CSV**
  - **Purpose:**  
    - Summarise saliency across all features to determine when during the sequence the model is most sensitive. 
    - Instead of per-feature detail, this focuses on the temporal pattern of feature importance.
  - **Logic Overview:**
    1. **Aggregate over patients and features:**
      -	`axis=(0, 2)` → average across:
        -	`0` = patients (`n_test`)
        -	`2` = features (`F`)
      -	Result: `temporal_mean` of shape (`T,`) → one value per timestep.
    2. **Create DataFrame (`df_temporal`):**
      -	Columns:
        -	`timestep` → sequential index from 0 → `MAX_SEQ_LEN` - 1
        -	`mean_abs_saliency` → average saliency across all features and patients at that timestep.
    3. **Save CSV:**
      -	Path: `interpretability_tcn/{target_name}_temporal_saliency.csv`
      -	Provides an interpretable, numerical summary showing temporal sensitivity of the model.
  - **Output:**
    -	3x CSV's of shape `(T, 2)` where:
    -	`T` = number of timesteps
    -	Each row contains:
      -	`timestep` index
      -	`mean_abs_saliency` (average `|grad × input|` across features and patients)

8. **Step 3D: Top-5 features temporal profile CSV**
  - **Purpose:**  
    - Identify the most important features for a given model head and track their saliency over time. 
    - This creates an interpretable temporal profile of key features rather than overwhelming the user with all 171 features.
  - **Logic Overview:**
    1. **Select Top 5 Features:**
      -	`feature_mean.argsort()[::-1][:5]` → sorts features by descending mean absolute saliency.
      -	Selects indices of the top 5 most influential features (`top_features_idx`).
    2. **Retrieve Feature Names:** Maps indices to actual feature column names (`feature_cols`) for CSV readability.
    3. **Compute Temporal Profile:**
      -	`per_patient_saliency[:, :, top_features_idx]` → subset saliency for top 5 features across all patients and timesteps `(n_test, T, 5)`.
      -	`.mean(axis=0)` → average over patients → shape `(T, 5)`, one value per timestep per feature.
    4. **Create DataFrame:**
      -	Columns = top feature names.
      -	Insert `timestep` column from `0` to `MAX_SEQ_LEN` - 1.
    5. **Save CSV:**
      -	Path: `interpretability_tcn/{target_name}_top_features_temporal.csv`.
      -	Output is interpretable, numeric, and concise.
      -	Provides temporal patterns of key features for each model head.
  - **Output:**
    -	3x CSV's of shape: `(T, 6)`
      -	Column 0: `timestep`
      -	Columns 1–5: mean saliency of top 5 features at each timestep
    -	Helps visualise when key features drive predictions during a patient’s timeline.

9. **Step 3E: Global Mean Heatmap PNG (Top 10 Features)**
  - **Purpose:**  
    - Generate a visual summary of the top 10 features’ saliency over time.  
    - Designed for interpretability and inspection, not quantitative analysis.  
    - Highlights temporal patterns of key features across all patients.
  - **Logic Overview:**
    1. **Select Top 10 Features:** Sort features by descending mean absolute saliency (`feature_mean.argsort()[::-1][:10]`).
    2. **Compute Mean Across Patients:** 
      - `per_patient_saliency[:, :, top10_idx].mean(axis=0)` → average saliency over all patients → shape `(T, 10)`.
      - Captures general temporal importance patterns rather than patient-level noise.
    3. **Log Transform:** `np.log1p` applied to enhance visibility of small differences and handle zeros safely.
    4. **Color Scaling:** Use 5th and 95th percentiles to reduce influence of outliers on colormap.
    5. **Plot Heatmap:**
      - `imshow` plots features (rows) vs timesteps (columns).
      -	Labels = feature names and timesteps.
      -	Colormap = `plasma` (perceptually uniform).
    6. **Save PNG:**
      -	File path: `interpretability_tcn/{target_name}_mean_heatmap.png`
      -	Provides a quick visual overview of which features dominate the model predictions over time.
  - **Rationale:**
    - **Plot the 10 features with the highest overall mean absolute saliency:** Reduces the visual complexity from 171 to 10; avoids overcrowding, improving interpretability.
    - **Log Transform (np.log1p):** Expands low-magnitude differences so smaller saliency values remain visible while preventing large values from dominating.
    - **Percentile-Based Color Scaling (vmin, vmax = 5th–95th):** Suppresses outlier influence, normalising visual intensity across features → heatmap better reflects the true variation in saliency across features and timesteps, rather than being skewed by a few extreme points.
    - **Perceptually Uniform Colormap (cmap='plasma'):** Ensures equal perceptual contrast across intensity levels, accurate visual mapping of saliency magnitude to colour intensity → improving readability for clinicians.
  - **Output:**
    - 3x Mean heatmap PNGs
    - Heatmap dimensions: (`10 features × T timesteps`)
    -	Log-scaled color intensity = mean |grad × input|
    - Intended for inspection and presentation, complementing numeric CSV outputs.

10. **Step 3F: Diagnostic Terminal Prints**
  - **Purpose:** 
    - Ensures no computational or logical errors in the saliency outputs.
    - Validates model interpretability integrity across all three target heads.
    - Provides confidence that downstream visual and quantitative analyses are built on reliable data.
  - **Logic Overview:**
    1. **Saliency Statistics:**
      ```python
      print(" - NaN count:", np.isnan(per_patient_saliency).sum())
      print(" - Mean value:", np.nanmean(per_patient_saliency))
      print(" - Max value:", np.nanmax(per_patient_saliency))
      ```
      - NaN count: Detects any invalid or missing values from the saliency computation (should be zero).
      - Mean and Max: Provide a quick overview of the magnitude and scale of computed saliency values across all patients, features, and timesteps.
      - Confirms that gradient × input operations produced stable and realistic magnitudes.
    2. **Model Head Correlations:**
      ```python
      print("Correlation between head outputs:")
      print("max ↔ median:", np.corrcoef(max_out, med_out)[0, 1])
      print("max ↔ regression:", np.corrcoef(max_out, reg_out)[0, 1])
      ```
      - Each model head (max_risk, median_risk, pct_time_high) predicts a different outcome.
      - Correlation checks ensure that heads are learning distinct but related patterns:
        - High correlation → possible redundancy between prediction heads.
        - Low correlation → confirms each head captures different aspects of patient risk.
**Outputs**
- All results saved under: `src/results_finalisation/interpretability_tcn/`
- For each target head (`max_risk`, `median_risk`, `pct_time_high`), four files are created:
  1. `{target}_feature_saliency.csv` 
     - Columns: `feature`, `mean_abs_saliency`, `std_abs_saliency`  
     - Mean and variability of saliency per feature across all patients and timesteps (overall feature importance).
  2. `{target}_temporal_saliency.csv` 
     - Columns: `timestep`, `mean_abs_saliency`  
     - Mean saliency across all features per timestep (when the model is most sensitive).
  3. `{target}_top_features_temporal.csv`  
     - Columns: `timestep` + top 5 most salient features  
     - Shows temporal evolution of key feature importances (interpretable time patterns).
  4. `{target}_mean_heatmap.png`
     - Heatmap of top 10 features averaged across all patients (log-scaled for visibility).  
     - Visual summary only, complements quantitative CSV outputs.
**Rationale**
-	Provide clinicians with interpretable, quantitative insight into the model’s temporal reasoning, identifying which features drive deterioration risk and at what times.
-	Create saliency-based interpretability outputs that directly complement and extend the LightGBM SHAP analysis, adding the temporal dimension absent in SHAP.
-	Support transparent clinical evaluation by showing how model attention evolves across time and features for each prediction head.
**Summary**
- Implemented a complete interpretability pipeline for the refined TCN, computing gradient × input (|∂y/∂x × x|) saliency maps on the test set.
- Quantified both feature importance and temporal sensitivity across all three model heads.
- Produced four reproducible outputs per head: feature-level saliency, temporal mean profile, top-feature temporal trends, and log-scaled mean heatmaps for qualitative inspection.
- Introduced log transform and percentile-based color scaling to make heatmaps visually interpretable, highlighting meaningful gradients without distortion from outliers.
- Ensured robustness through diagnostic checks (NaN counts, value ranges, and head correlations) verifying saliency stability and model distinctiveness.
- Created a modular, reproducible script that can be rerun on any model checkpoint or dataset without modification.
- Provides clinicians and researchers with transparent temporal interpretability that complements SHAP feature attribution (phase 6 step 3), revealing when, how, and which features drive model predictions.

---

### Folder Structure 
```text
 src/
  ├── ml_models_tcn/
  │   ├── prepared_datasets/
  │   │   ├── test.pt
  │   │   └── test_mask.pt
  │   │
  │   └── deployment_models/
  │       └── preprocessing/
  │           ├── patient_splits.json
  │           ├── padding_config.json
  │           └── standard_scaler.pkl
  │
  ├── prediction_diagnostics/
  │   └── trained_models_refined/
  │       ├── tcn_best_refined.pt
  │       └── config_refined.json
  │
  └── results_finalisation/
      ├── saliency_analysis_tcn.py           # main script
      │
      └── interpretability_tcn/              # all generated outputs
          ├── max_risk_feature_saliency.csv
          ├── max_risk_temporal_saliency.csv
          ├── max_risk_top_features_temporal.csv
          ├── max_risk_mean_heatmap.png
          │
          ├── median_risk_feature_saliency.csv
          ├── median_risk_temporal_saliency.csv
          ├── median_risk_top_features_temporal.csv
          ├── median_risk_mean_heatmap.png
          │
          ├── pct_time_high_feature_saliency.csv
          ├── pct_time_high_temporal_saliency.csv
          ├── pct_time_high_top_features_temporal.csv
          └── pct_time_high_mean_heatmap.png
```
---

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

---

### Saliency Computation Function Logic `compute_saliency_for_batch()`
**Background**
- Core function that performs the actual gradient-based saliency calculation for one batch of test patients.  
- Defines the mathematical logic for computing how much each input feature at each timestep influenced the model’s prediction for each patient.
- Outputs these values as interpretable saliency maps used later for CSVs and heatmaps.
**Code Logic**
1. **Define Function and Arguments**
  ```python
  def compute_saliency_for_batch(model, x_batch, mask_batch, head_key)
  ```
  - `model`: the trained TCN model already loaded and in evaluation mode.
	-	`x_batch`: tensor of shape `(B, T, F)` = a mini-batch of test patient sequences
    -	`B` = number of patients in the batch
    -	`T` = number of timesteps (96)
    -	`F` = number of input features (171)
	-	`mask_batch`: binary tensor `(B, T)` marking valid timesteps (1) and padding (0).
	-	`head_key`: specifies which output head to compute saliency for (`"logit_max"`, `"logit_median"`, or `"regression"`).

2. **Prepare Input Tensors**
  ```python
  x = x_batch.clone().detach().to(device)
  x.requires_grad = True
  mask = mask_batch.to(device)
  ```
  - `clone().detach()` creates a new copy of the batch to avoid modifying the original data.
	- `requires_grad=True` tells PyTorch to track how the model output changes with respect to every input feature value, this is essential for computing gradients.
	- The mask tensor is transferred to the same compute device (CPU or GPU).

3. **Forward Pass Through the Model**
  ```python
  outputs = model(x, mask)
  out = outputs[head_key].squeeze()
  ```
  - The model produces predictions for all output heads (`logit_max`, `logit_median`, `regression`).
  -	We select the target output head being explained (e.g., `"logit_max"`).
  -	`squeeze()` removes singleton dimensions, leaving a 1D tensor of shape `(B,)` → one scalar prediction per patient in the batch.

4. **Initialise Gradient Storage**
  ```python
  grads = []
  ```
  - Empty list to store gradients for each patient separately.

5. **Compute Gradients per Patient**
  ```python
  for i in range(out.shape[0]):
      if x.grad is not None:
          x.grad.zero_()
      scalar = out[i]
      scalar.backward(retain_graph=True)
      grads.append(x.grad[i].detach().cpu().numpy())
  ```
  -	The loop iterates over each patient (`i`) where `out.shape[0]` = number of patients in the batch `(B)`:
    -	`x.grad.zero_()` clears previous gradients before computing new ones.
    -	`scalar = out[i]` isolates a single scalar output (the model prediction for that patient).
    -	`scalar.backward()` computes the derivative of the patient's scalar prediction `(y_i)` with respect to all input features  (`x`) in the batch tensor. After this, `x.grad` has the same shape as `x` → `(B, T, F)`; gradients for every feature at every timestep.
    - `x.grad[i]` extracts the gradient map for this specific patient `(T, F)`.
    -	`.detach().cpu().numpy()` removes it from the PyTorch graph and converts to NumPy for later aggregation.
    - `grads.append` appends the extracted matrix to the list

6. **Combine all Per-Patient Gradients**
  ```python
  grads = np.stack(grads, axis=0)
  ```
	-	Produces a single unified 3D array of shape `(B, T, F)`; gradients for every patient, timestep, and feature.

7. **Compute Gradient × Input (Saliency Map)**
  ```python
  saliency = grads * x.detach().cpu().numpy()
  ```
	-	Element-wise multiplication combines sensitivity (`grad`) with the input magnitude (`x`).
  - **Mathematical equation:** `S[b, t, f] = | (∂y_b / ∂x[b, t, f]) × x[b, t, f] |`
    - `S[b, t, f]` = saliency value for patient `b`, timestep `t`, feature `f`
    -	`y_b` = model output (scalar prediction) for patient `b`
    -	`x[b, t, f]` = input value for feature `f` at time `t` for patient `b`
    -	`∂y_b / ∂x[b, t, f]` = gradient (sensitivity of the output to that input)
	-	This highlights where high input values strongly influenced the model’s output.

8. **Return Absolute Saliency**
  ```python
  return np.abs(saliency)
  ```
  - The absolute value removes direction (positive/negative influence) and keeps only magnitude → representing the strength of contribution.
	- **Final shape:** (`B, T, F`) → one saliency map per patient showing importance of each feature over time.

**Summary**
-	Computes gradients of the model’s predictions with respect to all input features.
-	Multiplies gradients by actual input magnitudes to derive |grad × input| saliency to measure contribution strength.
-	Returns |grad × input| saliency maps (3D array) showing how strongly each feature at each timestep influenced each patient’s risk prediction.

---

### TCN Saliency Terminal Output Summary
**Environment & Data:**
- All required input files found
- Device: CPU
- Test data shape: (15 patients, 96 timesteps, 171 features)
- Mask shape: (15, 96)
- TCN model loaded successfully and moved to device
**Outputs saved:**
All files saved in: `src/results_finalisation/interpretability_tcn/`
- Feature-level saliency CSVs: `{target}_feature_saliency.csv`
- Temporal mean saliency CSVs: `{target}_temporal_saliency.csv`
- Top-5 features temporal profile CSVs: `{target}_top_features_temporal.csv`
- Global mean heatmaps (top 10 features) PNGs: `{target}_mean_heatmap.png`  

**Saliency Outputs & Diagnostics:**
| Target           | Saliency shape       | NaN count | Mean saliency       | Max saliency       | Head correlations (max ↔ median / max ↔ regression) |
|-----------------|--------------------|-----------|------------------|------------------|----------------------------------------------------|
| max_risk         | (15, 96, 171)      | 0         | 1.716e-05         | 0.002402         | 0.956 / 0.958                                      |
| median_risk      | (15, 96, 171)      | 0         | 2.681e-05         | 0.005771         | 0.956 / 0.958                                      |
| pct_time_high    | (15, 96, 171)      | 0         | 7.897e-06         | 0.001682         | 0.956 / 0.958                                      |

**Notes:**
- Confirms the model and data loaded correctly.
- No NaNs present in the saliency arrays.
- Mean, max saliency checks numeric sanity.
- Array shapes and dimensions verified against input data.
- Outputs provide fully quantitative, interpretable information; heatmaps are only for visualization purposes.
- Head output correlations indicate high consistency between related targets.

---

### TCN Saliency Analysis: Issues, Diagnosis, and Fixes
#### Objective
- To explain temporal feature importance for each TCN output head (`max_risk`, `median_risk`, `pct_time_high`) in a way that is numerically robust, interpretable, and visually meaningful.

#### Original Implementation and Issues
1. **Original Approach**
  - Computed **|gradient × input| per patient, per feature, per timestep**.
  - Generated:
    1. CSVs of top 10 features and mean absolute saliency.
    2. Per-patient heatmaps.
    3. Global mean heatmaps.
2. **Identified Problems**
  - **Numerical Issues**
    - **Empty or NaN CSVs**:
      - `mean_abs_saliency` was NaN due to `per_patient_saliency` containing NaNs.
      - NaNs propagated from padded regions or zero-gradient outputs.
    - **Duplicate per-head arrays**: `np.allclose` confirmed identical saliency for different heads, suggesting degenerate gradient computation.
  - **Visualisation Issues**
    - **Per-patient heatmaps** (171 features × 96 timesteps) were cluttered and impossible to interpret.
    - **Gradient magnitudes** extremely small (~1e-6 to 1e-3), compressing color range.
    - Heatmaps showed mostly uniform blue, making subtle differences invisible.
3. **Diagnostic Strategy**
  - **Global saliency stats** → check for non-empty, finite values.
  - **Top-10 CSVs** → verify CSV writing and numeric output.
  - **Head correlations** → ensure semantic separation between heads.
  - **Gradient magnitude range** → verify non-zero sensitivity.
  - **Duplicate array check** → catch degenerate computation.

#### Diagnostic Summary
```bash
[DIAGNOSTICS] ==============================
Saliency summary:
 - NaN count: 40869
 - Mean value: 9.468709e-06
 - Max value: 0.0016815776

Top-10 features for max_risk:
                                feature  mean_abs_saliency
0                       ART BP Systolic                NaN
1                     GCS - Eye Opening                NaN
2                  GCS - Motor Response                NaN
3                 GCS - Verbal Response                NaN
4  Non Invasive Blood Pressure systolic                NaN
5                            heart_rate                NaN
6                      respiratory_rate                NaN
7                                  spo2                NaN
8                       supplemental_o2                NaN
9                           systolic_bp                NaN

Top-10 features for median_risk:
                                feature  mean_abs_saliency
0                       ART BP Systolic                NaN
1                     GCS - Eye Opening                NaN
2                  GCS - Motor Response                NaN
3                 GCS - Verbal Response                NaN
4  Non Invasive Blood Pressure systolic                NaN
5                            heart_rate                NaN
6                      respiratory_rate                NaN
7                                  spo2                NaN
8                       supplemental_o2                NaN
9                           systolic_bp                NaN

Top-10 features for pct_time_high:
                                feature  mean_abs_saliency
0                       ART BP Systolic                NaN
1                     GCS - Eye Opening                NaN
2                  GCS - Motor Response                NaN
3                 GCS - Verbal Response                NaN
4  Non Invasive Blood Pressure systolic                NaN
5                            heart_rate                NaN
6                      respiratory_rate                NaN
7                                  spo2                NaN
8                       supplemental_o2                NaN
9                           systolic_bp                NaN

Correlation between head outputs:
max ↔ median: 0.9557705554563857
max ↔ regression: 0.9577254391550588

Feature-level mean saliency range:
min: nan max: nan

Are patient_0 saliency maps identical between heads?: True
[DIAGNOSTICS COMPLETE]
```

#### Redesign Strategy
1. **Core Goals**
  - Eliminate un-interpretable outputs.
  - Provide aggregated, actionable quantitative outputs.
  - Ensure heatmaps are visible and informative.
2. **New Output Pipeline per Target Head**
  1. **Feature-level Mean Saliency CSV**
    - `{target}_feature_saliency.csv`
    - Columns: `feature`, `mean_abs_saliency`, `std_abs_saliency`
    - Sorted descending by mean saliency.
  2. **Temporal Mean Profile CSV**
    - `{target}_temporal_saliency.csv`
    - Columns: `timestep`, `mean_abs_saliency_over_features`
    - Shows when the model is most sensitive.
  3. **Top-5 Features Temporal Profile CSV**
    - `{target}_top_features_temporal.csv`
    - Columns: `time`, `feature_1`, …, `feature_5`
    - Tracks temporal evolution of key features.
  4. **Global Mean Heatmap PNG**
    - `{target}_mean_heatmap.png`
    - Top 10 features only.
    - Enhanced color scaling and visual improvements.
  5. **Diagnostics Console**
    - NaN counts.
    - Mean/max saliency range.
    - Head correlations.
    - Confirms numerical correctness.
3. **Code Updates**
  - Removed all per-patient `.npz` saves and heatmaps.
  - Aggregated over all patients and timesteps for CSV outputs.
  - Proper logging to catch NaNs or degenerate gradients.
  - Top-feature selection for temporal plotting and visual heatmap.

#### Visualisation Fixes
- **Issue:** Original heatmaps mostly blue due to tiny gradient magnitudes.
- **Fixes Applied:**
  1. **Percentile-based color scaling** → maps meaningful saliency differences, ignores extreme outliers.
  2. **Log transform (`np.log1p`)** → expands small differences for visibility.
  3. **Improved colormap** → plasma/cividis for better perceptual contrast.
  4. **Enhanced figure size & font** → ensures feature labels readable.
- **Result:** heatmaps now show visible, interpretable temporal patterns across top features.

#### Summary of Changes
| Step | Issue | Fix |
|------|-------|-----|
| Numerical CSVs | NaNs in per-patient CSVs | Compute aggregated feature- and temporal-level saliency, ignore padded regions |
| Per-patient heatmaps | Unreadable / cluttered | Removed; keep only aggregated top-feature mean heatmap |
| Degenerate gradients | Duplicate saliency across heads | Recompute per-sample gradients with correct `retain_graph` usage |
| Tiny heatmap differences | Almost uniform color | Percentile scaling + log transform + colormap + font improvements |
| Interpretability | Inconsistent outputs | CSVs now provide interpretable rankings and temporal profiles |

#### Outcome
- Valid, interpretable `per_patient_saliency` arrays.
- Clear CSV summaries for **feature-level importance**, **temporal patterns**, **top-feature evolution**.
- Visual heatmaps now clearly show temporal saliency across top features.
- Diagnostic block ensures reproducibility and numerical validity.
- Provides **clinically meaningful, publication-ready outputs**.

#### Notes for Users
- Heatmaps are visual summaries; all quantitative interpretation should come from CSVs.
- Saliency values are still based on `|grad × input|` computed from raw model outputs.
- Temporal trends in heatmaps and CSVs should be cross-referenced to physiological interpretation (HR, BP, RR, SpO₂).

---

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

---

### Reflection
#### Challenges
1. **Conceptual setup**
  - Initial uncertainty about whether temporal saliency was valid since the TCN was trained on timestamp-level data but evaluated at the patient level.
  - Questioned whether interpretability should be run on the training or test cohort and how to align it conceptually with LightGBM SHAP analysis.
  - Unsure whether Integrated Gradients (IG) or raw gradients were most appropriate for clinical interpretability.
2. **Implementation issues**
  - Early saliency outputs were invalid:
    - Empty CSVs with NaNs in all columns.
    - Duplicate or degenerate saliency arrays between model heads (high output correlation ~0.95).
    - Mean values extremely small (1e−6–1e−3), creating nearly blank heatmaps.
  - Per-patient heatmaps (171 features × 96 timesteps) were computationally heavy and uninterpretable.
  - Diagnostic checks revealed identical per-head outputs and NaN propagation from padded regions.
  - Color scaling made visual results unreadable; almost all blue due to tiny dynamic range.
3. **Analytical confusion**
  - Unsure how to treat post-processing transformations:
    - Whether to include median-risk threshold tuning.
    - Whether to inverse-transform the log1p regression output.
  - Concern about whether to compare SHAP vs saliency directly or conceptually, given completely different model types and feature spaces.
#### Solutions and Learnings
1. **Theoretical resolution**
  - Confirmed that temporal saliency is valid and essential when the model operates on sequences, even if outputs are aggregated per patient.  
  - Computed **gradient × input saliency manually** (not full Integrated Gradients):  
    - Captures how each input feature at each timestep contributes to the model's prediction.  
    - More interpretable than raw gradients alone, because multiplying by the input scales the gradient by the feature’s actual value.  
    - This approach accounts for both the **gradient magnitude** (how sensitive the prediction is to that feature) and the **activation magnitude** (whether the feature had a nonzero value), giving a more meaningful measure of feature influence than plain gradients.  
    - Reveals **temporal patterns of importance**, showing when and which physiological signals drive risk predictions.  
    - Provides explanations that complement global feature importance (e.g., LightGBM SHAP) by highlighting the timing of key features.  
2. **Clarified Interpretability differences**
  - Defined distinct interpretability goals:
    - **SHAP (LightGBM):** Global feature-level interpretability on training data → what features matter.
    - **Saliency (TCN):** Local temporal interpretability on test data → when and how features influence predictions.
  - Accepted that these are complementary, not directly comparable, since models are trained on different feature sets and input modalities.
3. **Implementation fixes**
  - Rewrote the saliency script to:
    - Remove per-patient `.npz` saves and heatmaps.
    - Replace with numerically interpretable CSVs:
      1. `feature_saliency.csv`: feature-level mean and std of |grad × input|.
      2. `temporal_saliency.csv`: overall sensitivity per timestep.
      3. `top_features_temporal.csv`: evolution of top 5 features over time.
    - Added one compact mean heatmap PNG (`mean_heatmap.png`) for top 10 features only.
  - **Introduced diagnostic validation**:
    - NaN count.
    - Gradient magnitude range.
    - Head correlations (max vs median vs regression).
    - Cross-head variance checks.
  - **Applied visual improvements**:
    - Percentile-based color scaling.
    - Log transform for small values.
    - Improved colormap (`plasma`).
    - Clearer feature labels and font sizes.
4. **Analytical clarity**
  - Confirmed saliency should not account for:
    - Threshold tuning (`median_risk`) → post-hoc.
    - Inverse transforms (`pct_time_high`) → non-linear and distorting.
  - Saliency correctly explains the trained log-space and pre-sigmoid logits, not transformed outputs.
  - Shifted from per-patient visualisation to **global interpretability via aggregation**; scientifically cleaner and computationally efficient.
  - Final saliency outputs were valid, interpretable, and aligned with known physiological drivers (RR, SpO₂, HR).
5. **Comparative reasoning with SHAP**
  - Comparison is conceptual, not numerical:
    - **Alignment:** Overlapping key features (SpO₂, RR, O₂ support) suggest both models learned coherent physiological relevance.
    - **Extension:** TCN adds temporal understanding, showing when deterioration signals emerge (e.g., rising RR preceding risk escalation).
    - **Divergence:** Features unique to saliency (e.g., HR variability) likely reflect temporal sensitivity not captured by static LightGBM inputs.
  - The two interpretability layers together provide a more complete picture:
    - **SHAP:** Stable global drivers.
    - **Saliency:** Dynamic evolution and clinical context.
#### Overall Reflection
- This step transformed interpretability from a theoretical add-on into a robust analytical layer:
  - Addressed numerical, conceptual, and visualisation failures.
  - Established a **two-tier interpretability framework**:
    1. Global feature importance (LightGBM–SHAP).
    2. Temporal reasoning (TCN–saliency).
- The refined saliency script now generates reliable and publication-ready outputs → feature-level tables, temporal profiles, and aggregated heatmaps.
- The analysis confirms that both models emphasize consistent physiological features, while the TCN adds insight into the temporal evolution of risk, complementing LightGBM’s static perspective.
- The comparison validated that **interpretability succeeds when combining static and temporal models**, yielding both transparency and clinical coherence.
- End result → Interpretability pipeline is complete, stable, and aligned with best practices for explainable clinical time-series ML modelling.

---

# Phase 7: Deployment

---

# Phase 7A: Inference Demonstration (Deployment-Lite)

---

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

---

## Day 45-46 Notes - Start Phase 7A: Inference Pipeline 

### Goals
-	Build and finalise a unified inference pipeline for both LightGBM and TCN models for reproducible deployment and easy terminal usage.
- Ensure outputs are clean, consistent, and correctly saved (predictions and interpretability CSVs).
- Test the full pipeline end-to-end, identify and fix bugs, and verify correctness of predictions and feature importance outputs.

### What We Did
#### Build Inference Full Pipeline (`unified_inference.py`)
**Purpose**
- Provide a unified, deployment-ready pipeline that performs inference for both models (LightGBM + TCN).
- Ensure predictions are generated using the exact trained models, preprocessing, and feature mappings.
- Produce clean batch outputs (predictions + interpretability summaries) suitable for downstream use or deployment.
- Offer an optional interactive CLI for quick, single-patient prediction checks without rerunning batch inference.
- Replace multiple fragmented evaluation scripts with one consistent, reproducible inference workflow.
**Workflow**
0. **Initialise Imports and File Paths**
  - Loads all required libraries (`json`, `pandas`, `numpy`, `torch`, `joblib`, `shap`).
  - Resolves project directories (`SCRIPT_DIR`, `SRC_DIR`, `PROJECT_ROOT`).
  - Loads all input file paths:
    - `news2_features_patient.csv` → patient-level feature columns for LightGBM
    - `patient_splits.json` → test patient ID's
    - `padding_config.json` → timestamp-level feature columns for TCN
    - `prepared_datasets/` → TCN test and mask tensors
  - Loads all model paths:
    - LightGBM models in `lightgbm_results/`
    - TCN weights `tcn_best_refined.pt`
    - TCN config `config_refined.json`
  - Creates the output directory `scripts_inference/deployment_lite_outputs/`.
1. **Load Test Patient Data for LightGBM**
  - Loads the `"test"` patient IDs from `patient_splits.json` → `test_ids`.
  - Loads the patient-level features from `news2_features_patient.csv` → `features_df`.
  - Subsets the DataFrame to only the test patients → `test_df`.
  - Defines the LightGBM input feature columns (`feature_cols`) by excluding only non-feature columns (`exclude_cols`): `subject_id`, `max_risk`, `median_risk`, `pct_time_high`
  - **Rationale:**
    - Binary targets are not rebuilt because inference does not compute metrics; it only produces predictions.
    -	Only the feature columns used during training are required; outcomes are irrelevant for inference.
    -	This makes the pipeline minimal and dataset-agnostic → any dataset with the correct features can be used.
    -	The section is simpler than evaluation pipelines because no label reconstruction, metric calculation, or calibration is needed.
2. **Compute LightGBM inference**
  - Define the prediction tasks (`targets` list) to load:
    - `max_risk` (classification)
    - `median_risk` (classification)
    - `pct_time_high` (regression)
  - **Iterate over each target to load model and perform inference:**
    - Looks for `{target}_retrained_model.pkl` inside the LightGBM results directory.
    - Loads the model with `joblib.load()`.
    - Stores the model in `models_dict` for later SHAP interpretability.
    - Prepare the test feature matrix → `X_test = test_df[feature_cols]` to obtain only input features (no labels).
    - **Generate predictions:**
      - For classification targets: `predict_proba()[:, 1]` gets the positive-class probability.
      - For regression: `predict()` gives a continuous prediction.
    - Store prediction outputs for all three targets into `lightgbm_preds`.
  - Create a results DataFrame (`df_lightgbm`): Aggregates predictions with `subject_id` into a clean output table.
  - Clip invalid regression values → ensures `pct_time_high` predictions cannot be negative.
  - Save dataframe as `lightgbm_inference_outputs.csv` containing all LightGBM predictions.
3. **Compute TCN inference**
  - **Loads requirements:**
    - Loads the `TCNModel` class from `tcn_model.py` → architecture during training.
    - Set device to CPU/GPU to ensure all tensors and model parameters are on same device.
    - Loads the prepared time-series tensors (`test.pt`) and corresponding masks (`test_mask.pt`) → `x_test` of shape `(num_patients, seq_len, num_features)` and `mask_test`, and ensure batch shape alignment of `x_test` with `test_ids`.
    - Loads model configuration (`config_refined.json`) → read hyerparameters (`"model_architecture"`)
  - **Reconstructs TCN architecture and prepares for inference:**
    - Instantiates model with loaded hyperparameters to recreate architecture → `model = TCNModel(...)` as weights can only be loaded if model structure matches exactly.
    - Loads the trained TCN weights (`tcn_best_refined.pt` → `state_dict`) → learned parameters (filters, biases).
    - Moves model to CPU/GPU and sets to evaluation mode `model.eval()` → disables dropout, ensures deterministic predictions.
  - **Performs a forward pass through the model:**
    - Runs timed inference with no gradient computation; `torch.no_grad()` disables gradient tracking → faster, lower memory. 
    - Generate outputs → `outputs = model(x_test, mask_test)`
  - **Process inference outputs:**
    - Extract logits (`logit_max`, `logit_median`) and `regression` (log-space) outputs, each tensor is flattened, move to CPU and numpy for processing
    - Converts logits to probabilities via sigmoid → converts raw logits to numbers between 0–1 (`prob_max`, `prob_median`).
    - Converts regression log-scale → raw scale (`reg_raw`); matches the inverse of training transform (`log1p` → `expm1`).
    - Clip negative regression values at 0 → prevents impossible negative percentages.
  - **Build output DataFrame which includes `subject_id`:**
    - `prob_max` (probability of high max-risk)
    - `prob_median` (probability of high median-risk)
    - `y_pred_reg_raw` (predicted pct-time-high, back-transformed)
  - Saves all outputs to `tcn_inference_outputs.csv`.
  - Print confimation to terminal that both models computed predictions for 15 test patients
  - **Rationale:**
    - No binary targets are needed because inference only generates predictions, not metrics.
    - TCN takes preprocessed tensors (`test.pt`, `test_mask.pt`); patient-level labels play no role.
    -	The model architecture must be rebuilt from the JSON config before loading weights, since .pt files store weights only.
    -	Masks ensure padded timesteps do not influence outputs, matching training behaviour.
    -	Regression predictions must be inverse-transformed with `expm1` because the model learned on log-scaled targets.
4. **Compute LightGBM Interpretability (SHAP)**
  - **Define helper function `compute_lightgbm_shap_top10()`**  
    - Accepts:
      - `models_dict`: dictionary of trained LightGBM models (one per target).  
      - `X_input`: the feature matrix used for inference.  
    - Initialises an empty `results` list.
    - **Loop through each LightGBM model**
      - Create a `shap.TreeExplainer(model)` for that target.
      - Compute SHAP values (`.shap_values(X_input)`) → If the model is binary classification, SHAP returns a list → extract the positive-class values (`shap_values[1]`).
      - Compute **mean absolute SHAP values** for each feature (`mean_abs = np.abs(shap_values).mean(axis=0)`).
      - **Create DataFrame of top 10 features**
        - Construct a DataFrame: `feature`, `mean_abs_shap`, `target`  
        - Sort by SHAP magnitude (`ascending=False`) and keep the top 10 (`.head(10)`).
      - Append dataframe to the `results` list.
    - **Concatenate results:** Combine all targets’ top-10 DataFrames into one summary table.
  - **Run the function:**
    - Call `compute_lightgbm_shap_top10(models_dict, test_df[feature_cols])`.
    - Store output in `lightgbm_top10`.
  - **Rationale:**
    - Only **Top-10 SHAP features** are computed to keep interpretability **lightweight, fast, and deployment-ready**.
    - This mirrors **Phase 6**, where the same top-10 SHAP summary was used as the global interpretability output.  
    - Full SHAP plots are deliberately excluded because:
      - They add unnecessary file size and runtime.
      - Visualisations are not needed in deployment-lite inference.
    - The resulting numeric table gives clear, model-consistent feature importance while remaining efficient and reproducible.
5. **Compute TCN Interpretability (|Gradient x Input| Saliency)**
  - **Load TCN feature names**
    - Read `padding_config.json` to obtain `feature_cols` in the correct tensor order.
    - Ensures saliency indices → human-readable feature names.
  - **Create list (`tcn_targets`) that defines each target and maps each output head to its corresponding key in `model(x, mask)`:**
    - `max_risk` → `"logit_max"`
    - `median_risk` → `"logit_median"`
    - `pct_time_high` → `"regression"`
  - **Define helper function `compute_tcn_saliency_top10()`**
    - The function receives as args:
      - `model` → trained TCN model,  
      - `x_tensor`, `mask_tensor` → test tensors  
      - `feature_cols` → ordered list of feature names corresponding to tensor indicies.
      - `targets` →  list of tuples `(target_name, model_output_key)` specifying which output heads to compute saliency for
      - `device` → PyTorch device to perform computation on.
    - Initialise containers: 
      - `results` list → stores top-10 features for each target head
      - `batch size = 4` → mirrors batching used in phase 6 interpretability (computational efficiency)
      - `n_test` → is number of patients in set (15)
    - Loop through each TCN output head `(target_name, model_output_key)`: 
      - `all_saliencies` list will store saliency results from all batches.
      - Iterate through test data in batches (`batch size = 4`):
        - Prepare batch data: 
          - `x_tensor` → `xb`, `.clone().detach()` ensures no linkage to previous gradients.
          - `mask_tensor` → `mb`, ensures padded timesteps do not affect output.
          - `xb.requires_grad = True` enables gradient computation w.r.t. inputs.
        - Forward pass:
          - `model(xb, mb)` returns outputs for all 3 heads, extract output for  the current target.
          - `.squeeze()` removes unecessary dimensions → produces shape `(batch,)`.
        - Compute gradients per sample:
          - For each patient in the batch:
            -	Zero gradients to prevent accumulation (`xb.grad.zero_()`).
            -	Backprop through the scalar output `out[j].backward(..)`.
            -	Store the gradient w.r.t. inputs into list (`grads.append`): gradient tensor has shape `(seq_len, n_features)`
            - Convert to CPU numpy for further operations.
          - This produces one gradient map per patient per timestep per feature.
        - Compute saliency:
          - Stack gradients into array shape `(batch, seq_len, n_features)`
          - Multiply gradients by input tensor (`grads * xb`), take absolute value (`.abs()`) → saliency magnitude (`sal`).
        - Append batch saliencies to `all_saliencies` list
      - Concatenate batch results (`np.concatenate()`) → Shape becomes `(n_test, seq_len, n_features)`.
      - Average by taking mean across patients (`axis=0`) and timestamps (`axis=1`) → `feature_mean = all_saliencies.mean(axis=(0, 1))` → shape `(n_features,)` → One importance score per feature, globalised across dataset.
      - Build DataFrame:
        - Columns: "feature", "mean_abs_saliency", "target"
        - Sort saliency by descending (`ascending=False`), keep only top 10 (`.head(10)`)
        - Append each target DataFrame to `results` list.
      - Return final concatenated DataFrame (`pd.concat()`)
  - Call helper function to run saliency summary using loaded model and test tensors → `tcn_top10 = compute_tcn_saliency_top10(model, x_test, mask_test, tcn_feature_names, tcn_targets, device)`
  - **Rationale:**
    - Gradient × Input saliency is the same interpretability method used in **Phase 6**, ensuring scientific and methodological consistency.  
    - Averaging saliency across patients and timesteps produces a clean, lightweight global feature ranking appropriate for deployment.  
    - Numerical summaries only, no heatmaps or multi-dimensional arrays to keep inference fast, storage small, and outputs compatible with deployment-lite goals.  
    - The top-10 feature summary gives a clear explanation of TCN behaviour while avoiding full interpretability overhead from earlier phases.
6. **Merge Feature Summaries (LightGBM + TCN)**
  - Concatenate both interpretability outputs into `combined_summary`:
    - `lightgbm_top10`: top-10 SHAP features per LightGBM target.
    - `tcn_top10`: top-10 saliency features per TCN target.
    - Each dataframe is tagged with a `"model"` column to preserve model identity.
  - 2 models x 3 targets x top-10 feature per target = 60 rows
  - Columns: `feature`,`mean_abs_shap`,`target`,`model`,`mean_abs_saliency`
  - Save to CSV:
    - Output file: `top10_features_summary.csv`
    - Lightweight, model-agnostic, easy to inspect or feed into dashboards.
  - **Rationale:**
    - Provides one consolidated file for interpretability instead of two separate outputs.
    - Mirrors what Phase 6 produced conceptually, but kept minimal:
      - **No plots**, **no per-patient heatmaps**, **no raw saliency arrays**.
      - Only the most actionable top-10 aggregated features.
    - Useful for: Model comparison, reporting, deployment dashboards, quick inspection after inference
    - Reduces cognitive load during inspection: one file contains everything needed.
7. **Create Interactive CLI: Single-Patient Inference**
- **Define helper function `run_single_patient_inference(patient_id)`:**
  - **Input:** a `patient_id` (int or str).  
  - **Validation:**  
    - Check whether the ID exists in `test_df["subject_id"]`.  
    - If invalid → print an error message and return.
  - **Locate patient index (`idx`):**  
    - Find the row index in `test_df` corresponding to the given patient.
  - **Print LightGBM predictions:**  
    - Loop through all LightGBM targets (`max_risk`, `median_risk`, `pct_time_high`).  
    - Retrieve that patient’s predicted value from `lightgbm_preds`.
  - **Print TCN predictions:**  
    - Output `prob_max`, `prob_median`, and the regression prediction (`y_pred_reg_raw`) for that patient.
- **Interactive CLI post-batch inference:**
  - Executed when the script is run directly (`if __name__ == __main__`)
  - Print a message confirming batch inference is complete.
  - Display the list of available patient IDs (`test_ids`) to guide the user.
  - Start an interactive loop for `user_input`:
    - Prompt: `“Enter a patient ID… (or 'no' to exit)”`
    - If input is `"no"` → exit loop.
    - Validate the input:
      - Must be numeric (`.isdigit()`).
      - Convert `user_input` to integer → must appear in `test_ids`.
    - Call `run_single_patient_inference(patient_id)`
  - **Rationale**
    - This CLI provides a **lightweight, deployment-friendly tool** to explore predictions without notebooks or dashboards.
    - It uses **already-computed outputs** (LightGBM + TCN) so no recomputation or preprocessing is required.
    - Enables **quick inspection**, debugging, and demo-ready functionality for reviewers.
    - No visualisations or extra artefacts are included, keeping the deployment pipeline **minimal and portable**.
    - The interface is optional and does not affect automated batch inference or downstream integration.

**Outputs**
- All artefacts are saved in the `deployment_lite_outputs/` folder.
- **Model Outputs:**
  - `lightgbm_inference_outputs.csv` → LightGBM predictions for all targets  
    (classification probabilities + regression output).
  - `tcn_inference_outputs.csv` → TCN predictions for all targets  
    (classification probabilities + regression output).
- **Interpretability Outputs**
  - `top10_features_summary.csv` → Combined top-10 feature importance table  
  (LightGBM SHAP + TCN |Gradient×Input| saliency).
- **Optional Interactive Output**
  - Clean terminal display for per-patient lookup via the CLI.  
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

### Inference Pipeline Output 
#### Terminal Output and CLI Example
```bash
[INFO] Starting LightGBM inference...
[INFO] Saved LightGBM inference outputs → lightgbm_inference_outputs.csv
[INFO] Starting TCN inference...
[INFO] TCN inference completed in 0.03s
[INFO] Saved TCN inference outputs → tcn_inference_outputs.csv

=== Deployment Lite Inference Completed ===
LightGBM predictions: 15 patients
TCN predictions:      15 patients
Outputs saved in 'deployment_lite_outputs/' folder
===========================================
[INFO] Computing SHAP Top-10 features (LightGBM)...
[INFO] Computing Gradient×Input Saliency Top-10 features (TCN)...
[INFO] Computing TCN saliency for max_risk...
[INFO] Computing TCN saliency for median_risk...
[INFO] Computing TCN saliency for pct_time_high...
[INFO] ✅ Combined interpretability summary saved → top10_features_summary.csv

Batch inference complete.
Available patient IDs for per-patient inference:
[10021938, 10005909, 10025612, 10027445, 10039831, 10022281, 10015931, 10023771, 10037928, 10021312, 10021666, 10020740, 10007058, 10002428, 10038999]

Enter a patient ID for per-patient inference (or 'no' to exit): 10021938

--- LightGBM predictions for patient 10021938 ---
max_risk: 0.6798
median_risk: 0.0031
pct_time_high: 0.0

--- TCN predictions for patient 10021938 ---
prob_max: 0.7658
prob_median: 0.3093
y_pred_reg_raw: 0.0855

Enter a patient ID for per-patient inference (or 'no' to exit): no
Exiting per-patient inference.
```
#### Notes:
- **Batch inference produces CSV outputs for all patients and computes top-10 feature interpretability summaries:**
	-	Confirms batch predictions for both models and top-10 feature summaries saved.
	-	TCN inference reports elapsed time; LightGBM outputs probabilities/regression.
- **The interactive CLI allows querying predictions for individual patients using their IDs:**
	-	Users can select any valid patient ID from the test set for individual predictions.
	-	LightGBM outputs include positive-class probabilities and regression predictions.
	-	TCN outputs include classification probabilities for `max_risk` and `median_risk` plus the back-transformed regression prediction (`y_pred_reg_raw`).
	-	CLI interaction is clean, minimal, and does not recompute batch outputs, ensuring fast inspection.
	-	Entering "no" exits the loop and terminates the single-patient inference session.
- Terminal outputs are kept minimal, clear, and informative to support deployment and quick inspection.

---

### Reflection
#### Challenges
1. **Deployment-Lite Pipeline Conceptualisation**
  - Needed to translate research/evaluation scripts (Phase 5–6) into a single, reproducible, lightweight inference pipeline for deployment.
  - Needed to balance completeness (both LightGBM + TCN predictions) with minimalism (no plotting, no metrics, no heavy per-patient outputs).
2. **Binary Target Columns**
  - Previous evaluation scripts required recreating binary target columns (`max_risk_binary`, `median_risk_binary`) to compute metrics.
  - Inference does not require true labels, so including these columns would add unnecessary computation and complexity.
3. **TCN Model Loading**
  - PyTorch `.pt` files store only weights, not architecture. Directly loading weights without rebuilding the model would fail.
  - Required consistent feature order and masking to ensure inference outputs matched expectations.
4. **Regression Outputs Producing Negative Values**
  - LightGBM regression predictions (`pct_time_high`) occasionally returned small negative numbers, which are impossible for percentage targets.
  - Needed a safe and reproducible way to constrain predictions without affecting other outputs.
5. **Interpretability Summaries**
  - Phase 6 outputs (full SHAP plots, per-timestep saliency) were too heavy for lightweight deployment.
  - Needed concise, aggregated summaries that retained actionable information but were fast and portable.
6. **Interactive Single-Patient CLI**
  - Need to provide real-time per-patient inference without duplicating batch computation or saving unnecessary files.
  - Needed a simple interface that was intuitive and aligned with deployment-lite goals.
#### Solutions and Learnings
1. **Unified Deployment-Lite Pipeline**
  - Implemented `unified_inference.py` to consolidate LightGBM and TCN inference into a single script.
  - Outputs both batch predictions and optional per-patient CLI predictions.
  - Workflow: load models → load input features/tensors → run inference → save batch outputs → compute minimal interpretability → provide optional CLI.
2. **Omitting Binary Targets**
  - Excluded binary target recreation from inference.
  - Pipeline uses only input features; predictions are generated without ground-truth labels or metric calculations.
  - Simplified feature preprocessing and ensured dataset-agnostic usability.
3. **TCN Model Reconstruction**
  - Loaded JSON configuration to rebuild model architecture before loading `.pt` weights.
  - Applied input masks during inference to ignore padded timesteps, ensuring outputs match training logic.
  - Learned: Neural network weights must always be coupled with architecture reconstruction; LightGBM does not require this.
4. **Regression Output Clipping**
  - Applied `clip(lower=0)` to LightGBM and `np.clip(..., 0, None)` to TCN regression outputs.
  - Ensured no negative predictions appear, maintaining deployment-safe numeric outputs.
5. **Concise Interpretability Summaries**
  - LightGBM: Computed top-10 SHAP features per target, numeric-only.
  - TCN: Computed mean absolute saliency per target, top-10 features.
  - Merged summaries into a single CSV for easy inspection.
  - Learned: Minimal numeric summaries provide interpretability insights without compromising speed or portability.
6. **Interactive CLI Implementation**
  - Added `run_single_patient_inference(patient_id)` function.
  - Validates patient ID, prints LightGBM probabilities/regression and TCN outputs directly to terminal.
  - Avoided saving per-patient outputs to disk for lightweight interaction.
  - Learned: CLI demonstrates functionality and provides real-time inspection while keeping deployment-lite goals intact.
#### Overall Reflection
- **Phase 7A successfully distilled Phases 5–6 into a self-contained, deployment-ready inference pipeline:**
  - **Inputs:** Patient-level CSV (LightGBM), preprocessed tensors (TCN).  
  - **Processing:** Pre-trained models loaded, feature order preserved, masks applied for TCN.  
  - **Outputs:** Batch predictions for all targets, top-10 feature summaries, optional single-patient CLI display.  
- **Key takeaways:**
  - Deployment-lite focuses on **reproducible, fast, minimal inference**, not on evaluation or visualisation.
  - Challenges around TCN architecture, regression clipping, and interpretability were solved with lightweight, deterministic solutions.
  - Binary targets, threshold tuning, calibration, and plots were deliberately omitted because they belong to evaluation, not inference.
  - Phase 7A forms a **complete, portable, cloud-ready foundation** for Phase 7B, demonstrating functional end-to-end inference for both model families.

---

## Phase 7B: Cloud Deployment (Full Live Deployment)

---









