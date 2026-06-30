# Master Prompt: Water Pump Functionality Prediction — End-to-End ML Project (v2, Production-Grade)

Use this prompt with Claude Code (or any coding agent with file/bash access) inside your project folder, with `features.csv` and `labels.csv` already placed in `data/raw/`. This version upgrades the original notebook-only plan into a full production-grade repo: modular `src/` code, tests, CI, and packaging — while keeping the notebook workflow your README specifies.

---

## PROMPT

You are a Senior Machine Learning Engineer, Data Scientist, and Technical Writer. Build a COMPLETE, production-quality, end-to-end machine learning project from scratch for **Water Pump Functionality Prediction** — a supervised multiclass classification task predicting the operational status of Tanzanian water pumps (`functional`, `functional needs repair`, `non functional`) from `features.csv` + `labels.csv`, merged on `id`. Do not leave placeholders anywhere — every file must contain real, working, executable code. The project must run end-to-end after `pip install -r requirements.txt && python main.py`, and the web app must run via `python app/app.py`.

### 0. Repository Structure

Create exactly this structure:

```
water-pump-prediction/
├── data/
│   ├── raw/                    # features.csv, labels.csv (already provided)
│   └── processed/              # train_clean.csv output
├── notebooks/
│   ├── 01_data_loading.ipynb
│   ├── 02_cleaning.ipynb
│   ├── 03_eda.ipynb
│   ├── 04_modelling.ipynb
│   └── 05_evaluation.ipynb
├── src/
│   ├── __init__.py
│   ├── config.py               # paths, constants, random_state, target classes
│   ├── data_loader.py          # load + merge features/labels
│   ├── preprocessing.py        # cleaning functions
│   ├── feature_engineering.py  # pump_age, date parts, cardinality reduction
│   ├── eda.py                  # reusable EDA/Cramér's V functions
│   ├── visualization.py        # plotting helpers, saves to reports/figures
│   ├── model_training.py       # pipeline + model definitions, CV, tuning
│   ├── evaluation.py           # metrics, confusion matrix, reports
│   ├── prediction.py           # load saved pipeline, predict on new input
│   └── utils.py                # logging setup, generic helpers
├── models/                     # best_model.pkl, label_encoder.pkl, metadata.json
├── reports/
│   └── figures/
├── app/
│   ├── app.py
│   ├── templates/
│   │   └── index.html
│   └── static/                 # css/ (bootstrap or custom)
├── tests/
│   ├── test_data_loader.py
│   ├── test_preprocessing.py
│   └── test_prediction.py
├── .github/
│   └── workflows/ci.yml        # lint + pytest on push/PR
├── requirements.txt
├── README.md
├── LICENSE                     # MIT
├── .gitignore
└── main.py                     # runs the full pipeline: load → clean → train → save
```

All `src/` modules use **type hints, docstrings, PEP8 formatting, logging (not print), and explicit exception handling**. No business logic should live only in notebooks — notebooks import from `src/` and call functions for anything reusable; raw exploration/plots can stay inline in the notebook.

### 1. `src/config.py`

Centralize: file paths (`RAW_DIR`, `PROCESSED_DIR`, `MODELS_DIR`, `FIGURES_DIR`), `RANDOM_STATE = 42`, `TARGET_COL = "status_group"`, `TARGET_CLASSES`, `TEST_SIZE`, `CV_FOLDS = 5`, and any column-group constants (e.g., redundant column groups, high-cardinality columns).

### 2. Notebook 01 — Data Loading & Exploration (`01_data_loading.ipynb`, uses `src/data_loader.py`)

- Load `features.csv` and `labels.csv`, merge on `id` (first step, no analysis before this).
- Inspect: shape, columns, dtypes, head/tail, missing values, duplicate rows, duplicate/near-duplicate columns, target distribution, unique value counts per column, summary statistics (`describe()` for numeric and categorical).

### 3. Notebook 02 — Data Cleaning & Feature Engineering (`02_cleaning.ipynb`, uses `src/preprocessing.py` + `src/feature_engineering.py`)

- Handle missing values per column with documented rationale (impute / flag-as-unknown / drop).
- Handle duplicate rows and any duplicate/redundant columns.
- Fix incorrect dtypes (e.g., `public_meeting`, `permit` to bool; `region_code`/`district_code` to category, not numeric).
- Treat zero/invalid values as missing where appropriate: zero lat/lon (ocean coordinates), `construction_year == 0`, anomalous `population` values.
- Resolve redundant column groups (extraction_type tiers, payment tiers, water_quality tiers, quantity tiers, waterpoint_type tiers, management tiers, source tiers) — keep the most useful granularity per group, drop the rest, document why.
- Reduce high-cardinality categoricals (`funder`, `installer`, `subvillage`, `wpt_name`, `scheme_name`) via top-N + "other" bucketing.
- Convert `date_recorded` to datetime; engineer `record_year`, `record_month`, `record_day`, `record_weekday`, and `pump_age = record_year - construction_year` (guarding against invalid construction years).
- Drop unneeded identifier columns where appropriate (e.g., `recorded_by` if constant, `id` from the modelling frame after merge is done).
- Encode `status_group` to numeric; save the encoder.
- Save cleaned data to `data/processed/train_clean.csv`.
- Document every cleaning decision in markdown.

### 4. Notebook 03 — EDA (`03_eda.ipynb`, uses `src/eda.py` + `src/visualization.py`)

Produce publication-quality plots, save each to `reports/figures/`, and write a 2–4 sentence insight after every figure:
- Target class distribution + imbalance analysis
- Histograms and boxplots for key numeric features
- Countplots for key categorical features
- Violin plots of numeric features by target class
- Missing-value heatmap (pre-cleaning, for documentation)
- Numeric correlation heatmap
- Cramér's V matrix/ranking for categorical features vs. target (implement in `src/eda.py` via scipy chi-squared contingency)
- Geographic scatter (latitude/longitude colored by target class)
- A short "most predictive features" summary that drives Step 5 feature selection

### 5. Feature Engineering Pipeline (`src/feature_engineering.py`, used in Notebook 04)

- Split numeric vs. categorical column lists based on EDA findings (exclude dropped/redundant columns).
- Build a `ColumnTransformer`: numeric → `SimpleImputer` + `StandardScaler`; categorical → `SimpleImputer(strategy="most_frequent")` + `OneHotEncoder(handle_unknown="ignore")`.
- Wrap as a reusable `build_preprocessor()` function importable by both training and inference code.

### 6. Notebook 04 — Modelling (`04_modelling.ipynb`, uses `src/model_training.py`)

- Stratified train/test split.
- Train, inside full `Pipeline`s (preprocessor + classifier): Logistic Regression, Decision Tree, Random Forest, Gradient Boosting, Extra Trees, and XGBoost/LightGBM/CatBoost if installed (gracefully skip with a logged message if a library isn't available — don't hard-fail).
- 5-fold stratified cross-validation per model; report accuracy, precision, recall, weighted F1, macro F1 in a comparison table.
- Hyperparameter-tune the top 1–2 models with `GridSearchCV` or `RandomizedSearchCV`; compare tuned vs. untuned performance explicitly.

### 7. Notebook 05 — Evaluation (`05_evaluation.ipynb`, uses `src/evaluation.py`)

- Confusion matrix and full classification report per model on the held-out test set.
- ROC-AUC (one-vs-rest) and precision-recall curves where applicable for multiclass.
- Learning curves for the best model.
- Cross-validation score comparison plot across models.
- Feature importance (model-native) and permutation importance; SHAP summary plot if `shap` is available (skip gracefully otherwise).
- Select and justify the final best model (metric trade-offs, per-class performance, interpretability vs. accuracy).
- Save the final pipeline (`models/best_model.pkl`), label encoder (`models/label_encoder.pkl`), and a `models/metadata.json` (feature list, model name, training date, key metrics) via `joblib`/`json`.

### 8. `main.py`

A single entry point that runs the full pipeline non-interactively: load data → clean → engineer features → train → evaluate → save best model. Should use `src/` modules only (no notebook dependency) and log progress at each stage.

### 9. Flask App (`app/`)

- `app/app.py`: loads `models/best_model.pkl`, `models/label_encoder.pkl`, and `models/metadata.json` at startup; `GET /` renders the form (only the most impactful features from EDA, not all 40 columns); `POST /predict` validates input, builds a one-row DataFrame matching the pipeline's expected schema, predicts, decodes the label, and re-renders with the result. Include basic input validation and error handling (e.g., missing/invalid fields → friendly error message, not a stack trace).
- `app/templates/index.html`: single responsive page using Bootstrap (via CDN), with a form section and a conditionally-rendered prediction-result section.
- `app/static/`: any custom CSS.

### 10. Tests (`tests/`)

Use `pytest`. At minimum:
- `test_data_loader.py`: merge logic produces expected shape/columns on a small fixture.
- `test_preprocessing.py`: cleaning functions handle missing/zero/invalid values as expected.
- `test_prediction.py`: the saved pipeline (or a stub) returns a valid class label for a well-formed input row.

### 11. CI (`.github/workflows/ci.yml`)

GitHub Actions workflow that on push/PR: sets up Python, installs `requirements.txt`, runs `pytest`, and optionally lints with `flake8`/`ruff`.

### 12. Packaging & Repo Hygiene

- `requirements.txt`: pin all libraries actually used (pandas, numpy, scikit-learn, matplotlib, seaborn, scipy, joblib, flask, jupyter, nbformat, pytest, flake8/ruff, optionally xgboost/lightgbm/catboost/shap).
- `.gitignore`: standard Python + venv + `.ipynb_checkpoints` + OS files (optionally exclude large `models/*.pkl` if size is a concern — note this as a choice).
- `LICENSE`: MIT.
- `environment.yml` (optional, for conda users).

### 13. README.md

Write a professional README with: Project Overview, Business Problem, Dataset Description, Installation, Folder Structure, Workflow, EDA Summary (key findings), Feature Engineering summary, Models trained + results table, Deployment instructions, Screenshot placeholders (`![screenshot](reports/figures/...)`), Future Improvements, References, and the completed deliverables checklist.

### Constraints & Style
- Real numbers only — run the actual code and report actual metrics; never fabricate results.
- Modular first: anything reused across notebooks belongs in `src/`, not copy-pasted.
- Graceful degradation: optional libraries (xgboost, shap, etc.) should not break the pipeline if absent — log a warning and skip.
- Keep the Flask form usable and focused, not a 40-field wall.
- Finish by initializing git (if needed), committing all files, creating the GitHub repo if it doesn't exist, and pushing to `main`.

---

*End of prompt.*
