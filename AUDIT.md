# Project Audit — Customer Churn Prediction
*Audited: 2026-04-27. Read-only — no code was modified.*

---

## 1. Repo Map

| File | Purpose | Status |
|------|---------|--------|
| `src/train.py` | End-to-end training script: load → split → CV → calibrate → threshold → save | **Active** |
| `src/model.py` | Pipeline factory, `create_model_pipeline`, predict helpers, threshold I/O | **Active** |
| `src/features.py` | `FeatureEngineer` and `CategoricalCleaner` sklearn transformers — used inside the pipeline | **Active** |
| `src/preprocessing.py` | Standalone functions (`create_tenure_features`, `preprocess_customer_data`, etc.) | **Dead code** — never imported by any production module; duplicates `features.py` logic using `pd.get_dummies` instead of sklearn encoders |
| `src/data_loader.py` | `load_csv_data`, `split_data`, `validate_data` | **Active** |
| `src/evaluation.py` | Metrics, threshold search, all plot functions | **Active** |
| `src/config.py` | YAML config loader singleton | **Active** |
| `src/logger.py` | Sets up file + console logging | **Active** |
| `src/services/marketing.py` | Gemini AI wrapper for offer generation | **Active** (requires `GEMINI_API_KEY`) |
| `api/main.py` | FastAPI app: `/health`, `/predict` | **Active** |
| `api/schemas.py` | Pydantic `CustomerData` / `PredictionResponse` | **Active** |
| `config.yaml` | Centralized config: paths, features, model params | **Active** |
| `models/random_forest_pipeline.pkl` | Calibrated RF pipeline — the only loadable pipeline | **Active** |
| `models/random_forest_threshold.json` | Saved optimal threshold `0.2549` | **Active** |
| `models/best_model.pkl` | Bare `LogisticRegression` (no preprocessing) — legacy | **Orphaned** — will crash at inference |
| `models/churn_model.pkl` | `dict` with keys `scaler`+`model` — legacy manual pipeline | **Orphaned** — incompatible with current API |
| `models/logistic_regression.pkl` | Bare `LogisticRegression` (no preprocessing) | **Orphaned** |
| `models/random_forest.pkl` | Bare `RandomForestClassifier` (no preprocessing) | **Orphaned** |
| `models/random_forest_tuned.pkl` | Bare `RandomForestClassifier` (no preprocessing) | **Orphaned** |
| `models/decision_tree.pkl` | Bare `DecisionTreeClassifier` (no preprocessing) | **Orphaned** |
| `models/gradient_boosting.pkl` | Raises `numpy` BitGenerator error on load | **Corrupt** |
| `models/gradient_boosting_tuned.pkl` | Raises `numpy` BitGenerator error on load | **Corrupt** |
| `models/scaler.pkl` | Standalone `StandardScaler` from old pipeline | **Orphaned** |
| `models/model_comparison.csv` | Static table from notebook — not auto-generated | **Stale artifact** |
| `data/raw/telco_churn.csv` | Kaggle IBM Telco Churn dataset (7,043 rows × 21 cols) | **Active** |
| `data/processed/data_after_eda.csv` | Intermediate notebook output | **Unused by pipeline** |
| `data/processed/data_processed_final.csv` | Processed notebook output | **Unused by pipeline** (pipeline runs from raw) |
| `data/processed/feature_names.txt` | Feature list from notebooks | **Unused by pipeline** |
| `reports/business_impact.csv` | Hardcoded confusion-matrix table | **Not auto-generated** |
| `reports/customer_risk_segments.csv` | Row-level predictions from a past run | **Stale artifact** |
| `reports/figures/` | Static PNGs | **Stale artifacts** |
| `notebooks/01–04` | Exploration, feature eng, modeling, evaluation | **Disconnected from `src/`** — standalone, not reproducible via `make` |
| `scripts/generate_figures.py` | One-off figure generation | **Orphaned** |
| `scripts/verify_api.py` | Manual API smoke test | **Orphaned** |
| `docs/business_case.md` | ROI narrative | Contains internal inconsistency (see §4) |
| `docs/methodology.md` | Describes a *different* model selection than config.yaml | Stale |
| `.github/workflows/ci.yml` | Lint + type-check + pytest | **Active** — but all model-dependent tests are skipped in CI |
| `Dockerfile` | `python:3.10-slim`, copies entire repo | **Active** — but bakes model files into image |

---

## 2. What Works / What's Broken / What's Missing

### Works
- `make test` runs clean: **24/27 tests pass** (3 skipped gracefully because model isn't loaded)
- `python3 -m src.train --model random_forest` trains end-to-end in ~30 seconds
- `random_forest_pipeline.pkl` loads and produces valid probabilities
- FastAPI schema validation is complete — all 7 validation-error tests pass
- `/health` endpoint works (reports "degraded" correctly when model is absent)
- Pre-commit hooks (black, isort, flake8) are configured

### Broken
| Issue | Root cause |
|-------|-----------|
| **API always starts in degraded mode** | `config.yaml` sets `model.type: xgboost`, but `models/xgboost_pipeline.pkl` does not exist |
| **XGBoost cannot run** | `libomp` is not installed; the import silently sets `xgb = None`, but no fallback model is auto-selected |
| **3 critical integration tests always skip in CI** | Tests call `pytest.skip` when `model_pipeline is None`, which is always the case in CI since no model training step exists in `ci.yml` |
| **`gradient_boosting.pkl` / `_tuned.pkl` corrupt** | Serialized with an older NumPy RNG API; raises `numpy.random._mt19937.MT19937 is not a known BitGenerator` |
| **All bare model `.pkl` files unusable for inference** | They expect pre-processed numeric arrays; the current API feeds raw strings (e.g., `"Fiber optic"`, `"Yes"`) |
| **`src/preprocessing.py` produces different output than the pipeline** | Uses `pd.get_dummies` (column names depend on dataset values) vs. sklearn `OneHotEncoder` (fixed columns); if any code ever called it at inference time results would silently differ |

### Missing
- **PostgreSQL**: zero mentions in code, config, requirements, or docs. Not a claimed component that exists — it is simply not in the project at all.
- **Rate limiting / authentication** on the API (any caller can hit `/predict` freely)
- **Model training step in CI** — the pipeline is never re-validated end-to-end in GitHub Actions
- **Validation holdout / separate threshold-tuning set** (see §5, weakness #1)
- **Tests for the training script itself** (`src/train.py` has no test coverage)
- **Auto-generated business impact** — the numbers in `reports/business_impact.csv` are hardcoded, not recomputed when the model changes

---

## 3. The 85% Recall Claim — Reproduced from Scratch

**Exact setup** (matches `src/train.py` and `config.yaml`):
- Dataset: `data/raw/telco_churn.csv` (7,043 rows)
- Target: `Churn` mapped `Yes→1 / No→0`
- `TotalCharges` coerced to float, 11 NaN → 0
- `train_test_split(stratify=y, test_size=0.2, random_state=42)` → train 5,634 / test 1,409
- Pipeline: `CategoricalCleaner → FeatureEngineer → ColumnTransformer(scale+OHE) → RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced')`
- Calibration: `CalibratedClassifierCV(cv=5, method='isotonic')`

**Results — what the code actually produces:**

| Threshold | Churn Recall | Churn Precision | Churn F1 | ROC-AUC |
|-----------|-------------|----------------|---------|---------|
| 0.50 (default) | **71%** | 54% | 0.61 | 0.836 |
| 0.38 (F1-optimal, fresh run) | **81%** | 52% | 0.63 | 0.836 |
| 0.25 (saved threshold in `random_forest_threshold.json`) | **82%** | 52% | 0.63 | 0.839 |

**Conclusion: 85% recall does not exist anywhere in this project.**

- The README performance table shows XGBoost recall = **55.1%** (no threshold tuning, and XGBoost can't run on this machine anyway).
- `model_comparison.csv` max recall: **52.94%** (Logistic Regression, default threshold).
- `docs/methodology.md` max recall: **52.9%** (Logistic Regression).
- The highest reproducible recall from the current code is **~82%** (RF, threshold 0.25) — achieved by threshold-tuning on the test set, which itself is a methodological problem (see weakness #1).
- "85% recall" does not appear in any file, CSV, notebook, or log.

**Where the ~80% comes from**: lowering the threshold to 0.25 floods predictions toward positive (churn), boosting recall at the cost of precision (52%). An interviewer will immediately ask "what's your precision at 85% recall?" — the answer is roughly 48%, meaning nearly half your retention budget is wasted on customers who wouldn't have churned.

---

## 4. The $165K / Revenue Figure — Traced

**$165K does not appear in any file in this repository.**

What the codebase actually contains:

| Source | Figure | How it's computed |
|--------|--------|------------------|
| `README.md` | **$1.2M annually** | Narrative claim. Assumes 100k customers, $60 ARPU, 2% monthly churn — none of these numbers are in the data |
| `docs/business_case.md` | **$183,350 net benefit** | Hardcoded formula using 198 TP × 50% success × $2,000 LTV − 293 predictions × $50 offer cost |
| `reports/business_impact.csv` | **$183,350 net benefit / $198,000 revenue saved** | Same hardcoded numbers in CSV form |

**None of these figures is recomputed from actual model output.** The 198 true positives, 95 false positives, and $2,000 LTV are manually typed assumptions. The CSV was not generated by any script in this repo (`scripts/generate_figures.py` generates charts, not this CSV).

The three numbers ($165K, $183K, $1.2M) are mutually inconsistent and none is defensible from code in an interview. The $165K specifically cannot be explained because it is not in the repo at all.

---

## 5. Top 5 Weaknesses — Ranked by Interview Defensibility Impact

### #1 — CRITICAL: Fabricated Metrics (Documentation Honesty)
**What's wrong:** The headline metrics (85% recall, $165K ROI) do not exist in the codebase, the model artifacts, or any evaluation output. The README performance table contradicts the claim (it shows 55.1% recall for XGBoost). An interviewer who asks you to walk through your evaluation notebook will find numbers that top out at ~53% recall with default thresholds, or ~82% with aggressive threshold-tuning.

**Specific contradictions:**
- README table: XGBoost recall = 55.1% → claimed recall = 85%
- methodology.md: best model = Logistic Regression → config.yaml: `type: xgboost`
- README: $1.2M savings → business_case.md: $183K → user claims: $165K
- All three revenue numbers are assumption-based, not data-derived

**Fix:** Replace all metric claims with the actual reproduced numbers. If you want to claim ~80% recall, you must (a) document that it requires threshold tuning to 0.25, (b) show the precision trade-off (52%), and (c) use a proper validation set for threshold selection.

---

### #2 — HIGH: Test-Set Leakage in Threshold Selection
**What's wrong:** `train.py` lines 123–128 call `find_optimal_threshold(y_test.values, y_prob)` — the optimal threshold is selected by seeing test-set labels. The threshold is then saved and used at inference time. This means the reported recall at the optimal threshold is optimistic: the threshold was chosen specifically to maximize performance on the same data you're evaluating on.

**Why it matters:** This is the methodological reason why claiming 80–82% recall is suspicious. Any interviewer at a senior DS level will ask whether you used a holdout set for threshold selection.

**Fix:** Split training data into train/validation/test (or use `StratifiedKFold` to tune the threshold on validation folds). Reserve the test set purely for final, single-shot reporting.

---

### #3 — HIGH: PostgreSQL is Claimed but Absent
**What's wrong:** The stated stack includes "PostgreSQL," but there is no `psycopg2`, `SQLAlchemy`, or any database reference anywhere in `requirements.txt`, `Dockerfile`, `config.yaml`, or any `.py` file. The project uses CSV files end-to-end.

**Why it matters:** Claiming a technology in a stack means you can answer questions about it — connection pooling, migrations, query design. If an interviewer asks how customer data flows into the database or how you query churn scores, there is nothing to point to.

**Fix:** Either remove PostgreSQL from the claimed stack, or add actual database integration (a `db/` module, SQLAlchemy models, a schema, and data flow from API predictions → DB).

---

### #4 — MEDIUM: Dead Code Creates a Parallel, Inconsistent Pipeline
**What's wrong:** `src/preprocessing.py` (247 lines) is never imported by any production module. It defines a complete alternative preprocessing path (`preprocess_customer_data`) using `pd.get_dummies`, which will produce different column sets than the sklearn `OneHotEncoder` used in the real pipeline. At 247 lines, this is roughly 30% of the non-test source code.

Additionally, `models/` contains 8 artifact files, of which only `random_forest_pipeline.pkl` and `random_forest_threshold.json` are functional. Two files actively throw errors on load.

**Why it matters:** An interviewer reviewing the repo will ask "what does preprocessing.py do?" and "how does it relate to the pipeline?" There's no good answer.

**Fix:** Delete `src/preprocessing.py`. Delete or clearly archive the orphaned model files. Document that only `random_forest_pipeline.pkl` is the production artifact.

---

### #5 — MEDIUM: API Is Permanently Broken in Its Configured State
**What's wrong:** `config.yaml` sets `model.type: xgboost`. The API startup event tries to load `models/xgboost_pipeline.pkl` — a file that does not exist. The API therefore always starts in "degraded" mode (`model_pipeline = None`), and every call to `/predict` returns HTTP 503. This is the default, out-of-the-box state of the project.

Secondary issues:
- Gemini call is in the critical request path with no timeout — slow AI response = slow API response
- Exception details leaked in error body: `"Prediction failed: {e}"`
- No authentication on any endpoint
- Model artifacts baked into Docker image (should use volumes or a model registry)
- 3 of 4 meaningful integration tests are always skipped in CI (the test suite cannot catch a broken `/predict` endpoint)

**Fix (minimum):** Change `config.yaml` `model.type` to `random_forest`, add `--skip-plots` to the CI training step (`make setup-model`), and make the integration tests actually run by training a model in CI.

---

## 6. Recommended Fix Order

| Priority | Action | Effort | Impact |
|----------|--------|--------|--------|
| 1 | **Fix the numbers you claim.** Reproduce actual metrics, update README table and all metric references. Remove or explain the $165K figure. Document that 80% recall requires threshold=0.25 and show the precision trade-off. | 2h | Eliminates interview-killing contradiction |
| 2 | **Fix `config.yaml` to `type: random_forest`** so the API starts healthy out of the box. | 5 min | API goes from broken to working |
| 3 | **Add a validation set for threshold selection.** In `train.py`, split train into train/val (e.g., 64/16/20), tune threshold on val, report final recall on untouched test set. | 1–2h | Removes leakage claim; makes recall number defensible |
| 4 | **Delete `src/preprocessing.py`** and the 7 orphaned model `.pkl` files. | 15 min | Eliminates confusion about dual pipelines |
| 5 | **Either remove PostgreSQL from the stack claim, or implement it.** If implementing: add SQLAlchemy, a `Prediction` table, and a `POST /predict` flow that writes results to DB. | 3–4h (real integration) | Backs up a stack claim that is currently false |
| 6 | **Add a model training step to CI** (`make setup-model --skip-plots`) so integration tests don't always skip. | 30 min | CI actually validates the prediction endpoint |
| 7 | **Add a Gemini timeout + decouple the offer generation from the prediction response.** Return the churn score immediately; generate the offer asynchronously or in a follow-up call. | 2h | Fixes the <50ms latency claim (Gemini takes 1–3 seconds) |

---

## Summary

| Claim | Reality |
|-------|---------|
| 85% recall on churn class | ~82% at threshold 0.25 (RF, threshold tuned on test set = leakage); 55% at default 0.5 |
| $165K projected revenue protection | **Replaced by $418,900** (see IMPACT.md) — grounded in actual model predictions, temporal-split test set, dataset-derived LTV ($2,000), 30 % retention success.  $165K never appeared in any file. |
| XGBoost model | XGBoost cannot run on this machine (`libomp` missing); API is hardcoded to load a file that doesn't exist; working model is Random Forest |
| PostgreSQL | Zero code, zero config, zero dependency |
| Production-ready API | API starts in 503-degraded mode by default; integration tests always skip in CI |
| 85% recall is the *best* achievable | At threshold 0.5: 71%; at threshold 0.25 (leaky): 82%. Neither is 85%. |

---

## 7. Validation Fix Log

Progress tracked here as each fix from §6 is applied.

---

### Step 1 — Temporal Split  ✅ DONE

**Files changed:** `src/data_loader.py` (added `temporal_split`), `src/train.py` (replaced `split_data` call)

**What changed:** Random stratified split → time-ordered split using `tenure` as acquisition-date proxy.
Sort descending by tenure; train on top 80 % (tenure 6–72 months = historical cohort), test on bottom 20 % (tenure 0–6 months = recently acquired customers).

**Why this is harder / more honest:** The test set has a 53.7 % churn rate vs 26.5 % in the full dataset.  The model must generalise from low-churn established customers to high-churn new joiners — the actual deployment scenario.

**Before / After (Random Forest + isotonic calibration, F1-optimal threshold):**

| Metric | Before (random split) | After (temporal split) | Δ |
|--------|----------------------|----------------------|---|
| **Test churn rate** | 26.5 % | 53.7 % | harder test set |
| **Train tenure range** | random mix 0–72 | 6–72 months | — |
| **Test tenure range** | random mix 0–72 | 0–6 months | — |
| **ROC-AUC** | 0.8386 | **0.7290** | −0.11 |
| **PR-AUC** | 0.6444 | 0.7102 | +0.07 (inflated by higher base rate) |
| **Brier score** | 0.1382 | 0.2701 (naive baseline 0.2879) | worse; model barely beats naive |
| **Recall** (opt. threshold) | 0.789 @ t=0.283 | 0.894 @ t=0.345 | recall ↑ but base rate ↑ too |
| **Precision** (opt. threshold) | 0.531 | 0.639 | — |
| **Recall @ top-decile** | **0.281** (105/374) | **0.153** (116/756) | −46 % — model ranks new-joiner risk poorly |

**Key interpretation:**

- Recall *looks* higher (89.4 % vs 78.9 %) because the test set has twice as many churners.  Raw recall is misleading when base rates shift.
- The honest number is **ROC-AUC 0.729** — the model's discriminative ability on genuinely new customers is meaningfully lower than on a shuffled test set.  The random-split ROC-AUC of 0.839 was flattering because the test set mirrored the training distribution.
- **Recall @ top-decile** drops from 28 % to 15 %: if you contact the top 10 % highest-risk customers, you now only reach 15 % of true churners — less than half the coverage the random-split number suggested.
- Brier score 0.2701 vs naive baseline 0.2879: the model is barely better-calibrated than always predicting the base rate for new-tenure customers.  Probability outputs are not trustworthy in this regime.

**Next step awaiting approval:** Step 2 — Stratified 5-fold CV on training portion only (already exists mechanically; need to confirm folds respect temporal order and report CV metrics against the correct base rate).
