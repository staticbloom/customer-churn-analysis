# Telco Customer Churn Classification

## Objective and Context

Predict customer churn from service, billing and customer attributes in the supplied v2 extract. The [notebook](notebooks/Telco-Churn.ipynb) compares five classifiers, selects an F1 threshold on development data, and interprets the selected Logistic Regression model.

## Data and Model Availability

The saved v2 model supports inference using the synthetic example below. Notebook training requires the matching v2 extract; see [data instructions](data/README.md). Source data and customer-level prediction exports are excluded from Git.

## Dataset and Source

The target is `Churn`, with `Yes` as the positive class. F1 is the primary metric; precision, recall, ROC-AUC, average precision and accuracy provide context.

The notebook expects `data/WA_Fn-UseC_-Telco-Customer-Churn v2.csv`. The source CSV is excluded from Git; `data/README.md` and the synthetic example remain in the repository. It includes support, billing, usage, reliability, NPS and competition fields beyond the standard IBM Telco dataset. A standard IBM download is not a drop-in replacement. Obtain the matching v2 extract and confirm its provenance and field definitions before reproducing the analysis.

## Methodology

Cleaning standardizes text, removes exact duplicates and checks customer IDs and target labels. Row-level features summarize support escalation, customer friction, reliability, tenure and charges. Learned imputation, scaling and categorical encoding remain inside training pipelines; CatBoost handles categories directly.

The stratified split is 40% training, 20% validation, 20% test and 20% reserve, with random state 42. Preliminary validation compares Logistic Regression, Random Forest, XGBoost, CatBoost and LightGBM. Five-fold out-of-fold predictions on the combined training/validation data then select the final model and threshold. The reserve is unused. Test results for all locked models are diagnostics, not the selection rule.

## Results

The saved Logistic Regression model uses `C=1.0` and threshold 0.353. Its recorded test metrics are:

| Metric | Test result |
|---|---:|
| Accuracy | 0.841 |
| Precision | 0.681 |
| Recall | 0.745 |
| F1 | 0.711 |
| ROC-AUC | 0.895 |
| Average precision | 0.793 |

These values match the [saved metadata](models/telco_churn_v2_40_20_20_model_metadata.json) and [test table](table/telco_churn_v2_final_test_metrics.csv). They describe this split and exceed the 0.70 F1 objective; they do not establish future retention performance.

## Installation and Execution

The existing environment uses Python 3.14.7. From this project directory:

```bash
python3.14 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip check
python -m jupyterlab notebooks/Telco-Churn.ipynb
```

Choose the project environment and use **Restart Kernel and Run All**. The kernel may start in the project root or `notebooks/`. A full run trains and tunes models and replaces generated tables, figures and model files; it is unnecessary just to review the saved results.

To score a matching v2 customer CSV with the saved model:

```bash
python predict.py data/example_customers_synthetic.csv --output outputs/customer_predictions.csv
```

The script checks required source columns, constructs features and applies the saved threshold. `customerID` is optional for scoring and `Churn` is not used as an input feature. The example contains one synthetic, unlabeled record for demonstrating the input schema. The default model path is relative to the script; input/output paths are relative to the working directory. Existing output files are not overwritten. Load only trusted model files.

### Reuse and Future Versions

Keep the 36 source fields listed in [`RAW_FEATURE_COLUMNS`](predict.py); the fitted pipeline expects the resulting 40 features. Extra CSV columns are ignored. Text is trimmed, blank inputs become missing values, and the fitted pipeline handles missing values and unseen categories. Numeric values must be finite when present. Charges, tenure, counts and day intervals must be nonnegative; the three indicator fields use 0/1. `nps_score` uses the extract's individual 0–10 scale, not an aggregate −100 to 100 NPS value. Dates must parse even though `contract_end_date` is excluded from modeling.

Use the same categorical spellings and numeric units as the supplied extract. `MonthlyCharges` has no verified currency declaration here. The measurement definitions, percentage-change conventions, customer observation date, churn horizon, and generation process for the added support/billing/usage/NPS/competition fields remain unverified. The column names and observed ranges are an input contract, not evidence that the variables were measured before churn. Confirm these points with the data supplier before applying the model to real customers.

The separate workflow accepts matching features plus `Churn=Yes/No`:

```bash
# Evaluate newly collected labels without fitting or changing the saved threshold.
python train.py new_customers.csv --evaluate-only --output-dir runs/customer_evaluation_v1

# Fit a new candidate with the same preprocessing and classifier settings.
python train.py new_customers.csv --output-dir runs/churn_candidate_v3
python predict.py data/example_customers_synthetic.csv --model runs/churn_candidate_v3/model.joblib --output outputs/candidate_predictions.csv
```

`train.py` requires a new output directory. It rejects duplicate customer IDs and repeated feature records, then uses a separate stratified 60/20/20 train/validation/test split with seed 42. The classifier fits on training rows, the threshold maximizes validation F1, and the test is evaluated once. It does not refit after evaluation. The report records source/model hashes, package versions, splits, a prevalence baseline, classification metrics and Brier score. This is a future-candidate protocol; it does not reproduce or replace the original 40/20/20/20 study.

Use new data with a prespecified evaluation protocol. Random splitting is appropriate only if customer observations can be treated as independent; repeated customers or time-dependent deployment need a group or chronological design before training. Review subgroup errors, probability calibration and retention costs before promoting a candidate. Brier score alone does not establish calibration. Changing the target or source schema requires adapting the analysis, not merely loading this model.

## Visualizations and Interpretation

See `visuals/telco_churn_v2_final_test_confusion_matrix.png`, ROC/precision–recall plots and the selected-model coefficient/SHAP figures. The [HTML report](Telco-Churn.html) provides a static view of the analysis. Individual-customer feature tables and the local explanation figure are omitted from the public notebook and HTML; local runs save these only to ignored files. Coefficients and SHAP describe model associations, not causal retention effects.

## Repository Structure

- `notebooks/Telco-Churn.ipynb`: active analysis and training workflow.
- `predict.py`: standalone scoring script.
- `train.py`: separate candidate training and labeled-data evaluation.
- `data/`: tracked instructions and a synthetic example; source extracts remain local.
- `models/telco_churn_v2_40_20_20_model.joblib`: selected model package.
- `models/telco_churn_v2_40_20_20_model_metadata.json`: selected settings and results.
- `table/`: exported analysis and evaluation tables; the existing singular name is retained.
- `visuals/`: saved figures.
- `outputs/`: generated scoring results.
- `Telco-Churn.html`: static notebook export.
- `requirements.txt`: pinned analysis and inference dependencies.

Other model files are retained historical artifacts; the scoring script defaults to the explicitly named v2 package above.

## Limitations and Reproducibility

The model provides a useful baseline on the supplied split. Full-source EDA includes later test customers, and development scores also serve model and threshold selection, so they are not independent estimates. The code keeps test labels out of fitting and threshold optimization; it cannot establish what influenced earlier human decisions.

Verify that all v2 fields were available before churn was observed. The random split does not test future-period generalization, probability calibration or the cost of retention actions. SHAP values and coefficients describe associations rather than causes. Further decisions should use a new evaluation protocol and newly collected data, without tuning against the existing test results.


Package versions match the recorded local environment. Installation in a fresh environment has not been established.
