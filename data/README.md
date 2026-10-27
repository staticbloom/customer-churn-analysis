# Data: telecom customer churn

The source dataset is not included in this repository. This directory is retained for documentation, directory placeholders, and explicitly identified synthetic examples or reproducibility metadata. Raw data and processed source records remain local.

Place the matching comma-delimited UTF-8 extract at `data/WA_Fn-UseC_-Telco-Customer-Churn v2.csv`. The notebook expects `customerID`, `Churn` (`Yes`/`No`) and the 36 source columns listed in `predict.py:RAW_FEATURE_COLUMNS`. The local file has 4,999 rows; numeric and categorical definitions are described in the root README and inference validator. `TotalCharges` is not part of this v2 contract.

`WA_Fn-UseC_-Telco-Customer-Churn 1.csv` is a historical extract, not the notebook input. `example_customers_synthetic.csv` contains an unlabeled synthetic input. The source-derived `example_customers.csv` is excluded from Git.

No verified download URL, redistribution license or generation process was found for the augmented v2 dataset. Request the exact extract and permission from its supplier; a standard IBM Telco CSV lacks the added columns and cannot reproduce this study. Confirm that added fields existed before churn was observed.

Both source CSVs and row-level prediction/SHAP exports are excluded from Git. Saved model inference works after cloning with the synthetic example. Training the original notebook requires the matching v2 extract. The separate `train.py` accepts independently supplied matching labeled data and does not reproduce the original study split.
