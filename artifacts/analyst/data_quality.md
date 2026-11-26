# ShopIntent AI · Data quality report

**Verdict: ✅ Ready for the Data Scientist crew** (decided by the validation tools)

_Explained by the Data Validator agent (gpt-4.1-mini) from `data_quality.json` (cb88f2bcc1d2)._

## Summary

The dataset contains 12,330 rows and 18 columns with no missing values and all contract checks passed. Duplicate rows were retained as they represent valid single-page bounce sessions from different users. Some month values were standardized for consistency, and the dataset includes a purchase rate of 15.5%, indicating class imbalance handled during modeling.

## Findings

### 1. Duplicate Rows Represent Valid Sessions

There are 125 identical rows in the raw data which were kept because they represent single-page bounce sessions from different users.

**Why it matters:** Retaining these duplicates ensures the dataset accurately reflects user behavior without artificially reducing data size, preserving model training quality.

### 2. Month Values Standardized for Consistency

The month value 'June' was renamed to 'Jun' in 288 rows to maintain consistent categorical values across the dataset.

**Why it matters:** Consistent categorical values prevent errors in encoding and improve model interpretability and performance.

### 3. No Missing Values and Contract Compliance

The dataset has zero missing values and passes all contract checks including presence of all 18 columns, 17 required features, numeric ranges within bounds, and categorical values within allowed values.

**Why it matters:** Complete and contract-compliant data ensures reliability and validity for downstream analysis and modeling.

### 4. Class Imbalance Present in Purchase Sessions

Only 15.5% of sessions ended in a purchase, indicating an imbalanced target variable.

**Why it matters:** Class imbalance requires stratified splits and appropriate metrics like PR-AUC to ensure fair and effective model evaluation.

### 5. Sessions with Zero Duration and Very Long Sessions Kept

720 sessions have zero total time (single-page visits) and 78 sessions last over 3 hours; both were retained as the model uses robust trees.

**Why it matters:** Keeping these sessions preserves data diversity and allows the model to learn from all user behaviors without bias from outlier removal.

## Cleaning steps

| Tool record | Explanation |
|---|---|
| Month: renamed 'June' -> 'Jun' in 288 rows | The month value 'June' was renamed to 'Jun' in 288 rows to standardize categorical data and avoid inconsistencies during analysis. |
| Kept 125 identical rows (single-page bounce sessions from different users) | 125 identical rows were kept because they represent valid single-page bounce sessions from different users, ensuring data completeness. |
| Final shape: 12330 rows x 18 columns | The final dataset shape remains 12,330 rows by 18 columns after cleaning, confirming no data loss occurred during processing. |

## Checks run by the tools

- ✅ All 18 contract columns present, no extra columns
- ✅ All 17 required features present
- ✅ No missing values
- ✅ Numeric ranges within contract bounds
- ✅ Categorical values within allowed_values
- ✅ Administrative: 0 pages but time recorded: 0 sessions have Administrative = 0 but Administrative_Duration > 0
- ✅ Informational: 0 pages but time recorded: 0 sessions have Informational = 0 but Informational_Duration > 0
- ✅ ProductRelated: 0 pages but time recorded: 0 sessions have ProductRelated = 0 but ProductRelated_Duration > 0
- ✅ Very long sessions (over 3 hours): 78 sessions last over 3 hours in total; kept, the model uses robust trees
- ✅ Sessions with no time recorded: 720 sessions have 0 seconds in total (single-page visits); kept
- ✅ Class balance: 15.5% of sessions are purchases; imbalanced, so splits are stratified and PR-AUC is reported
