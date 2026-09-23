"""
Data preprocessing functions for the <30-day hospital readmission prediction project
(Diabetes 130-US hospitals dataset). Used in notebooks/01_data_eda.ipynb.

"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

# ---- 1. Load data ----
def load_raw_data(input_dir="data/input"):
    df = pd.read_csv(f"{input_dir}/diabetic_data.csv")
    ids_mapping = pd.read_csv(f"{input_dir}/IDS_mapping.csv")
    return df, ids_mapping


# ---- 2. Cleaning RULE-BASED
EXPIRED_HOSPICE_IDS = [11, 13, 14, 19, 20, 21]  # from IDS_mapping.csv (Expired / Hospice)

def basic_clean(df):
    """
    Replace '?' with NaN and remove records of patients who expired or were transferred to hospice.
    These two steps are based only on fixed values rather than statistics learned from
    the entire dataset, so they can be safely applied before the train-test split
    without causing data leakage
    """
    df = df.copy()
    df = df.replace("?", np.nan)
    n_before = df.shape[0]
    df = df[~df["discharge_disposition_id"].isin(EXPIRED_HOSPICE_IDS)]
    print(f"[basic_clean] Removed {n_before - df.shape[0]} rows (expired/hospice). "
          f"{df.shape[0]} rows remaining.")
    return df


# ---- 3. Steps that MUST be fitted on train, then applied to test ----
NEAR_CONSTANT_CANDIDATES = [
    "examide", "citoglipton", "acetohexamide", "troglitazone",
    "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone",
]

def fit_near_constant_cols(train_df, candidates=NEAR_CONSTANT_CANDIDATES, threshold=0.99):
    """Inspect only the TRAIN set to determine which columns are nearly constant
    (>= threshold having the same value) and should be dropped. This prevents
    the decision from being influenced by the distribution of the test set"""
    dropped_cols = []
    for col in candidates:
        if col in train_df.columns:
            top_freq = train_df[col].value_counts(normalize=True, dropna=False).iloc[0]
            if top_freq >= threshold:
                dropped_cols.append(col)
    print(f"[fit_near_constant_cols] Columns to drop (fitted on train): {dropped_cols}")
    return dropped_cols


def drop_cols(df, cols):
    return df.drop(columns=[c for c in cols if c in df.columns])


OUTLIER_COLS = ["num_medications", "num_lab_procedures", "number_diagnoses"]

def fit_outlier_caps(train_df, cols=OUTLIER_COLS, factor=1.5):
    """Calculate capping thresholds using the IQR method ONLY on the train set
    to prevent the test set from influencing the thresholds"""
    caps = {}
    for col in cols:
        if col in train_df.columns:
            q1, q3 = train_df[col].quantile([0.25, 0.75])
            iqr = q3 - q1
            lower = max(train_df[col].min(), q1 - factor * iqr)
            upper = q3 + factor * iqr
            caps[col] = (lower, upper)
    print(f"[fit_outlier_caps] Capping thresholds (fitted on train): {caps}")
    return caps


def apply_outlier_caps(df, caps):
    df = df.copy()
    for col, (lower, upper) in caps.items():
        if col in df.columns:
            n_capped = ((df[col] < lower) | (df[col] > upper)).sum()
            if n_capped > 0:
                print(f"[apply_outlier_caps] {col}: capped {n_capped} rows "
                      f"outside [{lower:.1f}, {upper:.1f}]")
            df[col] = df[col].clip(lower=lower, upper=upper)
    return df


CATEGORICAL_COLS = [
    "race", "gender", "age", "admission_type_id", "discharge_disposition_id",
    "admission_source_id", "payer_code", "medical_specialty",
    "max_glu_serum", "A1Cresult", "change", "diabetesMed",
]

def fit_categories(train_df, cols=CATEGORICAL_COLS):
    """Record the set of category values for each categorical column using ONLY
    the train set. This ensures that the test set uses the same category set as
    the train set and prevents the category dtype from being inferred from the
    test set"""
    categories = {}
    for col in cols:
        if col in train_df.columns:
            categories[col] = sorted(train_df[col].dropna().astype(str).unique().tolist())
    return categories


def apply_categories(df, categories, unseen_label="Other_unseen"):
    """Apply the category set learned from the train set to a dataframe
    (train or test). Values in the test set that were never observed in the
    train set are grouped under 'unseen_label' instead of being converted
    to NaN, preventing silent data loss"""
    df = df.copy()
    for col, cats in categories.items():
        if col in df.columns:
            cats_with_fallback = list(cats) + [unseen_label]
            vals = df[col].astype(str)
            vals = vals.where(vals.isin(cats), unseen_label)
            df[col] = pd.Categorical(vals, categories=cats_with_fallback)
    return df


# ---- 4. Missing values: fill with fixed values (not learned from the data) ----
def handle_missing_values(df):
    """All replacement values are fixed labels ('Unknown', 'Not tested')
    rather than statistics calculated from the data (e.g., mean, median, or mode).
    Therefore, this function can safely be applied before or after the split,
    and produces consistent results on both train and test sets"""
    df = df.copy()
    if "weight" in df.columns:
        df = df.drop(columns=["weight"])

    for col in ["payer_code", "medical_specialty", "race"]:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown")

    for col in ["max_glu_serum", "A1Cresult"]:
        if col in df.columns:
            df[col] = df[col].fillna("Not tested")

    remaining_na = df.isna().sum()
    remaining_na = remaining_na[remaining_na > 0]
    if len(remaining_na):
        print(f"[handle_missing_values] Columns with remaining NaN values:\n{remaining_na}")
    return df


# ---- 5. Target (rule-based) ----
def create_target(df):
    df = df.copy()
    df["target"] = (df["readmitted"] == "<30").astype(int)
    df = df.drop(columns=["readmitted"])
    return df

# ---- Group Unknown / Not Available codes ----
UNKNOWN_CODES = {
    "discharge_disposition_id": [18, 25, 26],
    "admission_source_id": [9, 15, 17, 20, 21],
    "admission_type_id": [5, 6, 8],
}

def group_unknown_ids(df):
    df = df.copy()

    for col, codes in UNKNOWN_CODES.items():
        if col in df.columns:
            df[col] = df[col].where(
                ~df[col].isin(codes),
                "Unknown"
            )

    return df


# ---- 7. ICD-9 grouping cho diag_1 / diag_2 / diag_3 (rule-based, by chapter ICD-9-CM) ----
def _map_single_icd9(code):
    if pd.isna(code):
        return "Missing"

    code = str(code).strip()
    if not code:
        return "Missing"

    # V-code / E-code: keep them separate instead of grouping them into "Other"
    if code.startswith("V"):
        return "Supplementary_V"
    if code.startswith("E"):
        return "External_cause_E"

    try:
        code_num = float(code)
    except ValueError:
        return "Other"  # Invalid code format that cannot be parsed

    # ---  Special groups retained from the current approach (Strack et al.) ---
    if 250 <= code_num < 251:
        return "Diabetes"
    if 390 <= code_num <= 459 or code_num == 785:
        return "Circulatory"
    if 460 <= code_num <= 519 or code_num == 786:
        return "Respiratory"
    if 520 <= code_num <= 579 or code_num == 787:
        return "Digestive"
    if 800 <= code_num <= 999:
        return "Injury"
    if 710 <= code_num <= 739:
        return "Musculoskeletal"
    if 580 <= code_num <= 629 or code_num == 788:
        return "Genitourinary"
    if 140 <= code_num <= 239:
        return "Neoplasms"

    # --- Further divide the remaining codes by ICD-9-CM chapter instead of
    # grouping everything into "Other" ---
        return "Infectious"
    if 240 <= code_num <= 279:
        return "Endocrine_other"
    if 280 <= code_num <= 289:
        return "Blood"
    if 290 <= code_num <= 319:
        return "Mental"
    if 320 <= code_num <= 389:
        return "Nervous_SenseOrgans"
    if 630 <= code_num <= 677:
        return "Pregnancy_Childbirth"
    if 680 <= code_num <= 709:
        return "Skin"
    if 740 <= code_num <= 759:
        return "Congenital"
    if 760 <= code_num <= 779:
        return "Perinatal"
    if 780 <= code_num <= 799:          
        return "Symptoms_signs"

    return "Other"


def add_diag_groups(df):
    df = df.copy()
    for col in ["diag_1", "diag_2", "diag_3"]:
        if col in df.columns:
            df[f"{col}_group"] = df[col].apply(_map_single_icd9)
    return df


# ---- Split by patient_nbr (prevent leakage between admissions of the same patient) ----
def split_train_test(df, test_size=0.2, random_state=42, stratify_col="target"):
    splitter = GroupShuffleSplit(test_size=test_size, n_splits=1, random_state=random_state)
    train_idx, test_idx = next(splitter.split(df, groups=df["patient_nbr"]))
    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)
    print(f"[split_train_test] Train: {train_df.shape[0]} rows / Test: {test_df.shape[0]} rows")
    if stratify_col in df.columns:
        print(f"[split_train_test] Target rate - train: {train_df[stratify_col].mean():.4f}, "
              f"test: {test_df[stratify_col].mean():.4f}  "
              f"(GroupShuffleSplit does not support true stratification when grouping by "
              f"patient_nbr, so the target rate is checked after splitting; if the difference "
              f"is substantial, consider StratifiedGroupKFold in sklearn >=1.0)")
    return train_df, test_df


# ---- 9. Full pipeline: split first, fit data-dependent steps ONLY on train ----
def run_full_pipeline(input_dir="data/input", test_size=0.2, random_state=42):
    df, ids_mapping = load_raw_data(input_dir)

    # (a) Rule-based steps that are safe to apply before the split
    df = basic_clean(df)
    df = create_target(df)
    df = group_unknown_ids(df)
    df = add_diag_groups(df)

    df = df.drop(columns=["diag_1", "diag_2", "diag_3"])

    # (b) Split BEFORE fitting any statistics from the data
    train_df, test_df = split_train_test(df, test_size=test_size, random_state=random_state)

    # (c) Fit ONLY on train, then transform both train and test
    dropped_cols = fit_near_constant_cols(train_df)
    train_df = drop_cols(train_df, dropped_cols)
    test_df = drop_cols(test_df, dropped_cols)

    train_df = handle_missing_values(train_df)
    test_df = handle_missing_values(test_df)

    categories = fit_categories(train_df)
    train_df = apply_categories(train_df, categories)
    test_df = apply_categories(test_df, categories)

    outlier_caps = fit_outlier_caps(train_df)
    train_df = apply_outlier_caps(train_df, outlier_caps)
    test_df = apply_outlier_caps(test_df, outlier_caps)

    for col in categories:
        n_unseen = (test_df[col] == "Other_unseen").sum()
        if n_unseen > 0:
            print(f"[run_full_pipeline] Warning: column '{col}' contains values in the test set "
                  f"that were not observed in the train set "
                  f"({n_unseen} rows assigned the 'Other_unseen' label).")

    return train_df, test_df, ids_mapping