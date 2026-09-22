"""
Ham xu ly du lieu cho du an du doan tai nhap vien <30 ngay
(Diabetes 130-US hospitals dataset). Dung trong notebooks/01_data_eda.ipynb.

"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GroupShuffleSplit

# ---- 1. Load du lieu goc ----
def load_raw_data(input_dir="data/input"):
    df = pd.read_csv(f"{input_dir}/diabetic_data.csv")
    ids_mapping = pd.read_csv(f"{input_dir}/IDS_mapping.csv")
    return df, ids_mapping


# ---- 2. Cleaning RULE-BASED
EXPIRED_HOSPICE_IDS = [11, 13, 14, 19, 20, 21]  # tu IDS_mapping.csv (Expired / Hospice)

def basic_clean(df):
    """Thay '?' bang NaN va loai cac dong benh nhan da mat / chuyen hospice.
    Day la 2 buoc chi dua tren gia tri co dinh (khong phai thong ke hoc tu
    toan bo du lieu), nen ap dung truoc khi split khong gay leakage.
    """
    df = df.copy()
    df = df.replace("?", np.nan)
    n_before = df.shape[0]
    df = df[~df["discharge_disposition_id"].isin(EXPIRED_HOSPICE_IDS)]
    print(f"[basic_clean] Da loai {n_before - df.shape[0]} dong (expired/hospice). "
          f"Con lai {df.shape[0]} dong.")
    return df


# ---- 3. Cac buoc PHAI fit tren train, roi ap dung lai cho test ----
NEAR_CONSTANT_CANDIDATES = [
    "examide", "citoglipton", "acetohexamide", "troglitazone",
    "glimepiride-pioglitazone", "metformin-rosiglitazone", "metformin-pioglitazone",
]

def fit_near_constant_cols(train_df, candidates=NEAR_CONSTANT_CANDIDATES, threshold=0.99):
    """Chi nhin vao TRAIN de quyet dinh cot nao gan nhu hang so (>= threshold
    cung 1 gia tri) va nen bi drop. Tranh viec quyet dinh nay bi anh huong
    boi phan phoi cua tap test."""
    dropped_cols = []
    for col in candidates:
        if col in train_df.columns:
            top_freq = train_df[col].value_counts(normalize=True, dropna=False).iloc[0]
            if top_freq >= threshold:
                dropped_cols.append(col)
    print(f"[fit_near_constant_cols] Cot se drop (fit tren train): {dropped_cols}")
    return dropped_cols


def drop_cols(df, cols):
    return df.drop(columns=[c for c in cols if c in df.columns])


CATEGORICAL_COLS = [
    "race", "gender", "age", "admission_type_id", "discharge_disposition_id",
    "admission_source_id", "payer_code", "medical_specialty",
    "max_glu_serum", "A1Cresult", "change", "diabetesMed",
]

def fit_categories(train_df, cols=CATEGORICAL_COLS):
    """Ghi lai tap gia tri (categories) cua tung cot categorical, CHI dua
    tren train. Dung de ep test theo dung tap category cua train, tranh
    truong hop dtype 'category' duoc suy ra tu ca test (vo tinh 'nhin thay'
    test truoc khi train)."""
    categories = {}
    for col in cols:
        if col in train_df.columns:
            categories[col] = sorted(train_df[col].dropna().astype(str).unique().tolist())
    return categories


def apply_categories(df, categories, unseen_label="Other_unseen"):
    """Ap tap category cua train len df (train hoac test). Gia tri o test
    ma train CHUA TUNG THAY se duoc gop vao nhan 'unseen_label' thay vi
    bi bien thanh NaN, giu du lieu khong bi mat mot cach am tham."""
    df = df.copy()
    for col, cats in categories.items():
        if col in df.columns:
            cats_with_fallback = list(cats) + [unseen_label]
            vals = df[col].astype(str)
            vals = vals.where(vals.isin(cats), unseen_label)
            df[col] = pd.Categorical(vals, categories=cats_with_fallback)
    return df


# ---- 4. Missing values: dien gia tri CO DINH (khong hoc tu du lieu) ----
def handle_missing_values(df):
    """Tat ca gia tri dien vao deu la hang so co dinh ('Unknown', 'Not tested'),
    khong phai thong ke tinh tu du lieu (vd mean/median/mode), nen ham nay
    an toan de goi truoc hoac sau khi split deu duoc, tren train hay test
    deu cho ket qua nhat quan."""
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
        print(f"[handle_missing_values] Cac cot con NaN sau xu ly:\n{remaining_na}")
    return df


# ---- 5. Target (rule-based) ----
def create_target(df):
    df = df.copy()
    df["target"] = (df["readmitted"] == "<30").astype(int)
    df = df.drop(columns=["readmitted"])
    return df

# ---- 6. Gop cac code Unknown/Not Available ----
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


# ---- 7. ICD-9 grouping cho diag_1 / diag_2 / diag_3 (rule-based, theo chapter ICD-9-CM) ----
def _map_single_icd9(code):
    if pd.isna(code):
        return "Missing"

    code = str(code).strip()
    if not code:
        return "Missing"

    # V-code / E-code: tach rieng thay vi gop chung vao "Other"
    if code.startswith("V"):
        return "Supplementary_V"
    if code.startswith("E"):
        return "External_cause_E"

    try:
        code_num = float(code)
    except ValueError:
        return "Other"  # code loi dinh dang, khong parse duoc

    # --- Cac nhom "dac thu" giu nguyen nhu ban dang lam (Strack et al.) ---
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

    # --- Tach nho phan con lai theo chapter ICD-9-CM, thay vi don het vao "Other" ---
    if 1 <= code_num <= 139:
        return "Infectious"
    if 240 <= code_num <= 279:          # 250 da tach o tren
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
    if 780 <= code_num <= 799:          # 785-788 da tach o tren
        return "Symptoms_signs"

    return "Other"  # fallback that su hiem gap


def add_diag_groups(df):
    df = df.copy()
    for col in ["diag_1", "diag_2", "diag_3"]:
        if col in df.columns:
            df[f"{col}_group"] = df[col].apply(_map_single_icd9)
    return df


# ---- 8. Split theo patient_nbr (tranh leakage giua cac lan nam vien cua cung 1 nguoi) ----
def split_train_test(df, test_size=0.2, random_state=42, stratify_col="target"):
    splitter = GroupShuffleSplit(test_size=test_size, n_splits=1, random_state=random_state)
    train_idx, test_idx = next(splitter.split(df, groups=df["patient_nbr"]))
    train_df = df.iloc[train_idx].reset_index(drop=True)
    test_df = df.iloc[test_idx].reset_index(drop=True)
    print(f"[split_train_test] Train: {train_df.shape[0]} dong / Test: {test_df.shape[0]} dong")
    if stratify_col in df.columns:
        print(f"[split_train_test] Ty le target - train: {train_df[stratify_col].mean():.4f}, "
              f"test: {test_df[stratify_col].mean():.4f}  "
              f"(GroupShuffleSplit khong ho tro stratify that su khi group theo "
              f"patient_nbr, nen chi kiem tra lai ty le sau khi chia; neu lech "
              f"nhieu, can can nhac StratifiedGroupKFold cua sklearn >=1.0)")
    return train_df, test_df


# ---- 9. Pipeline day du: SPLIT SOM, fit cac buoc "hoc tu du lieu" CHI tren train ----
def run_full_pipeline(input_dir="data/input", test_size=0.2, random_state=42):
    df, ids_mapping = load_raw_data(input_dir)

    # (a) Cac buoc rule-based, an toan chay truoc split
    df = basic_clean(df)
    df = create_target(df)
    df = group_unknown_ids(df)
    df = add_diag_groups(df)

    df = df.drop(columns=["diag_1", "diag_2", "diag_3"])

    # (b) Split TRUOC khi fit bat ky thong ke nao tu du lieu
    train_df, test_df = split_train_test(df, test_size=test_size, random_state=random_state)

    # (c) Fit CHI tren train, roi transform ca train va test
    dropped_cols = fit_near_constant_cols(train_df)
    train_df = drop_cols(train_df, dropped_cols)
    test_df = drop_cols(test_df, dropped_cols)

    train_df = handle_missing_values(train_df)
    test_df = handle_missing_values(test_df)

    categories = fit_categories(train_df)
    train_df = apply_categories(train_df, categories)
    test_df = apply_categories(test_df, categories)

    for col in categories:
        n_unseen = (test_df[col] == "Other_unseen").sum()
        if n_unseen > 0:
            print(f"[run_full_pipeline] Canh bao: cot '{col}' co gia tri o test "
                  f"khong xuat hien trong train ({n_unseen} dong duoc gan nhan 'Other_unseen').")

    return train_df, test_df, ids_mapping