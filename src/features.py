def add_advanced_features(df):
    df = df.copy()
    # 1. Total number of previous healthcare visits
    df["total_previous_visits"] = (
        df["number_outpatient"]
        + df["number_emergency"]
        + df["number_inpatient"]
    )
    # 2. Whether the patient had any previous inpatient visits
    df["has_prior_inpatient"] = (
        df["number_inpatient"] > 0
    ).astype(int)
    # 3. Whether the patient had any previous emergency visits
    df["has_prior_emergency"] = (
        df["number_emergency"] > 0
    ).astype(int)
    # 4. Average number of medications per hospital day
    df["medications_per_day"] = (
        df["num_medications"]
        / df["time_in_hospital"].clip(lower=1)
    )
    # 5. Average number of lab procedures per hospital day
    df["labs_per_day"] = (
        df["num_lab_procedures"]
        / df["time_in_hospital"].clip(lower=1)
    )
    # 6. Average number of procedures per hospital day
    df["procedures_per_day"] = (
        df["num_procedures"]
        / df["time_in_hospital"].clip(lower=1)
    )
    # 7. Overall treatment intensity
    df["treatment_intensity"] = (
        df["num_lab_procedures"]
        + df["num_procedures"]
        + df["num_medications"]
    )
    return df
