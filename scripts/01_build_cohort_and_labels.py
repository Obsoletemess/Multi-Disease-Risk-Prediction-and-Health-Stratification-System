"""Build the adult NHANES cohort, derived form fields, labels, and triage tier.

Reads the P_*.csv files already in the workspace Data folder.
Writes data/interim/cohort_labeled.parquet, label_audit.json, and
missing_code_log.json.

Does not impute, split, or fit a model.
"Not enough measurements" is stored as a triage result and does not
remove a row from cvd_risk or diabetes_risk.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from config import (  # noqa: E402
    ACTIVITY_DOMAINS,
    ALCOHOL_FREQUENCY_PER_WEEK,
    CONTINUOUS_COLUMNS,
    CVD_SOURCE_COLUMNS,
    DATA_DIR,
    DISCRETE_MISSING_CODES,
    EDUCATION_LABELS,
    FAMILY_HISTORY_QUESTIONS,
    INSUFFICIENT_TIER,
    INTERIM_DIR,
    CDQ_QUESTIONS,
    LABEL_COLUMNS,
    LDL_TRIGLYCERIDE_MAX,
    MIN_AGE,
    MODEL_CANDIDATE_COLUMNS,
    NOT_MODEL_COLUMNS,
    REQUIRED_FILES,
    RULE_ONLY_COLUMNS,
    SEX_LABELS,
    SUBGROUP_ONLY_COLUMNS,
    WEIGHT_COLUMNS,
)
from severity_rules import (  # noqa: E402
    FORBIDDEN_TIER_INPUTS,
    TIER_INPUT_COLUMNS,
    assign_tier,
)

BYTE_LITERAL = re.compile(r"^b['\"](.*)['\"]$")
ZERO_ACTIVITY_CLAUSE = "activity minutes present and equal to 0"


def fail(message: str) -> None:
    print(f"FAIL: {message}")
    raise SystemExit(1)


def decode_byte_literals(frame: pd.DataFrame) -> dict:
    """Turn exported tokens such as b'R' back into plain text."""
    counts = {}
    for column in frame.columns:
        if frame[column].dtype != object:
            continue
        decoded = 0

        def _one(value, _decoded=None):
            nonlocal decoded
            if isinstance(value, bytes):
                decoded += 1
                return value.decode("utf-8", errors="replace")
            if isinstance(value, str):
                match = BYTE_LITERAL.match(value.strip())
                if match:
                    decoded += 1
                    return match.group(1)
            return value

        frame[column] = frame[column].map(_one)
        if decoded:
            counts[column] = decoded
    return counts


def normalize_seqn(frame: pd.DataFrame, file_name: str) -> pd.DataFrame:
    if "SEQN" not in frame.columns:
        fail(f"{file_name} has no SEQN column")
    seqn = pd.to_numeric(frame["SEQN"], errors="coerce")
    if seqn.isna().any() or (seqn % 1 != 0).any():
        fail(f"{file_name} has a SEQN that is blank or not a whole number")
    frame = frame.copy()
    frame["SEQN"] = seqn.astype("int64")
    if frame["SEQN"].duplicated().any():
        fail(f"{file_name} has duplicate SEQN values")
    return frame


def load_module(file_name: str) -> pd.DataFrame:
    path = DATA_DIR / file_name
    if not path.exists():
        fail(f"Required file is absent: {path}")
    frame = pd.read_csv(path, low_memory=False)
    frame = normalize_seqn(frame, file_name)
    decoded = decode_byte_literals(frame)
    if decoded:
        detail = ", ".join(f"{column}={count}" for column, count in decoded.items())
        print(f"  decoded b'...' tokens in {file_name}: {detail}")
    else:
        print(f"  {file_name}: no b'...' tokens")
    print(f"  {file_name}: {len(frame)} rows, {frame.shape[1]} columns")
    return frame


def apply_discrete_missing(frame: pd.DataFrame) -> dict:
    overlap = sorted(set(DISCRETE_MISSING_CODES) & set(CONTINUOUS_COLUMNS))
    if overlap:
        fail(
            "Discrete missing codes would have been applied to continuous columns: "
            + ", ".join(overlap)
        )
    print(
        "Discrete missing codes were not applied to continuous columns: "
        + ", ".join(CONTINUOUS_COLUMNS)
    )
    print(
        "ALQ121 codes 7 and 9 stay as answers "
        "(once a month; 3 to 6 times in the last year). "
        "ALQ130 values 7 and 9 stay as drink counts. "
        "RIDRETH3 value 7 stays as Other race, including multi-racial."
    )
    log = {}
    for column, codes in DISCRETE_MISSING_CODES.items():
        if column not in frame.columns:
            fail(f"Codebook column {column} is absent after the merge")
        numeric = pd.to_numeric(frame[column], errors="coerce")
        replaced = {}
        for code in codes:
            replaced[str(code)] = int((numeric == code).sum())
        frame[column] = numeric.where(~numeric.isin(list(codes)))
        log[column] = replaced
    return log


def null_nonpositive(frame: pd.DataFrame, columns: tuple[str, ...]) -> dict:
    log = {}
    for column in columns:
        numeric = pd.to_numeric(frame[column], errors="coerce")
        nonpositive = int((numeric <= 0).sum())
        log[column] = nonpositive
        frame[column] = numeric.where(numeric > 0)
    return log


def yes_no_to_binary(series: pd.Series) -> pd.Series:
    """Map NHANES 1/2 answers to 1/0. Anything else, including a skip, stays missing."""
    out = pd.Series(np.nan, index=series.index, dtype="float64")
    out = out.mask(series.eq(2), 0).mask(series.eq(1), 1)
    return out


def activity_minutes(frame: pd.DataFrame) -> pd.Series:
    """Weekly moderate-to-vigorous minutes.

    A domain answered No contributes 0. A domain that was refused, left
    blank, or only partly answered stays missing. The total is 0 only when
    every domain is known and each one is 0. A blank questionnaire stays missing.
    """
    domains = []
    for yes_col, day_col, minute_col in ACTIVITY_DOMAINS:
        yes = frame[yes_col]
        days = frame[day_col]
        minutes = frame[minute_col]
        domain = pd.Series(np.nan, index=frame.index, dtype="float64")
        domain = domain.mask(yes.eq(2), 0)
        known_yes = yes.eq(1) & days.notna() & minutes.notna()
        domain = domain.mask(known_yes, days * minutes)
        domains.append(domain)
    stacked = pd.concat(domains, axis=1)
    total = stacked.sum(axis=1, min_count=stacked.shape[1])
    return total


def alcohol_drinks_per_week(frame: pd.DataFrame) -> pd.Series:
    """Occasions per week, from the ALQ121 midpoints, times drinks per occasion.

    Never-drinkers are 0. A known frequency with a missing drink count stays
    missing, except 'never in the last year', which is 0. This is not a
    zero-fill of a blank questionnaire.
    """
    ever = frame["ALQ111"]
    frequency = frame["ALQ121"]
    drinks = frame["ALQ130"]
    out = pd.Series(np.nan, index=frame.index, dtype="float64")
    out = out.mask(ever.eq(2), 0)
    never_this_year = frequency.eq(0)
    out = out.mask(never_this_year, 0)
    for code, occasions in ALCOHOL_FREQUENCY_PER_WEEK.items():
        if code == 0:
            continue
        matched = frequency.eq(code) & drinks.notna()
        out = out.mask(matched, occasions * drinks)
    return out


def medication_flag(frame: pd.DataFrame, now: str, told: str, advised: str) -> pd.Series:
    """Current medicine, with questionnaire skips recorded as No.

    People who were never told they had the condition, or were never advised
    to take medicine, are not asked the 'now taking' item. That skip is No.
    An explicit Yes on the current-medicine item wins over a conflicting skip.
    """
    out = pd.Series(np.nan, index=frame.index, dtype="float64")
    skipped_or_no = frame[now].eq(2) | frame[told].eq(2) | frame[advised].eq(2)
    out = out.mask(skipped_or_no, 0).mask(frame[now].eq(1), 1)
    return out


def mean_of_readings(frame: pd.DataFrame, columns: tuple[str, ...]) -> pd.Series:
    block = frame.loc[:, list(columns)]
    return block.mean(axis=1, skipna=True)


def _rose_location(frame: pd.DataFrame) -> pd.Series:
    """Rose site rule: upper or lower sternum, or left chest and left arm."""
    return (
        frame["CDQ009D"].eq(4)
        | frame["CDQ009E"].eq(5)
        | (frame["CDQ009F"].eq(6) & frame["CDQ009G"].eq(7))
    )


def report_chest_pain_versions(frame: pd.DataFrame) -> None:
    """Print codebook text and flag counts. Does not change the live rule."""
    print("P_CDQ question text (2017-March 2020 codebook). Asked at age 40 and older.")
    for column, question in CDQ_QUESTIONS.items():
        print(f"  {column}: {question}")
    age40 = frame["age_years"].ge(40)
    ever = frame["CDQ001"].eq(1)
    on_hurry = frame["CDQ002"].eq(1)
    on_level = frame["CDQ003"].eq(1)
    rose_stem = (
        ever
        & on_hurry
        & frame["CDQ004"].eq(1)
        & frame["CDQ005"].eq(1)
        & frame["CDQ006"].eq(1)
        & _rose_location(frame)
    )
    versions = {
        "current rule, not changed: CDQ001 == 1, ever any chest pain": ever,
        "proposed: CDQ001 == 1 and CDQ002 == 1, pain when walking uphill or hurrying": ever & on_hurry,
        "broader exertion: uphill/hurry or ordinary pace on the level": ever & (on_hurry | on_level),
        "Rose grade 1 angina (exertional, not at ordinary pace, plus site)": rose_stem & frame["CDQ003"].ne(1),
        "Rose grade 2 angina (also at ordinary pace, plus site)": rose_stem & on_level,
        "CDQ008 == 1, severe pain across the chest for half an hour or more": frame["CDQ008"].eq(1),
    }
    print(
        "Chest-pain flag counts. The assigned triage field is still the current rule. "
        f"Cohort rows={len(frame)}. Age 40 or older={int(age40.sum())}."
    )
    for name, mask in versions.items():
        print(f"  {int(mask.sum())} cohort rows, {int((mask & age40).sum())} of them age >= 40 | {name}")
    print(
        "Proposal, not applied: set chest_pain from CDQ001 == 1 and CDQ002 == 1. "
        "That is pain or discomfort brought on by walking uphill or hurrying."
    )


def build_tier_inputs(frame: pd.DataFrame) -> list[dict]:
    """Measurements only. Diagnosis labels are not copied into this payload."""
    payloads = []
    for row in frame.itertuples(index=False):
        payloads.append(
            {
                "age_years": row.age_years,
                "bmi": row.bmi,
                "sbp_mean": row.sbp_mean,
                "dbp_mean": row.dbp_mean,
                "hba1c": row.hba1c,
                "fasting_glucose": row.fasting_glucose,
                "total_cholesterol": row.total_cholesterol,
                "current_smoker": row.current_smoker,
                "activity_minutes_week": row.activity_minutes_week,
                "chest_pain": row.chest_pain,
                "fatigue": None,
                "dizziness": None,
                "spo2": None,
            }
        )
    return payloads


def binary_summary(series: pd.Series) -> dict:
    labeled = series.dropna()
    n_positive = int((labeled == 1).sum())
    n_negative = int((labeled == 0).sum())
    n_excluded = int(series.isna().sum())
    if n_positive + n_negative == 0:
        fail(f"{series.name} has no labeled rows")
    prevalence = n_positive / (n_positive + n_negative)
    if prevalence in (0.0, 1.0):
        fail(f"{series.name} prevalence is {prevalence}")
    return {
        "positive": n_positive,
        "negative": n_negative,
        "excluded_missing_label": n_excluded,
        "prevalence_among_labeled": prevalence,
    }


def main() -> None:
    print("Files loaded from", DATA_DIR)
    missing_files = [name for name in REQUIRED_FILES if not (DATA_DIR / name).exists()]
    if missing_files:
        fail("Required files are absent: " + ", ".join(missing_files))

    modules = {name: load_module(name) for name in REQUIRED_FILES}
    demo = modules["P_DEMO.csv"]
    print(f"P_DEMO rows before filters: {len(demo)}")

    age = pd.to_numeric(demo["RIDAGEYR"], errors="coerce")
    n_missing_age = int(age.isna().sum())
    adults = demo.loc[age >= MIN_AGE].copy()
    print(f"Missing age (not treated as adults): {n_missing_age}")
    print(f"Adults age >= {MIN_AGE}: {len(adults)}")

    pregnancy = pd.to_numeric(adults["RIDEXPRG"], errors="coerce")
    n_pregnant = int(pregnancy.eq(1).sum())
    cohort = adults.loc[~pregnancy.eq(1)].copy()
    print(f"Pregnant participants dropped (RIDEXPRG == 1): {n_pregnant}")
    print(f"Cohort rows after the pregnancy filter: {len(cohort)}")
    n_locked = len(cohort)

    merge_plan = (
        ("P_BMX.csv", "BMXBMI"),
        ("P_BPXO.csv", "BPXOSY1"),
        ("P_GHB.csv", "LBXGH"),
        ("P_GLU.csv", "LBXGLU"),
        ("P_TCHOL.csv", "LBXTC"),
        ("P_HDL.csv", "LBDHDD"),
        ("P_TRIGLY.csv", "LBXTR"),
        ("P_MCQ.csv", "MCQ160B"),
        ("P_DIQ.csv", "DIQ010"),
        ("P_BPQ.csv", "BPQ020"),
        ("P_SMQ.csv", "SMQ020"),
        ("P_PAQ.csv", "PAQ605"),
        ("P_CDQ.csv", "CDQ001"),
        ("P_ALQ.csv", "ALQ121"),
    )
    for file_name, marker in merge_plan:
        extra = modules[file_name]
        overlap = [column for column in extra.columns if column != "SEQN" and column in cohort.columns]
        if overlap:
            print(f"  {file_name}: kept the copy already on the cohort for {overlap}")
            extra = extra.drop(columns=overlap)
        before = len(cohort)
        cohort = cohort.merge(extra, on="SEQN", how="left", validate="one_to_one")
        if len(cohort) != before:
            fail(f"{file_name} changed the row count from {before} to {len(cohort)}")
        matched = int(cohort[marker].notna().sum()) if marker in cohort.columns else 0
        print(f"  merged {file_name}: rows={len(cohort)}, non-missing {marker}={matched}")

    if len(cohort) != n_locked:
        fail("The merge changed the number of people in the cohort")
    if cohort["SEQN"].duplicated().any():
        fail("SEQN is not unique after the merge")

    for column, question in FAMILY_HISTORY_QUESTIONS.items():
        if column not in cohort.columns:
            fail(f"Family-history column {column} is absent")
        print(f"Family history {column}: {question}")

    missing_log = apply_discrete_missing(cohort)
    print("Missing-code replacements (code: rows set to missing):")
    for column, counts in missing_log.items():
        used = {code: n for code, n in counts.items() if n}
        if used:
            print(f"  {column}: {used}")

    physiologic_columns = (
        "BMXWT",
        "BMXHT",
        "BMXBMI",
        "BMXWAIST",
        "BPXOSY1",
        "BPXOSY2",
        "BPXOSY3",
        "BPXODI1",
        "BPXODI2",
        "BPXODI3",
        "BPXOPLS1",
        "BPXOPLS2",
        "BPXOPLS3",
    )
    physiologic_log = null_nonpositive(cohort, physiologic_columns)
    print("Readings at or below 0 set to missing (not a codebook code):")
    for column, count in physiologic_log.items():
        if count:
            print(f"  {column}: {count}")

    for column in ("LBXGH", "LBXGLU", "LBXTC", "LBDHDD", "LBXTR"):
        cohort[column] = pd.to_numeric(cohort[column], errors="coerce")

    age_years = pd.to_numeric(cohort["RIDAGEYR"], errors="coerce")
    sex = pd.to_numeric(cohort["RIAGENDR"], errors="coerce")
    education = pd.to_numeric(cohort["DMDEDUC2"], errors="coerce")
    print("Sex codes:", SEX_LABELS)
    print("Education codes:", EDUCATION_LABELS)

    height_m = cohort["BMXHT"] / 100.0
    bmi_from_parts = cohort["BMXWT"] / (height_m ** 2)
    height_cm = cohort["BMXHT"]
    weight_kg = cohort["BMXWT"]
    waist_cm = cohort["BMXWAIST"]
    bmi = cohort["BMXBMI"].where(cohort["BMXBMI"].notna(), bmi_from_parts)

    sbp_mean = mean_of_readings(cohort, ("BPXOSY1", "BPXOSY2", "BPXOSY3"))
    dbp_mean = mean_of_readings(cohort, ("BPXODI1", "BPXODI2", "BPXODI3"))
    heart_rate_mean = mean_of_readings(cohort, ("BPXOPLS1", "BPXOPLS2", "BPXOPLS3"))
    print("Blood pressure and heart rate are the mean of the readings that are present.")

    hba1c = cohort["LBXGH"]
    fasting_glucose = cohort["LBXGLU"]
    fasting_glucose_measured = fasting_glucose.notna().astype("int64")
    total_cholesterol = cohort["LBXTC"]
    hdl = cohort["LBDHDD"]
    triglycerides = cohort["LBXTR"]
    triglycerides_measured = triglycerides.notna().astype("int64")
    ldl_ready = (
        total_cholesterol.notna()
        & hdl.notna()
        & triglycerides.notna()
        & triglycerides.le(LDL_TRIGLYCERIDE_MAX)
    )
    ldl_friedewald = (total_cholesterol - hdl - (triglycerides / 5.0)).where(ldl_ready)
    n_ldl_skipped_tg = int((triglycerides.notna() & triglycerides.gt(LDL_TRIGLYCERIDE_MAX)).sum())

    told_high_bp = yes_no_to_binary(cohort["BPQ020"])
    on_bp_med = medication_flag(cohort, "BPQ050A", "BPQ020", "BPQ040A")
    told_high_chol = yes_no_to_binary(cohort["BPQ080"])
    on_chol_med = medication_flag(cohort, "BPQ100D", "BPQ080", "BPQ090D")

    smoking = pd.Series(pd.NA, index=cohort.index, dtype="object")
    smoking = smoking.mask(cohort["SMQ020"].eq(2), "never")
    smoking = smoking.mask(cohort["SMQ040"].eq(3), "former")
    smoking = smoking.mask(cohort["SMQ040"].isin([1, 2]), "current")
    current_smoker = smoking.map({"current": 1, "former": 0, "never": 0}).astype("float64")

    activity_minutes_week = activity_minutes(cohort)
    drinks_per_week = alcohol_drinks_per_week(cohort)
    family_hx_early_heart = yes_no_to_binary(cohort["MCQ300A"])
    family_hx_asthma = yes_no_to_binary(cohort["MCQ300B"])
    family_hx_diabetes = yes_no_to_binary(cohort["MCQ300C"])
    cdq001 = yes_no_to_binary(cohort["CDQ001"])
    cdq002 = yes_no_to_binary(cohort["CDQ002"])
    chest_pain = pd.Series(np.nan, index=cohort.index, dtype="float64")
    chest_pain = chest_pain.mask(cdq001.eq(0), 0)
    chest_pain = chest_pain.mask(cdq001.eq(1) & cdq002.eq(0), 0)
    chest_pain = chest_pain.mask(cdq001.eq(1) & cdq002.eq(1), 1)

    condition = cohort.loc[:, list(CVD_SOURCE_COLUMNS)]
    cvd_risk = pd.Series(np.nan, index=cohort.index, dtype="float64")
    cvd_risk = cvd_risk.mask(condition.notna().any(axis=1), 0).mask(condition.eq(1).any(axis=1), 1)
    diabetes_risk = pd.Series(np.nan, index=cohort.index, dtype="float64")
    diabetes_risk = diabetes_risk.mask(cohort["DIQ010"].isin([2, 3]), 0).mask(cohort["DIQ010"].eq(1), 1)

    derived = pd.DataFrame(
        {
            "age_years": age_years,
            "sex": sex,
            "education": education,
            "height_cm": height_cm,
            "weight_kg": weight_kg,
            "waist_cm": waist_cm,
            "bmi": bmi,
            "sbp_mean": sbp_mean,
            "dbp_mean": dbp_mean,
            "heart_rate_mean": heart_rate_mean,
            "hba1c": hba1c,
            "fasting_glucose": fasting_glucose,
            "fasting_glucose_measured": fasting_glucose_measured,
            "total_cholesterol": total_cholesterol,
            "hdl": hdl,
            "triglycerides": triglycerides,
            "triglycerides_measured": triglycerides_measured,
            "ldl_friedewald": ldl_friedewald,
            "told_high_bp": told_high_bp,
            "on_bp_med": on_bp_med,
            "told_high_chol": told_high_chol,
            "on_chol_med": on_chol_med,
            "smoking_status": smoking,
            "current_smoker": current_smoker,
            "activity_minutes_week": activity_minutes_week,
            "alcohol_drinks_per_week": drinks_per_week,
            "family_hx_early_heart": family_hx_early_heart,
            "family_hx_asthma": family_hx_asthma,
            "family_hx_diabetes": family_hx_diabetes,
            "chest_pain": chest_pain,
            "cvd_risk": cvd_risk,
            "diabetes_risk": diabetes_risk,
        },
        index=cohort.index,
    )
    cohort = pd.concat([cohort, derived], axis=1)

    print(
        f"Friedewald LDL computed for {int(ldl_ready.sum())} rows. "
        f"Skipped {n_ldl_skipped_tg} rows with triglycerides > {LDL_TRIGLYCERIDE_MAX} mg/dL. "
        "Raw LBDLDL stays on the file as a source column and is not a model candidate."
    )
    print(
        "Fasting glucose present="
        f"{int(cohort['fasting_glucose'].notna().sum())}, "
        f"missing={int(cohort['fasting_glucose'].isna().sum())}"
    )
    print(
        "Triglycerides present="
        f"{int(cohort['triglycerides'].notna().sum())}, "
        f"missing={int(cohort['triglycerides'].isna().sum())}"
    )
    print(
        "current_smoker is stored for the triage rule. "
        "The model candidate is smoking_status, so the two are not both offered as features."
    )
    n_activity_missing = int(cohort["activity_minutes_week"].isna().sum())
    n_activity_zero = int(cohort["activity_minutes_week"].eq(0).sum())
    print(
        "Activity minutes are the sum of five domains (vigorous work, moderate work, "
        "walk or cycle, vigorous recreation, moderate recreation). "
        f"Missing total={n_activity_missing}. Known zero={n_activity_zero}. "
        "A missing domain does not become zero."
    )
    print(
        "Alcohol drinks per week = ALQ121 occasions-per-week midpoint x ALQ130. "
        f"Midpoints: { {code: round(value, 4) for code, value in ALCOHOL_FREQUENCY_PER_WEEK.items()} }"
    )
    print(
        "MCQ300A is early heart attack or angina and is a model candidate. "
        "MCQ300C is diabetes and is a model candidate. "
        "MCQ300B is asthma: family_hx_asthma is stored and is not a model candidate."
    )
    print(
        "Chest pain for triage = CDQ001 == 1 and CDQ002 == 1 "
        "(pain when walking uphill or hurrying). Not asked or unanswered stays missing. "
        f"Flagged={int(cohort['chest_pain'].eq(1).sum())}, "
        f"no={int(cohort['chest_pain'].eq(0).sum())}, "
        f"missing={int(cohort['chest_pain'].isna().sum())}"
    )

    cvd_summary = binary_summary(cohort["cvd_risk"])
    diabetes_summary = binary_summary(cohort["diabetes_risk"])
    print(
        "cvd_risk is a recorded diagnosis (heart failure, coronary disease, angina, "
        "heart attack, or stroke), as profile similarity, not a forecast. "
        f"positive={cvd_summary['positive']}, negative={cvd_summary['negative']}, "
        f"label missing={cvd_summary['excluded_missing_label']}, "
        f"prevalence among labeled={cvd_summary['prevalence_among_labeled']:.4f}"
    )
    n_borderline = int((cohort["DIQ010"].eq(3) & cohort["diabetes_risk"].eq(0)).sum())
    lab_range = cohort["hba1c"].ge(6.5) | cohort["fasting_glucose"].ge(126)
    n_undiagnosed_labs = int((cohort["diabetes_risk"].eq(0) & lab_range).sum())
    print(
        "diabetes_risk is a recorded diagnosis (DIQ010 == 1). "
        "Borderline (DIQ010 == 3) is labeled negative. "
        "A No with HbA1c >= 6.5 or fasting glucose >= 126 is labeled negative. "
        f"positive={diabetes_summary['positive']}, negative={diabetes_summary['negative']}, "
        f"label missing={diabetes_summary['excluded_missing_label']}, "
        f"prevalence among labeled={diabetes_summary['prevalence_among_labeled']:.4f}, "
        f"borderline labeled negative={n_borderline}, "
        f"negative with lab-range glucose or HbA1c={n_undiagnosed_labs}"
    )

    print("assign_tier measurement columns:", ", ".join(TIER_INPUT_COLUMNS))
    print("assign_tier refuses these diagnosis fields:", ", ".join(FORBIDDEN_TIER_INPUTS))
    banned_overlap = set(TIER_INPUT_COLUMNS) & set(FORBIDDEN_TIER_INPUTS)
    if banned_overlap:
        fail("Tier inputs include a diagnosis field: " + ", ".join(sorted(banned_overlap)))
    try:
        assign_tier({"cvd_risk": 1})
    except ValueError:
        print("Confirmed: assign_tier rejects cvd_risk and does not read MCQ160* or an imputed column.")
    else:
        fail("assign_tier accepted cvd_risk")

    tier_labels = []
    tier_traces = []
    for payload in build_tier_inputs(cohort):
        label, trace = assign_tier(payload)
        tier_labels.append(label)
        tier_traces.append(trace)
    tier_frame = pd.DataFrame(
        {"severity_tier": tier_labels, "severity_trace": tier_traces},
        index=cohort.index,
    )
    cohort = pd.concat([cohort, tier_frame], axis=1)
    print("No imputation was applied. assign_tier received measured values only.")
    print("Triage histogram:")
    print(cohort["severity_tier"].value_counts(dropna=False).to_string())

    bad_zero = cohort["activity_minutes_week"].isna() & cohort["severity_trace"].str.contains(
        ZERO_ACTIVITY_CLAUSE, regex=False
    )
    if int(bad_zero.sum()):
        fail("A row with missing activity minutes was treated as zero")

    insufficient = cohort["severity_tier"].eq(INSUFFICIENT_TIER)
    n_insufficient_cvd = int((insufficient & cohort["cvd_risk"].notna()).sum())
    n_insufficient_diabetes = int((insufficient & cohort["diabetes_risk"].notna()).sum())
    print(f"Not enough measurements: {int(insufficient.sum())} rows")
    print(f"  of those, cvd_risk is present for {n_insufficient_cvd}")
    print(f"  of those, diabetes_risk is present for {n_insufficient_diabetes}")
    print(
        "Not enough measurements does not drop learned-target rows. "
        "A row leaves cvd_risk or diabetes_risk only when that label is missing."
    )
    if len(cohort) != n_locked:
        fail("Rows were removed after the pregnancy filter")

    candidate_banned = (
        set(WEIGHT_COLUMNS)
        | set(SUBGROUP_ONLY_COLUMNS)
        | set(CVD_SOURCE_COLUMNS)
        | set(LABEL_COLUMNS)
        | set(NOT_MODEL_COLUMNS)
        | {"DIQ010", "DIQ050", "DIQ070", "CDQ001", "current_smoker"}
    )
    leaked_candidates = sorted(candidate_banned & set(MODEL_CANDIDATE_COLUMNS))
    if leaked_candidates:
        fail("Model candidates include a forbidden column: " + ", ".join(leaked_candidates))
    print("Survey weights stored and not used as features:", ", ".join(WEIGHT_COLUMNS))
    print("Subgroup-only column stored and not a model candidate: RIDRETH3")
    print("Model candidates:", ", ".join(MODEL_CANDIDATE_COLUMNS))
    print("Rule-only stored column:", ", ".join(RULE_ONLY_COLUMNS))
    n_age_80 = int(cohort["age_years"].eq(80).sum())
    print(f"RIDAGEYR 80 is the public-file top code and means 80 or older. n={n_age_80}")

    for column in MODEL_CANDIDATE_COLUMNS:
        if column not in cohort.columns:
            fail(f"Model candidate {column} was not created")

    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    cohort_path = INTERIM_DIR / "cohort_labeled.parquet"
    audit_path = INTERIM_DIR / "label_audit.json"
    missing_path = INTERIM_DIR / "missing_code_log.json"

    audit = {
        "n_demo": int(len(demo)),
        "n_missing_age": n_missing_age,
        "n_adults": int(len(adults)),
        "n_pregnant_dropped": n_pregnant,
        "n_cohort": int(len(cohort)),
        "cvd_risk": cvd_summary,
        "diabetes_risk": diabetes_summary,
        "diabetes_borderline_labeled_negative": n_borderline,
        "diabetes_negative_with_hba1c_ge_6_5_or_fasting_glucose_ge_126": n_undiagnosed_labs,
        "fasting_glucose_present": int(cohort["fasting_glucose"].notna().sum()),
        "fasting_glucose_missing": int(cohort["fasting_glucose"].isna().sum()),
        "triglycerides_present": int(cohort["triglycerides"].notna().sum()),
        "triglycerides_missing": int(cohort["triglycerides"].isna().sum()),
        "ldl_friedewald_computed": int(ldl_ready.sum()),
        "ldl_skipped_triglycerides_above_400": n_ldl_skipped_tg,
        "tier_counts": {
            str(key): int(value)
            for key, value in cohort["severity_tier"].value_counts(dropna=False).items()
        },
        "insufficient_with_cvd_label": n_insufficient_cvd,
        "insufficient_with_diabetes_label": n_insufficient_diabetes,
        "retention_rule": (
            "Not enough measurements is a triage state only. "
            "A row leaves a learned target only when that target's label is missing."
        ),
        "n_age_topcoded_80": n_age_80,
        "age_topcode_means": "80 or older",
        "weights_stored_not_features": list(WEIGHT_COLUMNS),
        "family_history_questions": FAMILY_HISTORY_QUESTIONS,
        "tier_inputs": list(TIER_INPUT_COLUMNS),
        "model_candidates": list(MODEL_CANDIDATE_COLUMNS),
        "activity_missing": n_activity_missing,
        "activity_known_zero": n_activity_zero,
    }
    missing_document = {
        "discrete_codes_replaced": missing_log,
        "nonpositive_set_to_missing": physiologic_log,
        "continuous_columns_excluded_from_discrete_codes": list(CONTINUOUS_COLUMNS),
    }

    try:
        cohort.to_parquet(cohort_path, index=False)
    except ImportError as exc:
        fail(f"Writing parquet needs pyarrow or fastparquet ({exc}). Install pyarrow and rerun.")

    audit_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    missing_path.write_text(json.dumps(missing_document, indent=2), encoding="utf-8")
    print(f"Wrote {cohort_path}")
    print(f"Wrote {audit_path}")
    print(f"Wrote {missing_path}")
    print(f"Cohort rows={len(cohort)}, columns={cohort.shape[1]}")


if __name__ == "__main__":
    main()
