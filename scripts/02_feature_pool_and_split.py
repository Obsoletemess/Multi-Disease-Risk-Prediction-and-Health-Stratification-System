"""Script 02: form-feasible feature pool, per-target leakage rules, person-level split.

Reads data/interim/cohort_labeled.parquet.
Writes feature_pool.parquet, subgroup_side_table.parquet, split_indices.json,
leakage_drop_log.json. Fits no imputer, scaler, or selector.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from config import INTERIM_DIR, MODEL_CANDIDATE_COLUMNS, WEIGHT_COLUMNS  # noqa: E402
from leakage_rules import SUBGROUP_ONLY, apply_and_print  # noqa: E402

RANDOM_SEED = 42
TEST_SIZE = 0.20
TARGETS = ("cvd_risk", "diabetes_risk")
STRATIFY_KEY = (
    "cvd_risk and diabetes_risk as a 4-cell pattern where both are observed; "
    "one extra stratum for rows with either label missing"
)


def fail(message: str) -> None:
    print(f"FAIL: {message}")
    raise SystemExit(1)


def class_counts(frame: pd.DataFrame, target: str) -> dict:
    labeled = frame[target].dropna()
    pos = int((labeled == 1).sum())
    neg = int((labeled == 0).sum())
    return {
        "rows": int(len(frame)),
        "labeled": pos + neg,
        "positive": pos,
        "negative": neg,
        "missing_label": int(frame[target].isna().sum()),
        "prevalence": (pos / (pos + neg)) if (pos + neg) else None,
    }


def main() -> None:
    path = INTERIM_DIR / "cohort_labeled.parquet"
    if not path.exists():
        fail(f"Missing input: {path}. Run Script 01 first.")
    cohort = pd.read_parquet(path)
    print(f"Loaded {path.name}: {len(cohort)} rows, {cohort.shape[1]} columns")

    if cohort["SEQN"].duplicated().any():
        fail("SEQN is not unique")
    for column in (*TARGETS, *MODEL_CANDIDATE_COLUMNS, *SUBGROUP_ONLY, *WEIGHT_COLUMNS):
        if column not in cohort.columns:
            fail(f"Expected column is absent from the cohort: {column}")
    if "family_hx_asthma" in MODEL_CANDIDATE_COLUMNS:
        fail("family_hx_asthma must not be a model candidate")

    # 1. Leakage rules per target (printed)
    feature_lists, leakage_log = apply_and_print(MODEL_CANDIDATE_COLUMNS, cohort.columns)

    # 2. Fail checks on the feature lists
    for target, features in feature_lists.items():
        banned = set(SUBGROUP_ONLY) | set(WEIGHT_COLUMNS) | set(TARGETS) | {"family_hx_asthma"}
        if banned & set(features):
            fail(f"{target} features include a banned column: {sorted(banned & set(features))}")
        if "triglycerides" in features and "triglycerides_measured" not in features:
            fail(f"{target}: triglycerides is present without triglycerides_measured")
        if "fasting_glucose" in features and "fasting_glucose_measured" not in features:
            fail(f"{target}: fasting_glucose is present without fasting_glucose_measured")
    cvd_features = feature_lists["cvd_risk"]
    diabetes_features = feature_lists["diabetes_risk"]
    if any(c.startswith(("MCQ160", "CDQ")) for c in cvd_features):
        fail("A MCQ160* or CDQ* column remains in the cvd_risk features")
    if {"DIQ010", "DIQ050", "DIQ070"} & set(diabetes_features):
        fail("A DIQ010/DIQ050/DIQ070 column remains in the diabetes_risk features")
    for column in ("hba1c", "fasting_glucose", "fasting_glucose_measured"):
        if column not in diabetes_features:
            fail(f"{column} was dropped from the diabetes_risk features")
    print("Leakage checks passed.")

    # 3. Split persons once, before any fit
    both = cohort[list(TARGETS)].notna().all(axis=1)
    strata = pd.Series("missing_label", index=cohort.index, dtype="object")
    strata.loc[both] = (
        cohort.loc[both, "cvd_risk"].astype(int).astype(str)
        + cohort.loc[both, "diabetes_risk"].astype(int).astype(str)
    )
    strata_counts = strata.value_counts().to_dict()
    print("Stratum sizes:", strata_counts)
    if min(strata_counts.values()) < 2:
        fail("A stratum has fewer than 2 rows and cannot be stratified")

    positions = np.arange(len(cohort))
    train_pos, test_pos = train_test_split(
        positions, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=strata.values
    )
    split = pd.Series("train", index=cohort.index, dtype="object")
    split.iloc[test_pos] = "test"
    seqn_train = cohort["SEQN"].iloc[train_pos].astype(int).tolist()
    seqn_test = cohort["SEQN"].iloc[test_pos].astype(int).tolist()

    if set(seqn_train) & set(seqn_test):
        fail("Train and test SEQN overlap")
    test_fraction = len(seqn_test) / len(cohort)
    if abs(test_fraction - TEST_SIZE) > 0.01:
        fail(f"Test fraction {test_fraction:.3f} is not 0.20 +/- 0.01")
    print(f"Split: train={len(seqn_train)}, test={len(seqn_test)} (test fraction {test_fraction:.3f})")
    print(
        "Split done before any imputer, scaler, or selector was fit "
        "(this script fits none)."
    )

    # 4. Class counts per split. Missing-label rows stay in their split here
    #    and are dropped per target downstream, never moved between sets.
    per_target = {}
    for target in TARGETS:
        per_target[target] = {
            "train": class_counts(cohort.loc[split == "train"], target),
            "test": class_counts(cohort.loc[split == "test"], target),
        }
        for part in ("train", "test"):
            c = per_target[target][part]
            print(
                f"  {target} {part}: labeled={c['labeled']}, positive={c['positive']}, "
                f"negative={c['negative']}, label missing={c['missing_label']}, "
                f"prevalence={c['prevalence']:.4f}"
            )

    # 5. Write outputs
    INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    pool = pd.concat(
        [
            cohort[["SEQN", *TARGETS]],
            split.rename("split"),
            cohort[list(MODEL_CANDIDATE_COLUMNS)],
        ],
        axis=1,
    )
    side = pd.concat(
        [cohort[["SEQN", *SUBGROUP_ONLY, *WEIGHT_COLUMNS]], split.rename("split")], axis=1
    )
    pool.to_parquet(INTERIM_DIR / "feature_pool.parquet", index=False)
    side.to_parquet(INTERIM_DIR / "subgroup_side_table.parquet", index=False)

    split_doc = {
        "seed": RANDOM_SEED,
        "test_size": TEST_SIZE,
        "stratify_key": STRATIFY_KEY,
        "strata_counts": {str(k): int(v) for k, v in strata_counts.items()},
        "n_train": len(seqn_train),
        "n_test": len(seqn_test),
        "per_target_counts": per_target,
        "seqn_train": seqn_train,
        "seqn_test": seqn_test,
    }
    (INTERIM_DIR / "split_indices.json").write_text(json.dumps(split_doc), encoding="utf-8")
    (INTERIM_DIR / "leakage_drop_log.json").write_text(
        json.dumps({"targets": leakage_log}, indent=2), encoding="utf-8"
    )
    print(f"Wrote feature_pool.parquet ({pool.shape[0]} rows, {pool.shape[1]} columns)")
    print("Wrote subgroup_side_table.parquet (RIDRETH3 and survey weights, not features)")
    print("Wrote split_indices.json and leakage_drop_log.json")


if __name__ == "__main__":
    main()