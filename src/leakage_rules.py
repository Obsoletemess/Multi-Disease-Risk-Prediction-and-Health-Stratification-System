"""Per-target leakage rules for the NHANES likelihood models.

A column is removed from a target's inputs when it defines that target's
label or is a near-proxy for it. HbA1c, fasting glucose, BMI, height,
weight and waist are deliberately NOT excluded for diabetes (D4).
"""

from __future__ import annotations

SUBGROUP_ONLY = ("RIDRETH3",)

EXCLUSIONS = {
    "cvd_risk": {
        "MCQ160B": "label_definition",
        "MCQ160C": "label_definition",
        "MCQ160D": "label_definition",
        "MCQ160E": "label_definition",
        "MCQ160F": "label_definition",
    },
    "diabetes_risk": {
        "DIQ010": "label_definition",
        "DIQ050": "near_proxy",
        "DIQ070": "near_proxy",
    },
}

# Any cohort column starting with these prefixes is excluded for that target.
PREFIX_EXCLUSIONS = {
    "cvd_risk": (("MCQ160", "label_definition"), ("CDQ", "near_proxy")),
    "diabetes_risk": (),
}

# Stored on the cohort file but never offered to a model.
_NOT_FEATURES = {
    "family_hx_asthma": "not_on_form",
    "current_smoker": "not_on_form",
    "severity_tier": "not_on_form",
    "severity_trace": "not_on_form",
}
EXTRA_NOT_FEATURES = {
    "cvd_risk": {"chest_pain": "near_proxy", **_NOT_FEATURES},
    "diabetes_risk": {"chest_pain": "not_on_form", **_NOT_FEATURES},
}


def exclusion_table(target: str, cohort_columns) -> dict:
    cohort_columns = list(cohort_columns)
    table = dict(EXCLUSIONS[target])
    for prefix, reason in PREFIX_EXCLUSIONS.get(target, ()):
        for column in cohort_columns:
            if column.startswith(prefix):
                table.setdefault(column, reason)
    for column in SUBGROUP_ONLY:
        table[column] = "subgroup_only"
    for column, reason in EXTRA_NOT_FEATURES[target].items():
        table.setdefault(column, reason)
    return table


def apply_and_print(candidates, cohort_columns) -> tuple[dict, dict]:
    """Return (feature_lists, log) and print every exclusion per target."""
    cohort_columns = set(cohort_columns)
    candidates = list(candidates)
    feature_lists: dict[str, list[str]] = {}
    log: dict[str, dict] = {}
    for target in EXCLUSIONS:
        table = exclusion_table(target, cohort_columns)
        print(f"Leakage rules for {target}:")
        rows = []
        for column, reason in sorted(table.items(), key=lambda item: (item[1], item[0])):
            existed = column in cohort_columns
            was_candidate = column in candidates
            rows.append(
                {
                    "column": column,
                    "reason": reason,
                    "existed_in_cohort": existed,
                    "was_model_candidate": was_candidate,
                }
            )
            print(
                f"  {column:<26} {reason:<17} existed={existed!s:<5} "
                f"was_candidate={was_candidate}"
            )
        features = [column for column in candidates if column not in table]
        if any(column in table for column in features):
            raise SystemExit(f"FAIL: leakage column left in {target} features")
        feature_lists[target] = features
        log[target] = {"exclusions": rows, "features": features}
        print(f"  -> {target}: {len(features)} features kept of {len(candidates)} candidates")
    return feature_lists, log