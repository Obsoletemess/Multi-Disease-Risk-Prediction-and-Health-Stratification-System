"""Rule-based triage shared by the cohort audit and the live form.

The function reads measurements only. It does not read cvd_risk,
diabetes_risk, or the MCQ160 diagnosis items. A missing measurement
does not match a clause. "Not enough measurements" is a triage state,
not a reason to drop a learned-target row.
"""

from __future__ import annotations

import math

from config import INSUFFICIENT_TIER, TIER_NAMES, TIER_THRESHOLDS

TIER_INPUT_COLUMNS = (
    "age_years",
    "bmi",
    "sbp_mean",
    "dbp_mean",
    "hba1c",
    "fasting_glucose",
    "total_cholesterol",
    "current_smoker",
    "activity_minutes_week",
    "chest_pain",
    "fatigue",
    "dizziness",
    "spo2",
)

FORBIDDEN_TIER_INPUTS = (
    "cvd_risk",
    "diabetes_risk",
    "MCQ160B",
    "MCQ160C",
    "MCQ160D",
    "MCQ160E",
    "MCQ160F",
)


def _missing(value) -> bool:
    if value is None:
        return True
    try:
        if isinstance(value, str):
            return value.strip() == ""
        return math.isnan(float(value))
    except (TypeError, ValueError):
        return False


def _at_least(value, cut) -> bool:
    if _missing(value):
        return False
    return float(value) >= cut


def _below(value, cut) -> bool:
    if _missing(value):
        return False
    return float(value) < cut


def _is_yes(value) -> bool:
    if _missing(value):
        return False
    return float(value) == 1.0


def _is_zero(value) -> bool:
    if _missing(value):
        return False
    return float(value) == 0.0


def assign_tier(measurements: dict) -> tuple[str, str]:
    """Return (tier_label, rule_trace).

    tier_label is one of TIER_NAMES values or INSUFFICIENT_TIER.
    The trace lists only the clauses that set the returned tier.
    """
    forbidden = [key for key in measurements if key in FORBIDDEN_TIER_INPUTS]
    if forbidden:
        raise ValueError(
            "assign_tier received diagnosis fields it must not read: "
            + ", ".join(forbidden)
        )
    unknown = [key for key in measurements if key not in TIER_INPUT_COLUMNS]
    if unknown:
        raise ValueError(
            "assign_tier received columns outside the measurement list: "
            + ", ".join(unknown)
        )

    row = {key: measurements.get(key) for key in TIER_INPUT_COLUMNS}
    cuts = TIER_THRESHOLDS

    tier4 = []
    if _at_least(row["sbp_mean"], cuts["sbp_tier4"]):
        tier4.append(f"SBP {float(row['sbp_mean']):.0f} matched Tier 4: systolic >= {cuts['sbp_tier4']}")
    if _at_least(row["dbp_mean"], cuts["dbp_tier4"]):
        tier4.append(f"DBP {float(row['dbp_mean']):.0f} matched Tier 4: diastolic >= {cuts['dbp_tier4']}")
    if _at_least(row["hba1c"], cuts["hba1c_tier4"]):
        tier4.append(f"HbA1c {float(row['hba1c']):.1f} matched Tier 4: HbA1c >= {cuts['hba1c_tier4']}")
    if _at_least(row["fasting_glucose"], cuts["glucose_tier4"]):
        tier4.append(
            f"fasting glucose {float(row['fasting_glucose']):.0f} matched Tier 4: "
            f"fasting glucose >= {cuts['glucose_tier4']}"
        )
    if _below(row["spo2"], cuts["spo2_tier4"]):
        tier4.append(f"SpO2 {float(row['spo2']):.0f} matched Tier 4: SpO2 < {cuts['spo2_tier4']}")
    if _is_yes(row["chest_pain"]) and _is_yes(row["dizziness"]):
        tier4.append("chest pain and dizziness both present matched Tier 4")
    if tier4:
        return TIER_NAMES[4], " | ".join(tier4)

    tier3 = []
    if _at_least(row["sbp_mean"], cuts["sbp_tier3"]):
        tier3.append(f"SBP {float(row['sbp_mean']):.0f} matched Tier 3: systolic >= {cuts['sbp_tier3']}")
    if _at_least(row["dbp_mean"], cuts["dbp_tier3"]):
        tier3.append(f"DBP {float(row['dbp_mean']):.0f} matched Tier 3: diastolic >= {cuts['dbp_tier3']}")
    if _at_least(row["hba1c"], cuts["hba1c_tier3"]):
        tier3.append(f"HbA1c {float(row['hba1c']):.1f} matched Tier 3: HbA1c >= {cuts['hba1c_tier3']}")
    if _at_least(row["fasting_glucose"], cuts["glucose_tier3"]):
        tier3.append(
            f"fasting glucose {float(row['fasting_glucose']):.0f} matched Tier 3: "
            f"fasting glucose >= {cuts['glucose_tier3']}"
        )
    if _at_least(row["total_cholesterol"], cuts["cholesterol_tier3"]):
        tier3.append(
            f"total cholesterol {float(row['total_cholesterol']):.0f} matched Tier 3: "
            f"total cholesterol >= {cuts['cholesterol_tier3']}"
        )
    if _is_yes(row["chest_pain"]):
        tier3.append("chest pain matched Tier 3")
    if _is_yes(row["fatigue"]):
        tier3.append("unusual fatigue matched Tier 3")
    if tier3:
        return TIER_NAMES[3], " | ".join(tier3)

    tier2 = []
    if _at_least(row["bmi"], cuts["bmi_tier2"]):
        tier2.append(f"BMI {float(row['bmi']):.1f} matched Tier 2: BMI >= {cuts['bmi_tier2']}")
    if _at_least(row["sbp_mean"], cuts["sbp_tier2"]):
        tier2.append(f"SBP {float(row['sbp_mean']):.0f} matched Tier 2: systolic >= {cuts['sbp_tier2']}")
    if _at_least(row["dbp_mean"], cuts["dbp_tier2"]):
        tier2.append(f"DBP {float(row['dbp_mean']):.0f} matched Tier 2: diastolic >= {cuts['dbp_tier2']}")
    if _at_least(row["hba1c"], cuts["hba1c_tier2"]):
        tier2.append(f"HbA1c {float(row['hba1c']):.1f} matched Tier 2: HbA1c >= {cuts['hba1c_tier2']}")
    if _at_least(row["fasting_glucose"], cuts["glucose_tier2"]):
        tier2.append(
            f"fasting glucose {float(row['fasting_glucose']):.0f} matched Tier 2: "
            f"fasting glucose >= {cuts['glucose_tier2']}"
        )
    if _is_yes(row["current_smoker"]):
        tier2.append("current smoking matched Tier 2")
    if _is_zero(row["activity_minutes_week"]):
        tier2.append("activity minutes present and equal to 0 matched Tier 2")
    if tier2:
        return TIER_NAMES[2], " | ".join(tier2)

    core_missing = []
    if _missing(row["age_years"]):
        core_missing.append("age")
    if _missing(row["bmi"]):
        core_missing.append("BMI")
    if _missing(row["sbp_mean"]) or _missing(row["dbp_mean"]):
        core_missing.append("blood pressure")
    if core_missing:
        detail = ", ".join(core_missing)
        return INSUFFICIENT_TIER, f"Not enough measurements: {detail} missing"
    return TIER_NAMES[1], "No higher clause fired. Age, BMI, and both blood-pressure readings are present."
