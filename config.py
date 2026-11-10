"""Paths, codebook rules, and triage thresholds for the NHANES cohort.

Script 01 reads this module. Later scripts should import it rather than
copying constants.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT.parent / "Data"
INTERIM_DIR = PROJECT_ROOT / "data" / "interim"

RANDOM_SEED = 42
TEST_SIZE = 0.20
MIN_AGE = 18

REQUIRED_FILES = (
    "P_DEMO.csv",
    "P_BMX.csv",
    "P_BPXO.csv",
    "P_GHB.csv",
    "P_GLU.csv",
    "P_TCHOL.csv",
    "P_HDL.csv",
    "P_TRIGLY.csv",
    "P_MCQ.csv",
    "P_DIQ.csv",
    "P_BPQ.csv",
    "P_SMQ.csv",
    "P_PAQ.csv",
    "P_CDQ.csv",
    "P_ALQ.csv",
)

# Questionnaire items whose codebook uses refused / don't-know codes.
# Applied only to these columns. 7 and 9 are real answers on ALQ121
# (once a month; 3 to 6 times in the last year) and on ALQ130 (drink counts).
# RIDRETH3 value 7 is "Other race, including multi-racial", not a missing code.
DISCRETE_MISSING_CODES = {
    "DIQ010": (7, 9),
    "DIQ050": (7, 9),
    "DIQ070": (7, 9),
    "MCQ160B": (7, 9),
    "MCQ160C": (7, 9),
    "MCQ160D": (7, 9),
    "MCQ160E": (7, 9),
    "MCQ160F": (7, 9),
    "MCQ300A": (7, 9),
    "MCQ300B": (7, 9),
    "MCQ300C": (7, 9),
    "BPQ020": (7, 9),
    "BPQ040A": (7, 9),
    "BPQ050A": (7, 9),
    "BPQ080": (7, 9),
    "BPQ090D": (7, 9),
    "BPQ100D": (7, 9),
    "SMQ020": (7, 9),
    "SMQ040": (7, 9),
    "CDQ001": (7, 9),
    "DMDEDUC2": (7, 9),
    "ALQ111": (7, 9),
    "ALQ121": (77, 99),
    "ALQ130": (777, 999),
    "PAQ605": (7, 9),
    "PAQ610": (77, 99),
    "PAD615": (7777, 9999),
    "PAQ620": (7, 9),
    "PAQ625": (77, 99),
    "PAD630": (7777, 9999),
    "PAQ635": (7, 9),
    "PAQ640": (77, 99),
    "PAD645": (7777, 9999),
    "PAQ650": (7, 9),
    "PAQ655": (77, 99),
    "PAD660": (7777, 9999),
    "PAQ665": (7, 9),
    "PAQ670": (77, 99),
    "PAD675": (7777, 9999),
}

# Exam and lab fields. Blank means not measured. Never run discrete codes here.
CONTINUOUS_COLUMNS = (
    "RIDAGEYR",
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
    "LBXGH",
    "LBXGLU",
    "LBXTC",
    "LBDHDD",
    "LBXTR",
    "WTINTPRP",
    "WTMECPRP",
    "WTSAFPRP",
)

CVD_SOURCE_COLUMNS = ("MCQ160B", "MCQ160C", "MCQ160D", "MCQ160E", "MCQ160F")

# English text from the 2017–March 2020 P_CDQ codebook.
# The triage rule still uses CDQ001 only, until a stricter definition is approved.
CDQ_QUESTIONS = {
    "CDQ001": "Have you ever had any pain or discomfort in your chest?",
    "CDQ002": "Do you get it when you walk uphill or hurry?",
    "CDQ003": "Do you get it when you walk at an ordinary pace on level ground?",
    "CDQ004": "If you get it while walking, do you stop or slow down, or continue at the same pace?",
    "CDQ005": "If you stand still, is the pain or discomfort relieved?",
    "CDQ006": "How soon is the pain relieved? 10 minutes or less, or more than 10 minutes?",
    "CDQ008": "Have you ever had a severe pain across the front of your chest lasting for half an hour or more?",
    "CDQ009D": "Pain located in the upper sternum (code 4).",
    "CDQ009E": "Pain located in the lower sternum (code 5).",
    "CDQ009F": "Pain located in the left chest (code 6).",
    "CDQ009G": "Pain located in the left arm (code 7).",
    "CDQ010": "Have you had shortness of breath either when hurrying on the level or walking up a slight hill?",
}

FAMILY_HISTORY_QUESTIONS = {
    "MCQ300A": (
        "Including living and deceased, were any close biological relatives "
        "ever told they had a heart attack or angina before the age of 50?"
    ),
    "MCQ300B": (
        "Including living and deceased, were any close biological relatives "
        "ever told they had asthma?"
    ),
    "MCQ300C": (
        "Including living and deceased, were any close biological relatives "
        "ever told they had diabetes?"
    ),
}

# CDC English text, shortened only by dropping the {you/SP} interview branches.
# Source: NHANES 2017–March 2020 P_ALQ codebook.
ALCOHOL_FREQUENCY_PER_WEEK = {
    0: 0.0,          # never in the last year
    1: 7.0,          # every day
    2: 6.0,          # nearly every day
    3: 3.5,          # 3 to 4 times a week
    4: 2.0,          # 2 times a week
    5: 1.0,          # once a week
    6: 2.5 / 4.345,  # 2 to 3 times a month
    7: 1.0 / 4.345,  # once a month
    8: 9.0 / 52.0,   # 7 to 11 times in the last year (midpoint 9)
    9: 4.5 / 52.0,   # 3 to 6 times in the last year
    10: 1.5 / 52.0,  # 1 to 2 times in the last year
}

# Moderate and vigorous domains, including walking or cycling as moderate.
# (yes/no, days per week, minutes on a typical day)
ACTIVITY_DOMAINS = (
    ("PAQ605", "PAQ610", "PAD615"),
    ("PAQ620", "PAQ625", "PAD630"),
    ("PAQ635", "PAQ640", "PAD645"),
    ("PAQ650", "PAQ655", "PAD660"),
    ("PAQ665", "PAQ670", "PAD675"),
)

EDUCATION_LABELS = {
    1: "Less than 9th grade",
    2: "9th to 11th grade",
    3: "High school graduate or GED",
    4: "Some college or associate degree",
    5: "College graduate or above",
}

SEX_LABELS = {1: "Male", 2: "Female"}

# Friedewald is not used when triglycerides exceed this value (mg/dL).
LDL_TRIGLYCERIDE_MAX = 400

TIER_THRESHOLDS = {
    "sbp_tier4": 180,
    "dbp_tier4": 120,
    "hba1c_tier4": 10.0,
    "glucose_tier4": 250,
    "spo2_tier4": 90,
    "sbp_tier3": 140,
    "dbp_tier3": 90,
    "hba1c_tier3": 6.5,
    "glucose_tier3": 126,
    "cholesterol_tier3": 240,
    "bmi_tier2": 30,
    "sbp_tier2": 130,
    "dbp_tier2": 80,
    "hba1c_tier2": 5.7,
    "glucose_tier2": 100,
}

WEIGHT_COLUMNS = ("WTINTPRP", "WTMECPRP", "WTSAFPRP")

# Columns Script 02 is allowed to consider as model inputs.
# Race/ethnicity, labels, medications that define a label, chest pain, and
# survey weights are intentionally absent.
MODEL_CANDIDATE_COLUMNS = (
    "age_years",
    "sex",
    "education",
    "bmi",
    "height_cm",
    "weight_kg",
    "waist_cm",
    "sbp_mean",
    "dbp_mean",
    "heart_rate_mean",
    "total_cholesterol",
    "hdl",
    "triglycerides",
    "triglycerides_measured",
    "ldl_friedewald",
    "hba1c",
    "fasting_glucose",
    "fasting_glucose_measured",
    "told_high_bp",
    "on_bp_med",
    "told_high_chol",
    "on_chol_med",
    "smoking_status",
    "activity_minutes_week",
    "alcohol_drinks_per_week",
    "family_hx_early_heart",
    "family_hx_diabetes",
)

# MCQ300B is a close relative with asthma. It is stored on the cohort file
# and is not a model input.
NOT_MODEL_COLUMNS = ("family_hx_asthma",)

RULE_ONLY_COLUMNS = ("chest_pain",)
SUBGROUP_ONLY_COLUMNS = ("RIDRETH3",)
LABEL_COLUMNS = ("cvd_risk", "diabetes_risk")

INSUFFICIENT_TIER = "Not enough measurements"
TIER_NAMES = {
    4: "Immediate clinical flag",
    3: "Specialist consult",
    2: "Lifestyle change",
    1: "Healthy / routine",
}
