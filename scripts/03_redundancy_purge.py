"""Script 03: redundancy purge, per target, on TRAIN rows only.

Step 1: Spearman filter (|rho| > 0.85, pairwise-complete, no imputation).
Step 2: iterative VIF pruning (VIF <= 5.0) on a purge-only median-imputed copy.

Reads feature_pool.parquet, split_indices.json, leakage_drop_log.json.
Writes per-target pruned train/test parquet files (NOT imputed) and
redundancy_log.json. The imputation here exists only to compute VIF. It is
never saved and is not the model imputer (that is fit later inside CV folds).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from config import INTERIM_DIR  # noqa: E402

PROCESSED_DIR = ROOT / "data" / "processed"
TARGETS = ("cvd_risk", "diabetes_risk")
SPEARMAN_LIMIT = 0.85
VIF_LIMIT = 5.0
MIN_PAIR_ROWS = 200
SMOKING_CODES = {"never": 0, "former": 1, "current": 2}
FASTING_FLAGS = ("fasting_glucose_measured", "triglycerides_measured")
FASTING_LABS = ("fasting_glucose", "triglycerides", "ldl_friedewald")


def fail(message: str) -> None:
    print(f"FAIL: {message}")
    raise SystemExit(1)


def numeric_view(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Numeric copy for the purge only. smoking_status is coded never/former/current = 0/1/2."""
    out = pd.DataFrame(index=frame.index)
    for column in columns:
        series = frame[column]
        if column == "smoking_status":
            series = series.map(SMOKING_CODES)
        out[column] = pd.to_numeric(series, errors="coerce")
    return out


def spearman_purge(num: pd.DataFrame) -> tuple[list[str], list[dict]]:
    corr = num.corr(method="spearman", min_periods=MIN_PAIR_ROWS)
    abs_corr = corr.abs()
    missing_frac = num.isna().mean()
    keep = list(num.columns)
    log: list[dict] = []
    while True:
        arr = abs_corr.loc[keep, keep].to_numpy(copy=True)
        np.fill_diagonal(arr, 0.0)
        arr = np.nan_to_num(arr, nan=0.0)
        if arr.size == 0:
            break
        i, j = np.unravel_index(arr.argmax(), arr.shape)
        rho = float(arr[i, j])
        if rho <= SPEARMAN_LIMIT:
            break
        a, b = keep[i], keep[j]
        # Drop the column with the higher mean |rho| to the others; ties by missing share, then name.
        score_a = (float(arr[i].mean()), float(missing_frac[a]), a)
        score_b = (float(arr[j].mean()), float(missing_frac[b]), b)
        drop, kept = (a, b) if score_a >= score_b else (b, a)
        log.append(
            {
                "pair": [a, b],
                "rho": round(float(corr.loc[a, b]), 4),
                "dropped": drop,
                "kept": kept,
                "rule": "dropped the column with the higher mean |rho| to the remaining columns",
            }
        )
        keep.remove(drop)
    return keep, log


def vif_values(matrix: np.ndarray) -> np.ndarray:
    corr = np.corrcoef(matrix, rowvar=False)
    return np.diag(np.linalg.pinv(corr))


def vif_purge(num: pd.DataFrame, keep: list[str]) -> tuple[list[str], list[dict], list[dict]]:
    imputed = num[keep].copy()
    imputed = imputed.fillna(imputed.median())
    path: list[dict] = []
    constant_log: list[dict] = []
    keep = list(keep)
    for column in list(keep):
        if imputed[column].nunique(dropna=True) < 2:
            constant_log.append({"dropped": column, "reason": "constant on train rows"})
            keep.remove(column)
    while len(keep) >= 2:
        values = vif_values(imputed[keep].to_numpy(dtype=float))
        worst = int(np.argmax(values))
        if values[worst] <= VIF_LIMIT:
            break
        path.append({"dropped": keep[worst], "vif": round(float(values[worst]), 3)})
        keep.pop(worst)
    return keep, path, constant_log


def max_vif(num: pd.DataFrame, keep: list[str]) -> float:
    imputed = num[keep].fillna(num[keep].median())
    if len(keep) < 2:
        return 1.0
    return float(vif_values(imputed.to_numpy(dtype=float)).max())


def main() -> None:
    pool_path = INTERIM_DIR / "feature_pool.parquet"
    split_path = INTERIM_DIR / "split_indices.json"
    leak_path = INTERIM_DIR / "leakage_drop_log.json"
    for path in (pool_path, split_path, leak_path):
        if not path.exists():
            fail(f"Missing input: {path}. Run Script 02 first.")

    pool = pd.read_parquet(pool_path)
    split_doc = json.loads(split_path.read_text(encoding="utf-8"))
    leak_log = json.loads(leak_path.read_text(encoding="utf-8"))
    seqn_test = set(split_doc["seqn_test"])
    print(f"Loaded feature pool: {len(pool)} rows. Split seed={split_doc['seed']}.")

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    full_log: dict = {
        "spearman_limit": SPEARMAN_LIMIT,
        "vif_limit": VIF_LIMIT,
        "purge_imputation": "median of train rows, used only to compute VIF, never saved",
        "smoking_status_coding_for_purge": SMOKING_CODES,
        "targets": {},
    }
    kept_by_target: dict[str, list[str]] = {}

    for target in TARGETS:
        features = [c for c in leak_log["targets"][target]["features"] if c != "ldl_friedewald"]  # derived from fasting labs; keep measured total_cholesterol
        train_rows = pool[(pool["split"] == "train") & pool[target].notna()]
        test_rows = pool[(pool["split"] == "test") & pool[target].notna()]
        if set(train_rows["SEQN"]) & seqn_test:
            fail(f"{target}: a test SEQN is among the rows used for the purge")

        print(f"\n=== {target} ===")
        print(f"Columns in: {len(features)}. Train rows used: {len(train_rows)} (test rows not read for statistics).")

        num = numeric_view(train_rows, features)
        keep, spearman_log = spearman_purge(num)
        print(f"Spearman |rho| > {SPEARMAN_LIMIT}: dropped {len(spearman_log)}")
        for item in spearman_log:
            print(f"  dropped {item['dropped']:<26} kept {item['kept']:<26} rho={item['rho']}")

        keep, vif_path, constant_log = vif_purge(num, keep)
        print(f"VIF <= {VIF_LIMIT}: dropped {len(vif_path) + len(constant_log)}")
        for item in constant_log:
            print(f"  dropped {item['dropped']} ({item['reason']})")
        for item in vif_path:
            print(f"  dropped {item['dropped']:<26} VIF={item['vif']}")

        final_vif = max_vif(num, keep)
        print(f"Columns kept: {len(keep)}. Max VIF remaining: {final_vif:.3f}")
        print("Kept:", ", ".join(keep))

        if final_vif > VIF_LIMIT:
            fail(f"{target}: max VIF {final_vif:.2f} exceeds {VIF_LIMIT}")
        if "RIDRETH3" in keep:
            fail("RIDRETH3 is in the pruned features")
        if any(c in keep for c in FASTING_LABS) and not any(c in keep for c in FASTING_FLAGS):
            fail(f"{target}: a fasting lab survives but no measured-flag does (A1)")
        surviving_flags = [c for c in FASTING_FLAGS if c in keep]
        print(f"Measured-flags surviving (A1): {surviving_flags}")

        keep_in_order = [c for c in features if c in keep]
        cols = ["SEQN", target] + keep_in_order
        train_out = train_rows[cols].reset_index(drop=True)
        test_out = test_rows[cols].reset_index(drop=True)
        if list(train_out.columns) != list(test_out.columns):
            fail(f"{target}: train and test column sets differ")
        train_out.to_parquet(PROCESSED_DIR / f"features_pruned_{target}_train.parquet", index=False)
        test_out.to_parquet(PROCESSED_DIR / f"features_pruned_{target}_test.parquet", index=False)
        print(f"Wrote pruned train ({len(train_out)} rows) and test ({len(test_out)} rows).")
        print("Correlations and VIF were computed on train rows only; test rows were only column-filtered.")

        kept_by_target[target] = keep_in_order
        full_log["targets"][target] = {
            "n_columns_in": len(features),
            "n_train_rows": int(len(train_rows)),
            "n_dropped_spearman": len(spearman_log),
            "n_dropped_vif": len(vif_path) + len(constant_log),
            "spearman_drops": spearman_log,
            "vif_path": vif_path,
            "constant_drops": constant_log,
            "max_vif_remaining": round(final_vif, 4),
            "final_columns": keep_in_order,
            "measured_flags_surviving": surviving_flags,
        }

    cvd, dia = set(kept_by_target["cvd_risk"]), set(kept_by_target["diabetes_risk"])
    print("\n=== Difference between targets (purge is separate per target) ===")
    print("Only in cvd_risk matrix:", sorted(cvd - dia) or "none")
    print("Only in diabetes_risk matrix:", sorted(dia - cvd) or "none")

    (PROCESSED_DIR / "redundancy_log.json").write_text(
        json.dumps(full_log, indent=2), encoding="utf-8"
    )
    print(f"Wrote {PROCESSED_DIR / 'redundancy_log.json'}")


if __name__ == "__main__":
    main()