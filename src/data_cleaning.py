"""
data_cleaning.py
================
HydroSense-Kenya | ICS 2207 Scientific Computing Capstone
----------------------------------------------------------
Reusable data-cleaning functions for the three project datasets:
  - weather_daily.csv
  - soil_sensor_data.csv
  - crop_zone_parameters.csv

Each function is self-contained, well-documented, and logs every
cleaning decision so that all transformations are reproducible and
auditable (as required by the Level 4 and Level 6 rubric).

Usage
-----
    from data_cleaning import (
        load_datasets,
        clean_weather,
        clean_soil_sensors,
        clean_crop_params,
        get_cleaning_report,
    )

    weather, soil, params = load_datasets()
    weather_clean, w_log   = clean_weather(weather)
    soil_clean,    s_log   = clean_soil_sensors(soil)
    params_clean,  p_log   = clean_crop_params(params)
    report = get_cleaning_report(w_log, s_log, p_log)
    print(report)
"""

import pandas as pd
import numpy as np
from pathlib import Path


# ---------------------------------------------------------------------------
# 1. LOADING
# ---------------------------------------------------------------------------

def load_datasets(
    data_dir: str = "data/raw",
    weather_file1: str = "weatherdata (6).csv",
    weather_file2: str = "weatherdata (7).csv",
    weather_file3: str = "weatherdata (8).csv",
    weather_file4: str = "weatherdata (9).csv",
    weather_file5: str = "weatherdata (10).csv",
    weather_file6: str = "weatherdata (11).csv",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
   
    base = Path(data_dir)
    na_vals = ["NA", ""]

    weather1= pd.read_csv(base / weather_file1, na_values=na_vals)
    weather2 = pd.read_csv(base / weather_file2,    na_values=na_vals)
    weather3 = pd.read_csv(base / weather_file3,  na_values=na_vals)
    weather4 = pd.read_csv(base / weather_file4, na_values=na_vals)
    weather5 = pd.read_csv(base / weather_file5, na_values=na_vals)
    weather6 = pd.read_csv(base / weather_file6, na_values=na_vals)
    
    weather = pd.concat([weather1, weather2, weather3, weather4, weather5, weather6],
    ignore_index=True
    )

# Optional: sort by date and drop duplicates if any overlap
    weather["date"] = pd.to_datetime(merged_weather["date"])
    weather = (weather
                .sort_values("date")
                .drop_duplicates()
                .reset_index(drop=True))

    print(weather.shape)
    print(weather.head())

    # Parse date/timestamp columns
    weather["date"] = pd.to_datetime(weather["date"])
  

    return weather

# 2. WEATHER CLEANING
# Physical plausibility bounds for Kenyan farm conditions
_WEATHER_BOUNDS = {
    "rainfall_mm":     (0.0,  80.0),   # >80 mm/day is extreme; flag as outlier
    "temperature_c":   (10.0, 40.0),   # 45.8 °C is physically implausible here
    "humidity_pct":    (10.0, 100.0),
    "wind_speed_mps":  (0.0,  15.0),
    "solar_index":     (0.0,  1.0),
}


def detect_weather_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a boolean mask DataFrame (True = outlier) for each numeric
    weather column using the physical plausibility bounds in
    _WEATHER_BOUNDS.

    Parameters
    ----------
    df : pd.DataFrame
        Raw or partially-cleaned weather DataFrame.

    Returns
    -------
    pd.DataFrame
        Boolean DataFrame of the same shape; True where a value is
        outside its expected range.
    """
    mask = pd.DataFrame(False, index=df.index, columns=df.columns)
    for col, (lo, hi) in _WEATHER_BOUNDS.items():
        if col in df.columns:
            mask[col] = df[col].notna() & ((df[col] < lo) | (df[col] > hi))
    return mask


def clean_weather(
    df: pd.DataFrame,
    impute_method: str = "linear",
) -> tuple[pd.DataFrame, list[dict]]:
    """
    Clean the daily weather dataset.

    Cleaning steps (all logged):
      1. Outlier detection using physical bounds.
      2. Replace outliers with NaN so they are handled in step 3.
      3. Impute missing values (NaN / NA / replaced outliers) using
         linear interpolation (default) or forward-fill.
      4. Clip any remaining values to physical bounds.
      5. Ensure 'date' column is sorted and monotonically increasing.

    Parameters
    ----------
    df : pd.DataFrame
        Raw weather DataFrame produced by load_datasets().
    impute_method : str
        "linear"  – pandas interpolate(method="linear")  [default]
        "ffill"   – forward-fill then backward-fill

    Returns
    -------
    cleaned : pd.DataFrame
        Cleaned weather DataFrame with the same columns.
    log : list of dict
        One entry per cleaning action with keys:
        column, action, row_indices, old_values, new_values, reason.
    """
    cleaned = df.copy()
    log: list[dict] = []

    numeric_cols = list(_WEATHER_BOUNDS.keys())

    # --- Step 1 & 2: Detect and nullify outliers --------------------------
    outlier_mask = detect_weather_outliers(cleaned)
    for col in numeric_cols:
        if col not in cleaned.columns:
            continue
        bad_idx = outlier_mask.index[outlier_mask[col]].tolist()
        if bad_idx:
            lo, hi = _WEATHER_BOUNDS[col]
            log.append({
                "column":      col,
                "action":      "outlier → NaN",
                "row_indices": bad_idx,
                "old_values":  cleaned.loc[bad_idx, col].tolist(),
                "new_values":  [np.nan] * len(bad_idx),
                "reason":      f"Value outside physical range [{lo}, {hi}]",
            })
            cleaned.loc[bad_idx, col] = np.nan

    # --- Step 3: Impute missing values ------------------------------------
    for col in numeric_cols:
        if col not in cleaned.columns:
            continue
        missing_idx = cleaned.index[cleaned[col].isna()].tolist()
        if not missing_idx:
            continue

        if impute_method == "linear":
            cleaned[col] = cleaned[col].interpolate(
                method="linear", limit_direction="both"
            )
        else:
            cleaned[col] = cleaned[col].ffill().bfill()

        imputed_values = cleaned.loc[missing_idx, col].tolist()
        log.append({
            "column":      col,
            "action":      f"impute ({impute_method})",
            "row_indices": missing_idx,
            "old_values":  [np.nan] * len(missing_idx),
            "new_values":  imputed_values,
            "reason":      "Missing value filled by interpolation",
        })

    # --- Step 4: Final clip to physical bounds ----------------------------
    for col, (lo, hi) in _WEATHER_BOUNDS.items():
        if col not in cleaned.columns:
            continue
        clipped = cleaned[col].clip(lower=lo, upper=hi)
        changed_idx = cleaned.index[cleaned[col] != clipped].tolist()
        if changed_idx:
            log.append({
                "column":      col,
                "action":      "clip",
                "row_indices": changed_idx,
                "old_values":  cleaned.loc[changed_idx, col].tolist(),
                "new_values":  clipped.loc[changed_idx].tolist(),
                "reason":      f"Clipped to [{lo}, {hi}] after imputation",
            })
        cleaned[col] = clipped

    # --- Step 5: Sort by date ---------------------------------------------
    cleaned = cleaned.sort_values("date").reset_index(drop=True)

    return cleaned, log


# ---------------------------------------------------------------------------
# 4. CROP ZONE PARAMETERS CLEANING
# ---------------------------------------------------------------------------

def clean_crop_params(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict]]:
    """
    Clean the crop zone parameters dataset.

    Cleaning steps (all logged):
      1. Normalise zone_id and crop_type casing.
      2. Verify that min_moisture < target_moisture < field_capacity
         for every zone; log any violations.
      3. Confirm drainage_coefficient is in (0, 1); log anomalies.
      4. Drop fully duplicate rows.

    Parameters
    ----------
    df : pd.DataFrame
        Raw crop parameters DataFrame.

    Returns
    -------
    cleaned : pd.DataFrame
        Cleaned parameters DataFrame.
    log : list of dict
        Cleaning log entries.
    """
    cleaned = df.copy()
    log: list[dict] = []

    # --- Step 1: Normalise text columns -----------------------------------
    for col in ["zone_id", "crop_type"]:
        if col in cleaned.columns:
            cleaned[col] = cleaned[col].str.strip().str.title()

    # --- Step 2: Moisture threshold ordering check -----------------------
    moisture_cols = ["min_moisture_pct", "target_moisture_pct", "field_capacity_pct"]
    if all(c in cleaned.columns for c in moisture_cols):
        bad_order = cleaned.index[
            ~(
                (cleaned["min_moisture_pct"] < cleaned["target_moisture_pct"])
                & (cleaned["target_moisture_pct"] < cleaned["field_capacity_pct"])
            )
        ].tolist()
        if bad_order:
            log.append({
                "column":      "moisture thresholds",
                "action":      "flagged ordering violation",
                "row_indices": bad_order,
                "old_values":  cleaned.loc[bad_order, moisture_cols].values.tolist(),
                "new_values":  ["no change — manual review required"] * len(bad_order),
                "reason":      "Expected: min < target < field_capacity",
            })

    # --- Step 3: Drainage coefficient bounds -----------------------------
    if "drainage_coefficient" in cleaned.columns:
        bad_drain = cleaned.index[
            ~cleaned["drainage_coefficient"].between(0, 1, inclusive="neither")
        ].tolist()
        if bad_drain:
            log.append({
                "column":      "drainage_coefficient",
                "action":      "flagged out-of-range",
                "row_indices": bad_drain,
                "old_values":  cleaned.loc[bad_drain, "drainage_coefficient"].tolist(),
                "new_values":  ["no change — manual review required"] * len(bad_drain),
                "reason":      "drainage_coefficient must be in (0, 1)",
            })

    # --- Step 4: Drop exact duplicates -----------------------------------
    n_before = len(cleaned)
    cleaned = cleaned.drop_duplicates().reset_index(drop=True)
    n_dropped = n_before - len(cleaned)
    if n_dropped:
        log.append({
            "column":      "all",
            "action":      "drop duplicates",
            "row_indices": [],
            "old_values":  [],
            "new_values":  [],
            "reason":      f"{n_dropped} fully duplicate row(s) removed",
        })

    return cleaned, log


# ---------------------------------------------------------------------------
# 5. DESCRIPTIVE STATISTICS SUMMARY
# ---------------------------------------------------------------------------

def descriptive_stats(df: pd.DataFrame, label: str = "") -> pd.DataFrame:
    """
    Compute descriptive statistics for all numeric columns in a DataFrame.

    Returns mean, std, min, 25%, 50%, 75%, max, and count of missing values.

    Parameters
    ----------
    df : pd.DataFrame
        Any cleaned or raw DataFrame.
    label : str
        Optional label printed above the table (for display purposes).

    Returns
    -------
    pd.DataFrame
        Summary statistics table.
    """
    numeric = df.select_dtypes(include="number")
    stats = numeric.describe().T
    stats["missing"] = numeric.isna().sum()
    if label:
        print(f"\n{'─' * 60}")
        print(f"  Descriptive Statistics: {label}")
        print(f"{'─' * 60}")
        print(stats.to_string())
    return stats


# ---------------------------------------------------------------------------
# 6. CLEANING REPORT
# ---------------------------------------------------------------------------

def get_cleaning_report(
    *logs: list[dict],
    dataset_labels: list[str] | None = None,
) -> str:
    """
    Generate a human-readable cleaning report from one or more cleaning logs.

    Parameters
    ----------
    *logs : list of dict
        One or more cleaning logs returned by the clean_* functions.
    dataset_labels : list of str, optional
        Labels for each log (e.g. ["Weather", "Soil", "Params"]).
        Defaults to ["Dataset 1", "Dataset 2", ...].

    Returns
    -------
    str
        A formatted multi-line report string suitable for printing or
        writing to a markdown file.
    """
    labels = dataset_labels or [f"Dataset {i+1}" for i in range(len(logs))]
    lines: list[str] = ["# HydroSense-Kenya — Data Cleaning Report\n"]

    total_actions = 0

    for label, log in zip(labels, logs):
        lines.append(f"## {label}")
        if not log:
            lines.append("  No cleaning actions were necessary.\n")
            continue

        for entry in log:
            n = len(entry["row_indices"]) if entry["row_indices"] else "N/A"
            lines.append(
                f"  - **{entry['action']}** on `{entry['column']}` "
                f"({n} row(s)): {entry['reason']}"
            )
            total_actions += 1
        lines.append("")

    lines.append(f"---\n**Total cleaning actions logged:** {total_actions}")
    return "\n".join(lines)


def save_cleaned_datasets(
    weather_clean: pd.DataFrame,
    soil_clean: pd.DataFrame,
    params_clean: pd.DataFrame,
    output_dir: str = "data/processed",
) -> None:
    """
    Save cleaned DataFrames to the processed data directory as CSV files.

    Parameters
    ----------
    weather_clean : pd.DataFrame
    soil_clean : pd.DataFrame
    params_clean : pd.DataFrame
    output_dir : str
        Destination folder (created if it does not exist).
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    weather_clean.to_csv(out / "weather_daily_cleaned.csv", index=False)
    soil_clean.to_csv(out    / "soil_sensor_cleaned.csv",   index=False)
    params_clean.to_csv(out  / "crop_zone_params_cleaned.csv", index=False)

    print(f"Cleaned datasets saved to '{output_dir}/'")


# ---------------------------------------------------------------------------
# 7. QUICK DEMO (run as a script)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Loading datasets...")
    weather, soil, params = load_datasets()

    print("\nCleaning weather data...")
    weather_clean, w_log = clean_weather(weather)

    print("Cleaning soil sensor data...")
    soil_clean, s_log = clean_soil_sensors(soil)

    print("Cleaning crop zone parameters...")
    params_clean, p_log = clean_crop_params(params)

    # Print report
    report = get_cleaning_report(
        w_log, s_log, p_log,
        dataset_labels=["Weather (weather_daily.csv)",
                        "Soil Sensors (soil_sensor_data.csv)",
                        "Crop Parameters (crop_zone_parameters.csv)"],
    )
    print("\n" + report)

    # Print descriptive statistics
    descriptive_stats(weather_clean, "Weather (cleaned)")
    descriptive_stats(soil_clean,    "Soil Sensors (cleaned)")
    descriptive_stats(params_clean,  "Crop Parameters (cleaned)")

    # Save
    save_cleaned_datasets(weather_clean, soil_clean, params_clean)