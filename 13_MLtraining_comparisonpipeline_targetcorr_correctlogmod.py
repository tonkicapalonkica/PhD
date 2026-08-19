"""
Model comparison pipeline for predicting apparent drug solubility in surfactants.

Assumes a tidy dataframe with one row per (drug, surfactant) pair:
    drug_name | surfactant_name | <drug descriptors...> | <surfactant descriptors...> | log modulus solpow to handle negative values

This script supports one workflow:
1) With near-zero-variance feature reduction (both drug and surfactants descriptors filtered inside each CV training fold to avoid leakage)

It also includes:
- Hyperparameter tuning for all models
- Group k-fold evaluation (no holdout)
- Leave-one-drug-out evaluation (no holdout)
- Nested holdout evaluation (outer holdout + inner tuning)
- Scaling only for models where it is needed

Important leakage-safety note:
Feature filtering is performed inside each pipeline fit, so in CV (cross-validation) each training fold
computes its own selected features. Test fold rows are transformed using only training
fold decisions.
"""
# import all the necessary libraries
import numpy as np # for numerical operations
import pandas as pd # for data manipulation
from collections import Counter # for counting occurrences of features across folds
import ctypes # for preventing Windows sleep mode during long runs
import sys # so the script can check if it's running on Windows and apply specific settings to prevent sleep mode during long computations.
from pathlib import Path # for creating output directory and saving result tables
from datetime import datetime # for timestamping exported result files
from sklearn.base import BaseEstimator, TransformerMixin, clone # for creating custom transformers
from sklearn.model_selection import RepeatedKFold, LeaveOneGroupOut, GroupKFold, GridSearchCV, cross_validate # for cross-validation and hyperparameter tuning
from sklearn.pipeline import Pipeline # for building machine learning pipelines
from sklearn.preprocessing import StandardScaler # for feature scaling for linear models
from sklearn.feature_selection import VarianceThreshold # for variance-based feature selection
from sklearn.linear_model import Ridge, Lasso # for linear regression models
from sklearn.cross_decomposition import PLSRegression # for Partial Least Squares regression
from sklearn.gaussian_process import GaussianProcessRegressor # for Gaussian Process regression (GP)
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel # for defining GP kernels
from sklearn.svm import SVR # for Support Vector Regression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor # for tree-based ensemble models
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score # for nested CV metric calculation

try:
    from xgboost import XGBRegressor # XGBoost - good for tabular data so included in the model comparison
    HAS_XGBOOST = True
except ImportError:
    XGBRegressor = None
    HAS_XGBOOST = False

print("Libraries imported.")

# ---------------------------------------------------------------------------
# 0. Configuration
# ---------------------------------------------------------------------------
DATA_FILE = "13_drugsurf_moldescr_withnames_zerovariationfiltered_logmodsolpow_correct.csv" # name of file with rdkit descriptors for drugs, manual for surfs and solpow values
TARGET_COL = "log-mod(solpow)"  # column name for the target variable (log-modulus of solubilisation power, umol/g: )
GROUP_COL = "drug_NAME"  # used for leave-one-drug-out CV (cross-validation)
SURF_GROUP_COL = "surf_NAME"  # used to define unique surfactants for surf descriptor filtering

# Run all validation modes in one pass and compare them:
# 1) Standard no-holdout flow - GroupKfold and LOGO
# 2) Nested holdout flow - each repeat creates a never-seen holdout test split, tunes only on the
# corresponding training split, then evaluates on holdout.
# if True, the script runs all three validation modes and prints a side-by-side comparison.
# If False, only the standard no-holdout flow is run (1).
# Note: always set to True as code modified to exclude other options to simplify the script.
USE_COMPARE_ALL_VALIDATION_MODES = True

# Near-zero-variance filtering threshold. - if variance is below this threshold, the descriptor is dropped.
# (exactly zero variance reduction already performed previously in a separate script)

# Drug-side feature reduction thresholds. - if variance is below this threshold, the descriptor is dropped. 
# If correlation is above this threshold, one of the correlated descriptors is dropped.
DRUG_VAR_THRESHOLD = 0.01 # variance cut off for drug descriptors 
# Variance measures how much a descriptor changes across compounds. 
# If a drug descriptor barely changes, it is unlikely to help the model much. 
# This removes drug descriptors that are almost constant across the drugs in the current training fold.
DRUG_CORR_THRESHOLD = 0.95
# Target-correlation relevance threshold, applied AFTER variance filtering and BEFORE
# interfeature correlation dedup. Drops any drug descriptor whose |correlation| with
# the target (within the current training fold's unique drugs) falls below this value.
# Based on Cohen (1988) conventions, 0.2 sits at the boundary between "negligible" and
# "weak" effect sizes for a correlation coefficient. Based on Hemphill, J. F. (2003) this is the lower third (small).
DRUG_TARGET_CORR_THRESHOLD = 0.2

# Surfactant-side feature reduction thresholds. - same thing just for surfactants
SURF_VAR_THRESHOLD = 0.01
SURF_CORR_THRESHOLD = 0.95

# Target correlation for surfactant descriptors is computed at ROW level (all drug-surfactant pairings), not on the 5 deduplicated surfactants, 
# so it uses the full replication in the data and is statistically valid to threshold on (unlike a correlation computed from only 5 points).
# A formal significance test (row-level correlation p-value) would be an even more defensible, non-arbitrary alternative to a fixed magnitude cutoff. 
# However, 0.2 is not arbitrary (Hemphill 2003, and Cohen 1998). p kinda complex - with ~109 drug descriptors, interfeature correlation involves testing 
# ~5,900 pairs simultaneously. At an uncorrected α=0.05, you'd expect roughly 295 "significant" results purely by chance, even if no real correlations existed.
# If significance-testing route chosen, a multiple-comparisons correction — Benjamini-Hochberg (FDR) needs to be done for this many tests
SURF_TARGET_CORR_THRESHOLD = 0.2

# Cross-validation settings. - used for tuning inside GridSearchCV to choose hyperparameters and for final evaluation
# after tuning to estimate model performance on unseen data.
# each repeat makes a new random 5-fold split of the data, and the results are averaged across all repeats.
# 25 total fits per hyperparameter setting (5 folds x 5 repeats) for tuning, and 
# 100 total fits per model for evaluation (5 folds x 20 repeats).
# Purpose in evaluation is to give a more stable estimate of RMSE/MAE/R² with less split-to-split noise
RANDOM_STATE = 31 # makes split generation reproducible, can be any number. Different random states will give different splits and slightly different results.
TUNE_N_SPLITS = 5
TUNE_N_REPEATS = 5 # Repeating (5 for tuning, 20 for evaluation) reduces variance from random splits.
EVAL_N_SPLITS = 5
EVAL_N_REPEATS = 20 # evaluation needs higher stability for reporting

# Nested holdout settings.
# In grouped mode with GroupKFold, this is the number of outer folds.
# With 5 folds, each drug group appears in the held-out set exactly once.
NESTED_HOLDOUT_N_REPEATS = 5

# Nested grouped-inner settings. - controls how hyperparameter tuning is done inside each training split.
# Outer split is grouped: train/test split is done by drug group, so the test set contains drugs not seen in that
# outer training set.
# Inner loop also uses GroupKFold on drug groups, aligning tuning with the unseen-drug evaluation goal.
NESTED_GROUPED_INNER_N_SPLITS = 5 # reducing inner folds from 5 to 4 can make tuning more stable, if needed

# For compact output when printing coefficient/importance tables.
TOP_N_FEATURES_TO_PRINT = 20
TOP_N_TUNING_ROWS_TO_EXPORT = 10

# CSV export settings.
RESULTS_OUTPUT_DIR = "13_ML_results"

print("Configuration complete.")

# make sure the laptop doesn't go into sleep mode while script is running
def _set_windows_sleep_prevention(enable: bool) -> None:
    """Prevent or restore Windows sleep while the script is running."""
    if sys.platform != "win32":
        return

    kernel32 = ctypes.windll.kernel32
    if enable:
        # Keep the system and display awake while the Python process is active.
        kernel32.SetThreadExecutionState(0x80000000 | 0x00000001 | 0x00000002)
    else:
        # Clear the execution-state flags so Windows can resume normal sleep behavior.
        kernel32.SetThreadExecutionState(0x80000000)

print("Sleep prevention function defined.")

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
df = pd.read_csv(DATA_FILE)

# Molecular descriptors have been generated separately.
# Expected data format: one row per (drug, surfactant) pair, with drug descriptor columns
# prefixed "drug_" and surfactant descriptor columns prefixed "surf_"
# Descriptor columns are identified by prefix, but metadata name/label columns
# that share those prefixes must be excluded from model features.
NON_FEATURE_NAME_COLS = {"drug_NAME", "surf_NAME", "surf_ABB"}
descriptor_cols = [
    c
    for c in df.columns
    if (c.startswith("drug_") or c.startswith("surf_"))
    and c not in NON_FEATURE_NAME_COLS
    and pd.api.types.is_numeric_dtype(df[c])
]  # keep only numeric descriptor features
drug_cols = [c for c in descriptor_cols if c.startswith("drug_")]
surf_cols = [c for c in descriptor_cols if c.startswith("surf_")]

# Keep rows where descriptors, target, and grouping label are present.
# tehcnically not required as I know all rows have all of these, but keeping it is usually good practice because it protects 
# against future CSVs with unexpected missing values and runtime overhead is tiny for my dataset size.
required_cols = descriptor_cols + [TARGET_COL, GROUP_COL]
if SURF_GROUP_COL in df.columns:
    required_cols.append(SURF_GROUP_COL)

# core data prep handoff from pandas to NumPy for model training and evaluation.
df = df.dropna(subset=required_cols).reset_index(drop=True) # removes any rows with missing values in required columns
# (descriptors, target, grouping labels), then resets row indices after row removal. This ensures that the dataset is clean and ready for model training and evaluation.
y = df[TARGET_COL].to_numpy() # extracts target column (logsolpow) into a NumPy array for model training
groups = df[GROUP_COL].to_numpy() # extracts drug grouping column into a NumPy array for grouped splitting methods

# X_df pandas dataframe includes descriptors and grouping columns because the leakage-safe selector
# needs these to compute unique-drug and unique-surfactant filtering per training fold.
selector_meta_cols = [GROUP_COL]
if SURF_GROUP_COL in df.columns:
    selector_meta_cols.append(SURF_GROUP_COL)

X_df = df[descriptor_cols + selector_meta_cols].copy()

# code to build filename-safe text for exported result files.
# used so exported CSV filenames are clean and consistent across operating systems.
def _slugify(text: str) -> str: #defines a helper function named _slugify that takes a string text as input and returns a modified version of the string that is safe to use in filenames.
    """Build filesystem-friendly names for exported result files.""" # explains the function
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in text).strip("_") # does the conversion: keeps letters and 
# digits, converts them to lowercase, replaces non-alphanumeric characters with underscores, and removes leading/trailing underscores.

# define and run CSV export helper
def export_result_table(df_to_save: pd.DataFrame, mode_name: str, table_tag: str) -> None:
    """Export one result table to CSV.""" # explains the function

    output_dir = Path(RESULTS_OUTPUT_DIR) # where to save the CSV file, using the RESULTS_OUTPUT_DIR variable defined earlier.
    output_dir.mkdir(parents=True, exist_ok=True) # creates parent folder if it doesn't exist, and avoids error if it already exists.

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") # builds the timestamp for the file name
    file_name = f"{timestamp}__{_slugify(mode_name)}__{table_tag}.csv" # builds the file name using the timestamp, mode name, and table tag, all separated by double underscores.
    output_path = output_dir / file_name # Combines folder + filename into full output path.
    df_to_save.to_csv(output_path, index=False) # saves the DataFrame to a CSV file at the specified path, without including row index column in the CSV.
    print(f"Saved: {output_path}") # prints confirmation of where the file was saved

# the below code defines a function to flag unusually bad test splits based on RMSE thresholds, and another function to add 
# percentage error columns relative to target spread scales. These functions are useful for analyzing model performance and
# identifying outliers in the results.
def build_split_outlier_flags(
    split_scores_df: pd.DataFrame,
    split_label_col: str,
    rmse_col: str = "RMSE",
) -> pd.DataFrame:
    """Flag unusually bad test splits using a per-model RMSE threshold.

    Threshold rule: RMSE > median(RMSE) + 2 * IQR(RMSE), computed separately
    for each model inside the provided split-score table.
    """
    # Interquartile Range (IQR) is a measure of statistical dispersion that describes the spread of the middle 50% of the data
    # set. It is calculated as the difference between the 75th percentile (Q₃) and the 25th percentile (Q₁): 
    # IQR = Q₃ - Q₁
    # Because it only looks at the middle half of the data, the IQR is highly resistant to extreme values or outliers. 
    # This makes it a preferred metric for skewed distributions or datasets where you want to ignore noise at the extremes.
    # RMSE > median(RMSE) + 2 * IQR(RMSE) is a statistical anomaly detector to identify abnormally high errors (poor 
    # performance) without requiring a subjective, hardcoded threshold.
    # median(RMSE) - establishes the "typical" or expected error of the model, which is highly robust against massive outliers. 
    # IQR(RMSE) - the Interquartile Range measures the spread of the middle 50% of your error data. It represents the natural, 
    # expected variance in model performance. 
    # 2 x IQR(RMSE) - Acts as a buffer to account for natural fluctuation. 
    # The Threshold: Anything falling above this line (median(RMSE) + 2 * IQR(RMSE)) is considered a statistical outlier—
    # an error so severe it deviates significantly from the model's normal behavior.
 
    if split_scores_df.empty:
        return pd.DataFrame()

    flagged_blocks = []
    for model_name, model_df in split_scores_df.groupby("Model", dropna=False):
        rmse_values = model_df[rmse_col].astype(float)
        median_rmse = float(rmse_values.median())
        q1 = float(rmse_values.quantile(0.25))
        q3 = float(rmse_values.quantile(0.75))
        iqr = q3 - q1
        threshold = median_rmse + (2.0 * iqr)

        block = model_df.copy()
        block["RMSE_Median"] = median_rmse
        block["RMSE_Q1"] = q1
        block["RMSE_Q3"] = q3
        block["RMSE_IQR"] = iqr
        block["OutlierThreshold_MedianPlus2IQR"] = threshold
        block["RMSE_AboveThreshold"] = block[rmse_col].astype(float) - threshold
        block["IsSplitOutlier"] = block[rmse_col].astype(float) > threshold
        block["Model"] = model_name
        flagged_blocks.append(block)

    flagged_df = pd.concat(flagged_blocks, ignore_index=True)
    flagged_df = flagged_df.sort_values(["Model", "IsSplitOutlier", "RMSE"], ascending=[True, False, False])
    # Keep column order readable by placing summary flags at the end.
    preferred_order = [
        "Model",
        split_label_col,
        "RMSE",
        "MAE",
        "R2",
        "RMSE_Median",
        "RMSE_Q1",
        "RMSE_Q3",
        "RMSE_IQR",
        "OutlierThreshold_MedianPlus2IQR",
        "RMSE_AboveThreshold",
        "IsSplitOutlier",
    ]
    ordered_cols = [c for c in preferred_order if c in flagged_df.columns]
    remaining_cols = [c for c in flagged_df.columns if c not in ordered_cols]
    return flagged_df[ordered_cols + remaining_cols]


def add_error_percentage_columns(
    table_df: pd.DataFrame,
    y_reference: np.ndarray,
) -> pd.DataFrame:
    """Add RMSE/MAE percentage columns relative to key target spread scales.

    For every numeric RMSE/MAE column, this appends percentage-normalized
    companions versus:
    - mean absolute target value
    - target standard deviation
    - full target range
    - target IQR (Q3-Q1)
    """
    if table_df.empty:
        return table_df

    target_arr = np.asarray(y_reference, dtype=float)
    scales = {
        "PctOfMeanAbsTarget": float(np.mean(np.abs(target_arr))),
        "PctOfTargetStd": float(np.std(target_arr)),
        "PctOfTargetRange": float(np.max(target_arr) - np.min(target_arr)),
        "PctOfTargetIQR": float(np.quantile(target_arr, 0.75) - np.quantile(target_arr, 0.25)),
    }

    out_df = table_df.copy()
    for col in out_df.columns:
        col_name = str(col)
        if "Pct" in col_name:
            continue
        if ("RMSE" not in col_name) and ("MAE" not in col_name):
            continue
        if not pd.api.types.is_numeric_dtype(out_df[col]):
            continue

        for suffix, scale in scales.items():
            if scale <= 0.0:
                continue
            pct_col = f"{col}_{suffix}"
            out_df[pct_col] = (out_df[col].astype(float) / scale) * 100.0

    return out_df

print("Data loaded and preprocessed.")

# ---------------------------------------------------------------------------
# 2. Leakage-safe feature selector
# ---------------------------------------------------------------------------
# Feature reduction — done SEPARATELY for drug and surfactant descriptors.
# Why separately: drug descriptors only vary meaningfully across the 20 unique drugs, and surfactant descriptors only vary 
# meaningfully across the 5 unique surfactants. Filtering on the merged (drug x surfactant) table mixes these scales and, 
# more importantly, with only 5 surfactants, variance/correlation estimates on the surfactant side are too noisy to
# trust with the same automated thresholds used for drugs (5 points is not enough to reliably estimate a correlation coefficient).

# This custom transformer is used inside the model pipeline to perform feature selection in a leakage-safe manner. 
# It ensures that feature selection is based only on the training data within each cross-validation fold, 
# preventing information from the test set from influencing the selection process.
class DrugSurfFeatureSelector(BaseEstimator, TransformerMixin): # class = custom scikit-learn transformer for leakage-safe 
    # feature selection to be done safely inside cross-validation folds
    """Feature selector applied inside CV folds to avoid leakage. # comments explaining the purpose of the class

     1) On drug descriptors only, use variance thresholding and correlation filtering
         based on unique drugs in the training fold.
     2) On surfactant descriptors only, use variance thresholding and correlation
         filtering based on unique surfactants in the training fold.

    Input X must be a DataFrame containing descriptor columns and group_col.
    """
    # function to store all configuration values on the class instance so sklearn can clone
    # the transformer cleanly inside pipelines and CV folds.
    def __init__(
        self,
        descriptor_columns: list[str], # full descriptor set that can be used by models.
        drug_columns: list[str], # split descriptor block for drugs
        surfactant_columns: list[str], # split descriptor block for surfactants - surf and drug blocks are filtered independently.
        group_col: str,
        surf_group_col: str, # identifiers used to build unique drug/surf rows inside each training fold 
        # for leakage-safe filtering.
        drug_var_threshold: float, # variance threshold for drug descriptors
        drug_corr_threshold: float, # correlation threshold for drug descriptors
        surf_var_threshold: float, # variance threshold for surfactant descriptors
        surf_corr_threshold: float, # correlation threshold for surfactant descriptors
        drug_target_corr_threshold: float = 0.0, # min |target correlation| for a drug descriptor to be kept
        surf_target_corr_threshold: float = 0.0, # min |target correlation| for a surf descriptor to be kept
        # all thresholds defined earlier
    ):
        
        # below lines store the input settings on the class instance so they can be used later in fit() and transform().
        # they are the object setup which takes constructor inputs and stores them so other methods like fit and transform can use them later
        self.descriptor_columns = descriptor_columns # save the descriptors list
        self.drug_columns = drug_columns # save drug descriptor columns
        self.surfactant_columns = surfactant_columns # save surfactant descriptor columns
        self.group_col = group_col # save the drug grouping column name
        self.surf_group_col = surf_group_col # save the surfactant grouping column name
        self.drug_var_threshold = drug_var_threshold # save the variance threshold for drug descriptors
        self.drug_corr_threshold = drug_corr_threshold # save the correlation threshold for drug descriptors
        self.surf_var_threshold = surf_var_threshold # save the variance threshold for surfactant descriptors
        self.surf_corr_threshold = surf_corr_threshold # save the correlation threshold for surfactant descriptors
        self.drug_target_corr_threshold = drug_target_corr_threshold # save the target-correlation relevance threshold for drug descriptors
        self.surf_target_corr_threshold = surf_target_corr_threshold # save the target-correlation relevance threshold for surf descriptors

    # small helper function that performs a specific reusable task: filtering a block of descriptors by variance and 
    # correlation thresholds. It is static because it does not depend on the instance state (thus starts with @staticmethod).
    @staticmethod # decorator to indicate that the following function is a static method (behaves like a plain function scoped in the class)
    def _filter_by_variance_and_correlation( 
        block_df: pd.DataFrame,
        block_columns: list[str],
        var_threshold: float,
        corr_threshold: float,
        target_column: str | None = None,
        target_corr_threshold: float = 0.0,
        target_corr_block_df: pd.DataFrame | None = None,
        target_corr_column: str | None = None,
    ) -> list[str]:
        """Apply variance then correlation filtering to one descriptor block."""
        # If this block has no descriptors, return an empty selection immediately.
        if len(block_columns) == 0:
            return []

        # 1) Variance filtering: remove near-constant descriptors.
        # VarianceThreshold expects a numeric array, so we convert from DataFrame.
        var_selector = VarianceThreshold(threshold=var_threshold)
        var_selector.fit(block_df[block_columns].to_numpy())
        retained = list(np.array(block_columns)[var_selector.get_support()])

        # If 0 or 1 descriptors remain, correlation filtering is not applicable.
        if len(retained) <= 1:
            return retained

        # 2) Correlation filtering: remove one descriptor from highly correlated pairs.
        # If a target column is available, keep the descriptor that is more strongly
        # associated with solubility and drop the weaker one.
        # another way is to have order-dependent selection. This is simple, but can be arbitrary. It makes more sense
        # to keep the descriptor that is more strongly associated with the target variable (solubility) and drop the weaker one.
        # If no target column is available, we fall back to the order-dependent selection rule. This should not be the case.
        # We use absolute correlation (magnitude, ignoring the sign - positive and negative correlation) and only inspect the 
        # upper triangle to avoid duplicate pair checks (A-B and B-A).
        # upper triangle - we only need to check one half of the correlation matrix because it is symmetric.
        with np.errstate(divide="ignore", invalid="ignore"):
            corr = block_df[retained].corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

        if target_column is not None and target_column in block_df.columns:
            # Target correlation can optionally be computed against a different,
            # row-level (non-deduplicated) dataframe than the one used for variance
            # and interfeature correlation. This matters specifically for the
            # surfactant block: a surfactant descriptor only takes 5 distinct values,
            # but each surfactant appears in ~20 rows (once per drug), each with a
            # genuinely different target value. Computing target correlation on the
            # deduplicated 5-row table wastes that replication (df=3, statistically
            # underpowered); computing it on the full row-level data uses all the
            # real information collected (df~95) without changing what "relevant"
            # means. Interfeature correlation does NOT need this fix — two
            # descriptors that are each constant within a group have the same
            # correlation whether measured on the deduplicated or row-level table,
            # since equal repetition doesn't change a correlation coefficient.
            tc_df = target_corr_block_df if target_corr_block_df is not None else block_df
            tc_target_column = target_corr_column if target_corr_column is not None else target_column
            target_series = tc_df[tc_target_column].astype(float)
            target_std = float(target_series.std(ddof=0))

            if np.isfinite(target_std) and target_std > 0.0:
                with np.errstate(divide="ignore", invalid="ignore"):
                    target_corr = tc_df[retained].corrwith(target_series).abs()
                target_corr = target_corr.fillna(0.0)
            else:
                # If target variance is zero in this fold, correlation is undefined.
                target_corr = pd.Series(0.0, index=retained)

            # 2a) Relevance filtering: drop descriptors whose |target correlation| falls
            # below target_corr_threshold. Done here — after variance filtering, before
            # interfeature correlation dedup — so the dedup step only ever has to choose
            # between descriptors that have already cleared the relevance bar, rather
            # than risk keeping a redundant-but-weak descriptor over a relevant one.
            # Only applied when target_corr_threshold > 0.0 and target correlation could
            # actually be computed (target_std > 0.0) — otherwise every descriptor would
            # be dropped, since an undefined/zero target_corr is not a genuine "irrelevant"
            # signal.
            if target_corr_threshold > 0.0 and np.isfinite(target_std) and target_std > 0.0:
                retained = [col for col in retained if target_corr.get(col, 0.0) >= target_corr_threshold]
                target_corr = target_corr.loc[retained]
                if len(retained) <= 1:
                    return retained
                corr = corr.loc[retained, retained]
                upper = upper.loc[retained, retained]

            order_lookup = {col: idx for idx, col in enumerate(retained)}
            ordered = sorted(
                retained,
                key=lambda col: (-target_corr.get(col, 0.0), order_lookup[col]),
            )

            kept = []
            for col in ordered:
                if not any(corr.loc[col, kept_col] > corr_threshold for kept_col in kept):
                    kept.append(col)
            return kept

        # Fallback: if no target column is available, use the original order-based rule.
        # Drop any descriptor that has correlation above threshold with at least one
        # previously considered descriptor - the selection is order-dependent, but this is a common and simple approach.
        to_drop = [col for col in upper.columns if any(upper[col] > corr_threshold)]
        return [c for c in retained if c not in to_drop]

    @staticmethod
    def _build_selection_audit(
        block_df: pd.DataFrame,
        block_columns: list[str],
        retained_columns: list[str],
        var_threshold: float,
        corr_threshold: float,
        target_column: str | None = None,
        target_corr_threshold: float = 0.0,
        target_corr_block_df: pd.DataFrame | None = None,
        target_corr_column: str | None = None,
    ) -> pd.DataFrame:
        """Build a per-feature audit table with variance and correlation stats."""
        if len(block_columns) == 0:
            return pd.DataFrame(
                columns=[
                    "Feature",
                    "Variance",
                    "Abs_Target_Correlation",
                    "Max_Abs_InterFeature_Correlation",
                    "Retained",
                    "Filtered_By_Variance",
                    "Filtered_By_Correlation",
                    "Selection_Reason",
                ]
            )

        corr = None
        if len(block_columns) > 1:
            with np.errstate(divide="ignore", invalid="ignore"):
                corr = block_df[block_columns].corr().abs()

        retained_set = set(retained_columns)
        rows = []
        for feature_name in block_columns:
            feature_series = block_df[feature_name].astype(float)
            variance = float(feature_series.var(ddof=0))

            abs_target_corr = np.nan
            tc_df = target_corr_block_df if target_corr_block_df is not None else block_df
            tc_target_column = target_corr_column if target_corr_column is not None else target_column
            if tc_target_column is not None and tc_target_column in tc_df.columns and feature_name in tc_df.columns:
                tc_feature_series = tc_df[feature_name].astype(float)
                target_series = tc_df[tc_target_column].astype(float)
                feature_std = float(tc_feature_series.std(ddof=0))
                target_std = float(target_series.std(ddof=0))
                if (
                    np.isfinite(feature_std)
                    and np.isfinite(target_std)
                    and feature_std > 0.0
                    and target_std > 0.0
                ):
                    with np.errstate(divide="ignore", invalid="ignore"):
                        abs_target_corr = float(abs(tc_feature_series.corr(target_series)))

            max_interfeature_corr = np.nan
            if corr is not None and feature_name in corr.index:
                pairwise_corr = corr.loc[feature_name].drop(labels=[feature_name], errors="ignore")
                pairwise_corr = pairwise_corr[np.isfinite(pairwise_corr)]
                if not pairwise_corr.empty:
                    max_interfeature_corr = float(pairwise_corr.max())

            filtered_by_variance = bool(variance <= var_threshold)
            filtered_by_target_corr = bool(
                (not filtered_by_variance)
                and (feature_name not in retained_set)
                and target_corr_threshold > 0.0
                and np.isfinite(abs_target_corr)
                and abs_target_corr < target_corr_threshold
            )
            filtered_by_correlation = bool(
                (not filtered_by_variance)
                and (not filtered_by_target_corr)
                and (feature_name not in retained_set)
            )

            if feature_name in retained_set:
                selection_reason = "Retained"
            elif filtered_by_variance:
                selection_reason = "Variance"
            elif filtered_by_target_corr:
                selection_reason = "TargetCorrelation"
            elif filtered_by_correlation:
                selection_reason = "Correlation"
            else:
                selection_reason = "Other"

            rows.append(
                {
                    "Feature": feature_name,
                    "Variance": variance,
                    "Abs_Target_Correlation": abs_target_corr,
                    "Max_Abs_InterFeature_Correlation": max_interfeature_corr,
                    "Retained": feature_name in retained_set,
                    "Filtered_By_Variance": filtered_by_variance,
                    "Filtered_By_Correlation": filtered_by_correlation,
                    "Selection_Reason": selection_reason,
                }
            )

        return pd.DataFrame(rows)

    def fit(self, X, y=None):
        # Enforce DataFrame input because this selector depends on column names and
        # group identifiers, not just raw array positions.
        if not isinstance(X, pd.DataFrame):
            raise TypeError("DrugSurfFeatureSelector expects a pandas DataFrame as input.")

        # Build the list of required columns for safe operation.
        # At minimum we need descriptors + drug group column.
        required = self.descriptor_columns + [self.group_col]
        if self.surf_group_col in X.columns:
            required.append(self.surf_group_col)

        # Fail early with a clear error if expected columns are missing.
        missing = [c for c in required if c not in X.columns]
        if missing:
            raise ValueError(f"Missing required columns for feature selection: {missing}")

        if y is None:
            raise ValueError("DrugSurfFeatureSelector requires y when feature reduction is enabled.")

        # Attach the target to the current training fold so we can score descriptors
        # against solubility without leaking information from outside the fold.
        training_df = X.copy()
        training_df["_target"] = np.asarray(y)

        # Unique-drug rows from the current training fold only.
        # This is leakage-safe because fit() is called separately per training fold
        # inside sklearn CV pipelines.
        unique_drug_df = training_df.drop_duplicates(subset=self.group_col).copy()
        drug_target = training_df.groupby(self.group_col, as_index=False)["_target"].mean()
        unique_drug_df = unique_drug_df.merge(drug_target, on=self.group_col, how="left", suffixes=("", "_target"))

        # Drug block filtering:
        # first variance thresholding, then correlation-based pruning.
        retained_drug = self._filter_by_variance_and_correlation(
            block_df=unique_drug_df,
            block_columns=self.drug_columns,
            var_threshold=self.drug_var_threshold,
            corr_threshold=self.drug_corr_threshold,
            target_column="_target_target",
            target_corr_threshold=self.drug_target_corr_threshold,
        )
        drug_audit_df = self._build_selection_audit(
            block_df=unique_drug_df,
            block_columns=self.drug_columns,
            retained_columns=retained_drug,
            var_threshold=self.drug_var_threshold,
            corr_threshold=self.drug_corr_threshold,
            target_column="_target_target",
            target_corr_threshold=self.drug_target_corr_threshold,
        )

        # Surfactant block filtering:
        # if surf group labels exist, deduplicate by surf group; otherwise fallback
        # to deduplicating on surf descriptor patterns.
        if self.surf_group_col in X.columns:
            unique_surf_df = training_df.drop_duplicates(subset=self.surf_group_col).copy()
            surf_target = training_df.groupby(self.surf_group_col, as_index=False)["_target"].mean()
            unique_surf_df = unique_surf_df.merge(surf_target, on=self.surf_group_col, how="left", suffixes=("", "_target"))
            surf_target_column = "_target_target"
        else:
            # Fallback when surf group column is unavailable: use unique surf descriptor rows.
            unique_surf_df = training_df.drop_duplicates(subset=self.surfactant_columns).copy()
            surf_target = training_df.groupby(self.surfactant_columns, as_index=False)["_target"].mean()
            unique_surf_df = unique_surf_df.merge(surf_target, on=self.surfactant_columns, how="left", suffixes=("", "_target"))
            surf_target_column = "_target_target"

        # Surfactant descriptors use variance-then-correlation filtering on the
        # deduplicated surfactant table (variance and interfeature correlation are
        # unaffected by deduplication, since equal repetition preserves both), but
        # target correlation is computed on the full row-level training_df — each
        # surfactant appears in many rows (once per drug), each with a genuinely
        # different target value, so this uses all the real replication collected
        # rather than collapsing to 5 points before testing relevance.
        retained_surf = self._filter_by_variance_and_correlation(
            block_df=unique_surf_df,
            block_columns=self.surfactant_columns,
            var_threshold=self.surf_var_threshold,
            corr_threshold=self.surf_corr_threshold,
            target_column=surf_target_column,
            target_corr_threshold=self.surf_target_corr_threshold,
            target_corr_block_df=training_df,
            target_corr_column="_target",
        )
        surf_audit_df = self._build_selection_audit(
            block_df=unique_surf_df,
            block_columns=self.surfactant_columns,
            retained_columns=retained_surf,
            var_threshold=self.surf_var_threshold,
            corr_threshold=self.surf_corr_threshold,
            target_column=surf_target_column,
            target_corr_threshold=self.surf_target_corr_threshold,
            target_corr_block_df=training_df,
            target_corr_column="_target",
        )

        # Final selected feature list is block-wise concatenation of retained drug and
        # retained surfactant descriptors.
        self.selected_feature_names_ = retained_drug + retained_surf
        self.selection_audit_ = {"drug": drug_audit_df, "surf": surf_audit_df}
        return self

    def transform(self, X):
        # Ensure fit() has already run and produced selected feature names.
        if not hasattr(self, "selected_feature_names_"):
            raise RuntimeError("Call fit before transform.")

        # Keep DataFrame requirement here too for consistent column-based selection.
        if not isinstance(X, pd.DataFrame):
            raise TypeError("DrugSurfFeatureSelector expects a pandas DataFrame as input.")

        # Return only selected columns as a NumPy array so downstream sklearn
        # estimators receive numeric matrix input.
        return X[self.selected_feature_names_].to_numpy()

print("Leakage-safe feature selector defined.")

# ---------------------------------------------------------------------------
# 3. Model definitions and tuning grids
# ---------------------------------------------------------------------------
def model_specs(max_pls_components: int) -> dict[str, dict]:
    """Return model specs with estimator, scaling requirement, and tuning grid."""

    # Kernel for Gaussian Process:
    # - ConstantKernel scales the signal amplitude. It multiplies the RBF kernel and allows the GP to learn the overall output scale.
    # - RBF models smooth nonlinear relationships and has length_scale. RBF = Radial Basis Function kernel, which is 
    # a common choice for GP regression. It assumes that points closer in input space have more similar outputs.
    # - WhiteKernel adds observation noise.
    kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0)  # commonly chosen values. GP will 
    # learn best values during training anyway.

# a kernel is the rule your model uses to measure similarity, and that rule strongly shapes the kind of patterns 
# the model can learn. It is the model's definition of closeness - tells a model how similar two data points are.

# constant kernel scales the RBF, white kernel adds noise term. These are hyperparameters that could be tuned further if desired.
# number in bracket of constantkernel  is the starting value of the kernel’s constant term, usually called the amplitude or scale.
# ConstantKernel(1.0) means the kernel contributes a constant level of covariance. 1.0 is the initial value the model starts 
# from before fitting. During training, that value is usually optimized unless you have fixed it.
# constant kernel sets the overall amplitude of the covariance function. It controls how large the function values can vary. 

# rbf kernel is a common choice for GP regression, stands for "radial basis function" and allows for smooth, 
# non-linear relationships between features and target. controls smoothness and how quickly correlation drops as points get farther apart.
# number in bracket is the length scale, which controls how quickly the function can change. A smaller length scale 
# allows for more rapid changes in the function, while a larger length scale results in smoother, more slowly changing functions. 
# 
# The white kernel adds a noise term to the model - represents observation noise. The noise can help account for measurement 
# error or other sources of noise in the data. Does not model the function shape, just tells the model how noisy the measured data are. 
# The number in the bracket is the initial noise level, which can be optimized during training.

    specs = {
        # Ridge regression is a linear model that adds L2 regularisation to the ordinary least squares loss function (linear regression).
        # L2 penalises large coefficient magnitudes and helps with multicollinearity.
        # Scaling is needed so penalties are applied fairly across descriptors.
        # L2 uses the sum of the squared coefficients.
        # Shrinks coefficients towards zero but does not set them exactly to zero, so all descriptors remain in the model
        # but with smaller weights. It is useful when many descriptors are relevant but we want to reduce overfitting.
        "Ridge": {
            "estimator": Ridge(),
            "needs_scaling": True,
            "param_grid": {
                "model__alpha": [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000],
            },
        },
    # L2 regularization penalizes large coefficients, which can help prevent overfitting and improve generalization to new data.
    # L2 means the model adds a penalty based on the squared size of the coefficients.
    # useful because many rdkit descriptors are correlated, and ridge regression can help stabilize the coefficient estimates
    # in the presence of multicollinearity.
    # data needs to be scaled before ridge, otherwise variables with larger scales will get larger penalties and dominate the model.
    # alpha is the regularization strength. Higher alpha means more regularization (more shrinkage of coefficients). 
    # 1.0 is a common default value, but it can be tuned based on cross-validation or other methods.

        # Lasso regression is a linear model that adds L1 regularisation to the ordinary least squares loss function.
        # L1 adds a penalty to the model for having large coefficient values. It can set some coefficients exactly to zero,
        # acting like feature selection.
        # It can make the model simpler and reduce overfitting.
        # It encourages simplicity by pushing unhelpful descriptors out of the model entirely.
        # L1 affects coefficients for descriptors in the Lasso model.
        # L1 uses the sum of the absolute values of the coefficients. 
        "Lasso": {
            "estimator": Lasso(max_iter=100000, random_state=RANDOM_STATE),
            "needs_scaling": True,
            "param_grid": { # hyperparameter grid for tuning the Lasso model - testing different values and choosing the best one based on cross-validation performance.
                "model__alpha": [1e-3, 1e-2, 1e-1, 1, 5, 10, 50], # alpha controls the strength of the L1 penalty. Smaller values 
                # allow more flexibility, larger values enforce more coefficients to go to 0.
            },
        },
        # L1 regularisation penalises the absolute size of the coefficients, which can lead to sparse models
    # where some coefficients are exactly zero. This can be useful for feature selection, as it can effectively remove 
    # irrelevant features from the model. Lasso is also useful when you have many correlated features, as it tends to select 
    # one feature from a group and ignore the others.
    # alpha is the regularisation strength. Higher alpha means more regularisation (more shrinkage of coefficients).
    # max_iter is the maximum number of iterations for the optimization algorithm. Lasso can sometimes require more 
    # iterations to converge, especially with larger datasets or more complex models.

        # PLS is useful with many correlated features and relatively small sample size.
        # PLS = Partial Least Squares regression. Finds latent variables that capture variance in both descriptors and target.
        # n_components is the number of latent variables (too low underfits, too high overfits).
        "PLS": {
            "estimator": PLSRegression(),
            "needs_scaling": True,
            "param_grid": { # hyperparameter grid for tuning the PLS model
                "model__n_components": list(range(1, max_pls_components + 1)),
            },
        },
        # PLS is a linear regression technique that is particularly useful when you have many correlated features and 
        # relatively few samples. It builds new latent variables with each one being a weighted combination of the original 
        # descriptors, so they capture the most important information in the predictors and the response variable. 
    # Unlike PCA, it does not only look at variation in X, but chooses components that explain X while also being strongly related 
    # to y. It then fits regression using those components (latent variables) instead of all the raw descriptors directly.
    # This can help improve predictive performance and interpretability when dealing with
    # high-dimensional and collinear data.
    # n_components is the number of latent variables to use in the model. If too low, it will underfit, too high overfit.
    # should be chosen via cross-validation.
    # data needs to be scaled inside cross-validation folds to avoid data leakage, which has been done below.

        # Gaussian Process regression is a nonparametric and probabilistic model. It can capture both nonlinear relationships
        # and uncertainty in predictions. It predicts a distribution over possible outputs for each input, rather than a single point estimate.
        # It predicts both mean and uncertainty, based on kernel-defined similarity.
        "GaussianProcess": {
            "estimator": GaussianProcessRegressor(kernel=kernel, random_state=RANDOM_STATE),
            "needs_scaling": True,
            "param_grid": {
                "model__alpha": [1e-10, 1e-8, 1e-6, 1e-4],
                "model__normalize_y": [True, False],
            },
        },
        # Gaussian Process (GP) regression is a probabilistic, nonparametric way to model a function. Instead of assuming one
    # fixed equation form, GP assumes that similar inputs should have similar outputs and it uses a kernel to define what
    # “similar” means. It puts a probability distribution over possible functions. After seeing data, it updates that 
    # distribution. For each new molecule, it gives both a predicted mean value andan uncertainty (variance/confidence). 
    # can be useful but should do strong preprocessing and feature selection first to avoid overfitting (NZV/correlation filtering, scaling)
    # GP asks: “what family of smooth functions could explain the data, and how uncertain am I at each point?”

        # SVR fits within an epsilon-insensitive tube and penalizes errors outside it. SVR = Support Vector Regression. 
        # It is a kernel-based method that can capture nonlinear relationships.
        # It uses support vectors (critical training points) to define the regression function.
        # support vectors are the training points that lie on or outside the epsilon-insensitive tube. 
        # They are the most informative points for defining the regression function.
        # Epsilon-insensitive tube means that errors within a certain margin (epsilon) are ignored, which can make the model 
        # more robust to noise.
        # C controls fit rigidity, epsilon controls tube width, gamma controls RBF curvature.
        "SVR": {
            "estimator": SVR(),
            "needs_scaling": True,
            "param_grid": {
                "model__kernel": ["rbf", "linear"], # kernel type for SVR. RBF is nonlinear, linear is linear. RBF = Radial Basis Function kernel, which allows SVR to capture nonlinear relationships.
                "model__C": [0.1, 1, 10, 50, 100], # C is the regularization parameter that controls the trade-off between achieving a low training error and a low testing error. A smaller C makes the decision surface smoother, while a larger C aims to classify all training examples correctly.
                "model__epsilon": [0.01, 0.05, 0.1, 0.2, 0.5, 0.75], # epsilon defines a margin of tolerance where no penalty is given to errors. It controls the width of the epsilon-insensitive tube. A larger epsilon allows more deviation from the true values without penalty, which can make the model more robust to noise.
                "model__gamma": ["scale", 0.01, 0.1, 1.0], # gamma defines how far the influence of a single training example reaches. Low values mean 'far' and high values mean 'close'. A small gamma means that the model is influenced by points far away, leading to a smoother decision boundary, while a large gamma means that the model is influenced only by points close to it, leading to a more complex decision boundary.
            },
        },
        # SVR tries to find a function that is as flat/simple as possible while keeping prediction errors within a tolerance band 
        # called epsilon (ϵ). does not try to fit every point exactly. It focuses on fitting within an acceptable error margin.
    # C: penalty strength for points outside the tolerance band. high C = tighter fit to training data, higher overfitting risk
    # low C = smoother/softer fit.
    # epsilon: width of the no-penalty tolerance band zone around true values. larger epsilon ignores small errors.
    # smaller epsilon tries to fit more precisely. 
    # kernel (e.g., RBF): linear kernel for linear relationships, RBF for nonlinear relationships.

        # Random Forest is an ensemble of decision trees.
        # Each tree is trained on a bootstrap sample of the data and uses random feature subsets for splitting.
        # Decision tree is a nonparametric model that splits the feature space into regions based on feature values. 
        # Each tree makes predictions based on the average (for regression) of its leaves.
        # bootstrap sampling means that each tree is trained on a random sample of the training data with replacement. 
        # This introduces diversity among the trees and helps reduce overfitting.
        # n_estimators = number of trees; more trees improve stability but cost more.
        # max_depth, min_samples_split, min_samples_leaf help control overfitting.
        # max_features controls per-split randomness and tree diversity.
        "RandomForest": {
            "estimator": RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
            "needs_scaling": False,
            "param_grid": {
                "model__n_estimators": [300, 500, 800],
                "model__max_depth": [2, 3, 4, 5, None],
                "model__min_samples_split": [2, 4, 6, 8],
                "model__min_samples_leaf": [1, 2, 3, 4],
                "model__max_features": ["sqrt", 0.3, 0.5, 0.7, 1.0],
                "model__bootstrap": [True],
            },
        },
        # RF- ensemble model made of many decision trees. creates many different training subsets by random sampling with replacement (bootstrap)
    # trains one decision tree on each subset. at each split in each tree, it only considers a random subset of features.
    # final prediction is the average of all trees' predictions. This reduces overfitting and improves generalization.
    # becasue a single tree is usually unstable and overfits. averaging many trees reduces variance and improves robustness.
    # hyperparameters:
    #   n_estimators: number of trees in the forest. more trees = better performance but increased computational cost.
    #   max_depth: maximum depth of each tree. limits tree complexity. small depth reduces overfitting. 
    # deeper trees can model more complex relationships but may overfit by memorising training data.
    #   min_samples_split: minimum number of samples required to split an internal node. higher values reduce overfitting
    # by making the tree more conservative and smoother - less overall splits. 
    #   min_samples_leaf: minimum number of samples required in a leaf node. very important for small datasets. 
    # higher values reduce overfitting by reducing noisy, overly specific leaves. (less leaves in total, more generalization)
    #   max_features: how many features are considered at each split. lower values increase tree diversity, higher values
    # reduce bias. default is "sqrt" or "all" for regression.
    #   bootstrap: Whether each tree is trained on a bootstrap sample (sample with replacement). True is standard 
    # Random Forest behavior and usually preferred.
    #   oob_score: Out-of-bag validation estimate using unsampled points from each bootstrap draw.
    # Useful as a quick internal performance check when bootstrap is True.
    #   criterion: how to measure the quality of a split. For regression, it is squared_error or absolute_error.
    #   random_state: seed for reproducibility, so results are repeatable.
    #   n_jobs: how many CPU cores to use. -1 means use all available cores. Can speed up training on large datasets.
    # for small dataset, most influential anti-overfitting settings are max_depth, min_samples_split, min_samples_leaf
    # and max_features.
    #   practical small-data starting point: n_estimators = 500, max_depth = 2 to 6, min_samples_leaf = 2 to 4, 
    # min_samples_split = 4 to 8, max_features = sqrt or a small fraction, bootstrap = True, random_state fixed

    # decision tree is a model that makes predictions by asking a sequence of yes/no questions about features.
    # each split asks something like “Is descriptor X <= threshold?” data are split into two groups repeatedly
    # at the end (leaf), prediction is usually the average target value of training points in that leaf

        # Gradient Boosting builds trees sequentially, each correcting residuals.
        # learning_rate and n_estimators trade off fitting speed vs complexity.
        # max_depth, min_samples_split, min_samples_leaf regularize tree complexity.
        "GradientBoosting": {
            "estimator": GradientBoostingRegressor(random_state=RANDOM_STATE),
            "needs_scaling": False, # because it is a tree-based model
            "param_grid": {
                "model__n_estimators": [100, 200, 400],  # number of trees to fit. More trees can improve performance but increase training time and risk of overfitting.
                "model__learning_rate": [0.005, 0.01, 0.05, 0.1], # learning rate shrinks the contribution of each tree. Lower values require more trees but can improve generalization.
                "model__max_depth": [2, 3, 4], # maximum depth of each tree. Controls model complexity. Deeper trees can capture more complex patterns but may overfit.
                "model__min_samples_split": [2, 4, 8], # minimum number of samples required to split an internal node. Higher values reduce overfitting by making the tree more conservative.
                "model__min_samples_leaf": [1, 2, 4], # minimum number of samples required to be at a leaf node. Higher values reduce overfitting by making the tree more conservative.
                "model__subsample": [0.7, 0.85, 1.0], # fraction of samples used for fitting each tree. Lower values can improve generalization but may increase bias.
            },
        },
    }

    # Gradient Boosting chosen over XGBoost because dataset is small and there are many descriptors. 
# XGBoost is more complex and can overfit small datasets. Gradient Boosting is simpler and more interpretable for this case.
# Therefore, it is easier to tune and less likely to overfit with small data.
# keeps the workflow self-contained within scikit-learn - avoids adding an extra library and another layer of complexity.
# Additionally, XGBoost has more hyperparameters to tune, which can be challenging with small datasets.

# XGBoost added as David suggested it often performs well on tabular data.

    # Add XGBoost only when the dependency is available in the current environment.
    if HAS_XGBOOST:
        specs["XGBoost"] = {
            "estimator": XGBRegressor(
                objective="reg:squarederror",
                random_state=RANDOM_STATE,
                n_jobs=-1, # use all available CPU cores for parallel processing during training. This can speed up training significantly, especially for large datasets or complex models.
                verbosity=0, # suppresses detailed training output. 0 = silent, 1 = warning, 2 = info, 3 = debug. Setting to 0 keeps the console output clean during model training.
            ),
            "needs_scaling": False, # because it is a tree-based model and does not require feature scaling.
            "param_grid": {
                "model__n_estimators": [100, 200, 300, 500], # number of trees to fit. More trees can improve performance but increase training time and risk of overfitting.
                "model__learning_rate": [0.01, 0.03, 0.1], # learning rate shrinks the contribution of each tree. Lower values require more trees but can improve generalization.
                # controls how much each individual tree is allowed to contribute to the overall prediction during boosting
                "model__max_depth": [2, 3, 4, 5], # maximum depth of each tree. Controls model complexity. Deeper trees can capture more complex patterns but may overfit.
                "model__min_child_weight": [1, 2, 3, 5], # minimum sum of instance weight (hessian) needed in a child. Controls model complexity. Higher values prevent the model from learning overly specific patterns, reducing overfitting.
                "model__subsample": [0.6, 0.8, 1.0], # fraction of samples used for fitting each tree. Lower values can improve generalization but may increase bias.
                "model__colsample_bytree": [0.5, 0.7, 1.0], # fraction of features used for fitting each tree. Lower values can improve generalization but may increase bias.
            },
        }

    return specs

print("Model specifications defined.")

# ---------------------------------------------------------------------------
# 4. Pipeline builder
# ---------------------------------------------------------------------------
def make_pipeline(estimator, needs_scaling: bool) -> Pipeline:
    """Build model pipeline with leakage-safe feature selection and optional scaling."""
    feature_selector = DrugSurfFeatureSelector(
        descriptor_columns=descriptor_cols,
        drug_columns=drug_cols,
        surfactant_columns=surf_cols,
        group_col=GROUP_COL,
        surf_group_col=SURF_GROUP_COL,
        drug_var_threshold=DRUG_VAR_THRESHOLD,
        drug_corr_threshold=DRUG_CORR_THRESHOLD,
        surf_var_threshold=SURF_VAR_THRESHOLD,
        surf_corr_threshold=SURF_CORR_THRESHOLD,
        drug_target_corr_threshold=DRUG_TARGET_CORR_THRESHOLD,
        surf_target_corr_threshold=SURF_TARGET_CORR_THRESHOLD,
    )

    steps = [("feature_selector", feature_selector)]

    # Scaling is required for ridge/lasso/pls/gp/svr and not needed for tree models.
    if needs_scaling: # "needs_scaling": False/True is in every model in section 3.
        steps.append(("scaler", StandardScaler())) # StandardScaler standardises features by removing the mean and scaling to 
        # unit variance. This means that each feature will have a mean of 0 and SD of 1, so all features contribute equally
        # to the model.
        # unit variance is (definition) the variance of a variable divided by the square of its mean. 
        # It is a measure of relative variability. (square root of variance is SD)

    steps.append(("model", estimator))
    return Pipeline(steps)

print("Pipeline builder defined.")

# ---------------------------------------------------------------------------
# 5. Tuning and evaluation
# ---------------------------------------------------------------------------
SCORING = {
    "RMSE": "neg_root_mean_squared_error", # RMSE = Root Mean Squared Error. It is the square root of the average of the 
    # squared differences between predicted and actual values.
    "MAE": "neg_mean_absolute_error", # MAE = Mean Absolute Error. It is the average of the absolute differences between 
    # predicted and actual values.
    "R2": "r2", # R2 = Coefficient of Determination. It is a measure of how well the model explains the variance in the 
    # target variable.
}
# RMSE vs MAE: RMSE penalizes larger errors more than MAE because it squares the errors before averaging.
# We use them both because RMSE is more sensitive to outliers, while MAE gives a more balanced view of overall prediction accuracy.


def _build_repeated_group_kfold_splits(
    groups_vec: np.ndarray,
    n_splits: int,
    n_repeats: int,
    random_state: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Build deterministic repeated grouped K-fold splits.

    Each repeat shuffles unique groups, partitions them into ``n_splits`` folds,
    and returns train/test row indices with strict group separation.
    """
    groups_arr = np.asarray(groups_vec)
    unique_groups = np.unique(groups_arr)

    if n_splits < 2:
        raise ValueError("n_splits must be at least 2 for repeated grouped K-fold tuning.")
    if n_repeats < 1:
        raise ValueError("n_repeats must be at least 1 for repeated grouped K-fold tuning.")
    if len(unique_groups) < n_splits:
        raise ValueError(
            f"Repeated grouped K-fold requested {n_splits} folds, "
            f"but only {len(unique_groups)} unique groups are available."
        )

    rng = np.random.RandomState(random_state)
    split_list: list[tuple[np.ndarray, np.ndarray]] = []

    for _ in range(n_repeats):
        shuffled_groups = unique_groups.copy()
        rng.shuffle(shuffled_groups)
        fold_groups = np.array_split(shuffled_groups, n_splits)

        for test_groups in fold_groups:
            test_mask = np.isin(groups_arr, test_groups)
            test_idx = np.where(test_mask)[0]
            train_idx = np.where(~test_mask)[0]
            split_list.append((train_idx, test_idx))

    return split_list

# for tuning of each model, we use repeated k-fold cross-validation to get a more stable estimate of performance across different splits of the data.
def tune_models(
    X_data: pd.DataFrame,
    y_vec: np.ndarray, # y_vec is the target/dependent variable as a 1D array of solpow for each sample in X_data
    groups_vec: np.ndarray,
    specs: dict[str, dict], # dictionary of model specifications, including estimator, scaling requirement, and hyperparameter grid for tuning
) -> tuple[dict[str, Pipeline], pd.DataFrame, pd.DataFrame]:
    """Tune each model with repeated grouped k-fold and return tuned estimators plus summary."""
    # Build repeated grouped CV for no-holdout tuning: 5 grouped folds repeated 5 times.
    unique_groups = np.unique(groups_vec)
    n_tune_splits = TUNE_N_SPLITS
    if len(unique_groups) < n_tune_splits:
        raise ValueError(
            f"No-holdout grouped tuning requested {n_tune_splits} folds, "
            f"but only {len(unique_groups)} unique drug groups are available."
        )
    tune_cv = _build_repeated_group_kfold_splits(
        groups_vec=groups_vec,
        n_splits=n_tune_splits,
        n_repeats=TUNE_N_REPEATS,
        random_state=RANDOM_STATE,
    )

    # Store the best fitted pipeline for each model and a table of tuning results.
    tuned_estimators = {}
    tuning_rows = []
    tuning_detail_tables = []

    for model_name, spec in specs.items():
        # Build the full pipeline for this model, including leakage-safe feature selection and optional scaling.
        pipe = make_pipeline(spec["estimator"], spec["needs_scaling"])

        # GridSearchCV uses a single tuning metric here: RMSE. That keeps model selection aligned with the main optimisation
        # target, while the broader SCORING dict below is used later for multi-metric evaluation.
        search = GridSearchCV(
            estimator=pipe,
            param_grid=spec["param_grid"],
            scoring="neg_root_mean_squared_error",
            cv=tune_cv,
            n_jobs=-1,
            refit=True,
        )

        # Fit the search object on the full dataset. During fitting, the data are split internally by the repeated CV object,
        # so each parameter setting is tested fairly on multiple folds.
        search.fit(X_data, y_vec)

        # Default single-stage tuning path for non-XGBoost models.
        tuned_estimators[model_name] = search.best_estimator_
        tuning_rows.append(
            {
                "Model": model_name,
                "BestCV_RMSE": -search.best_score_,
                "BestParams": search.best_params_,
            }
        )

        cv_results_df = pd.DataFrame(search.cv_results_).copy()
        cv_results_df.insert(0, "Model", model_name)
        cv_results_df.insert(1, "TuneStage", "single_stage")
        if "mean_test_score" in cv_results_df.columns:
            cv_results_df["mean_test_RMSE"] = -cv_results_df["mean_test_score"]
        if "std_test_score" in cv_results_df.columns:
            cv_results_df["std_test_RMSE"] = cv_results_df["std_test_score"]
        if "rank_test_score" in cv_results_df.columns:
            cv_results_df = cv_results_df.sort_values("rank_test_score")
        tuning_detail_tables.append(cv_results_df)

    # Sort the summary table so the best tuning result appears first.
    tuning_df = pd.DataFrame(tuning_rows).sort_values("BestCV_RMSE")
    tuning_details_df = pd.concat(tuning_detail_tables, ignore_index=True)
    return tuned_estimators, tuning_df, tuning_details_df

print("Model tuning function defined.")

def evaluate_models(
    X_data: pd.DataFrame,
    y_vec: np.ndarray,
    groups_vec: np.ndarray,
    tuned_estimators: dict[str, Pipeline],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate tuned models using repeated k-fold and leave-one-drug-out CV."""
    # RepeatedKFold gives a more stable estimate of general performance, while
    # LeaveOneGroupOut gives a strict unseen-drug evaluation.
    eval_cv = RepeatedKFold(
        n_splits=EVAL_N_SPLITS,
        n_repeats=EVAL_N_REPEATS,
        random_state=RANDOM_STATE,
    )
    logo_cv = LeaveOneGroupOut()

    # Store the aggregated scores from each evaluation strategy.
    repeated_rows = []
    logo_rows = []
    repeated_split_rows = []
    logo_split_rows = []
    repeated_oof_rows = []
    logo_oof_rows = []

    for model_name, estimator in tuned_estimators.items():
        repeated_splits = list(eval_cv.split(X_data, y_vec))

        # Evaluate the tuned model across repeated k-fold splits.
        repeated_results = cross_validate(
            estimator,
            X_data,
            y_vec,
            cv=eval_cv,
            scoring=SCORING,
            n_jobs=-1,
        )

        repeated_baseline_rmse = []
        repeated_improvement_pct = []

        for split_idx, (train_idx, test_idx) in enumerate(repeated_splits, start=1):
            y_train = y_vec[train_idx]
            y_test = y_vec[test_idx]
            held_out_drugs = np.unique(groups_vec[test_idx])
            held_out_drugs_text = " | ".join(map(str, held_out_drugs))

            X_test_fold = X_data.iloc[test_idx]
            if SURF_GROUP_COL in X_test_fold.columns:
                held_out_surfs = np.unique(X_test_fold[SURF_GROUP_COL].astype(str).to_numpy())
                held_out_surfs_text = " | ".join(map(str, held_out_surfs))

                held_out_pairs = np.unique(
                    [
                        f"{drug} || {surf}"
                        for drug, surf in zip(
                            groups_vec[test_idx],
                            X_test_fold[SURF_GROUP_COL].astype(str).to_numpy(),
                        )
                    ]
                )
                held_out_pairs_text = " | ".join(map(str, held_out_pairs))
            else:
                held_out_surfs_text = "NA"
                held_out_pairs_text = "NA"

            model_rmse = -repeated_results["test_RMSE"][split_idx - 1]
            baseline_pred = np.full(shape=y_test.shape, fill_value=float(np.mean(y_train)), dtype=float)
            baseline_rmse = float(np.sqrt(mean_squared_error(y_test, baseline_pred)))
            improvement_pct = float(((baseline_rmse - model_rmse) / baseline_rmse) * 100.0) if baseline_rmse > 0 else np.nan

            repeated_baseline_rmse.append(baseline_rmse)
            repeated_improvement_pct.append(improvement_pct)

            repeated_split_rows.append(
                {
                    "Model": model_name,
                    "Split": split_idx,
                    "Repeat": ((split_idx - 1) // EVAL_N_SPLITS) + 1,
                    "Fold": ((split_idx - 1) % EVAL_N_SPLITS) + 1,
                    "TestRows": len(test_idx),
                    "HeldOutDrugs": held_out_drugs_text,
                    "HeldOutSurfactants": held_out_surfs_text,
                    "HeldOutDrugSurfPairs": held_out_pairs_text,
                    "RMSE": model_rmse,
                    "MAE": -repeated_results["test_MAE"][split_idx - 1],
                    "R2": repeated_results["test_R2"][split_idx - 1],
                    "Baseline_RMSE_TrainMean": baseline_rmse,
                    "RMSE_ImprovementVsBaseline_Pct": improvement_pct,
                }
            )

        # Convert the negative scoring convention used by scikit-learn into positive RMSE/MAE values for reporting.
        repeated_rmse_values = -repeated_results["test_RMSE"]
        repeated_mae_values = -repeated_results["test_MAE"]
        repeated_r2_values = repeated_results["test_R2"]
        repeated_rows.append(
            {
                "Model": model_name,
                "RMSE": repeated_rmse_values.mean(),
                "RMSE_std": repeated_rmse_values.std(),
                "RMSE_min": repeated_rmse_values.min(),
                "RMSE_max": repeated_rmse_values.max(),
                "RMSE_range": repeated_rmse_values.max() - repeated_rmse_values.min(),
                "MAE": repeated_mae_values.mean(),
                "MAE_std": repeated_mae_values.std(),
                "MAE_min": repeated_mae_values.min(),
                "MAE_max": repeated_mae_values.max(),
                "MAE_range": repeated_mae_values.max() - repeated_mae_values.min(),
                "R2": repeated_r2_values.mean(),
                "R2_std": repeated_r2_values.std(),
                "R2_min": repeated_r2_values.min(),
                "R2_max": repeated_r2_values.max(),
                "R2_range": repeated_r2_values.max() - repeated_r2_values.min(),
                "Baseline_RMSE_TrainMean": float(np.mean(repeated_baseline_rmse)),
                "Baseline_RMSE_TrainMean_std": float(np.std(repeated_baseline_rmse)),
                "Baseline_RMSE_TrainMean_min": float(np.min(repeated_baseline_rmse)),
                "Baseline_RMSE_TrainMean_max": float(np.max(repeated_baseline_rmse)),
                "Baseline_RMSE_TrainMean_range": float(np.max(repeated_baseline_rmse) - np.min(repeated_baseline_rmse)),
                "RMSE_ImprovementVsBaseline_Pct": float(np.nanmean(repeated_improvement_pct)),
                "RMSE_ImprovementVsBaseline_Pct_std": float(np.nanstd(repeated_improvement_pct)),
                "RMSE_ImprovementVsBaseline_Pct_min": float(np.nanmin(repeated_improvement_pct)),
                "RMSE_ImprovementVsBaseline_Pct_max": float(np.nanmax(repeated_improvement_pct)),
                "RMSE_ImprovementVsBaseline_Pct_range": float(np.nanmax(repeated_improvement_pct) - np.nanmin(repeated_improvement_pct)),
            }
        )

        for split_idx, (train_idx, test_idx) in enumerate(repeated_splits, start=1):
            fold_estimator = clone(estimator)
            X_train = X_data.iloc[train_idx]
            X_test = X_data.iloc[test_idx]
            y_train = y_vec[train_idx]
            y_test = y_vec[test_idx]
            groups_test = groups_vec[test_idx]
            if SURF_GROUP_COL in X_test.columns:
                surf_test = X_test[SURF_GROUP_COL].astype(str).to_numpy()
            else:
                surf_test = np.array(["NA"] * len(test_idx), dtype=object)
            y_pred = np.ravel(fold_estimator.fit(X_train, y_train).predict(X_test))

            repeat_number = ((split_idx - 1) // EVAL_N_SPLITS) + 1
            fold_number = ((split_idx - 1) % EVAL_N_SPLITS) + 1

            for local_row_idx, original_row_idx in enumerate(test_idx):
                repeated_oof_rows.append(
                    {
                        "Model": model_name,
                        "Split": split_idx,
                        "Repeat": repeat_number,
                        "Fold": fold_number,
                        "RowIndex": int(original_row_idx),
                        "Drug": str(groups_test[local_row_idx]),
                        "Surfactant": str(surf_test[local_row_idx]),
                        "Observed": float(y_test[local_row_idx]),
                        "Predicted": float(y_pred[local_row_idx]),
                    }
                )

        logo_splits = list(logo_cv.split(X_data, y_vec, groups_vec))

        # Evaluate the same tuned model with leave-one-drug-out CV.
        logo_results = cross_validate(
            estimator,
            X_data,
            y_vec,
            cv=logo_cv,
            groups=groups_vec,
            scoring=SCORING,
            n_jobs=-1,
        )

        logo_baseline_rmse = []
        logo_improvement_pct = []

        for split_idx, (train_idx, test_idx) in enumerate(logo_splits, start=1):
            y_train = y_vec[train_idx]
            y_test = y_vec[test_idx]
            model_rmse = -logo_results["test_RMSE"][split_idx - 1]
            baseline_pred = np.full(shape=y_test.shape, fill_value=float(np.mean(y_train)), dtype=float)
            baseline_rmse = float(np.sqrt(mean_squared_error(y_test, baseline_pred)))
            improvement_pct = float(((baseline_rmse - model_rmse) / baseline_rmse) * 100.0) if baseline_rmse > 0 else np.nan

            logo_baseline_rmse.append(baseline_rmse)
            logo_improvement_pct.append(improvement_pct)

            held_out_groups = np.unique(groups_vec[test_idx])
            logo_split_rows.append(
                {
                    "Model": model_name,
                    "Split": split_idx,
                    "TestRows": len(test_idx),
                    "UniqueTestDrugs": len(held_out_groups),
                    "HeldOutDrugs": " | ".join(map(str, held_out_groups)),
                    "RMSE": model_rmse,
                    "MAE": -logo_results["test_MAE"][split_idx - 1],
                    "R2": logo_results["test_R2"][split_idx - 1],
                    "Baseline_RMSE_TrainMean": baseline_rmse,
                    "RMSE_ImprovementVsBaseline_Pct": improvement_pct,
                }
            )

        # Store the leave-one-drug-out summary for comparison.
        logo_rmse_values = -logo_results["test_RMSE"]
        logo_mae_values = -logo_results["test_MAE"]
        logo_r2_values = logo_results["test_R2"]
        logo_rows.append(
            {
                "Model": model_name,
                "RMSE": logo_rmse_values.mean(),
                "RMSE_std": logo_rmse_values.std(),
                "RMSE_min": logo_rmse_values.min(),
                "RMSE_max": logo_rmse_values.max(),
                "RMSE_range": logo_rmse_values.max() - logo_rmse_values.min(),
                "MAE": logo_mae_values.mean(),
                "MAE_std": logo_mae_values.std(),
                "MAE_min": logo_mae_values.min(),
                "MAE_max": logo_mae_values.max(),
                "MAE_range": logo_mae_values.max() - logo_mae_values.min(),
                "R2": logo_r2_values.mean(),
                "R2_std": logo_r2_values.std(),
                "R2_min": logo_r2_values.min(),
                "R2_max": logo_r2_values.max(),
                "R2_range": logo_r2_values.max() - logo_r2_values.min(),
                "Baseline_RMSE_TrainMean": float(np.mean(logo_baseline_rmse)),
                "Baseline_RMSE_TrainMean_std": float(np.std(logo_baseline_rmse)),
                "Baseline_RMSE_TrainMean_min": float(np.min(logo_baseline_rmse)),
                "Baseline_RMSE_TrainMean_max": float(np.max(logo_baseline_rmse)),
                "Baseline_RMSE_TrainMean_range": float(np.max(logo_baseline_rmse) - np.min(logo_baseline_rmse)),
                "RMSE_ImprovementVsBaseline_Pct": float(np.nanmean(logo_improvement_pct)),
                "RMSE_ImprovementVsBaseline_Pct_std": float(np.nanstd(logo_improvement_pct)),
                "RMSE_ImprovementVsBaseline_Pct_min": float(np.nanmin(logo_improvement_pct)),
                "RMSE_ImprovementVsBaseline_Pct_max": float(np.nanmax(logo_improvement_pct)),
                "RMSE_ImprovementVsBaseline_Pct_range": float(np.nanmax(logo_improvement_pct) - np.nanmin(logo_improvement_pct)),
            }
        )

        for split_idx, (train_idx, test_idx) in enumerate(logo_splits, start=1):
            fold_estimator = clone(estimator)
            X_train = X_data.iloc[train_idx]
            X_test = X_data.iloc[test_idx]
            y_train = y_vec[train_idx]
            y_test = y_vec[test_idx]
            groups_test = groups_vec[test_idx]
            if SURF_GROUP_COL in X_test.columns:
                surf_test = X_test[SURF_GROUP_COL].astype(str).to_numpy()
            else:
                surf_test = np.array(["NA"] * len(test_idx), dtype=object)
            held_out_drugs_text = " | ".join(map(str, np.unique(groups_test)))
            y_pred = np.ravel(fold_estimator.fit(X_train, y_train).predict(X_test))

            for local_row_idx, original_row_idx in enumerate(test_idx):
                logo_oof_rows.append(
                    {
                        "Model": model_name,
                        "Split": split_idx,
                        "HeldOutDrugs": held_out_drugs_text,
                        "RowIndex": int(original_row_idx),
                        "Drug": str(groups_test[local_row_idx]),
                        "Surfactant": str(surf_test[local_row_idx]),
                        "Observed": float(y_test[local_row_idx]),
                        "Predicted": float(y_pred[local_row_idx]),
                    }
                )

    # Sort both summary tables so the best-performing model appears first.
    repeated_df = pd.DataFrame(repeated_rows).sort_values("RMSE")
    logo_df = pd.DataFrame(logo_rows).sort_values("RMSE")
    repeated_splits_df = pd.DataFrame(repeated_split_rows).sort_values(["Model", "Split"])
    logo_splits_df = pd.DataFrame(logo_split_rows).sort_values(["Model", "Split"])
    repeated_oof_df = pd.DataFrame(repeated_oof_rows).sort_values(["Model", "Split", "RowIndex"])
    repeated_oof_agg_df = (
        repeated_oof_df
        .groupby(["Model", "RowIndex", "Drug", "Surfactant"], as_index=False)
        .agg(
            Observed=("Observed", "mean"),
            Predicted_mean=("Predicted", "mean"),
            Predicted_std=("Predicted", "std"),
            Predicted_count=("Predicted", "count"),
        )
        .sort_values(["Model", "RowIndex"])
    )
    repeated_oof_agg_df["Predicted_std"] = repeated_oof_agg_df["Predicted_std"].fillna(0.0)

    logo_oof_df = pd.DataFrame(logo_oof_rows).sort_values(["Model", "Split", "RowIndex"])
    pooled_logo_r2_df = (
        logo_oof_df
        .groupby("Model", as_index=False)
        .apply(lambda model_df: pd.Series({"Pooled_R2": float(r2_score(model_df["Observed"], model_df["Predicted"]))}))
        .reset_index(drop=True)
    )
    logo_df = logo_df.merge(pooled_logo_r2_df, on="Model", how="left")
    return (
        repeated_df,
        logo_df,
        repeated_splits_df,
        logo_splits_df,
        repeated_oof_df,
        repeated_oof_agg_df,
        logo_oof_df,
    )

print("Model evaluation function defined.")

def report_feature_selection_stability(
    X_data: pd.DataFrame,
    y_vec: np.ndarray,
    estimator: Pipeline,
    mode_name: str, # run mode label for exported files
) -> None:
    """Report fold-to-fold stability of leakage-safe feature selection.

    This fits the provided pipeline across repeated CV folds and inspects which
    features were retained by the feature selector in each fold.
    """
    stability_cv = RepeatedKFold(
        n_splits=EVAL_N_SPLITS,
        n_repeats=EVAL_N_REPEATS,
        random_state=RANDOM_STATE,
    )

    # return_estimator=True exposes each fold-fitted pipeline, including the feature_selector step with selected_feature_names_.
    stability_results = cross_validate(
        estimator,
        X_data,
        y_vec,
        cv=stability_cv,
        scoring="neg_root_mean_squared_error", # useful for consistency but not necessary for the stability result itself
        n_jobs=-1,
        return_estimator=True,
    )

    # Extract the selected feature names from each fold-fitted pipeline.
    selected_lists = [
        fold_estimator.named_steps["feature_selector"].selected_feature_names_
        for fold_estimator in stability_results["estimator"]
    ]

    drug_audit_frames = []
    surf_audit_frames = []
    for fold_idx, fold_estimator in enumerate(stability_results["estimator"], start=1):
        selector = fold_estimator.named_steps["feature_selector"]
        fold_audit = getattr(selector, "selection_audit_", {})

        if "drug" in fold_audit and isinstance(fold_audit["drug"], pd.DataFrame) and not fold_audit["drug"].empty:
            audit_df = fold_audit["drug"].copy()
            audit_df.insert(0, "Fold", fold_idx)
            drug_audit_frames.append(audit_df)

        if "surf" in fold_audit and isinstance(fold_audit["surf"], pd.DataFrame) and not fold_audit["surf"].empty:
            audit_df = fold_audit["surf"].copy()
            audit_df.insert(0, "Fold", fold_idx)
            surf_audit_frames.append(audit_df)

    # Count how many features were kept in each fold, then summarise the spread.
    fold_counts = np.array([len(features) for features in selected_lists])
    total_folds = len(selected_lists)

    print("\nFeature-selection stability across repeated CV folds:")
    print(f"Folds checked: {total_folds}")
    print(f"Retained feature count (mean ± std): {fold_counts.mean():.2f} ± {fold_counts.std():.2f}")
    print(f"Retained feature count range: {fold_counts.min()} to {fold_counts.max()}")

    # Save the fold-count summary to a CSV so the stability check is archived
    # alongside the other model comparison outputs.
    stability_summary_df = pd.DataFrame(
        [
            {"Metric": "Folds checked", "Value": total_folds},
            {"Metric": "Retained feature count mean", "Value": fold_counts.mean()},
            {"Metric": "Retained feature count std", "Value": fold_counts.std()},
            {"Metric": "Retained feature count min", "Value": fold_counts.min()},
            {"Metric": "Retained feature count max", "Value": fold_counts.max()},
        ]
    )
    export_result_table(stability_summary_df, mode_name, "feature_selection_stability_summary")

    print("Results table exported - feature_selection_stability_summary.csv")

    # Count how often each drug descriptor survived the feature-selection process.
    drug_feature_counter = Counter()
    surf_feature_counter = Counter()
    for feature_list in selected_lists:
        drug_feature_counter.update([f for f in feature_list if f.startswith("drug_")])
        surf_feature_counter.update([f for f in feature_list if f.startswith("surf_")])

    if drug_feature_counter:
        # Show the descriptors that were most consistently retained across folds.
        print("\nMost frequently retained drug descriptors across folds:")
        stability_descriptor_rows = []
        for feature_name, count in drug_feature_counter.most_common(15):
            frequency_pct = (count / total_folds) * 100
            print(f"{feature_name}: retained in {count}/{total_folds} folds ({frequency_pct:.1f}%)")
            stability_descriptor_rows.append(
                {
                    "Feature": feature_name,
                    "Retained_Folds": count,
                    "Retention_Pct": frequency_pct,
                }
            )

        export_result_table(
            pd.DataFrame(stability_descriptor_rows),
            mode_name,
            "feature_selection_stability_descriptors",
        )

    # Export a full drug-descriptor audit so retained and filtered descriptors
    # are both documented across all CV folds.
    if drug_audit_frames:
        drug_audit_df = pd.concat(drug_audit_frames, ignore_index=True)
        drug_audit_df["Retained_Int"] = drug_audit_df["Retained"].astype(int)
        drug_audit_df["Filtered_By_Variance_Int"] = drug_audit_df["Filtered_By_Variance"].astype(int)
        drug_audit_df["Filtered_By_TargetCorrelation_Int"] = (
            drug_audit_df["Selection_Reason"].eq("TargetCorrelation").astype(int)
        )
        drug_audit_df["Filtered_By_Correlation_Int"] = drug_audit_df["Filtered_By_Correlation"].astype(int)

        drug_audit_summary = (
            drug_audit_df.groupby("Feature", as_index=False)
            .agg(
                Retained_Folds=("Retained_Int", "sum"),
                Mean_Variance=("Variance", "mean"),
                Std_Variance=("Variance", "std"),
                Mean_Abs_Target_Correlation=("Abs_Target_Correlation", "mean"),
                Std_Abs_Target_Correlation=("Abs_Target_Correlation", "std"),
                Mean_Max_Abs_InterFeature_Correlation=("Max_Abs_InterFeature_Correlation", "mean"),
                Std_Max_Abs_InterFeature_Correlation=("Max_Abs_InterFeature_Correlation", "std"),
                Filtered_By_Variance_Folds=("Filtered_By_Variance_Int", "sum"),
                Filtered_By_TargetCorrelation_Folds=("Filtered_By_TargetCorrelation_Int", "sum"),
                Filtered_By_Correlation_Folds=("Filtered_By_Correlation_Int", "sum"),
            )
            .sort_values(["Retained_Folds", "Feature"], ascending=[False, True])
        )
        drug_audit_summary["Retained_Pct"] = (drug_audit_summary["Retained_Folds"] / total_folds) * 100
        drug_audit_summary["Filtered_Folds"] = total_folds - drug_audit_summary["Retained_Folds"]
        drug_audit_summary["Filtered_Pct"] = (drug_audit_summary["Filtered_Folds"] / total_folds) * 100
        drug_audit_summary["Status"] = np.where(
            drug_audit_summary["Retained_Folds"] == total_folds,
            "Always retained",
            np.where(drug_audit_summary["Retained_Folds"] == 0, "Always filtered", "Sometimes filtered"),
        )
        export_result_table(
            drug_audit_summary,
            mode_name,
            "feature_selection_drug_retained_vs_filtered",
        )

    if surf_feature_counter:
        # Show surfactant descriptors that were consistently retained across folds.
        print("\nMost frequently retained surfactant descriptors across folds:")
        surf_stability_rows = []
        for feature_name, count in surf_feature_counter.most_common(15):
            frequency_pct = (count / total_folds) * 100
            print(f"{feature_name}: retained in {count}/{total_folds} folds ({frequency_pct:.1f}%)")
            surf_stability_rows.append(
                {
                    "Feature": feature_name,
                    "Retained_Folds": count,
                    "Retention_Pct": frequency_pct,
                }
            )

        export_result_table(
            pd.DataFrame(surf_stability_rows),
            mode_name,
            "feature_selection_stability_surf_descriptors",
        )

    # Export a full surf-descriptor audit so retained and filtered descriptors
    # are both documented across all CV folds.
    if surf_audit_frames:
        surf_audit_df = pd.concat(surf_audit_frames, ignore_index=True)
        surf_audit_df["Retained_Int"] = surf_audit_df["Retained"].astype(int)
        surf_audit_df["Filtered_By_Variance_Int"] = surf_audit_df["Filtered_By_Variance"].astype(int)
        surf_audit_df["Filtered_By_TargetCorrelation_Int"] = (
            surf_audit_df["Selection_Reason"].eq("TargetCorrelation").astype(int)
        )
        surf_audit_df["Filtered_By_Correlation_Int"] = surf_audit_df["Filtered_By_Correlation"].astype(int)

        surf_audit_summary = (
            surf_audit_df.groupby("Feature", as_index=False)
            .agg(
                Retained_Folds=("Retained_Int", "sum"),
                Mean_Variance=("Variance", "mean"),
                Std_Variance=("Variance", "std"),
                Mean_Abs_Target_Correlation=("Abs_Target_Correlation", "mean"),
                Std_Abs_Target_Correlation=("Abs_Target_Correlation", "std"),
                Mean_Max_Abs_InterFeature_Correlation=("Max_Abs_InterFeature_Correlation", "mean"),
                Std_Max_Abs_InterFeature_Correlation=("Max_Abs_InterFeature_Correlation", "std"),
                Filtered_By_Variance_Folds=("Filtered_By_Variance_Int", "sum"),
                Filtered_By_TargetCorrelation_Folds=("Filtered_By_TargetCorrelation_Int", "sum"),
                Filtered_By_Correlation_Folds=("Filtered_By_Correlation_Int", "sum"),
            )
            .sort_values(["Retained_Folds", "Feature"], ascending=[False, True])
        )
        surf_audit_summary["Retained_Pct"] = (surf_audit_summary["Retained_Folds"] / total_folds) * 100
        surf_audit_summary["Filtered_Folds"] = total_folds - surf_audit_summary["Retained_Folds"]
        surf_audit_summary["Filtered_Pct"] = (surf_audit_summary["Filtered_Folds"] / total_folds) * 100
        surf_audit_summary["Status"] = np.where(
            surf_audit_summary["Retained_Folds"] == total_folds,
            "Always retained",
            np.where(surf_audit_summary["Retained_Folds"] == 0, "Always filtered", "Sometimes filtered"),
        )
        export_result_table(
            surf_audit_summary,
            mode_name,
            "feature_selection_surf_retained_vs_filtered",
        )

print("Results table exported - feature_selection_stability_descriptors.csv")

def run_nested_holdout(
    X_data: pd.DataFrame,
    y_vec: np.ndarray,
    groups_vec: np.ndarray,
    specs: dict[str, dict],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run repeated nested holdout with inner tuning and unseen holdout testing."""
    # Outer grouped split: unseen-drug holdout folds.
    unique_groups = np.unique(groups_vec)
    n_outer_splits = NESTED_HOLDOUT_N_REPEATS
    if n_outer_splits < 2:
        raise ValueError("Grouped nested holdout requires at least 2 outer folds.")
    if len(unique_groups) < n_outer_splits:
        raise ValueError(
            f"Grouped nested holdout requested {n_outer_splits} outer folds, "
            f"but only {len(unique_groups)} unique drug groups are available."
        )

    holdout_splitter = GroupKFold(n_splits=n_outer_splits)
    holdout_splits = list(holdout_splitter.split(X_data, y_vec, groups_vec))

    # Store one row per model for the averaged holdout performance,
    # plus a more detailed row for each repeat.
    holdout_rows = []
    holdout_repeat_rows = []
    holdout_prediction_rows = []
    inner_tuning_rows = []

    for model_name, spec in specs.items():
        holdout_rmse = []
        holdout_mae = []
        holdout_r2 = []
        holdout_baseline_rmse = []
        holdout_improvement_pct = []

        for repeat_idx, (train_idx, test_idx) in enumerate(holdout_splits, start=1):
            X_train = X_data.iloc[train_idx]
            X_test = X_data.iloc[test_idx]
            y_train = y_vec[train_idx]
            y_test = y_vec[test_idx]
            groups_train = groups_vec[train_idx]
            groups_test = groups_vec[test_idx]
            if SURF_GROUP_COL in X_test.columns:
                surf_test = X_test[SURF_GROUP_COL].astype(str).to_numpy()
            else:
                surf_test = np.array(["NA"] * len(test_idx), dtype=object)
            held_out_drugs_text = " | ".join(map(str, np.unique(groups_test)))
            held_out_drug_count = int(len(np.unique(groups_test)))

            unique_train_groups = np.unique(groups_train)
            n_inner_splits = min(NESTED_GROUPED_INNER_N_SPLITS, len(unique_train_groups))
            if n_inner_splits < 2:
                raise ValueError(
                    "Nested grouped inner tuning requires at least 2 unique drug groups in outer-train data."
                )
            inner_cv = GroupKFold(n_splits=n_inner_splits)

            pipe = make_pipeline(spec["estimator"], spec["needs_scaling"])

            search = GridSearchCV(
                estimator=pipe,
                param_grid=spec["param_grid"],
                scoring="neg_root_mean_squared_error",
                cv=inner_cv,
                n_jobs=-1,
                refit=True,
            )

            search.fit(X_train, y_train, groups=groups_train)

            # Archive split-level inner tuning results for this outer repeat.
            repeat_cv_results_df = pd.DataFrame(search.cv_results_).copy()
            repeat_cv_results_df.insert(0, "Model", model_name)
            repeat_cv_results_df.insert(1, "OuterRepeat", repeat_idx)
            repeat_cv_results_df.insert(2, "OuterHeldOutDrugs", held_out_drugs_text)
            repeat_cv_results_df.insert(3, "OuterUniqueTestDrugs", held_out_drug_count)
            if "mean_test_score" in repeat_cv_results_df.columns:
                repeat_cv_results_df["mean_test_RMSE"] = -repeat_cv_results_df["mean_test_score"]
            if "std_test_score" in repeat_cv_results_df.columns:
                repeat_cv_results_df["std_test_RMSE"] = repeat_cv_results_df["std_test_score"]
            if "rank_test_score" in repeat_cv_results_df.columns:
                repeat_cv_results_df = repeat_cv_results_df.sort_values("rank_test_score")
            inner_tuning_rows.append(repeat_cv_results_df)

            best_model = search.best_estimator_
            y_pred = np.ravel(best_model.predict(X_test))

            repeat_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            repeat_mae = mean_absolute_error(y_test, y_pred)
            repeat_r2 = r2_score(y_test, y_pred)
            baseline_pred = np.full(shape=y_test.shape, fill_value=float(np.mean(y_train)), dtype=float)
            baseline_rmse = float(np.sqrt(mean_squared_error(y_test, baseline_pred)))
            improvement_pct = float(((baseline_rmse - repeat_rmse) / baseline_rmse) * 100.0) if baseline_rmse > 0 else np.nan

            holdout_rmse.append(repeat_rmse)
            holdout_mae.append(repeat_mae)
            holdout_r2.append(repeat_r2)
            holdout_baseline_rmse.append(baseline_rmse)
            holdout_improvement_pct.append(improvement_pct)

            holdout_repeat_rows.append(
                {
                    "Model": model_name,
                    "Repeat": repeat_idx,
                    "TestRows": len(test_idx),
                    "UniqueTestDrugs": held_out_drug_count,
                    "HeldOutDrugs": held_out_drugs_text,
                    "RMSE": float(repeat_rmse),
                    "MAE": float(repeat_mae),
                    "R2": float(repeat_r2),
                    "Baseline_RMSE_TrainMean": baseline_rmse,
                    "RMSE_ImprovementVsBaseline_Pct": improvement_pct,
                }
            )

            for local_row_idx, original_row_idx in enumerate(test_idx):
                holdout_prediction_rows.append(
                    {
                        "Model": model_name,
                        "Repeat": repeat_idx,
                        "UniqueTestDrugs": held_out_drug_count,
                        "HeldOutDrugs": held_out_drugs_text,
                        "RowIndex": int(original_row_idx),
                        "Drug": str(groups_test[local_row_idx]),
                        "Surfactant": str(surf_test[local_row_idx]),
                        "Observed": float(y_test[local_row_idx]),
                        "Predicted": float(y_pred[local_row_idx]),
                    }
                )

        holdout_rows.append(
            {
                "Model": model_name,
                "Holdout_RMSE": float(np.mean(holdout_rmse)),
                "Holdout_RMSE_std": float(np.std(holdout_rmse)),
                "Holdout_RMSE_min": float(np.min(holdout_rmse)),
                "Holdout_RMSE_max": float(np.max(holdout_rmse)),
                "Holdout_RMSE_range": float(np.max(holdout_rmse) - np.min(holdout_rmse)),
                "Holdout_MAE": float(np.mean(holdout_mae)),
                "Holdout_MAE_std": float(np.std(holdout_mae)),
                "Holdout_MAE_min": float(np.min(holdout_mae)),
                "Holdout_MAE_max": float(np.max(holdout_mae)),
                "Holdout_MAE_range": float(np.max(holdout_mae) - np.min(holdout_mae)),
                "Holdout_R2": float(np.mean(holdout_r2)),
                "Holdout_R2_std": float(np.std(holdout_r2)),
                "Holdout_R2_min": float(np.min(holdout_r2)),
                "Holdout_R2_max": float(np.max(holdout_r2)),
                "Holdout_R2_range": float(np.max(holdout_r2) - np.min(holdout_r2)),
                "Baseline_RMSE_TrainMean": float(np.mean(holdout_baseline_rmse)),
                "Baseline_RMSE_TrainMean_std": float(np.std(holdout_baseline_rmse)),
                "Baseline_RMSE_TrainMean_min": float(np.min(holdout_baseline_rmse)),
                "Baseline_RMSE_TrainMean_max": float(np.max(holdout_baseline_rmse)),
                "Baseline_RMSE_TrainMean_range": float(np.max(holdout_baseline_rmse) - np.min(holdout_baseline_rmse)),
                "RMSE_ImprovementVsBaseline_Pct": float(np.nanmean(holdout_improvement_pct)),
                "RMSE_ImprovementVsBaseline_Pct_std": float(np.nanstd(holdout_improvement_pct)),
                "RMSE_ImprovementVsBaseline_Pct_min": float(np.nanmin(holdout_improvement_pct)),
                "RMSE_ImprovementVsBaseline_Pct_max": float(np.nanmax(holdout_improvement_pct)),
                "RMSE_ImprovementVsBaseline_Pct_range": float(np.nanmax(holdout_improvement_pct) - np.nanmin(holdout_improvement_pct)),
            }
        )

    summary_df = pd.DataFrame(holdout_rows).sort_values("Holdout_RMSE")
    repeats_df = pd.DataFrame(holdout_repeat_rows).sort_values(["Model", "Repeat"])
    predictions_df = pd.DataFrame(holdout_prediction_rows).sort_values(["Model", "Repeat", "RowIndex"])
    inner_tuning_df = pd.concat(inner_tuning_rows, ignore_index=True)

    pooled_r2_df = (
        predictions_df
        .groupby("Model", as_index=False)
        .apply(lambda model_df: pd.Series({"Pooled_R2": float(r2_score(model_df["Observed"], model_df["Predicted"]))}))
        .reset_index(drop=True)
    )
    summary_df = summary_df.merge(pooled_r2_df, on="Model", how="left")
    return summary_df, repeats_df, predictions_df, inner_tuning_df

print("Nested holdout function defined.")


def run_standard_flow(
    X_data: pd.DataFrame,
    y_vec: np.ndarray,
    groups_vec: np.ndarray,
    specs: dict[str, dict],
    mode_name: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the original no-holdout workflow (tuning + repeated CV + LOGO)."""
    print("\nTuning models...")
    tuned_estimators, tuning_df, tuning_details_df = tune_models(X_data, y_vec, groups_vec, specs)
    print(tuning_df[["Model", "BestCV_RMSE"]].round(4).to_string(index=False))

    print("\nBest parameters by model:")
    for _, row in tuning_df.iterrows():
        print(f"{row['Model']}: {row['BestParams']}")

    print("\nRepeated 5-fold CV results (5x20):")
    (
        repeated_df,
        logo_df,
        repeated_splits_df,
        logo_splits_df,
        repeated_oof_df,
        repeated_oof_agg_df,
        logo_oof_df,
    ) = evaluate_models(X_data, y_vec, groups_vec, tuned_estimators)
    print(repeated_df.round(4).to_string(index=False))

    print("\nLeave-One-Drug-Out CV results:")
    print(logo_df.round(4).to_string(index=False))

    best_model_name = repeated_df.iloc[0]["Model"]
    best_estimator = tuned_estimators[best_model_name]
    print(f"\nBest model by repeated CV RMSE: {best_model_name}")

    best_estimator.fit(X_data, y_vec)
    fitted_model = best_estimator.named_steps["model"]
    selected_features = best_estimator.named_steps["feature_selector"].selected_feature_names_

    if hasattr(fitted_model, "coef_"):
        coef_values = np.ravel(fitted_model.coef_)
        if len(coef_values) == len(selected_features):
            coefs = pd.Series(coef_values, index=selected_features)
            print("\nTop coefficients by absolute magnitude:")
            print(coefs.reindex(coefs.abs().sort_values(ascending=False).index).head(TOP_N_FEATURES_TO_PRINT))
    elif hasattr(fitted_model, "feature_importances_"):
        importances = pd.Series(fitted_model.feature_importances_, index=selected_features)
        print("\nTop feature importances:")
        print(importances.sort_values(ascending=False).head(TOP_N_FEATURES_TO_PRINT))

    report_feature_selection_stability(X_data, y_vec, best_estimator, mode_name)

    return (
        repeated_df,
        logo_df,
        repeated_splits_df,
        logo_splits_df,
        repeated_oof_df,
        repeated_oof_agg_df,
        logo_oof_df,
        tuning_df,
        tuning_details_df,
    )

print("Standard workflow function defined.")

# ---------------------------------------------------------------------------
# 6. Run the feature-reduction workflow
# ---------------------------------------------------------------------------
def run_mode(mode_name: str) -> None:
    # This wrapper runs the complete comparison workflow with feature reduction.
    run_log_rows = []

    print("\n" + "=" * 80)
    print(f"MODE: {mode_name}")
    print("=" * 80)
    run_log_rows.append({"Stage": "Mode", "Detail": "Mode name", "Value": mode_name})

    print(
        "Feature reduction is ON (leakage-safe): drug and surf descriptors are filtered inside each CV training fold."
    )
    run_log_rows.append({"Stage": "Feature reduction", "Detail": "Setting", "Value": "ON"})

    max_pls_components = min(6, len(descriptor_cols), len(y) - 1)
    # Build the set of model definitions and tuning grids using the available
    # number of descriptors and samples.
    specs = model_specs(max_pls_components=max_pls_components)
    run_log_rows.append({"Stage": "Model setup", "Detail": "Max PLS components", "Value": max_pls_components})

    # In this cleaned version, the script is intentionally set up to always run
    # the all-validation comparison workflow.
    if not USE_COMPARE_ALL_VALIDATION_MODES:
        raise ValueError(
            "This cleaned script version expects USE_COMPARE_ALL_VALIDATION_MODES = True. "
            "Set it back to True to run the all-validation workflow."
        )

    print("\nAll-validation comparison mode is ON.")
    print("This run will execute and compare: no-holdout and nested holdout.")
    run_log_rows.append({"Stage": "Validation workflow", "Detail": "All-validation mode", "Value": "ON"})

    # Run the standard workflow first: tune models, then evaluate with repeated
    # CV and leave-one-drug-out CV.
    (
        repeated_df,
        logo_df,
        repeated_splits_df,
        logo_splits_df,
        repeated_oof_df,
        repeated_oof_agg_df,
        logo_oof_df,
        tuning_df,
        tuning_details_df,
    ) = run_standard_flow(X_df, y, groups, specs, mode_name)
    tuning_df = add_error_percentage_columns(tuning_df, y)
    tuning_details_df = add_error_percentage_columns(tuning_details_df, y)
    repeated_df = add_error_percentage_columns(repeated_df, y)
    logo_df = add_error_percentage_columns(logo_df, y)
    repeated_splits_df = add_error_percentage_columns(repeated_splits_df, y)
    logo_splits_df = add_error_percentage_columns(logo_splits_df, y)
    export_result_table(tuning_df, mode_name, "tuning_summary")
    export_result_table(tuning_details_df, mode_name, "tuning_gridsearch_details")

    # Export a shorter tuning shortlist so the strongest parameter settings are
    # easy to review without scanning the full grid-search table.
    tuning_top_df = (
        tuning_details_df
        .sort_values(["Model", "rank_test_score", "mean_test_RMSE"])
        .groupby("Model", group_keys=False)
        .head(TOP_N_TUNING_ROWS_TO_EXPORT)
    )
    export_result_table(tuning_top_df, mode_name, "tuning_gridsearch_top_rows")
    repeated_split_flags_df = build_split_outlier_flags(
        split_scores_df=repeated_splits_df,
        split_label_col="Split",
        rmse_col="RMSE",
    )
    logo_split_flags_df = build_split_outlier_flags(
        split_scores_df=logo_splits_df,
        split_label_col="Split",
        rmse_col="RMSE",
    )
    repeated_split_flags_df = add_error_percentage_columns(repeated_split_flags_df, y)
    logo_split_flags_df = add_error_percentage_columns(logo_split_flags_df, y)
    export_result_table(repeated_df, mode_name, "standard_repeated_cv")
    export_result_table(logo_df, mode_name, "standard_logo_cv")
    export_result_table(repeated_split_flags_df, mode_name, "standard_repeated_cv_per_split")
    export_result_table(logo_split_flags_df, mode_name, "standard_logo_cv_per_split")
    export_result_table(repeated_split_flags_df, mode_name, "standard_repeated_cv_split_outlier_flags")
    export_result_table(logo_split_flags_df, mode_name, "standard_logo_cv_split_outlier_flags")
    export_result_table(repeated_oof_df, mode_name, "standard_repeated_oof_predictions_raw")
    export_result_table(repeated_oof_agg_df, mode_name, "standard_repeated_oof_predictions_aggregated")
    export_result_table(logo_oof_df, mode_name, "standard_logo_oof_predictions")

        # Run the nested holdout workflow, where each repeat creates a fresh outer
    # holdout split and performs tuning only inside the corresponding training set.
    print("\nNested holdout mode:")
    print(f"Mode: grouped, Outer folds: {NESTED_HOLDOUT_N_REPEATS}")
    run_log_rows.append(
        {
            "Stage": "Nested holdout",
                "Detail": "Split mode / outer folds",
                "Value": f"grouped, outer_folds={NESTED_HOLDOUT_N_REPEATS}",
        }
    )
    nested_holdout_df, nested_holdout_repeats_df, nested_holdout_predictions_df, nested_inner_tuning_df = run_nested_holdout(X_df, y, groups, specs)
    nested_holdout_df = add_error_percentage_columns(nested_holdout_df, y)
    nested_holdout_repeats_df = add_error_percentage_columns(nested_holdout_repeats_df, y)
    nested_inner_tuning_df = add_error_percentage_columns(nested_inner_tuning_df, y)
    print("\nNested holdout results (summary):")
    print(nested_holdout_df.round(4).to_string(index=False))
    print("\nNested holdout per-repeat results:")
    print(nested_holdout_repeats_df.round(4).to_string(index=False))
    nested_repeat_flags_df = build_split_outlier_flags(
        split_scores_df=nested_holdout_repeats_df,
        split_label_col="Repeat",
        rmse_col="RMSE",
    )
    nested_repeat_flags_df = add_error_percentage_columns(nested_repeat_flags_df, y)
    export_result_table(nested_holdout_df, mode_name, "nested_holdout_summary")
    export_result_table(nested_repeat_flags_df, mode_name, "nested_holdout_per_repeat")
    export_result_table(nested_holdout_predictions_df, mode_name, "nested_holdout_predictions")
    export_result_table(nested_inner_tuning_df, mode_name, "nested_holdout_inner_tuning_details")
    export_result_table(nested_repeat_flags_df, mode_name, "nested_holdout_split_outlier_flags")
    nested_inner_tuning_top_df = (
        nested_inner_tuning_df
        .sort_values(["Model", "OuterRepeat", "rank_test_score", "mean_test_RMSE"])
        .groupby(["Model", "OuterRepeat"], group_keys=False)
        .head(TOP_N_TUNING_ROWS_TO_EXPORT)
    )
    export_result_table(nested_inner_tuning_top_df, mode_name, "nested_holdout_inner_tuning_top_rows")
    best_nested_model = nested_holdout_df.iloc[0]["Model"]
    best_nested_predictions = nested_holdout_predictions_df[nested_holdout_predictions_df["Model"] == best_nested_model]
    
    run_log_rows.append(
        {
            "Stage": "Nested holdout",
            "Detail": "Best summary model",
            "Value": nested_holdout_df.iloc[0]["Model"],
        }
    )

    # Merge the summary tables so the three validation strategies can be compared
    # side by side for each model.
    standard_cmp = repeated_df[[
        "Model",
        "RMSE",
        "RMSE_std",
        "MAE",
        "MAE_std",
        "R2",
        "R2_std",
        "Baseline_RMSE_TrainMean",
        "Baseline_RMSE_TrainMean_std",
        "RMSE_ImprovementVsBaseline_Pct",
        "RMSE_ImprovementVsBaseline_Pct_std",
    ]].rename(
        columns={
            "RMSE": "Standard_RMSE",
            "RMSE_std": "Standard_RMSE_std",
            "MAE": "Standard_MAE",
            "MAE_std": "Standard_MAE_std",
            "R2": "Standard_R2",
            "R2_std": "Standard_R2_std",
            "Baseline_RMSE_TrainMean": "Standard_Baseline_RMSE_TrainMean",
            "Baseline_RMSE_TrainMean_std": "Standard_Baseline_RMSE_TrainMean_std",
            "RMSE_ImprovementVsBaseline_Pct": "Standard_RMSE_ImprovementVsBaseline_Pct",
            "RMSE_ImprovementVsBaseline_Pct_std": "Standard_RMSE_ImprovementVsBaseline_Pct_std",
        }
    )
    logo_cmp = logo_df[[
        "Model",
        "RMSE",
        "RMSE_std",
        "MAE",
        "MAE_std",
        "R2",
        "R2_std",
        "Baseline_RMSE_TrainMean",
        "Baseline_RMSE_TrainMean_std",
        "RMSE_ImprovementVsBaseline_Pct",
        "RMSE_ImprovementVsBaseline_Pct_std",
    ]].rename(
        columns={
            "RMSE": "LOGO_RMSE",
            "RMSE_std": "LOGO_RMSE_std",
            "MAE": "LOGO_MAE",
            "MAE_std": "LOGO_MAE_std",
            "R2": "LOGO_R2",
            "R2_std": "LOGO_R2_std",
            "Baseline_RMSE_TrainMean": "LOGO_Baseline_RMSE_TrainMean",
            "Baseline_RMSE_TrainMean_std": "LOGO_Baseline_RMSE_TrainMean_std",
            "RMSE_ImprovementVsBaseline_Pct": "LOGO_RMSE_ImprovementVsBaseline_Pct",
            "RMSE_ImprovementVsBaseline_Pct_std": "LOGO_RMSE_ImprovementVsBaseline_Pct_std",
        }
    )

    comparison_df = (
        standard_cmp
        .merge(logo_cmp, on="Model", how="left")
        .merge(nested_holdout_df, on="Model", how="left")
    )
    comparison_df["NestedHoldout_minus_Standard_RMSE"] = (
        comparison_df["Holdout_RMSE"] - comparison_df["Standard_RMSE"]
    )
    comparison_df = add_error_percentage_columns(comparison_df, y)

    # Print and export the combined comparison table so the results are easy to
    # inspect later.
    print("\nComparison across all validation modes:")
    print(comparison_df.sort_values("Standard_RMSE").round(4).to_string(index=False))
    export_result_table(comparison_df.sort_values("Standard_RMSE"), mode_name, "comparison_all_validation_modes")

    # Export a compact run log with the key configuration and headline outcomes.
    export_result_table(pd.DataFrame(run_log_rows), mode_name, "run_log")


if __name__ == "__main__":
    _set_windows_sleep_prevention(True)
    try:
        run_mode("With Feature Reduction")
    finally:
        _set_windows_sleep_prevention(False)

print("\nAll workflows completed. Check the output folder for results tables and logs.")
print("🌹🌻🌷")