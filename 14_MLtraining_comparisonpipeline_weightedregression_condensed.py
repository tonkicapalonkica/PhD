"""
Condensed weighted-regression pipeline for drug-surfactant solubility modeling.

This script runs three validation streams with fixed hyperparameters only:
1) No-holdout repeated K-fold
2) No-holdout LOGO (leave-one-drug-out)
3) Nested holdout (outer GroupKFold only; no inner tuning)

Feature filtering is performed inside each pipeline fit to avoid leakage.
Weights are derived from propagated SE in column SEtransformed.

Assumed tidy input structure (one row per drug-surfactant pair):
    drug_name | surfactant_name | <drug descriptors...> |
    <surfactant descriptors...> | target | propagated SE

Leakage-safety principle:
Feature filtering is executed inside each pipeline fit. During CV, each
training fold computes its own selected features, and test-fold rows are
transformed using only those training-fold decisions.
"""

import ctypes # to prevent Windows sleep during long runs
import inspect # to check if model.fit accepts sample_weight
import sys # to check platform for sleep prevention
from datetime import datetime # to timestamp output files
from pathlib import Path # to manage output directories and file paths

import numpy as np # to handle arrays and numerical operations
import pandas as pd # to handle dataframes and CSV I/O
from sklearn.base import BaseEstimator, TransformerMixin, clone # to build custom feature selector and clone pipelines
from sklearn.ensemble import RandomForestRegressor # to use Random Forest regression
from sklearn.feature_selection import VarianceThreshold # to filter near-zero-variance features
from sklearn.gaussian_process import GaussianProcessRegressor # to use Gaussian Process regression
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel # to define kernels for Gaussian Process
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score # to compute regression metrics
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut, RepeatedKFold # to perform cross-validation
from sklearn.pipeline import Pipeline # to build machine learning pipelines
from sklearn.preprocessing import StandardScaler # to standardize features for models that require it

try:
    from xgboost import XGBRegressor # to use XGBoost regression

    HAS_XGBOOST = True
except ImportError:
    XGBRegressor = None
    HAS_XGBOOST = False

print("Libraries imported.")

# ---------------------------------------------------------------------------
# 0. Configuration
# ---------------------------------------------------------------------------
DATA_FILE = "14_drugsurf_moldescr_withnames_zerovariationfiltered_logmodsolpow+SE.csv" # name of file with rdkit descriptors for drugs and surfs, and log-modulus solpow values with propagated SE
TARGET_COL = "log-mod(solpow)" # column name for the target variable (log-modulus of solubilisation power, umol/g)
SE_COL = "SEtransformed" # column name for the propagated standard error of the log-modulus solpow values
GROUP_COL = "drug_NAME" # used for leave-one-drug-out CV (cross-validation)
SURF_GROUP_COL = "surf_NAME" # used to define unique surfactants for surf descriptor filtering

# Feature-selection thresholds.
# Variance filtering removes near-constant descriptors that carry little signal.
# Correlation filtering deduplicates highly collinear descriptors.
# Target-correlation filtering keeps descriptors with at least weak relevance.
DRUG_VAR_THRESHOLD = 0.01 # Minimum variance threshold for drug descriptors; features with variance below this are dropped.
DRUG_CORR_THRESHOLD = 0.95 # Maximum absolute correlation threshold for drug descriptors; features with pairwise correlation above this are dropped.
# Drug target-correlation relevance threshold, applied after variance filtering and
# before interfeature deduplication. A value around 0.2 corresponds to a weak but
# non-negligible effect-size range often used as a practical lower bound.
DRUG_TARGET_CORR_THRESHOLD = 0.2 # Minimum absolute correlation with the target for drug descriptors; features with lower correlation are dropped.
SURF_VAR_THRESHOLD = 0.01 # Minimum variance threshold for surfactant descriptors; features with variance below this are dropped.
SURF_CORR_THRESHOLD = 0.95 # Maximum absolute correlation threshold for surfactant descriptors; features with pairwise correlation above this are dropped.
# Surfactant target-correlation relevance is computed on row-level replication
# (all drug-surfactant pairs), not just deduplicated surf rows, to use available
# information when estimating descriptor-target association.
SURF_TARGET_CORR_THRESHOLD = 0.2 # Minimum absolute correlation with the target for surfactant descriptors; features with lower correlation are dropped.

# CV settings.
# Repetition reduces split-to-split variance and stabilizes reported RMSE/MAE/R2.
RANDOM_STATE = 31 # Random seed for reproducibility of CV splits and model training.
EVAL_N_SPLITS = 5 # Number of splits for K-fold cross-validation (both repeated K-fold and nested holdout outer CV).
EVAL_N_REPEATS = 20 # Number of repeats for repeated K-fold cross-validation.
# In grouped outer CV, each drug group is held out once across the folds.
NESTED_HOLDOUT_N_SPLITS = 5 # Number of splits for nested holdout outer CV.

# Output
RESULTS_OUTPUT_DIR = "14_ML_results" # Folder to save CSV output files containing evaluation results, predictions, and summaries.

# Weighting config - used to avoid division by zero in inverse-variance weighting
WEIGHT_EPS = 1e-12 # Small constant added to propagated SE values to prevent division by zero when computing inverse-variance weights.

# Fixed model lists per stream (no tuning)
REPEATED_MODELS = { # Fixed hyperparameters for no holdout repeated K-fold evaluation stream.
    "XGBoost": {
        "model__colsample_bytree": 0.5, # Fraction of features to consider at each split for XGBoost.
        "model__learning_rate": 0.01, # small values for more stable convergence
        "model__max_depth": 5, # deeper trees can capture more complex relationships but may overfit
        "model__min_child_weight": 1, # minimum sum of instance weight (hessian) needed in a child; smaller values allow more splits
        "model__n_estimators": 200, # number of boosting rounds (trees)
        "model__subsample": 0.8, # fraction of samples to use for each boosting round; helps prevent overfitting
    },
    "GaussianProcess": {
        "model__alpha": 0.0001, # small value to add to the diagonal of the kernel matrix for numerical stability
        "model__normalize_y": True, # normalize target values to zero mean and unit variance before fitting
    },
}

LOGO_MODELS = { # Fixed hyperparameters for no holdout leave-one-drug-out (LOGO) evaluation stream.
    "RandomForest": {
        "model__bootstrap": True, # bootstrap samples when building trees means each tree is built on a random sample of the data with replacement
        "model__max_depth": 2, # limit the depth of the trees to prevent overfitting
        "model__max_features": 0.3, # fraction of features to consider when looking for the best split
        "model__min_samples_leaf": 1, # minimum number of samples required to be at a leaf node
        "model__min_samples_split": 4, # minimum number of samples required to split an internal node
        "model__n_estimators": 300, # number of trees in the forest
    },
    "GaussianProcess": {
        "model__alpha": 0.0001,
        "model__normalize_y": True,
    },
}

NESTED_MODELS = { # Fixed hyperparameters for nested holdout evaluation stream (outer GroupKFold only; no inner tuning).
    "GaussianProcess": {
        "model__alpha": 0.0001,
        "model__normalize_y": True,
    },
    "XGBoost": {
        "model__colsample_bytree": 0.5, # fraction of features to consider at each split for XGBoost.
        "model__learning_rate": 0.01, # small values for more stable convergence
        "model__max_depth": 5, # deeper trees can capture more complex relationships but may overfit
        "model__min_child_weight": 1, # minimum sum of instance weight (hessian) needed in a child; smaller values allow more splits
        "model__n_estimators": 200, # number of boosting rounds (trees)
        "model__subsample": 0.6, # fraction of samples to use for each boosting round; helps prevent overfitting
    },
}

print("Configuration complete.")

# to make sure Windows doesn't go to sleep during long runs, which can cause the script to hang or terminate unexpectedly
def _set_windows_sleep_prevention(enable: bool) -> None:
    """Prevent or restore Windows sleep while the script is running."""
    if sys.platform != "win32":
        return

    kernel32 = ctypes.windll.kernel32
    if enable:
        # Keep system and display awake while Python is actively running.
        kernel32.SetThreadExecutionState(0x80000000 | 0x00000001 | 0x00000002)
    else:
        # Restore default Windows power-management behavior.
        kernel32.SetThreadExecutionState(0x80000000)


print("Sleep prevention function defined.")


# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
df = pd.read_csv(DATA_FILE)

# Descriptor columns are identified by prefix. Name/label columns that share
# the same prefixes are metadata and must be excluded from model features.
NON_FEATURE_NAME_COLS = {"drug_NAME", "surf_NAME", "surf_ABB"}
descriptor_cols = [
    c
    for c in df.columns
    if (c.startswith("drug_") or c.startswith("surf_"))
    and c not in NON_FEATURE_NAME_COLS
    and pd.api.types.is_numeric_dtype(df[c])
]
drug_cols = [c for c in descriptor_cols if c.startswith("drug_")]
surf_cols = [c for c in descriptor_cols if c.startswith("surf_")]

# Keep rows where all required inputs for modeling and weighting are present.
# This is a defensive check for future CSVs with missing values.
required_cols = descriptor_cols + [TARGET_COL, GROUP_COL, SE_COL]
if SURF_GROUP_COL in df.columns:
    required_cols.append(SURF_GROUP_COL)

df = df.dropna(subset=required_cols).reset_index(drop=True)

# Target and group labels
y = df[TARGET_COL].to_numpy(dtype=float)
groups = df[GROUP_COL].to_numpy()

# Inverse-variance weights from propagated SE.
# Smaller SE implies larger weight, so more precise observations contribute more.
# N.B SE values are propagated as log-modulus of solpow, so they are already on the same scale as the target, and they are already
# in a way normalised by the target (SEpropagated = SEraw / (1 + absolute solpow raw)) so a large SE on a smaller solpow is 
# more significant than a large SE on a larger solpow. This is why we can use the propagated SE directly for weighting.
se_values = df[SE_COL].to_numpy(dtype=float) # Propagated standard error values for each observation, used to compute weights for weighted regression.
se_values = np.maximum(np.abs(se_values), WEIGHT_EPS) # Ensure no SE values are below WEIGHT_EPS to avoid division by zero in weight calculation.
weights = 1.0 / np.square(se_values) # Compute inverse-variance weights for each observation, which will be used in weighted regression to give more importance to observations with lower uncertainty.

# Keep grouping metadata in X_df because the leakage-safe selector needs them
# when constructing unique-drug and unique-surfactant views per fold.
selector_meta_cols = [GROUP_COL] # Columns used for grouping in the feature selector to avoid leakage. These columns are not used as features but are necessary for the feature selection process.
if SURF_GROUP_COL in df.columns:
    selector_meta_cols.append(SURF_GROUP_COL)
X_df = df[descriptor_cols + selector_meta_cols].copy()

print("Data loaded and preprocessed.")
print(f"Rows used: {len(df)}")
print(f"Weight range: min={weights.min():.6g}, max={weights.max():.6g}")

weight_summary_df = pd.DataFrame( # Create a summary DataFrame containing statistics about the weights and propagated SE values for the dataset. This summary includes the number of rows, the name of the SE column, the WEIGHT_EPS value, and various quantiles and statistics for both SE and weight values.
    [
        {
            "NRows": int(len(df)),
            "SE_Column": SE_COL,
            "WEIGHT_EPS": float(WEIGHT_EPS),
            "SE_min": float(np.min(se_values)),
            "SE_p25": float(np.quantile(se_values, 0.25)),
            "SE_median": float(np.median(se_values)),
            "SE_p75": float(np.quantile(se_values, 0.75)),
            "SE_max": float(np.max(se_values)),
            "Weight_min": float(np.min(weights)),
            "Weight_p25": float(np.quantile(weights, 0.25)),
            "Weight_median": float(np.median(weights)),
            "Weight_mean": float(np.mean(weights)),
            "Weight_p75": float(np.quantile(weights, 0.75)),
            "Weight_p95": float(np.quantile(weights, 0.95)),
            "Weight_p99": float(np.quantile(weights, 0.99)),
            "Weight_max": float(np.max(weights)),
        }
    ]
)


# ---------------------------------------------------------------------------
# 2. Helpers
# ---------------------------------------------------------------------------
# code to build filename-safe text for exported result files.
# used so exported CSV filenames are clean and consistent across operating systems.
def _slugify(text: str) -> str: #defines a helper function named _slugify that takes a string text as input and returns a modified version of the string that is safe to use in filenames.
    """Build filesystem-friendly names for exported result files.""" # explains the function
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in text).strip("_") # does the conversion: keeps letters and 
# digits, converts them to lowercase, replaces non-alphanumeric characters with underscores, and removes leading/trailing underscores.

# define and run CSV export helper
def export_result_table(df_to_save: pd.DataFrame, mode_name: str, table_tag: str) -> Path:
    """Export one result table to CSV and return the output path.

    File naming convention:
    <timestamp>__<mode_name_slug>__<table_tag>.csv
    """
    # Create output directory if needed; safe if it already exists.
    output_dir = Path(RESULTS_OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Timestamp each table to preserve run history.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"{timestamp}__{_slugify(mode_name)}__{table_tag}.csv"
    output_path = output_dir / file_name
    # Export without DataFrame index to keep CSV structure tidy.
    df_to_save.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")
    return output_path


def add_error_percentage_columns(table_df: pd.DataFrame, y_reference: np.ndarray) -> pd.DataFrame:
    """Add RMSE/MAE percentage columns relative to key target spread scales.

    For each numeric RMSE/MAE column, append percentage-normalized companions
    versus:
    - mean absolute target
    - target standard deviation
    - target range
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


# ---------------------------------------------------------------------------
# 3. Leakage-safe feature selector
# ---------------------------------------------------------------------------
class DrugSurfFeatureSelector(BaseEstimator, TransformerMixin):
    """Feature selector applied inside CV folds to avoid leakage.

    Filtering is done separately for drug and surfactant descriptor blocks.
    Drug descriptors vary across unique drugs, and surfactant descriptors vary
    across unique surfactants, so block-wise filtering avoids mixing scales.

    Pipeline order within each block:
    1) variance threshold
    2) target-correlation relevance filter (on if threshold is set to > 0 - currently 0.2)
    3) interfeature correlation deduplication
    """

    def __init__(
        self,
        descriptor_columns: list[str],
        drug_columns: list[str],
        surfactant_columns: list[str],
        group_col: str,
        surf_group_col: str,
        drug_var_threshold: float,
        drug_corr_threshold: float,
        surf_var_threshold: float,
        surf_corr_threshold: float,
        drug_target_corr_threshold: float = 0.0,
        surf_target_corr_threshold: float = 0.0,
    ):
        self.descriptor_columns = descriptor_columns
        self.drug_columns = drug_columns
        self.surfactant_columns = surfactant_columns
        self.group_col = group_col
        self.surf_group_col = surf_group_col
        self.drug_var_threshold = drug_var_threshold
        self.drug_corr_threshold = drug_corr_threshold
        self.surf_var_threshold = surf_var_threshold
        self.surf_corr_threshold = surf_corr_threshold
        self.drug_target_corr_threshold = drug_target_corr_threshold
        self.surf_target_corr_threshold = surf_target_corr_threshold

    @staticmethod
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
        # No descriptors in this block means no candidates to keep.
        if len(block_columns) == 0:
            return []

        # 1) Variance filtering: remove near-constant descriptors.
        var_selector = VarianceThreshold(threshold=var_threshold)
        var_selector.fit(block_df[block_columns].to_numpy())
        retained = list(np.array(block_columns)[var_selector.get_support()])

        # If 0 or 1 descriptors remain, interfeature correlation filtering is moot.
        if len(retained) <= 1:
            return retained

        # 2) Correlation filtering: remove one feature from highly correlated pairs.
        # Use absolute correlation and upper triangle to avoid duplicate pair checks
        # (A-B and B-A carry the same information in symmetric correlation matrices).
        with np.errstate(divide="ignore", invalid="ignore"):
            corr = block_df[retained].corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))

        if target_column is not None and target_column in block_df.columns:
            # Target-correlation can be computed on an alternate dataframe.
            # This is useful for surfactant relevance filtering, where row-level
            # replication provides a more stable target-correlation estimate.
            tc_df = target_corr_block_df if target_corr_block_df is not None else block_df
            tc_target_column = target_corr_column if target_corr_column is not None else target_column
            target_series = tc_df[tc_target_column].astype(float)
            target_std = float(target_series.std(ddof=0))

            if np.isfinite(target_std) and target_std > 0.0:
                with np.errstate(divide="ignore", invalid="ignore"):
                    target_corr = tc_df[retained].corrwith(target_series).abs()
                target_corr = target_corr.fillna(0.0)
            else:
                # Zero target variance in a fold makes correlation undefined.
                target_corr = pd.Series(0.0, index=retained)

            # Optional relevance filter before interfeature deduplication.
            # This prevents keeping redundant features that are weakly related
            # to the target when stronger alternatives exist.
            if target_corr_threshold > 0.0 and np.isfinite(target_std) and target_std > 0.0:
                retained = [col for col in retained if target_corr.get(col, 0.0) >= target_corr_threshold]
                target_corr = target_corr.loc[retained]
                if len(retained) <= 1:
                    return retained
                corr = corr.loc[retained, retained]

            order_lookup = {col: idx for idx, col in enumerate(retained)}
            ordered = sorted(retained, key=lambda col: (-target_corr.get(col, 0.0), order_lookup[col]))

            kept = []
            for col in ordered:
                if not any(corr.loc[col, kept_col] > corr_threshold for kept_col in kept):
                    kept.append(col)
            return kept

        # Fallback if target correlation is unavailable: order-based deduplication.
        to_drop = [col for col in upper.columns if any(upper[col] > corr_threshold)]
        return [c for c in retained if c not in to_drop]

    def fit(self, X, y=None):
        # DataFrame input is required because the selector is column-name aware.
        if not isinstance(X, pd.DataFrame):
            raise TypeError("DrugSurfFeatureSelector expects a pandas DataFrame as input.")

        required = self.descriptor_columns + [self.group_col]
        if self.surf_group_col in X.columns:
            required.append(self.surf_group_col)

        missing = [c for c in required if c not in X.columns]
        if missing:
            raise ValueError(f"Missing required columns for feature selection: {missing}")

        if y is None:
            raise ValueError("DrugSurfFeatureSelector requires y when feature reduction is enabled.")

        # Attach fold-local target values for leakage-safe relevance calculations.
        training_df = X.copy()
        training_df["_target"] = np.asarray(y)

        # Drug block filtering is based on unique-drug rows within this fold.
        unique_drug_df = training_df.drop_duplicates(subset=self.group_col).copy()
        drug_target = training_df.groupby(self.group_col, as_index=False)["_target"].mean()
        unique_drug_df = unique_drug_df.merge(drug_target, on=self.group_col, how="left", suffixes=("", "_target"))

        retained_drug = self._filter_by_variance_and_correlation(
            block_df=unique_drug_df,
            block_columns=self.drug_columns,
            var_threshold=self.drug_var_threshold,
            corr_threshold=self.drug_corr_threshold,
            target_column="_target_target",
            target_corr_threshold=self.drug_target_corr_threshold,
        )

        if self.surf_group_col in X.columns:
            # Preferred surfactant deduplication path when explicit surf labels exist.
            unique_surf_df = training_df.drop_duplicates(subset=self.surf_group_col).copy()
            surf_target = training_df.groupby(self.surf_group_col, as_index=False)["_target"].mean()
            unique_surf_df = unique_surf_df.merge(
                surf_target,
                on=self.surf_group_col,
                how="left",
                suffixes=("", "_target"),
            )
            surf_target_column = "_target_target"
        else:
            # Fallback if surf labels are unavailable: deduplicate by descriptor pattern.
            unique_surf_df = training_df.drop_duplicates(subset=self.surfactant_columns).copy()
            surf_target = training_df.groupby(self.surfactant_columns, as_index=False)["_target"].mean()
            unique_surf_df = unique_surf_df.merge(
                surf_target,
                on=self.surfactant_columns,
                how="left",
                suffixes=("", "_target"),
            )
            surf_target_column = "_target_target"

        # Surfactant interfeature filtering uses deduplicated surf rows, while
        # target-correlation relevance is computed on row-level fold data to use
        # all observed replication across drug-surfactant pairs.
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

        self.selected_feature_names_ = retained_drug + retained_surf
        return self

    def transform(self, X):
        # Enforce fit-before-transform and DataFrame input for safe column lookup.
        if not hasattr(self, "selected_feature_names_"):
            raise RuntimeError("Call fit before transform.")
        if not isinstance(X, pd.DataFrame):
            raise TypeError("DrugSurfFeatureSelector expects a pandas DataFrame as input.")
        return X[self.selected_feature_names_].to_numpy()


print("Leakage-safe feature selector defined.")


# ---------------------------------------------------------------------------
# 4. Model/pipeline utilities
# ---------------------------------------------------------------------------
def make_model(model_name: str):
    """Instantiate base estimator for a given model name."""
    if model_name == "GaussianProcess":
        # Constant * RBF models smooth nonlinear signal; WhiteKernel absorbs noise.
        kernel = ConstantKernel(1.0) * RBF(length_scale=1.0) + WhiteKernel(noise_level=1.0)
        return GaussianProcessRegressor(kernel=kernel, random_state=RANDOM_STATE)

    if model_name == "RandomForest":
        return RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)

    if model_name == "XGBoost":
        if not HAS_XGBOOST:
            raise RuntimeError(
                "XGBoost requested but not available in this environment. Install xgboost first."
            )
        # objective="reg:squarederror" is the standard regression objective.
        return XGBRegressor(
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0,
        )

    raise ValueError(f"Unsupported model name: {model_name}")


def needs_scaling(model_name: str) -> bool:
    """Return True if model should receive standardized features."""
    # Keep scaling decision explicit and model-dependent.
    return model_name in {"GaussianProcess"}


def make_pipeline_for_model(model_name: str, fixed_params: dict) -> Pipeline:
    """Build leakage-safe pipeline and set fixed model hyperparameters."""
    # Feature selection is part of the pipeline so it is re-fit inside each
    # training fold, preventing information leakage.
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
    if needs_scaling(model_name):
        # Scale only when model assumptions benefit from standardized inputs.
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", make_model(model_name)))

    pipe = Pipeline(steps)
    pipe.set_params(**fixed_params)
    return pipe


def model_accepts_sample_weight(fitted_or_unfitted_pipeline: Pipeline) -> bool:
    """Check whether underlying estimator fit method accepts sample_weight."""
    model = fitted_or_unfitted_pipeline.named_steps["model"]
    return "sample_weight" in inspect.signature(model.fit).parameters


def fit_pipeline(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    w_train: np.ndarray,
    se_train: np.ndarray,
    model_name: str,
    warning_once: set[str],
) -> Pipeline:
    """Fit with sample_weight when supported; otherwise use model-specific SE-aware fallback."""
    if model_accepts_sample_weight(pipeline):
        # Direct weighted fitting path for estimators that expose sample_weight.
        pipeline.fit(X_train, y_train, model__sample_weight=w_train)
    else:
        if model_name == "GaussianProcess":
            # GP has no sample_weight argument; emulate SE-aware weighting using heteroscedastic alpha. Heteroscedastic means that the noise level can vary across observations, which is useful when we have different levels of uncertainty (SE) for each observation. The alpha parameter in GaussianProcessRegressor controls the noise level, and by setting it to a vector derived from the SE values, we can effectively weight the observations according to their uncertainty.
            base_alpha = float(pipeline.get_params().get("model__alpha", 0.0))
            alpha_vector = np.square(np.maximum(np.abs(se_train), WEIGHT_EPS)) + base_alpha
            pipeline.set_params(model__alpha=alpha_vector)
            pipeline.fit(X_train, y_train)
        else:
            if model_name not in warning_once:
                print(
                    f"Warning: {model_name} does not support sample_weight in sklearn API; fitting unweighted."
                )
                warning_once.add(model_name)
            pipeline.fit(X_train, y_train)
    return pipeline


print("Model and pipeline utilities defined.")


# ---------------------------------------------------------------------------
# 5. Stream evaluators (fixed hyperparameters, no tuning)
# ---------------------------------------------------------------------------
def evaluate_stream(
    stream_name: str,
    model_param_map: dict[str, dict],
    split_tuples: list[tuple[np.ndarray, np.ndarray]],
    split_label: str,
    include_repeat_fold: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Evaluate one stream and return summary, split-level scores, and OOF predictions."""
    # summary_rows: averaged per-model performance for the stream
    # split_rows: one record per fold/split
    # oof_rows: out-of-fold predictions for later diagnostics/parity plots
    summary_rows = []
    split_rows = []
    oof_rows = []
    warning_once: set[str] = set()

    for model_name, fixed_params in model_param_map.items():
        # Accumulate split-level metrics, then summarize by mean and standard deviation.
        rmse_list = []
        mae_list = []
        r2_list = []

        for split_idx, (train_idx, test_idx) in enumerate(split_tuples, start=1):
            estimator = make_pipeline_for_model(model_name, fixed_params)

            X_train = X_df.iloc[train_idx]
            X_test = X_df.iloc[test_idx]
            y_train = y[train_idx]
            y_test = y[test_idx]
            w_train = weights[train_idx]
            se_train = se_values[train_idx]

            estimator = fit_pipeline(
                estimator,
                X_train,
                y_train,
                w_train,
                se_train,
                model_name,
                warning_once,
            )
            y_pred = np.ravel(estimator.predict(X_test))

            # Metrics are computed on held-out rows only for this split.
            rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
            mae = float(mean_absolute_error(y_test, y_pred))
            r2 = float(r2_score(y_test, y_pred))

            rmse_list.append(rmse)
            mae_list.append(mae)
            r2_list.append(r2)

            held_out_drugs = np.unique(groups[test_idx])
            split_row = {
                "Stream": stream_name,
                "Model": model_name,
                split_label: split_idx,
                "TestRows": int(len(test_idx)),
                "HeldOutDrugs": " | ".join(map(str, held_out_drugs)),
                "RMSE": rmse,
                "MAE": mae,
                "R2": r2,
            }

            if include_repeat_fold:
                # Decode repeated-kfold split index into repeat/fold components.
                split_row["Repeat"] = ((split_idx - 1) // EVAL_N_SPLITS) + 1
                split_row["Fold"] = ((split_idx - 1) % EVAL_N_SPLITS) + 1

            split_rows.append(split_row)

            if SURF_GROUP_COL in X_test.columns:
                surf_test = X_test[SURF_GROUP_COL].astype(str).to_numpy()
            else:
                # Keep schema stable if surf labels are unavailable in X_test.
                surf_test = np.array(["NA"] * len(test_idx), dtype=object)

            for local_idx, row_idx in enumerate(test_idx):
                oof_rows.append(
                    {
                        "Stream": stream_name,
                        "Model": model_name,
                        split_label: split_idx,
                        "RowIndex": int(row_idx),
                        "Drug": str(groups[test_idx][local_idx]),
                        "Surfactant": str(surf_test[local_idx]),
                        "Observed": float(y_test[local_idx]),
                        "Predicted": float(y_pred[local_idx]),
                    }
                )

        summary_rows.append(
            {
                "Stream": stream_name,
                "Model": model_name,
                "RMSE": float(np.mean(rmse_list)),
                "RMSE_std": float(np.std(rmse_list)),
                "MAE": float(np.mean(mae_list)),
                "MAE_std": float(np.std(mae_list)),
                "R2": float(np.mean(r2_list)),
                "R2_std": float(np.std(r2_list)),
                "N_Splits": int(len(split_tuples)),
            }
        )

    summary_df = pd.DataFrame(summary_rows).sort_values("RMSE")
    split_df = pd.DataFrame(split_rows).sort_values(["Model", split_label])
    oof_df = pd.DataFrame(oof_rows).sort_values(["Model", split_label, "RowIndex"])

    # Add percentage-normalized error columns to improve interpretability.
    summary_df = add_error_percentage_columns(summary_df, y)
    split_df = add_error_percentage_columns(split_df, y)

    return summary_df, split_df, oof_df


print("Stream evaluators defined.")


# ---------------------------------------------------------------------------
# 6. Run all requested streams
# ---------------------------------------------------------------------------
def run_mode(mode_name: str) -> None:
    if ("XGBoost" in REPEATED_MODELS or "XGBoost" in NESTED_MODELS) and not HAS_XGBOOST:
        raise RuntimeError("XGBoost is required by this script but is not installed.")

    # Repeated K-fold stream (no holdout): repeated random 5-fold splits for
    # stable average performance estimates.
    repeated_cv = RepeatedKFold(
        n_splits=EVAL_N_SPLITS,
        n_repeats=EVAL_N_REPEATS,
        random_state=RANDOM_STATE,
    )
    repeated_splits = list(repeated_cv.split(X_df, y))

    repeated_summary, repeated_splits_df, repeated_oof_df = evaluate_stream(
        stream_name="NoHoldout_RepeatedKFold",
        model_param_map=REPEATED_MODELS,
        split_tuples=repeated_splits,
        split_label="Split",
        include_repeat_fold=True,
    )

    # LOGO stream (no holdout): each split leaves one drug group out.
    logo_cv = LeaveOneGroupOut()
    logo_splits = list(logo_cv.split(X_df, y, groups))

    logo_summary, logo_splits_df, logo_oof_df = evaluate_stream(
        stream_name="NoHoldout_LOGO",
        model_param_map=LOGO_MODELS,
        split_tuples=logo_splits,
        split_label="Split",
        include_repeat_fold=False,
    )

    # Nested holdout stream (outer GroupKFold only, fixed hyperparameters).
    # Outer grouping ensures held-out drugs are unseen during training.
    # No inner tuning is performed in this condensed fixed-parameter workflow.
    unique_groups = np.unique(groups)
    if len(unique_groups) < NESTED_HOLDOUT_N_SPLITS:
        raise ValueError(
            f"Nested holdout requested {NESTED_HOLDOUT_N_SPLITS} splits, "
            f"but only {len(unique_groups)} unique drug groups are available."
        )

    nested_cv = GroupKFold(n_splits=NESTED_HOLDOUT_N_SPLITS)
    nested_splits = list(nested_cv.split(X_df, y, groups))

    nested_summary, nested_splits_df, nested_oof_df = evaluate_stream(
        stream_name="NestedHoldout",
        model_param_map=NESTED_MODELS,
        split_tuples=nested_splits,
        split_label="Repeat",
        include_repeat_fold=False,
    )

    # Comparison table across streams for side-by-side review.
    comparison_df = pd.concat(
        [
            repeated_summary.assign(StreamOrder=1),
            logo_summary.assign(StreamOrder=2),
            nested_summary.assign(StreamOrder=3),
        ],
        ignore_index=True,
    ).sort_values(["StreamOrder", "RMSE"]).drop(columns=["StreamOrder"])

    # Fixed-parameter audit export to document exactly what was run.
    fixed_param_rows = []
    for stream_name, config in [
        ("NoHoldout_RepeatedKFold", REPEATED_MODELS),
        ("NoHoldout_LOGO", LOGO_MODELS),
        ("NestedHoldout", NESTED_MODELS),
    ]:
        for model_name, params in config.items():
            fixed_param_rows.append(
                {
                    "Stream": stream_name,
                    "Model": model_name,
                    "FixedParams": str(params),
                }
            )
    fixed_params_df = pd.DataFrame(fixed_param_rows)

    # Export one CSV per table for traceability and downstream plotting/reporting.
    export_result_table(weight_summary_df, mode_name, "weight_summary")
    export_result_table(fixed_params_df, mode_name, "fixed_model_hyperparameters")

    export_result_table(repeated_summary, mode_name, "repeatedkfold_summary")
    export_result_table(repeated_splits_df, mode_name, "repeatedkfold_per_split")
    export_result_table(repeated_oof_df, mode_name, "repeatedkfold_oof_predictions")

    export_result_table(logo_summary, mode_name, "logo_summary")
    export_result_table(logo_splits_df, mode_name, "logo_per_split")
    export_result_table(logo_oof_df, mode_name, "logo_oof_predictions")

    export_result_table(nested_summary, mode_name, "nestedholdout_summary")
    export_result_table(nested_splits_df, mode_name, "nestedholdout_per_split")
    export_result_table(nested_oof_df, mode_name, "nestedholdout_oof_predictions")

    export_result_table(comparison_df, mode_name, "comparison_all_streams")

    print("\nComparison across streams:")
    print(comparison_df.round(4).to_string(index=False))


if __name__ == "__main__":
    _set_windows_sleep_prevention(True)
    try:
        run_mode("Weighted Fixed Models")
    finally:
        _set_windows_sleep_prevention(False)

print("\nWorkflow completed.")
