"""
Plot RMSE (with SD error bars) for each ML model, read from ML_1.xlsx.

Two side-by-side panels separate feature reduction (a: without, b: with).
Within each panel, the four evaluation strategies are the x-axis groups,
and each model is a coloured point (dodged within the group) so it can be
tracked visually across both panels via its pastel colour.
"""

import colorsys
import itertools

import numpy as np
import pandas as pd
import scikit_posthocs as sp
from scipy.stats import friedmanchisquare, wilcoxon
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------------------------------------------------------------------
# 1. Load the raw sheet with no header, since it uses a custom layout with
#    two side-by-side tables ("Without"/"With feature reduction"), each
#    containing four stacked sections (one per CV scheme).
# ---------------------------------------------------------------------------
DATA_FILE = Path(__file__).parent / "ML_1.xlsx"
raw = pd.read_excel(DATA_FILE, sheet_name="thesis", header=None)


def parse_value(cell):
    """Split a 'value ± sd' cell into (value, sd); holdout cells have no SD."""
    if pd.isna(cell):
        return None, None
    if isinstance(cell, str) and "±" in cell:
        value_str, sd_str = cell.split("±")
        return float(value_str.strip()), float(sd_str.strip())
    return float(cell), np.nan  # single holdout has no SD (only one split, no replicates)


def parse_block(df, col_start, fr_label, ncols=4):
    """Parse one feature-reduction block (4 columns) into tidy RMSE records.

    Each block repeats: a section-title row (e.g. 'Standard LOGO CV'), a
    header row ('Model', 'RMSE ± SD', ...), then one data row per model,
    separated by blank rows.
    """
    # the spreadsheet has an inconsistent capitalisation for this one model
    model_name_fixes = {"Partial least squares": "Partial Least Squares"}
    sub = df.iloc[:, col_start:col_start + ncols]
    records = []
    section = None
    for i in range(len(sub)):
        model_cell = sub.iat[i, 0]
        rmse_cell = sub.iat[i, 1]
        if pd.isna(model_cell) or i == 0:  # blank row, or the block title row
            continue
        if pd.isna(rmse_cell):  # section-title row, e.g. 'Standard LOGO CV'
            section = str(model_cell).strip()
            continue
        if str(model_cell).strip().lower() == "model":  # header row
            continue
        rmse, sd = parse_value(rmse_cell)
        model_name = str(model_cell).strip()
        model_name = model_name_fixes.get(model_name, model_name)
        records.append({
            "FeatureReduction": fr_label,
            "CVScheme": section,
            "Model": model_name,
            "RMSE": rmse,
            "RMSE_SD": sd,
        })
    return records


records = (
    parse_block(raw, 0, "No feature reduction")
    + parse_block(raw, 5, "With feature reduction")
)
data = pd.DataFrame(records)

# ---------------------------------------------------------------------------
# 2. Fix a consistent order for models (-> colours) and evaluation strategies
#    (-> x-axis groups); "Condition" is no longer needed since feature
#    reduction now splits the data into two separate panels instead.
# ---------------------------------------------------------------------------
model_order = ["Gaussian Process", "XGBoost", "Gradient Boosting", "Random Forest",
               "Ridge", "Support Vector Regression", "Lasso", "Partial Least Squares"]
cv_order = ["Standard repeated CV", "Standard LOGO CV", "Single holdout", "Nested holdout"]
cv_display = {  # shorter labels for the x-axis ticks
    "Standard repeated CV": "Repeated CV",
    "Standard LOGO CV": "LOGO CV",
    "Single holdout": "Single holdout",
    "Nested holdout": "Nested holdout",
}
fr_order = ["No feature reduction", "With feature reduction"]
panel_titles = {
    "No feature reduction": "(a) Without feature reduction",
    "With feature reduction": "(b) With feature reduction",
}

data["Model"] = pd.Categorical(data["Model"], categories=model_order, ordered=True)

# ---------------------------------------------------------------------------
# 2b. Statistical significance: Friedman test + Nemenyi post-hoc, computed
#     separately for each (feature reduction x CV scheme) combination where
#     the same folds/repeats are reused identically across all 8 models
#     (verified beforehand) - this matched/blocked structure is required for
#     a valid Friedman test. "Single holdout" has no repeated folds, so it
#     is skipped (no significance can be computed for it).
# ---------------------------------------------------------------------------
RAW_DIR = Path(r"C:\Users\anton\Desktop\PhD\data\ML\07_ML1_results_firstgo")
# each entry maps (feature reduction, CV scheme) -> (raw per-fold CSV, block/fold id column)
raw_fold_files = {
    ("No feature reduction", "Standard repeated CV"): (RAW_DIR / "20260722_012352__without_feature_reduction__standard_repeated_cv_per_split.csv", "Split"),
    ("No feature reduction", "Standard LOGO CV"): (RAW_DIR / "20260722_012352__without_feature_reduction__standard_logo_cv_per_split.csv", "Split"),
    ("No feature reduction", "Nested holdout"): (RAW_DIR / "20260722_035057__without_feature_reduction__nested_holdout_per_repeat.csv", "Repeat"),
    ("With feature reduction", "Standard repeated CV"): (RAW_DIR / "20260722_063411__with_feature_reduction__standard_repeated_cv_per_split.csv", "Split"),
    ("With feature reduction", "Standard LOGO CV"): (RAW_DIR / "20260722_063411__with_feature_reduction__standard_logo_cv_per_split.csv", "Split"),
    ("With feature reduction", "Nested holdout"): (RAW_DIR / "20260722_093551__with_feature_reduction__nested_holdout_per_repeat.csv", "Repeat"),
}
# the raw CSVs use model names without spaces; map them onto the display names used above
raw_model_name_map = {
    "GaussianProcess": "Gaussian Process",
    "XGBoost": "XGBoost",
    "GradientBoosting": "Gradient Boosting",
    "RandomForest": "Random Forest",
    "Ridge": "Ridge",
    "SVR": "Support Vector Regression",
    "Lasso": "Lasso",
    "PLS": "Partial Least Squares",
}


def compact_letter_display(avg_ranks, pvals, alpha=0.05):
    """Assign compact-letter-display groups from ALL maximal cliques of the
    'not significantly different' graph (the same method used by R's
    multcompView::multcompLetters). This guarantees both directions of
    correctness: two models sharing a letter are never significantly
    different, AND two models that are NOT significantly different always
    share at least one letter. A simpler single-pass greedy grouping (tried
    first, then discarded) fails the second guarantee whenever compatibility
    isn't transitive - e.g. it can place two models with p=0.74 (clearly not
    significantly different) into disjoint letter groups just because of the
    order models happened to be processed in.
    """
    models = list(avg_ranks.keys())

    def compatible(a, b):
        return a == b or pvals.loc[a, b] > alpha

    # brute-force maximal-clique search (only 8 models per test here, so the
    # 2**8 = 256 subsets checked is trivial); scanning largest subsets first
    # and skipping any subset already covered by a found clique guarantees
    # only genuinely maximal cliques are kept.
    cliques = []
    for size in range(len(models), 0, -1):
        for combo in itertools.combinations(models, size):
            if all(compatible(a, b) for a, b in itertools.combinations(combo, 2)):
                if not any(set(combo) <= c for c in cliques):
                    cliques.append(set(combo))

    cliques.sort(key=lambda c: min(avg_ranks[m] for m in c))  # best (lowest rank) clique labelled 'a' first
    letters = {m: "" for m in models}
    for letter_idx, clique in enumerate(cliques):
        letter = chr(ord("a") + letter_idx)
        for m in clique:
            letters[m] += letter
    return letters


def merged_significance_lines(models_in_x_order, pvals, alpha=0.05):
    """Represent all significant pairs (in fixed x-position order) as the
    fewest possible lines, each with tick marks at every model it flags,
    via two merge passes:
      1. Row-merge: for each anchor model, any contiguous run of models to
         its right that are all significantly different from it becomes one
         line (instead of one bracket per pair).
      2. Column-merge: any row-merge lines left as a single-model reach (no
         further merging possible in pass 1) often share the same lone
         target (e.g. several models all differing from one outlier) - a
         contiguous run of such anchors sharing the same target is merged
         into one line spanning that shared target, instead of one line per
         anchor.
    Each line carries a per-tick Nemenyi p-value (None at the line's own
    reference position - the anchor for a row-merge line, the shared target
    for a column-merge line - since that position has no p-value of its own,
    only the ticks it is compared against do), so the chart can show one
    significance rating per pair even when several pairs share one line.
    Returns a list of (left_idx, right_idx, tick_indices, tick_pvalues) tuples.
    """
    n = len(models_in_x_order)

    def sig(i, j):
        return pvals.loc[models_in_x_order[i], models_in_x_order[j]] <= alpha

    def p_of(i, j):
        return pvals.loc[models_in_x_order[i], models_in_x_order[j]]

    # pass 1: row-merge (contiguous run of significant models to the right of each anchor)
    row_lines = []  # (anchor_idx, run_start_idx, run_end_idx)
    for i in range(n):
        run = []
        for j in range(i + 1, n):
            if sig(i, j):
                run.append(j)
            elif run:
                row_lines.append((i, run[0], run[-1]))
                run = []
        if run:
            row_lines.append((i, run[0], run[-1]))

    # lines that already reach a wide contiguous range are kept as-is; only
    # single-target lines are candidates for pass 2 (column-merge)
    wide_lines = [(a, rs, re, [a] + list(range(rs, re + 1))) for a, rs, re in row_lines if rs != re]
    singleton_lines = [(a, re) for a, rs, re in row_lines if rs == re]

    # pass 2: column-merge (contiguous run of anchors sharing the same lone target)
    by_target = {}
    for anchor, target in singleton_lines:
        by_target.setdefault(target, []).append(anchor)
    merged_singletons = []
    for target, anchors in by_target.items():
        anchors.sort()
        run = []
        for a in anchors:
            if run and a == run[-1] + 1:
                run.append(a)
            else:
                if run:
                    merged_singletons.append((run[0], run[-1], target, run + [target]))
                run = [a]
        if run:
            merged_singletons.append((run[0], run[-1], target, run + [target]))

    lines = [
        (a, re, ticks, [None] + [p_of(a, j) for j in range(rs, re + 1)])
        for a, rs, re, ticks in wide_lines
    ]
    lines += [
        (rs, target, ticks, [p_of(a, target) for a in range(rs, re + 1)] + [None])
        for rs, re, target, ticks in merged_singletons
    ]
    return lines


sig_letters = {}  # (FeatureReduction, CVScheme, Model) -> compact letter display string, kept for the CSV export
sig_lines = {}  # (FeatureReduction, CVScheme) -> list of (left_idx, right_idx, tick_indices, tick_pvalues), used for the chart
nemenyi_pvals = {}  # (FeatureReduction, CVScheme) -> full pairwise Nemenyi p-value DataFrame, reused for the CSV export below
friedman_results = {}  # (FeatureReduction, CVScheme) -> {chi2, p, n_folds}, reused for the CSV export below
wide_by_condition = {}  # (FeatureReduction, CVScheme) -> wide block x model RMSE frame, reused in section 2c below
for (fr_label, cv_scheme), (csv_path, block_col) in raw_fold_files.items():
    raw_folds = pd.read_csv(csv_path)
    raw_folds["Model"] = raw_folds["Model"].map(raw_model_name_map)
    wide = raw_folds.pivot(index=block_col, columns="Model", values="RMSE")
    wide_by_condition[(fr_label, cv_scheme)] = wide
    chi2, p_omnibus = friedmanchisquare(*[wide[c] for c in wide.columns])
    print(f"Friedman test - {fr_label} / {cv_scheme}: chi2={chi2:.2f}, p={p_omnibus:.4g}, n_folds={len(wide)}")
    friedman_results[(fr_label, cv_scheme)] = {"chi2": chi2, "p": p_omnibus, "n_folds": len(wide)}
    avg_ranks = wide.rank(axis=1).mean().to_dict()  # rank 1 = lowest (best) RMSE in that fold
    pvals = sp.posthoc_nemenyi_friedman(wide)
    nemenyi_pvals[(fr_label, cv_scheme)] = pvals
    for model, letter in compact_letter_display(avg_ranks, pvals).items():
        sig_letters[(fr_label, cv_scheme, model)] = letter
    sig_lines[(fr_label, cv_scheme)] = merged_significance_lines(model_order, pvals)

# ---------------------------------------------------------------------------
# 2c. Statistical significance: does feature reduction itself change a given
#     model's RMSE, within a given CV scheme? Verified beforehand that the
#     with/without feature-reduction runs reuse identical folds (matching
#     TestRows counts for repeated CV, matching HeldOutDrugs identity for
#     LOGO CV/nested holdout), so each model's two RMSE series are paired -
#     a Wilcoxon signed-rank test on that pairing is the non-parametric
#     equivalent of a paired t-test. The 8 per-CV-scheme p-values are then
#     Holm-Bonferroni corrected, since 8 tests are run within each scheme.
# ---------------------------------------------------------------------------
def holm_bonferroni(pvals, alpha=0.05):
    # step-down correction: sort p-values ascending, reject while p <= alpha / (n - rank);
    # once one comparison fails, it and all remaining (larger p) are treated as non-significant
    ordered = sorted(pvals, key=lambda k: pvals[k])
    n = len(ordered)
    significant = {}
    still_significant = True
    for rank, key in enumerate(ordered):
        if still_significant and pvals[key] <= alpha / (n - rank):
            significant[key] = True
        else:
            still_significant = False
            significant[key] = False
    return significant


fr_effect_sig = {}  # (CVScheme, Model) -> True/False, whether feature reduction significantly changed RMSE
fr_effect_pval = {}  # (CVScheme, Model) -> raw Wilcoxon p-value, reused for the CSV export below
for cv_scheme in ["Standard repeated CV", "Standard LOGO CV", "Nested holdout"]:
    wide_no_fr = wide_by_condition[("No feature reduction", cv_scheme)]
    wide_with_fr = wide_by_condition[("With feature reduction", cv_scheme)]
    pvals = {}
    for model in model_order:
        x = wide_no_fr[model]
        y = wide_with_fr[model].reindex(x.index)  # align by fold/split id, same numbering in both runs
        _, p = wilcoxon(x, y)
        pvals[model] = p
    significant = holm_bonferroni(pvals)
    for model in model_order:
        fr_effect_sig[(cv_scheme, model)] = significant[model]
        fr_effect_pval[(cv_scheme, model)] = pvals[model]
        flag = " (Holm-sig)" if significant[model] else ""
        print(f"Wilcoxon FR-effect - {cv_scheme} / {model}: p={pvals[model]:.4g}{flag}")

# ---------------------------------------------------------------------------
# 2d. Export all statistical results (one row per Model x FeatureReduction x
#     CVScheme) to a CSV, alongside the summary RMSE data, for use in the
#     thesis text/appendix without needing to re-run this script.
# ---------------------------------------------------------------------------
export_rows = []
for row in data.itertuples(index=False):
    friedman = friedman_results.get((row.FeatureReduction, row.CVScheme), {})
    export_rows.append({
        "FeatureReduction": row.FeatureReduction,
        "CVScheme": row.CVScheme,
        "Model": row.Model,
        "RMSE": row.RMSE,
        "RMSE_SD": row.RMSE_SD,
        "Friedman_Chi2": friedman.get("chi2"),
        "Friedman_P": friedman.get("p"),
        "Friedman_NFolds": friedman.get("n_folds"),
        "Nemenyi_Letter": sig_letters.get((row.FeatureReduction, row.CVScheme, row.Model)),
        "Wilcoxon_FREffect_P": fr_effect_pval.get((row.CVScheme, row.Model)),
        "Wilcoxon_FREffect_HolmSignificant": fr_effect_sig.get((row.CVScheme, row.Model)),
    })
export_df = pd.DataFrame(export_rows)
export_df.to_csv(Path(__file__).parent / "RMSE_ML_1_statistics.csv", index=False)

# Nemenyi post-hoc is pairwise, so its 28 p-values per condition don't fit the
# one-row-per-model export above - written out separately in long format instead.
pairwise_rows = []
for (fr_label, cv_scheme), pvals in nemenyi_pvals.items():
    for i, model_a in enumerate(model_order):
        for model_b in model_order[i + 1:]:
            p = pvals.loc[model_a, model_b]
            pairwise_rows.append({
                "FeatureReduction": fr_label,
                "CVScheme": cv_scheme,
                "Model_A": model_a,
                "Model_B": model_b,
                "Nemenyi_P": p,
                "Significant": p <= 0.05,
            })
pairwise_df = pd.DataFrame(pairwise_rows)
pairwise_df.to_csv(Path(__file__).parent / "RMSE_ML_1_pairwise_pvalues.csv", index=False)

def boost_saturation(rgb, factor=1.8, value_scale=0.95):
    """Make a pastel colour more visible by boosting its saturation, keeping the hue."""
    h, s, v = colorsys.rgb_to_hsv(*rgb)
    return colorsys.hsv_to_rgb(h, min(s * factor, 1.0), v * value_scale)


# one consistent pastel colour per model, reused across every panel
pastel_colors = plt.get_cmap("Pastel2").colors
vivid_pastel_colors = [boost_saturation(c) for c in pastel_colors]
model_colors = dict(zip(model_order, vivid_pastel_colors))
model_markers = {
    "Gaussian Process": "o",
    "XGBoost": "s",
    "Gradient Boosting": "^",
    "Random Forest": "D",
    "Ridge": "v",
    "Support Vector Regression": "<",
    "Lasso": ">",
    "Partial Least Squares": "p",
}
# matplotlib normalises each marker shape to a different visual area for the
# same markersize (e.g. triangles/diamonds render smaller than circles/
# squares), so sizes are tuned per-shape here to look visually consistent.
model_markersizes = {
    "Gaussian Process": 16,  # o
    "XGBoost": 15,  # s
    "Gradient Boosting": 17,  # ^
    "Random Forest": 15,  # D
    "Ridge": 17,  # v
    "Support Vector Regression": 17,  # <
    "Lasso": 17,  # >
    "Partial Least Squares": 18,  # p
}

# ---------------------------------------------------------------------------
# 3. Plot: two panels (feature reduction), each with the four evaluation
#    strategies as x-axis groups and one dodged point per model.
# ---------------------------------------------------------------------------
n_models = len(model_order)
dodge = np.linspace(-0.7, 0.7, n_models)  # spread the 8 models within each group
x_base = np.arange(len(cv_order)) * 2.2  # extra spacing so wider group labels don't collide
min_rmse = data["RMSE"].min()  # single lowest RMSE across both panels
ylim_top = 2000  # RMSE axis is clipped here; letters are kept below this so they stay visible

fig, axes = plt.subplots(1, 2, figsize=(20, 8), sharey=True)

for ax, fr_label in zip(axes, fr_order):
    panel_data = data[data["FeatureReduction"] == fr_label]
    rmse_by_model = {}  # model -> array over cv_order, reused below to draw the significance lines per CV scheme
    sd_by_model = {}
    for m_idx, model in enumerate(model_order):
        subset = (
            panel_data[panel_data["Model"] == model]
            .set_index("CVScheme")
            .reindex(cv_order)
        )
        rmse = subset["RMSE"].to_numpy()
        sd = subset["RMSE_SD"].to_numpy()
        rmse_by_model[model] = rmse
        sd_by_model[model] = sd
        lower_err = np.minimum(sd, rmse)  # clip the lower whisker at zero
        ax.errorbar(
            x_base + dodge[m_idx], rmse, yerr=np.vstack([lower_err, sd]),
            marker=model_markers[model], linestyle="none", markersize=model_markersizes[model], color=model_colors[model],
            markeredgecolor="grey", markeredgewidth=0.5,
            ecolor=model_colors[model], elinewidth=1.2, capsize=3,
            label=model,
        )
        # Wilcoxon signed-rank marker: flags models whose RMSE changed
        # significantly between the with/without feature-reduction runs
        # (same marker shown in both panels, since it compares across them).
        # Star count follows the same p-value thresholds as the Nemenyi ticks.
        # Offset alternates by model so dodged neighbours' stars don't overlap.
        star_y_offset = 9 if m_idx % 2 == 0 else 22
        for cv_scheme, x_pos, y_val in zip(cv_order, x_base + dodge[m_idx], rmse):
            if fr_effect_sig.get((cv_scheme, model)) and not np.isnan(y_val):
                p_value = fr_effect_pval[(cv_scheme, model)]
                if p_value < 0.001:
                    stars = "***"
                elif p_value < 0.01:
                    stars = "**"
                else:
                    stars = "*"
                ax.annotate(
                    stars, xy=(x_pos, y_val), xytext=(0, star_y_offset), textcoords="offset points",
                    fontsize=15, fontweight="bold", color="dimgrey", ha="center", va="bottom", clip_on=True,
                )
    # Friedman + Nemenyi significance, drawn as merged horizontal lines
    # (row-merge and column-merge, see merged_significance_lines) with a
    # tick mark at every model each line flags (skipped for "Single
    # holdout", which has no repeated folds to test). Multiple lines in the
    # same CV-scheme group are stacked into non-overlapping tracks.
    line_gap = 50  # vertical gap (RMSE units) between the tallest error bar in a group and its first line - leaves room for the stars above it
    line_step = 85  # vertical spacing between stacked line tracks - wide enough for the alternating stars to clear the neighbouring track
    for cv_idx, cv_scheme in enumerate(cv_order):
        lines = sig_lines.get((fr_label, cv_scheme), [])
        if not lines:
            continue
        x_positions = x_base[cv_idx] + dodge  # one x position per model_order index
        # base line placement on the tallest error bar that is actually visible within the
        # axes (ignoring any model whose bar already exceeds ylim_top and is clipped there
        # anyway) - otherwise a single off-chart outlier (e.g. Lasso in LOGO CV) would force
        # every line in the group to be pinned uselessly high, far above the visible data.
        visible_tops = [
            rmse_by_model[m][cv_idx] + (0 if np.isnan(sd_by_model[m][cv_idx]) else sd_by_model[m][cv_idx])
            for m in model_order if not np.isnan(rmse_by_model[m][cv_idx])
        ]
        visible_tops = [t for t in visible_tops if t <= ylim_top] or visible_tops
        y_top_of_group = max(visible_tops)
        # stack lines into the lowest non-overlapping track (greedy interval scheduling on
        # the [left, right] index span), widest spans first so short lines can tuck in below
        ordered_lines = sorted(lines, key=lambda l: (l[0], l[1] - l[0]))
        track_ends = []  # track_ends[t] = farthest model index currently occupied on track t
        levels = []
        for left_idx, right_idx, ticks, tick_pvals in ordered_lines:
            placed = False
            for t, end in enumerate(track_ends):
                if left_idx > end:
                    track_ends[t] = right_idx
                    levels.append(t)
                    placed = True
                    break
            if not placed:
                levels.append(len(track_ends))
                track_ends.append(right_idx)
        for (left_idx, right_idx, ticks, tick_pvals), level in zip(ordered_lines, levels):
            y_unclamped = y_top_of_group + line_gap + level * line_step
            # clamp near the top edge (same trick as before) so lines for models whose
            # error bars already reach/exceed ylim_top still render inside the visible axes
            y_line = min(y_unclamped, ylim_top - 10 - (len(track_ends) - 1 - level) * line_step)
            ax.plot([x_positions[left_idx], x_positions[right_idx]], [y_line, y_line],
                    color="dimgrey", linewidth=1.5, clip_on=True, zorder=1)
            # a line with only one flagged pair gets its star centred on the line instead
            # of at the tick's own x-position, which would otherwise sit at one end of it
            single_pair = sum(p is not None for p in tick_pvals) == 1
            mid_x = (x_positions[left_idx] + x_positions[right_idx]) / 2
            star_count = 0  # alternates star placement above/below the line so closely-spaced ticks don't overlap
            for tick_idx, p_value in zip(ticks, tick_pvals):
                ax.plot([x_positions[tick_idx], x_positions[tick_idx]], [y_line - 7, y_line],
                        color="dimgrey", linewidth=1.5, clip_on=True, zorder=1)
                if p_value is None:  # the line's own reference model (anchor/shared target) - no self p-value
                    continue
                # significance stars for this specific tick, using the conventional
                # thresholds, one rating per pair so a merged line can show several
                # different star counts instead of a single rating for the whole line
                if p_value < 0.001:
                    stars = "***"
                elif p_value < 0.01:
                    stars = "**"
                else:
                    stars = "*"
                if star_count % 2 == 0:
                    # "baseline" (not "bottom") - "bottom" reserves room for descenders
                    # the asterisk glyph doesn't have, which left a visible gap above the line
                    offset, va = (0, 0), "baseline"
                else:
                    offset, va = (0, -2.5), "top"  # hugging just below the line
                star_x = mid_x if single_pair else x_positions[tick_idx]
                ax.annotate(
                    stars, xy=(star_x, y_line), xytext=offset, textcoords="offset points",
                    fontsize=15, fontweight="bold", color="dimgrey", ha="center", va=va, clip_on=True,
                )
                star_count += 1
    # shared marker for the single lowest RMSE value found in either panel
    ax.axhline(
        min_rmse, color="grey", linestyle=":", linewidth=1.5, zorder=0,
        label="Lowest RMSE" if ax is axes[0] else None,
    )
    ax.set_xticks(x_base)
    ax.set_xticklabels([cv_display[cv] for cv in cv_order], fontsize=19)
    ax.set_title(panel_titles[fr_label], fontsize=21, loc="left")
    ax.set_ylim(bottom=0, top=ylim_top)  # RMSE axis starts at zero, clipped at ylim_top
    ax.tick_params(axis="y", labelsize=19)

axes[0].set_ylabel("RMSE (μmol/g)", fontsize=22, labelpad=15)

# reorder handles into matplotlib's column-major legend fill so that, with
# ncol=8, the 8 models occupy the full top row and the dotted line ends up
# alone in the row below (see the common 'legend flip' recipe). Models and
# the axhline are stored in separate artist categories internally, so their
# handles/labels aren't necessarily returned in the order they were added -
# explicitly move the line to the end first to guarantee "models, then line".
handles, labels = list(axes[0].get_legend_handles_labels()[0]), list(axes[0].get_legend_handles_labels()[1])
line_idx = labels.index("Lowest RMSE")
handles.append(handles.pop(line_idx))
labels.append(labels.pop(line_idx))

ncol = len(model_order) // 2  # wrap models into two rows of 4 so the legend fits within the figure width
flipped_handles = list(itertools.chain(*[handles[i::ncol] for i in range(ncol)]))
flipped_labels = list(itertools.chain(*[labels[i::ncol] for i in range(ncol)]))

fig.tight_layout()  # finalise axes positions first, so we can align the legend to them
left_x = axes[0].get_position().x0  # left edge of the plot area, not the figure
legend = fig.legend(
    flipped_handles, flipped_labels, title="Model", loc="upper left", bbox_to_anchor=(left_x, 0),
    fontsize=19, title_fontsize=20, markerscale=1.3, ncol=ncol, borderaxespad=0,
)
legend._legend_box.align = "left"  # left-align the "Model" title above the entries

# explain both significance symbols, left-aligned under the "Lowest RMSE" row
# (the bottom-left entry of the legend, per the flip trick above), matching
# the legend's own font size/position rather than spanning the full figure
fig.text(
    left_x, -0.23,
    "Lines: Friedman test + Nemenyi post-hoc; a line from a model to its tick marks flags model(s) it is significantly different from \n(within the same panel and evaluation condition).\n"
    "Asterisks above each tick: Nemenyi p-value for that specific pair.\n"
    "Asterisks above a point: significant with-vs-without feature-reduction RMSE difference for that model (Wilcoxon signed-rank test, \nHolm\u2013Bonferroni corrected, \u03b1 = 0.05).\n* p < 0.05, ** p < 0.01, *** p < 0.001",
    ha="left", va="top", fontsize=18, color="black",
)

fig.savefig(Path(__file__).parent / "RMSE_ML_1.png", dpi=300, bbox_inches="tight")
plt.show()
