"""Parts 2-3: cell frequencies and exploratory response-group comparisons."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests


ROOT_DIR = Path(__file__).resolve().parent
DATABASE_PATH = ROOT_DIR / "cell_counts.db"
OUTPUT_PATH = ROOT_DIR / "outputs" / "summary_table.csv"
STATISTICAL_RESULTS_PATH = ROOT_DIR / "outputs" / "statistical_results.csv"
BOXPLOT_PATH = ROOT_DIR / "outputs" / "responder_vs_nonresponder_boxplot.png"
BASELINE_COHORT_PATH = ROOT_DIR / "outputs" / "baseline_melanoma_miraclib_pbmc.csv"
BASELINE_PROJECT_COUNTS_PATH = ROOT_DIR / "outputs" / "baseline_samples_by_project.csv"
BASELINE_RESPONSE_COUNTS_PATH = ROOT_DIR / "outputs" / "baseline_subjects_by_response.csv"
BASELINE_SEX_COUNTS_PATH = ROOT_DIR / "outputs" / "baseline_subjects_by_sex.csv"
BASELINE_BCELL_PATH = ROOT_DIR / "outputs" / "melanoma_male_responder_baseline_bcell.csv"

POPULATION_ORDER = (
    "b_cell",
    "cd8_t_cell",
    "cd4_t_cell",
    "nk_cell",
    "monocyte",
)
EXPECTED_SAMPLE_COUNT = 10_500
EXPECTED_SUMMARY_ROWS = 52_500
EXPECTED_POPULATIONS_PER_SAMPLE = 5
PERCENTAGE_SUM_TOLERANCE = 1e-9
SIGNIFICANCE_ALPHA = 0.05


def get_frequency_summary(connection: sqlite3.Connection) -> pd.DataFrame:
    """Return full-precision long-form cell frequencies from normalized tables."""
    query = """
        SELECT
            s.sample_id AS sample,
            SUM(cc.cell_count) OVER (PARTITION BY s.sample_id) AS total_count,
            cp.population_name AS population,
            cc.cell_count AS count
        FROM samples AS s
        JOIN cell_counts AS cc
          ON cc.sample_id = s.sample_id
        JOIN cell_populations AS cp
          ON cp.population_id = cc.population_id
        ORDER BY
            s.sample_id,
            CASE cp.population_name
                WHEN 'b_cell' THEN 1
                WHEN 'cd8_t_cell' THEN 2
                WHEN 'cd4_t_cell' THEN 3
                WHEN 'nk_cell' THEN 4
                WHEN 'monocyte' THEN 5
                ELSE 6
            END,
            cp.population_name
    """
    summary = pd.read_sql_query(query, connection)
    summary["percentage"] = summary["count"] / summary["total_count"] * 100.0
    return summary[["sample", "total_count", "population", "count", "percentage"]]


def validate_frequency_summary(summary: pd.DataFrame) -> None:
    """Raise a clear error if the Part 2 source or calculations are inconsistent."""
    expected_columns = ["sample", "total_count", "population", "count", "percentage"]
    if summary.columns.tolist() != expected_columns:
        raise ValueError(
            f"Unexpected summary columns: {summary.columns.tolist()}; "
            f"expected {expected_columns}."
        )

    if len(summary) != EXPECTED_SUMMARY_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_SUMMARY_ROWS:,} sample-population rows, "
            f"found {len(summary):,}."
        )

    unique_samples = summary["sample"].nunique(dropna=False)
    if unique_samples != EXPECTED_SAMPLE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_SAMPLE_COUNT:,} unique samples, found {unique_samples:,}."
        )

    if summary[["sample", "population", "count", "total_count"]].isna().any().any():
        raise ValueError("Null sample, population, population-count, or total-count values found.")

    duplicate_pairs = summary.duplicated(["sample", "population"], keep=False)
    if duplicate_pairs.any():
        examples = summary.loc[duplicate_pairs, ["sample", "population"]].head().to_dict("records")
        raise ValueError(f"Duplicate sample/population pairs found; examples: {examples}")

    rows_per_sample = summary.groupby("sample", sort=False).size()
    invalid_row_counts = rows_per_sample.ne(EXPECTED_POPULATIONS_PER_SAMPLE)
    if invalid_row_counts.any():
        examples = rows_per_sample[invalid_row_counts].head().to_dict()
        raise ValueError(f"Every sample must have exactly five population rows; found: {examples}")

    population_sets = summary.groupby("sample", sort=False)["population"].agg(frozenset)
    expected_populations = frozenset(POPULATION_ORDER)
    invalid_sets = population_sets.ne(expected_populations)
    if invalid_sets.any():
        examples = population_sets[invalid_sets].head().to_dict()
        raise ValueError(f"Samples do not contain the expected five populations; found: {examples}")

    if summary["total_count"].le(0).any():
        samples = summary.loc[summary["total_count"].le(0), "sample"].unique()[:5].tolist()
        raise ValueError(f"Every sample total_count must be greater than zero; examples: {samples}")

    total_values_per_sample = summary.groupby("sample", sort=False)["total_count"].nunique()
    if total_values_per_sample.ne(1).any():
        raise ValueError("A sample has inconsistent total_count values across populations.")

    calculated_totals = summary.groupby("sample", sort=False)["count"].sum()
    reported_totals = summary.groupby("sample", sort=False)["total_count"].first()
    if not calculated_totals.equals(reported_totals):
        raise ValueError("Reported total_count values do not equal summed population counts.")

    percentage_sums = summary.groupby("sample", sort=False)["percentage"].sum()
    outside_tolerance = percentage_sums.sub(100.0).abs().gt(PERCENTAGE_SUM_TOLERANCE)
    if outside_tolerance.any():
        examples = percentage_sums[outside_tolerance].head().to_dict()
        raise ValueError(f"Population percentages do not sum to approximately 100; found: {examples}")


def save_frequency_summary(summary: pd.DataFrame, output_path: Path) -> None:
    """Write a readable CSV without modifying the full-precision DataFrame."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output_path, index=False, float_format="%.6f")


def get_analysis_metadata(connection: sqlite3.Connection) -> pd.DataFrame:
    """Return sample and subject metadata needed for the Part 3 comparison."""
    query = """
        SELECT
            s.sample_id AS sample,
            sub.subject_id AS subject,
            sub.project_id AS project,
            sub.condition,
            sub.treatment,
            sub.response,
            s.sample_type,
            s.time_from_treatment_start,
            sub.sex
        FROM samples AS s
        JOIN subjects AS sub
          ON sub.subject_id = s.subject_id
        ORDER BY s.sample_id
    """
    return pd.read_sql_query(query, connection)


def build_response_comparison_data(
    frequency_summary: pd.DataFrame,
    metadata: pd.DataFrame,
) -> pd.DataFrame:
    """Join full-precision frequencies to metadata and apply Part 3 filters."""
    if metadata["sample"].duplicated().any():
        raise ValueError("Analysis metadata must contain exactly one row per sample.")

    comparison = frequency_summary[["sample", "population", "percentage"]].merge(
        metadata,
        on="sample",
        how="left",
        validate="many_to_one",
    )
    metadata_columns = [
        "sample",
        "subject",
        "project",
        "condition",
        "treatment",
        "response",
        "sample_type",
        "time_from_treatment_start",
        "sex",
        "population",
        "percentage",
    ]
    comparison = comparison[metadata_columns]
    if comparison.drop(columns="response").isna().any().any():
        raise ValueError("Frequency rows could not be matched to complete sample metadata.")

    eligible = comparison.loc[
        comparison["condition"].eq("melanoma")
        & comparison["treatment"].eq("miraclib")
        & comparison["sample_type"].eq("PBMC")
        & comparison["response"].isin(("yes", "no"))
    ].copy()
    return eligible.sort_values(["sample", "population"], kind="stable").reset_index(drop=True)


def aggregate_subject_frequencies(comparison: pd.DataFrame) -> pd.DataFrame:
    """Average repeated samples so each subject contributes once per population.

    Samples at times 0, 7, and 14 are repeated measurements from the same subject.
    Treating them as independent would create pseudo-replication, so the statistical
    unit is the subject's mean percentage across all eligible PBMC samples.
    """
    if comparison.empty:
        raise ValueError("No records satisfy the Part 3 eligibility filters.")
    if comparison["percentage"].isna().any():
        raise ValueError("Null percentages cannot be used for Part 3 statistics.")
    if set(comparison["response"].unique()) != {"yes", "no"}:
        raise ValueError("Both responder ('yes') and non-responder ('no') groups are required.")
    if set(comparison["population"].unique()) != set(POPULATION_ORDER):
        raise ValueError("Part 3 requires exactly the five expected cell populations.")

    subject_response_counts = comparison.groupby("subject")["response"].nunique()
    if subject_response_counts.gt(1).any():
        raise ValueError("Response must be constant within each eligible subject.")

    subject_means = (
        comparison.groupby(["subject", "response", "population"], as_index=False)["percentage"]
        .mean()
        .rename(columns={"percentage": "mean_percentage"})
    )
    if subject_means.duplicated(["subject", "population"]).any():
        raise ValueError("Subject-level aggregation left duplicate subject/population rows.")

    expected_pairs = comparison["subject"].nunique() * len(POPULATION_ORDER)
    if len(subject_means) != expected_pairs:
        raise ValueError("Every eligible subject must have one mean for every population.")

    group_counts = subject_means.groupby(["population", "response"]).size().unstack(fill_value=0)
    if not {"yes", "no"}.issubset(group_counts.columns) or group_counts[["yes", "no"]].eq(0).any().any():
        raise ValueError("Every population must have observations in both response groups.")
    return subject_means


def run_population_statistics(subject_means: pd.DataFrame) -> pd.DataFrame:
    """Run two-sided Mann-Whitney tests and Benjamini-Hochberg correction."""
    rows: list[dict[str, float | int | str]] = []
    for population in POPULATION_ORDER:
        population_data = subject_means.loc[subject_means["population"].eq(population)]
        responders = population_data.loc[
            population_data["response"].eq("yes"), "mean_percentage"
        ]
        non_responders = population_data.loc[
            population_data["response"].eq("no"), "mean_percentage"
        ]
        test = mannwhitneyu(responders, non_responders, alternative="two-sided")
        rows.append(
            {
                "population": population,
                "responder_n": len(responders),
                "non_responder_n": len(non_responders),
                "responder_mean_percentage": responders.mean(),
                "non_responder_mean_percentage": non_responders.mean(),
                "responder_median_percentage": responders.median(),
                "non_responder_median_percentage": non_responders.median(),
                "mean_difference_percentage_points": responders.mean() - non_responders.mean(),
                "responder_std": responders.std(),
                "non_responder_std": non_responders.std(),
                "mann_whitney_u": float(test.statistic),
                "p_value": float(test.pvalue),
            }
        )

    results = pd.DataFrame(rows)
    rejected, adjusted_p_values, _, _ = multipletests(
        results["p_value"].to_numpy(), alpha=SIGNIFICANCE_ALPHA, method="fdr_bh"
    )
    results["adjusted_p_value"] = adjusted_p_values
    # Express the requested strict adjusted-p threshold directly; `rejected` is
    # checked as a guard against an unexpected library/API discrepancy.
    results["significant"] = results["adjusted_p_value"].lt(SIGNIFICANCE_ALPHA)
    if not (results["significant"].to_numpy() == rejected).all():
        raise RuntimeError("Multiple-testing correction returned inconsistent decisions.")
    return results


def save_statistical_results(results: pd.DataFrame, output_path: Path) -> None:
    """Save readable statistics without changing the full-precision results."""
    output = results.copy()
    descriptive_columns = [
        "responder_mean_percentage",
        "non_responder_mean_percentage",
        "responder_median_percentage",
        "non_responder_median_percentage",
        "mean_difference_percentage_points",
        "responder_std",
        "non_responder_std",
        "mann_whitney_u",
    ]
    output[descriptive_columns] = output[descriptive_columns].round(6)
    output[["p_value", "adjusted_p_value"]] = output[
        ["p_value", "adjusted_p_value"]
    ].round(12)
    output.to_csv(output_path, index=False)


def create_response_boxplot(subject_means: pd.DataFrame, output_path: Path) -> None:
    """Plot the same subject-level observations used by the statistical tests."""
    colors = {"no": "#4C78A8", "yes": "#F58518"}
    offsets = {"no": -0.18, "yes": 0.18}
    labels = {"no": "Non-responder", "yes": "Responder"}
    positions = range(1, len(POPULATION_ORDER) + 1)

    figure, axis = plt.subplots(figsize=(12, 7))
    for response in ("no", "yes"):
        values = [
            subject_means.loc[
                subject_means["population"].eq(population)
                & subject_means["response"].eq(response),
                "mean_percentage",
            ].to_numpy()
            for population in POPULATION_ORDER
        ]
        boxplot = axis.boxplot(
            values,
            positions=[position + offsets[response] for position in positions],
            widths=0.30,
            patch_artist=True,
            showfliers=True,
            medianprops={"color": "black", "linewidth": 1.4},
        )
        for box in boxplot["boxes"]:
            box.set_facecolor(colors[response])
            box.set_alpha(0.72)

    axis.set_xticks(list(positions), POPULATION_ORDER)
    axis.set_xlabel("Immune-cell population")
    axis.set_ylabel("Subject mean relative frequency (%)")
    axis.set_title(
        "PBMC Cell-Population Frequencies by Response\n"
        "Melanoma subjects receiving miraclib"
    )
    axis.legend(
        handles=[
            Patch(facecolor=colors["no"], alpha=0.72, label=labels["no"]),
            Patch(facecolor=colors["yes"], alpha=0.72, label=labels["yes"]),
        ],
        title="Response group",
    )
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def describe_eligible_data(comparison: pd.DataFrame) -> dict[str, object]:
    """Return compact counts used for console reporting and verification."""
    subject_responses = comparison[["subject", "response"]].drop_duplicates()
    sample_responses = comparison[["sample", "response"]].drop_duplicates()
    return {
        "subjects": subject_responses["subject"].nunique(),
        "responders": subject_responses["response"].eq("yes").sum(),
        "non_responders": subject_responses["response"].eq("no").sum(),
        "samples": sample_responses["sample"].nunique(),
        "samples_by_response": sample_responses["response"].value_counts().to_dict(),
        "sample_population_rows": comparison.groupby(["response", "population"]).size(),
    }


BASELINE_FILTER_PARAMS = ("melanoma", "miraclib", "PBMC", 0)


def get_baseline_melanoma_miraclib_pbmc(
    connection: sqlite3.Connection,
) -> pd.DataFrame:
    """Query the Part 4A sample cohort directly from normalized SQLite tables."""
    query = """
        SELECT
            p.project_id AS project,
            sub.subject_id AS subject,
            s.sample_id AS sample,
            sub.condition,
            sub.sex,
            sub.treatment,
            sub.response,
            s.sample_type,
            s.time_from_treatment_start
        FROM projects AS p
        JOIN subjects AS sub
          ON sub.project_id = p.project_id
        JOIN samples AS s
          ON s.subject_id = sub.subject_id
        WHERE sub.condition = ?
          AND sub.treatment = ?
          AND s.sample_type = ?
          AND s.time_from_treatment_start = ?
        ORDER BY p.project_id, sub.subject_id, s.sample_id
    """
    return pd.read_sql_query(query, connection, params=BASELINE_FILTER_PARAMS)


def get_baseline_samples_by_project(connection: sqlite3.Connection) -> pd.DataFrame:
    """Count Part 4A samples by project in SQL."""
    query = """
        SELECT
            p.project_id AS project,
            COUNT(s.sample_id) AS sample_count
        FROM projects AS p
        JOIN subjects AS sub
          ON sub.project_id = p.project_id
        JOIN samples AS s
          ON s.subject_id = sub.subject_id
        WHERE sub.condition = ?
          AND sub.treatment = ?
          AND s.sample_type = ?
          AND s.time_from_treatment_start = ?
        GROUP BY p.project_id
        ORDER BY p.project_id
    """
    return pd.read_sql_query(query, connection, params=BASELINE_FILTER_PARAMS)


def get_baseline_subjects_by_response(connection: sqlite3.Connection) -> pd.DataFrame:
    """Count distinct Part 4A subjects by response in SQL, retaining unexpected NULLs."""
    query = """
        SELECT
            sub.response,
            COUNT(DISTINCT sub.subject_id) AS subject_count
        FROM subjects AS sub
        JOIN samples AS s
          ON s.subject_id = sub.subject_id
        WHERE sub.condition = ?
          AND sub.treatment = ?
          AND s.sample_type = ?
          AND s.time_from_treatment_start = ?
        GROUP BY sub.response
        ORDER BY sub.response
    """
    return pd.read_sql_query(query, connection, params=BASELINE_FILTER_PARAMS)


def get_baseline_subjects_by_sex(connection: sqlite3.Connection) -> pd.DataFrame:
    """Count distinct Part 4A subjects by sex in SQL."""
    query = """
        SELECT
            sub.sex,
            COUNT(DISTINCT sub.subject_id) AS subject_count
        FROM subjects AS sub
        JOIN samples AS s
          ON s.subject_id = sub.subject_id
        WHERE sub.condition = ?
          AND sub.treatment = ?
          AND s.sample_type = ?
          AND s.time_from_treatment_start = ?
        GROUP BY sub.sex
        ORDER BY sub.sex
    """
    return pd.read_sql_query(query, connection, params=BASELINE_FILTER_PARAMS)


def get_melanoma_male_responder_baseline_bcell(
    connection: sqlite3.Connection,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Calculate the special raw B-cell average across all treatments/sample types.

    Deliberately absent from this SQL: predicates on treatment and sample_type.
    Those dimensions are audited with distinct counts to guard against accidentally
    narrowing this question to the Part 4A cohort.
    """
    filters = ("melanoma", "M", "yes", 0, "b_cell")
    query = """
        SELECT
            AVG(cc.cell_count) AS average_b_cell_count,
            COUNT(DISTINCT s.sample_id) AS sample_count,
            COUNT(DISTINCT sub.treatment) AS treatment_count,
            COUNT(DISTINCT s.sample_type) AS sample_type_count,
            SUM(
                CASE WHEN sub.condition = ?
                       AND sub.sex = ?
                       AND sub.response = ?
                       AND s.time_from_treatment_start = ?
                       AND cp.population_name = ?
                     THEN 0 ELSE 1 END
            ) AS invalid_row_count
        FROM subjects AS sub
        JOIN samples AS s
          ON s.subject_id = sub.subject_id
        JOIN cell_counts AS cc
          ON cc.sample_id = s.sample_id
        JOIN cell_populations AS cp
          ON cp.population_id = cc.population_id
        WHERE sub.condition = ?
          AND sub.sex = ?
          AND sub.response = ?
          AND s.time_from_treatment_start = ?
          AND cp.population_name = ?
    """
    aggregate = pd.read_sql_query(query, connection, params=(*filters, *filters))
    if len(aggregate) != 1:
        raise ValueError("The special B-cell query must return exactly one aggregate row.")
    row = aggregate.iloc[0]
    if pd.isna(row["average_b_cell_count"]) or int(row["sample_count"]) <= 0:
        raise ValueError("No samples contributed to the special baseline B-cell average.")
    if int(row["invalid_row_count"]) != 0:
        raise ValueError("The special B-cell query returned rows outside its required filters.")
    if int(row["sample_count"]) != connection.execute(
        """
        SELECT COUNT(*) FROM (
            SELECT s.sample_id
            FROM subjects AS sub
            JOIN samples AS s ON s.subject_id = sub.subject_id
            JOIN cell_counts AS cc ON cc.sample_id = s.sample_id
            JOIN cell_populations AS cp ON cp.population_id = cc.population_id
            WHERE sub.condition = ? AND sub.sex = ? AND sub.response = ?
              AND s.time_from_treatment_start = ? AND cp.population_name = ?
            GROUP BY s.sample_id
            HAVING COUNT(*) = 1
        )
        """,
        filters,
    ).fetchone()[0]:
        raise ValueError("Special B-cell rows are not unique by sample.")

    output = aggregate[["average_b_cell_count", "sample_count"]].copy()
    output["sample_count"] = output["sample_count"].astype("int64")
    audit = {
        "treatment_count": int(row["treatment_count"]),
        "sample_type_count": int(row["sample_type_count"]),
    }
    return output, audit


def validate_part4_results(
    cohort: pd.DataFrame,
    project_counts: pd.DataFrame,
    response_counts: pd.DataFrame,
    sex_counts: pd.DataFrame,
) -> None:
    """Validate cohort predicates, uniqueness, and aggregate reconciliation."""
    if cohort.empty:
        raise ValueError("The Part 4A baseline cohort is empty.")
    required_values = {
        "condition": {"melanoma"},
        "treatment": {"miraclib"},
        "sample_type": {"PBMC"},
        "time_from_treatment_start": {0},
    }
    for column, expected in required_values.items():
        actual = set(cohort[column].dropna().unique())
        if actual != expected or cohort[column].isna().any():
            raise ValueError(f"Invalid Part 4A {column} values: {actual}; expected {expected}.")
    if cohort["sample"].duplicated().any():
        raise ValueError("Part 4A sample IDs must be unique.")

    sample_count = len(cohort)
    subject_count = cohort["subject"].nunique()
    if int(project_counts["sample_count"].sum()) != sample_count:
        raise ValueError("Project sample counts do not reconcile to the baseline cohort.")
    if response_counts["response"].isna().any():
        null_count = int(response_counts.loc[response_counts["response"].isna(), "subject_count"].sum())
        raise ValueError(f"Baseline responder counts unexpectedly include {null_count} NULL responses.")
    if set(response_counts["response"]) != {"yes", "no"}:
        raise ValueError(f"Unexpected baseline response categories: {response_counts['response'].tolist()}")
    if int(response_counts["subject_count"].sum()) != subject_count:
        raise ValueError("Response subject counts do not reconcile to the baseline cohort.")
    if set(sex_counts["sex"]) != {"M", "F"}:
        raise ValueError(f"Unexpected baseline sex categories: {sex_counts['sex'].tolist()}")
    if int(sex_counts["subject_count"].sum()) != subject_count:
        raise ValueError("Sex subject counts do not reconcile to the baseline cohort.")


def run_part4_analysis(
    connection: sqlite3.Connection,
    output_dir: Path,
) -> dict[str, object]:
    """Run, validate, and save all SQL-driven Part 4 outputs."""
    cohort = get_baseline_melanoma_miraclib_pbmc(connection)
    project_counts = get_baseline_samples_by_project(connection)
    response_counts = get_baseline_subjects_by_response(connection)
    sex_counts = get_baseline_subjects_by_sex(connection)
    bcell_result, bcell_audit = get_melanoma_male_responder_baseline_bcell(connection)
    validate_part4_results(cohort, project_counts, response_counts, sex_counts)

    output_dir.mkdir(parents=True, exist_ok=True)
    cohort.to_csv(output_dir / BASELINE_COHORT_PATH.name, index=False)
    project_counts.to_csv(output_dir / BASELINE_PROJECT_COUNTS_PATH.name, index=False)
    response_counts.to_csv(output_dir / BASELINE_RESPONSE_COUNTS_PATH.name, index=False)
    sex_counts.to_csv(output_dir / BASELINE_SEX_COUNTS_PATH.name, index=False)
    bcell_result.to_csv(
        output_dir / BASELINE_BCELL_PATH.name,
        index=False,
        float_format="%.2f",
    )
    return {
        "cohort": cohort,
        "project_counts": project_counts,
        "response_counts": response_counts,
        "sex_counts": sex_counts,
        "bcell_result": bcell_result,
        "bcell_audit": bcell_audit,
    }


def main() -> None:
    if not DATABASE_PATH.is_file():
        raise FileNotFoundError(
            f"Database not found: {DATABASE_PATH}. Run load_data.py before analysis.py."
        )

    database_uri = f"file:{DATABASE_PATH}?mode=ro"
    with sqlite3.connect(database_uri, uri=True) as connection:
        summary = get_frequency_summary(connection)
        metadata = get_analysis_metadata(connection)

    validate_frequency_summary(summary)
    save_frequency_summary(summary, OUTPUT_PATH)

    print("Part 2 complete.")
    print()
    print(f"Samples analyzed: {summary['sample'].nunique():,}")
    print(f"Summary rows: {len(summary):,}")
    print(f"Populations per sample: {EXPECTED_POPULATIONS_PER_SAMPLE}")
    print(f"Output: {OUTPUT_PATH}")
    print()
    print("Validation passed.")

    comparison = build_response_comparison_data(summary, metadata)
    subject_means = aggregate_subject_frequencies(comparison)
    results = run_population_statistics(subject_means)
    eligible_counts = describe_eligible_data(comparison)
    save_statistical_results(results, STATISTICAL_RESULTS_PATH)
    create_response_boxplot(subject_means, BOXPLOT_PATH)

    print()
    print("Part 3 complete.")
    print()
    print(f"Eligible subjects: {eligible_counts['subjects']:,}")
    print(f"Responders: {eligible_counts['responders']:,}")
    print(f"Non-responders: {eligible_counts['non_responders']:,}")
    print(f"Eligible samples: {eligible_counts['samples']:,}")
    samples_by_response = eligible_counts["samples_by_response"]
    print(f"Responder samples: {samples_by_response.get('yes', 0):,}")
    print(f"Non-responder samples: {samples_by_response.get('no', 0):,}")
    print()
    print("Statistical results (one mean observation per subject and population):")
    display_columns = [
        "population",
        "responder_n",
        "non_responder_n",
        "mean_difference_percentage_points",
        "p_value",
        "adjusted_p_value",
        "significant",
    ]
    print(
        results[display_columns].to_string(
            index=False,
            formatters={
                "mean_difference_percentage_points": lambda value: f"{value:.6f}",
                "p_value": lambda value: f"{value:.6g}",
                "adjusted_p_value": lambda value: f"{value:.6g}",
            },
        )
    )
    print()
    significant = results.loc[results["significant"], "population"].tolist()
    print("Significant populations after Benjamini-Hochberg correction:")
    if significant:
        for population in significant:
            print(f"- {population}")
    else:
        print("- None")
    print()
    print("Outputs:")
    print(f"- {STATISTICAL_RESULTS_PATH}")
    print(f"- {BOXPLOT_PATH}")
    print()
    print("Validation passed.")

    with sqlite3.connect(database_uri, uri=True) as connection:
        part4 = run_part4_analysis(connection, OUTPUT_PATH.parent)

    cohort = part4["cohort"]
    project_counts = part4["project_counts"]
    response_counts = part4["response_counts"]
    sex_counts = part4["sex_counts"]
    bcell_result = part4["bcell_result"]
    bcell_audit = part4["bcell_audit"]
    average_b_cell_count = float(bcell_result.iloc[0]["average_b_cell_count"])
    bcell_sample_count = int(bcell_result.iloc[0]["sample_count"])

    print()
    print("Part 4 complete.")
    print()
    print("Baseline melanoma + miraclib + PBMC:")
    print(f"Samples: {len(cohort):,}")
    print(f"Subjects: {cohort['subject'].nunique():,}")
    print()
    print("Samples by project:")
    print(project_counts.to_string(index=False))
    print()
    print("Subjects by response:")
    print(response_counts.to_string(index=False))
    print()
    print("Subjects by sex:")
    print(sex_counts.to_string(index=False))
    print()
    print("Melanoma male responders at baseline")
    print("(all treatments and sample types):")
    print(f"Samples included: {bcell_sample_count:,}")
    print(f"Average B-cell count: {average_b_cell_count:.2f}")
    print(
        "Coverage audit: "
        f"{bcell_audit['treatment_count']} treatments, "
        f"{bcell_audit['sample_type_count']} sample types"
    )
    print()
    print("Outputs:")
    for path in (
        BASELINE_COHORT_PATH,
        BASELINE_PROJECT_COUNTS_PATH,
        BASELINE_RESPONSE_COUNTS_PATH,
        BASELINE_SEX_COUNTS_PATH,
        BASELINE_BCELL_PATH,
    ):
        print(f"- {path}")
    print()
    print("Validation passed.")


if __name__ == "__main__":
    main()
