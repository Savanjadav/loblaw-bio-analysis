"""Interactive Streamlit dashboard for the immune-cell assessment results."""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from analysis import (
    POPULATION_ORDER,
    aggregate_subject_frequencies,
    build_response_comparison_data,
    get_analysis_metadata,
    get_baseline_melanoma_miraclib_pbmc,
    get_baseline_samples_by_project,
    get_baseline_subjects_by_response,
    get_baseline_subjects_by_sex,
    get_frequency_summary,
    get_melanoma_male_responder_baseline_bcell,
    run_population_statistics,
    validate_frequency_summary,
    validate_part4_results,
)


ROOT_DIR = Path(__file__).resolve().parent
DATABASE_PATH = ROOT_DIR / "cell_counts.db"

RESPONSE_LABELS = {"no": "Non-responder", "yes": "Responder"}
RESPONSE_COLORS = {"Non-responder": "#4C78A8", "Responder": "#F58518"}


def open_readonly_database(database_path: str) -> sqlite3.Connection:
    """Open SQLite without allowing dashboard reads to create or modify a database."""
    return sqlite3.connect(f"file:{Path(database_path).resolve()}?mode=ro", uri=True)


@st.cache_data(show_spinner=False)
def load_overview(database_path: str) -> tuple[dict[str, int], list[str], dict[str, pd.DataFrame]]:
    with open_readonly_database(database_path) as connection:
        metrics = {
            "Projects": connection.execute("SELECT COUNT(*) FROM projects").fetchone()[0],
            "Subjects": connection.execute("SELECT COUNT(*) FROM subjects").fetchone()[0],
            "Samples": connection.execute("SELECT COUNT(*) FROM samples").fetchone()[0],
            "Cell populations": connection.execute(
                "SELECT COUNT(*) FROM cell_populations"
            ).fetchone()[0],
        }
        populations = [
            row[0]
            for row in connection.execute(
                "SELECT population_name FROM cell_populations ORDER BY population_id"
            )
        ]
        distributions = {
            "Condition": pd.read_sql_query(
                """
                SELECT condition AS category, COUNT(*) AS subjects
                FROM subjects GROUP BY condition ORDER BY subjects DESC, condition
                """,
                connection,
            ),
            "Treatment": pd.read_sql_query(
                """
                SELECT treatment AS category, COUNT(*) AS subjects
                FROM subjects GROUP BY treatment ORDER BY subjects DESC, treatment
                """,
                connection,
            ),
            "Sample type": pd.read_sql_query(
                """
                SELECT sample_type AS category, COUNT(*) AS samples
                FROM samples GROUP BY sample_type ORDER BY samples DESC, sample_type
                """,
                connection,
            ),
        }
    return metrics, populations, distributions


@st.cache_data(show_spinner=False)
def load_frequency_data(database_path: str) -> pd.DataFrame:
    with open_readonly_database(database_path) as connection:
        summary = get_frequency_summary(connection)
    validate_frequency_summary(summary)
    return summary


@st.cache_data(show_spinner=False)
def load_response_analysis(
    database_path: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    with open_readonly_database(database_path) as connection:
        summary = get_frequency_summary(connection)
        metadata = get_analysis_metadata(connection)
    validate_frequency_summary(summary)
    comparison = build_response_comparison_data(summary, metadata)
    subject_means = aggregate_subject_frequencies(comparison)
    results = run_population_statistics(subject_means)
    return comparison, subject_means, results


@st.cache_data(show_spinner=False)
def load_baseline_analysis(
    database_path: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    with open_readonly_database(database_path) as connection:
        cohort = get_baseline_melanoma_miraclib_pbmc(connection)
        projects = get_baseline_samples_by_project(connection)
        responses = get_baseline_subjects_by_response(connection)
        sexes = get_baseline_subjects_by_sex(connection)
    validate_part4_results(cohort, projects, responses, sexes)
    return cohort, projects, responses, sexes


@st.cache_data(show_spinner=False)
def load_bcell_analysis(database_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    with open_readonly_database(database_path) as connection:
        result, _ = get_melanoma_male_responder_baseline_bcell(connection)
        coverage = pd.read_sql_query(
            """
            SELECT
                sub.treatment,
                s.sample_type,
                COUNT(DISTINCT s.sample_id) AS sample_count
            FROM subjects AS sub
            JOIN samples AS s ON s.subject_id = sub.subject_id
            JOIN cell_counts AS cc ON cc.sample_id = s.sample_id
            JOIN cell_populations AS cp ON cp.population_id = cc.population_id
            WHERE sub.condition = ?
              AND sub.sex = ?
              AND sub.response = ?
              AND s.time_from_treatment_start = ?
              AND cp.population_name = ?
            GROUP BY sub.treatment, s.sample_type
            ORDER BY sub.treatment, s.sample_type
            """,
            connection,
            params=("melanoma", "M", "yes", 0, "b_cell"),
        )
    if int(coverage["sample_count"].sum()) != int(result.iloc[0]["sample_count"]):
        raise ValueError("B-cell coverage does not reconcile to the calculated sample count.")
    return result, coverage


def render_overview(database_path: str) -> None:
    st.header("1. Data Overview")
    metrics, populations, distributions = load_overview(database_path)
    columns = st.columns(4)
    for column, (label, value) in zip(columns, metrics.items()):
        column.metric(label, f"{value:,}")

    st.markdown("**Immune-cell populations:** " + " · ".join(f"`{p}`" for p in populations))
    with st.expander("Dataset composition"):
        columns = st.columns(3)
        for column, (label, distribution) in zip(columns, distributions.items()):
            column.markdown(f"**{label}**")
            column.dataframe(distribution, hide_index=True, use_container_width=True)


def render_sample_analysis(database_path: str) -> None:
    st.header("2. Sample Cell Frequencies")
    st.caption("Explore raw counts and relative frequencies calculated from the SQLite database.")
    summary = load_frequency_data(database_path)
    samples = summary["sample"].drop_duplicates().tolist()
    selected_sample = st.selectbox("Select a sample", samples, index=0)
    selected = summary.loc[summary["sample"].eq(selected_sample)].copy()
    selected["population"] = pd.Categorical(
        selected["population"], categories=POPULATION_ORDER, ordered=True
    )
    selected = selected.sort_values("population")

    table = selected[["population", "count", "percentage", "total_count"]].copy()
    st.dataframe(
        table,
        hide_index=True,
        use_container_width=True,
        column_config={
            "count": st.column_config.NumberColumn("Raw count", format="%d"),
            "percentage": st.column_config.NumberColumn("Relative frequency (%)", format="%.2f"),
            "total_count": st.column_config.NumberColumn("Total count", format="%d"),
        },
    )
    chart = px.bar(
        selected,
        x="population",
        y="percentage",
        category_orders={"population": list(POPULATION_ORDER)},
        labels={"population": "Immune-cell population", "percentage": "Relative frequency (%)"},
        title=f"Cell-population profile for {selected_sample}",
        color="population",
        color_discrete_sequence=px.colors.qualitative.Safe,
    )
    chart.update_layout(showlegend=False, margin=dict(t=55, l=10, r=10, b=10))
    st.plotly_chart(chart, use_container_width=True)


def render_response_analysis(database_path: str) -> None:
    st.header("3. Miraclib Response Analysis")
    st.markdown("**Fixed cohort:** Melanoma patients receiving miraclib, PBMC samples only.")
    st.caption(
        "Repeated samples at times 0, 7, and 14 are averaged to one relative-frequency "
        "value per subject and population before statistical comparison."
    )
    comparison, subject_means, results = load_response_analysis(database_path)
    subjects = comparison[["subject", "response"]].drop_duplicates()
    columns = st.columns(3)
    columns[0].metric("Eligible subjects", f"{subjects['subject'].nunique():,}")
    columns[1].metric("Responders", f"{subjects['response'].eq('yes').sum():,}")
    columns[2].metric("Non-responders", f"{subjects['response'].eq('no').sum():,}")

    plot_data = subject_means.copy()
    plot_data["Response"] = plot_data["response"].map(RESPONSE_LABELS)
    chart = px.box(
        plot_data,
        x="population",
        y="mean_percentage",
        color="Response",
        category_orders={
            "population": list(POPULATION_ORDER),
            "Response": ["Non-responder", "Responder"],
        },
        color_discrete_map=RESPONSE_COLORS,
        labels={
            "population": "Immune-cell population",
            "mean_percentage": "Subject mean relative frequency (%)",
        },
        title="Subject-level PBMC frequencies by response group",
        points="outliers",
    )
    chart.update_layout(boxmode="group", legend_title_text="Response group")
    st.plotly_chart(chart, use_container_width=True)

    display = results[
        [
            "population",
            "responder_mean_percentage",
            "non_responder_mean_percentage",
            "mean_difference_percentage_points",
            "p_value",
            "adjusted_p_value",
            "significant",
        ]
    ].rename(
        columns={
            "responder_mean_percentage": "responder_mean",
            "non_responder_mean_percentage": "non_responder_mean",
            "mean_difference_percentage_points": "mean_difference",
            "adjusted_p_value": "adjusted_p",
        }
    )
    st.dataframe(
        display,
        hide_index=True,
        use_container_width=True,
        column_config={
            "responder_mean": st.column_config.NumberColumn(format="%.3f"),
            "non_responder_mean": st.column_config.NumberColumn(format="%.3f"),
            "mean_difference": st.column_config.NumberColumn(format="%+.3f"),
            "p_value": st.column_config.NumberColumn("Raw p-value", format="%.4g"),
            "adjusted_p": st.column_config.NumberColumn("Adjusted p-value", format="%.4g"),
            "significant": st.column_config.CheckboxColumn("Significant"),
        },
    )
    st.caption(
        "Two-sided Mann–Whitney U tests; Benjamini–Hochberg false-discovery-rate "
        "correction across five populations; α = 0.05."
    )
    significant = results.loc[results["significant"], "population"].tolist()
    if significant:
        st.info(
            "Populations significant after correction: " + ", ".join(significant) + ". "
            "These exploratory associations do not establish predictive performance."
        )
    else:
        smallest = results.loc[results["p_value"].idxmin()]
        st.info(
            "No cell population reached statistical significance after multiple-testing "
            f"correction. `{smallest['population']}` had the smallest raw p-value "
            f"({smallest['p_value']:.4f}), but its adjusted p-value "
            f"({smallest['adjusted_p_value']:.4f}) was above 0.05."
        )


def render_baseline_analysis(database_path: str) -> None:
    st.header("4. Baseline Subset Analysis")
    st.markdown(
        "**Fixed filters:** `condition = melanoma`, `treatment = miraclib`, "
        "`sample_type = PBMC`, `time_from_treatment_start = 0`."
    )
    cohort, projects, responses, sexes = load_baseline_analysis(database_path)
    columns = st.columns(2)
    columns[0].metric("Baseline samples", f"{len(cohort):,}")
    columns[1].metric("Unique subjects", f"{cohort['subject'].nunique():,}")

    chart_columns = st.columns(3)
    chart_inputs = (
        ("Samples by project", projects, "project", "sample_count", "#4C78A8"),
        ("Subjects by response", responses, "response", "subject_count", "#F58518"),
        ("Subjects by sex", sexes, "sex", "subject_count", "#72B7B2"),
    )
    for column, (title, data, category, value, color) in zip(chart_columns, chart_inputs):
        chart = px.bar(
            data,
            x=category,
            y=value,
            text=value,
            title=title,
            color_discrete_sequence=[color],
        )
        chart.update_traces(textposition="outside")
        chart.update_layout(showlegend=False, margin=dict(t=55, l=5, r=5, b=5), height=340)
        column.plotly_chart(chart, use_container_width=True)


def render_bcell_analysis(database_path: str) -> None:
    st.header("5. Baseline B-cell Analysis")
    st.markdown(
        "**Cohort:** melanoma, male, responder, time = 0, across **all treatments and "
        "all sample types**."
    )
    result, coverage = load_bcell_analysis(database_path)
    average = float(result.iloc[0]["average_b_cell_count"])
    sample_count = int(result.iloc[0]["sample_count"])
    columns = st.columns(2)
    columns[0].metric("Average B-cell count", f"{average:,.2f}")
    columns[1].metric("Samples included", f"{sample_count:,}")
    st.caption("This calculation uses raw B-cell counts, not relative frequencies.")

    coverage_display = coverage.copy()
    coverage_display["cohort"] = (
        coverage_display["treatment"] + " + " + coverage_display["sample_type"]
    )
    chart = px.bar(
        coverage_display,
        x="cohort",
        y="sample_count",
        text="sample_count",
        labels={"cohort": "Treatment and sample type", "sample_count": "Samples"},
        title="Samples contributing to the B-cell average",
        color="treatment",
    )
    chart.update_traces(textposition="outside")
    st.plotly_chart(chart, use_container_width=True)


def main() -> None:
    st.set_page_config(
        page_title="Loblaw Bio Immune Cell Analysis",
        page_icon="🧬",
        layout="wide",
    )
    st.title("Loblaw Bio Immune Cell Analysis")
    st.write(
        "Explore immune-cell population frequencies, treatment-response comparisons, "
        "and assessment-defined baseline cohorts. Results are exploratory and do not "
        "establish that any cell population predicts treatment response."
    )

    if not DATABASE_PATH.is_file():
        st.error("Database not found. Run `python load_data.py` first.")
        st.stop()

    try:
        database_path = str(DATABASE_PATH)
        render_overview(database_path)
        st.divider()
        render_sample_analysis(database_path)
        st.divider()
        render_response_analysis(database_path)
        st.divider()
        render_baseline_analysis(database_path)
        st.divider()
        render_bcell_analysis(database_path)
    except Exception as error:
        st.error(f"Dashboard data could not be loaded: {error}")
        st.info("Rebuild the database with `python load_data.py`, then refresh this page.")
        st.stop()


if __name__ == "__main__":
    main()
