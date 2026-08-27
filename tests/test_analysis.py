"""Regression tests for Parts 2-4 and dashboard import safety."""

from __future__ import annotations

import importlib
import sqlite3
import sys
from pathlib import Path

import pandas as pd
import pytest

import analysis


@pytest.fixture(scope="module")
def frequency_summary(test_database_path) -> pd.DataFrame:
    with sqlite3.connect(test_database_path) as connection:
        summary = analysis.get_frequency_summary(connection)
    analysis.validate_frequency_summary(summary)
    return summary


@pytest.fixture(scope="module")
def response_analysis(test_database_path, frequency_summary):
    with sqlite3.connect(test_database_path) as connection:
        metadata = analysis.get_analysis_metadata(connection)
    comparison = analysis.build_response_comparison_data(frequency_summary, metadata)
    subject_means = analysis.aggregate_subject_frequencies(comparison)
    results = analysis.run_population_statistics(subject_means)
    return comparison, subject_means, results


def test_sample00000_frequency_values(frequency_summary: pd.DataFrame) -> None:
    sample = frequency_summary.loc[frequency_summary["sample"].eq("sample00000")]
    assert sample["total_count"].unique().tolist() == [93_214]
    expected_counts = {
        "b_cell": 10_908,
        "cd8_t_cell": 24_440,
        "cd4_t_cell": 20_491,
        "nk_cell": 13_864,
        "monocyte": 23_511,
    }
    expected_percentages = {
        "b_cell": 11.7021048340,
        "cd8_t_cell": 26.2192374536,
        "cd4_t_cell": 21.9827493724,
        "nk_cell": 14.8733022936,
        "monocyte": 25.2226060463,
    }
    actual = sample.set_index("population")
    assert actual["count"].to_dict() == expected_counts
    for population, expected in expected_percentages.items():
        assert actual.loc[population, "percentage"] == pytest.approx(expected, abs=1e-9)
    assert sample["percentage"].sum() == pytest.approx(100.0, abs=1e-12)


def test_frequency_summary_cardinality(frequency_summary: pd.DataFrame) -> None:
    assert frequency_summary.shape == (52_500, 5)
    assert frequency_summary["sample"].nunique() == 10_500
    assert frequency_summary.groupby("sample").size().eq(5).all()


def test_response_cohort_filters_and_counts(response_analysis) -> None:
    comparison, _, _ = response_analysis
    assert set(comparison["condition"]) == {"melanoma"}
    assert set(comparison["treatment"]) == {"miraclib"}
    assert set(comparison["sample_type"]) == {"PBMC"}
    assert set(comparison["response"]) == {"yes", "no"}
    subjects = comparison[["subject", "response"]].drop_duplicates()
    assert subjects["subject"].nunique() == 656
    assert subjects["response"].value_counts().to_dict() == {"yes": 331, "no": 325}
    assert comparison["sample"].nunique() == 1_968


def test_subject_level_aggregation(response_analysis) -> None:
    _, subject_means, _ = response_analysis
    assert len(subject_means) == 656 * 5
    assert not subject_means.duplicated(["subject", "population"]).any()
    assert set(subject_means["population"]) == set(analysis.POPULATION_ORDER)


def test_adjusted_statistical_results(response_analysis) -> None:
    _, _, results = response_analysis
    indexed = results.set_index("population")
    expected_adjusted = {
        "b_cell": 0.432245,
        "cd8_t_cell": 0.622144,
        "cd4_t_cell": 0.062111,
        "nk_cell": 0.316862,
        "monocyte": 0.432245,
    }
    assert set(indexed.index) == set(analysis.POPULATION_ORDER)
    for population, expected in expected_adjusted.items():
        assert indexed.loc[population, "adjusted_p_value"] == pytest.approx(expected, abs=1e-6)
    assert results["significant"].sum() == 0
    assert indexed.loc["cd4_t_cell", "p_value"] == pytest.approx(0.012422, abs=1e-6)
    assert not bool(indexed.loc["cd4_t_cell", "significant"])


def test_baseline_cohort_and_grouped_counts(test_database_path) -> None:
    with sqlite3.connect(test_database_path) as connection:
        cohort = analysis.get_baseline_melanoma_miraclib_pbmc(connection)
        projects = analysis.get_baseline_samples_by_project(connection)
        responses = analysis.get_baseline_subjects_by_response(connection)
        sexes = analysis.get_baseline_subjects_by_sex(connection)
    analysis.validate_part4_results(cohort, projects, responses, sexes)
    assert len(cohort) == 656
    assert cohort["subject"].nunique() == 656
    assert set(cohort["condition"]) == {"melanoma"}
    assert set(cohort["treatment"]) == {"miraclib"}
    assert set(cohort["sample_type"]) == {"PBMC"}
    assert set(cohort["time_from_treatment_start"]) == {0}
    assert projects.set_index("project")["sample_count"].to_dict() == {"prj1": 384, "prj3": 272}
    assert responses.set_index("response")["subject_count"].to_dict() == {"no": 325, "yes": 331}
    assert sexes.set_index("sex")["subject_count"].to_dict() == {"F": 312, "M": 344}
    assert projects["sample_count"].sum() == 656
    assert responses["subject_count"].sum() == 656
    assert sexes["subject_count"].sum() == 656


def test_special_bcell_query_includes_all_treatments_and_sample_types(
    test_database_path,
) -> None:
    with sqlite3.connect(test_database_path) as connection:
        result, audit = analysis.get_melanoma_male_responder_baseline_bcell(connection)
    row = result.iloc[0]
    assert int(row["sample_count"]) == 485
    assert row["average_b_cell_count"] == pytest.approx(10_206.150515463918, abs=1e-10)
    assert f"{row['average_b_cell_count']:.2f}" == "10206.15"
    assert audit == {"treatment_count": 2, "sample_type_count": 2}


def test_analysis_generates_all_outputs_in_temporary_directory(
    test_database_path,
    tmp_path: Path,
) -> None:
    with sqlite3.connect(test_database_path) as connection:
        summary = analysis.get_frequency_summary(connection)
        metadata = analysis.get_analysis_metadata(connection)
    analysis.validate_frequency_summary(summary)
    analysis.save_frequency_summary(summary, tmp_path / "summary_table.csv")
    comparison = analysis.build_response_comparison_data(summary, metadata)
    subject_means = analysis.aggregate_subject_frequencies(comparison)
    results = analysis.run_population_statistics(subject_means)
    analysis.save_statistical_results(results, tmp_path / "statistical_results.csv")
    analysis.create_response_boxplot(
        subject_means,
        tmp_path / "responder_vs_nonresponder_boxplot.png",
    )
    with sqlite3.connect(test_database_path) as connection:
        analysis.run_part4_analysis(connection, tmp_path)

    expected = {
        "summary_table.csv",
        "statistical_results.csv",
        "responder_vs_nonresponder_boxplot.png",
        "baseline_melanoma_miraclib_pbmc.csv",
        "baseline_samples_by_project.csv",
        "baseline_subjects_by_response.csv",
        "baseline_subjects_by_sex.csv",
        "melanoma_male_responder_baseline_bcell.csv",
    }
    assert {path.name for path in tmp_path.iterdir()} == expected
    assert all((tmp_path / name).stat().st_size > 0 for name in expected)


def test_dashboard_import_does_not_rewrite_pipeline_outputs() -> None:
    output_files = sorted(analysis.OUTPUT_PATH.parent.glob("*"))
    before = {path: path.stat().st_mtime_ns for path in output_files}
    sys.modules.pop("dashboard", None)
    imported = importlib.import_module("dashboard")
    after = {path: path.stat().st_mtime_ns for path in output_files}
    assert imported.DATABASE_PATH == analysis.DATABASE_PATH
    assert after == before
