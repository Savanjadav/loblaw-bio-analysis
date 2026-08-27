"""Tests for CSV validation and normalized SQLite loading."""

from __future__ import annotations

import sqlite3

import pandas as pd
import pytest

from load_data import EXPECTED_COUNTS, build_database, validate_dataframe


EXPECTED_TABLES = {
    "projects",
    "subjects",
    "samples",
    "cell_populations",
    "cell_counts",
}


def test_database_schema_counts_and_integrity(test_database_path) -> None:
    with sqlite3.connect(test_database_path) as connection:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            )
        }
        assert tables == EXPECTED_TABLES
        for table, expected in EXPECTED_COUNTS.items():
            assert connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == expected

        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
        assert connection.execute(
            "SELECT COUNT(*) = COUNT(DISTINCT sample_id) FROM samples"
        ).fetchone()[0]
        invalid_count_groups = connection.execute(
            """
            SELECT COUNT(*) FROM (
                SELECT sample_id FROM cell_counts
                GROUP BY sample_id HAVING COUNT(*) <> 5
            )
            """
        ).fetchone()[0]
        assert invalid_count_groups == 0


def test_database_build_is_idempotent(test_database_path, validated_source) -> None:
    first_counts = build_database(validated_source, test_database_path)
    second_counts = build_database(validated_source, test_database_path)
    assert first_counts == EXPECTED_COUNTS
    assert second_counts == EXPECTED_COUNTS
    with sqlite3.connect(test_database_path) as connection:
        assert {
            table: connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in EXPECTED_COUNTS
        } == EXPECTED_COUNTS


def test_missing_required_column_fails(validated_source: pd.DataFrame) -> None:
    invalid = validated_source.drop(columns="monocyte")
    with pytest.raises(ValueError, match="missing required columns: monocyte"):
        validate_dataframe(invalid)


def test_duplicate_sample_id_fails(validated_source: pd.DataFrame) -> None:
    invalid = validated_source.copy()
    invalid.loc[invalid.index[1], "sample"] = invalid.loc[invalid.index[0], "sample"]
    with pytest.raises(ValueError, match="Sample IDs must be unique"):
        validate_dataframe(invalid)


def test_negative_cell_count_fails(validated_source: pd.DataFrame) -> None:
    invalid = validated_source.copy()
    invalid.loc[invalid.index[0], "b_cell"] = -1
    with pytest.raises(ValueError, match="must be non-negative"):
        validate_dataframe(invalid)


def test_inconsistent_subject_metadata_fails(validated_source: pd.DataFrame) -> None:
    invalid = validated_source.copy()
    invalid.loc[invalid.index[1], "condition"] = "conflicting_condition"
    with pytest.raises(ValueError, match="Conflicting subject-level metadata"):
        validate_dataframe(invalid)
