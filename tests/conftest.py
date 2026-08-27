"""Shared pytest fixtures for isolated database and analysis tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from load_data import CSV_PATH, build_database, validate_dataframe


@pytest.fixture(scope="session")
def validated_source() -> pd.DataFrame:
    return validate_dataframe(pd.read_csv(CSV_PATH))


@pytest.fixture(scope="session")
def test_database_path(
    tmp_path_factory: pytest.TempPathFactory,
    validated_source: pd.DataFrame,
) -> Path:
    database_path = tmp_path_factory.mktemp("database") / "cell_counts.db"
    build_database(validated_source, database_path)
    return database_path
