"""Validate cell-count.csv and rebuild the normalized SQLite database."""

from __future__ import annotations

import math
import os
import sqlite3
from pathlib import Path

import pandas as pd


ROOT_DIR = Path(__file__).resolve().parent
CSV_PATH = ROOT_DIR / "cell-count.csv"
DATABASE_PATH = ROOT_DIR / "cell_counts.db"

REQUIRED_COLUMNS = (
    "project",
    "subject",
    "condition",
    "age",
    "sex",
    "treatment",
    "response",
    "sample",
    "sample_type",
    "time_from_treatment_start",
    "b_cell",
    "cd8_t_cell",
    "cd4_t_cell",
    "nk_cell",
    "monocyte",
)
SUBJECT_COLUMNS = ("project", "condition", "age", "sex", "treatment", "response")
POPULATION_NAMES = ("b_cell", "cd8_t_cell", "cd4_t_cell", "nk_cell", "monocyte")
INTEGER_COLUMNS = ("age", "time_from_treatment_start", *POPULATION_NAMES)
EXPECTED_COUNTS = {
    "projects": 3,
    "subjects": 3_500,
    "samples": 10_500,
    "cell_populations": 5,
    "cell_counts": 52_500,
}


SCHEMA_SQL = """
CREATE TABLE projects (
    project_id TEXT PRIMARY KEY
        CHECK (length(trim(project_id)) > 0)
);

CREATE TABLE subjects (
    subject_id TEXT PRIMARY KEY
        CHECK (length(trim(subject_id)) > 0),
    project_id TEXT NOT NULL,
    condition TEXT NOT NULL CHECK (length(trim(condition)) > 0),
    age INTEGER NOT NULL CHECK (age BETWEEN 0 AND 130),
    sex TEXT NOT NULL CHECK (sex IN ('M', 'F')),
    treatment TEXT NOT NULL CHECK (length(trim(treatment)) > 0),
    response TEXT CHECK (response IS NULL OR response IN ('yes', 'no')),
    FOREIGN KEY (project_id) REFERENCES projects (project_id)
        ON UPDATE CASCADE ON DELETE RESTRICT
);

CREATE TABLE samples (
    sample_id TEXT PRIMARY KEY
        CHECK (length(trim(sample_id)) > 0),
    subject_id TEXT NOT NULL,
    sample_type TEXT NOT NULL CHECK (length(trim(sample_type)) > 0),
    time_from_treatment_start INTEGER NOT NULL
        CHECK (time_from_treatment_start >= 0),
    FOREIGN KEY (subject_id) REFERENCES subjects (subject_id)
        ON UPDATE CASCADE ON DELETE CASCADE
);

CREATE TABLE cell_populations (
    population_id INTEGER PRIMARY KEY,
    population_name TEXT NOT NULL UNIQUE
        CHECK (length(trim(population_name)) > 0)
);

CREATE TABLE cell_counts (
    sample_id TEXT NOT NULL,
    population_id INTEGER NOT NULL,
    cell_count INTEGER NOT NULL CHECK (cell_count >= 0),
    PRIMARY KEY (sample_id, population_id),
    FOREIGN KEY (sample_id) REFERENCES samples (sample_id)
        ON UPDATE CASCADE ON DELETE CASCADE,
    FOREIGN KEY (population_id) REFERENCES cell_populations (population_id)
        ON UPDATE CASCADE ON DELETE RESTRICT
);

CREATE INDEX idx_subjects_project_id
    ON subjects (project_id);
CREATE INDEX idx_subjects_condition_treatment_response
    ON subjects (condition, treatment, response);
CREATE INDEX idx_subjects_condition_response_sex
    ON subjects (condition, response, sex);
CREATE INDEX idx_samples_subject_id
    ON samples (subject_id);
CREATE INDEX idx_samples_sample_type_time_subject
    ON samples (sample_type, time_from_treatment_start, subject_id);
CREATE INDEX idx_cell_counts_population_sample
    ON cell_counts (population_id, sample_id);
"""


def _require_nonempty_strings(dataframe: pd.DataFrame, columns: tuple[str, ...]) -> None:
    for column in columns:
        values = dataframe[column]
        invalid = values.isna() | values.astype(str).str.strip().eq("")
        if invalid.any():
            rows = (invalid[invalid].index[:5] + 2).tolist()
            raise ValueError(f"Column {column!r} contains empty values at CSV rows {rows}.")


def _validate_and_convert_integers(dataframe: pd.DataFrame) -> None:
    for column in INTEGER_COLUMNS:
        numeric = pd.to_numeric(dataframe[column], errors="coerce")
        invalid = numeric.isna() | ~numeric.map(lambda value: math.isfinite(value))
        non_integer = numeric.notna() & numeric.mod(1).ne(0)
        if invalid.any() or non_integer.any():
            bad = invalid | non_integer
            rows = (bad[bad].index[:5] + 2).tolist()
            raise ValueError(
                f"Column {column!r} must contain finite integers; invalid CSV rows: {rows}."
            )
        dataframe[column] = numeric.astype("int64")


def validate_dataframe(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Validate source values and return a type-normalized copy."""
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in dataframe.columns]
    if missing_columns:
        raise ValueError(f"CSV is missing required columns: {', '.join(missing_columns)}")

    dataframe = dataframe.loc[:, REQUIRED_COLUMNS].copy()
    _require_nonempty_strings(
        dataframe,
        (
            "project",
            "subject",
            "condition",
            "sex",
            "treatment",
            "sample",
            "sample_type",
        ),
    )
    _validate_and_convert_integers(dataframe)

    if dataframe["sample"].duplicated().any():
        duplicate_ids = dataframe.loc[dataframe["sample"].duplicated(False), "sample"].unique()
        raise ValueError(f"Sample IDs must be unique; duplicates include: {duplicate_ids[:5].tolist()}")

    invalid_ages = ~dataframe["age"].between(0, 130)
    if invalid_ages.any():
        raise ValueError("Age values must be integers between 0 and 130.")

    invalid_times = dataframe["time_from_treatment_start"].lt(0)
    if invalid_times.any():
        raise ValueError("time_from_treatment_start values must be non-negative integers.")

    invalid_sex = ~dataframe["sex"].isin(("M", "F"))
    if invalid_sex.any():
        values = sorted(dataframe.loc[invalid_sex, "sex"].astype(str).unique())
        raise ValueError(f"Sex values must be 'M' or 'F'; found: {values}")

    # Empty CSV fields are parsed as NaN and are valid only for response.
    invalid_response = dataframe["response"].notna() & ~dataframe["response"].isin(("yes", "no"))
    if invalid_response.any():
        values = sorted(dataframe.loc[invalid_response, "response"].astype(str).unique())
        raise ValueError(f"Response values must be 'yes', 'no', or NULL; found: {values}")

    for population in POPULATION_NAMES:
        if dataframe[population].lt(0).any():
            raise ValueError(f"Cell counts in {population!r} must be non-negative.")

    grouped = dataframe.groupby("subject", sort=False, dropna=False)
    conflicts: list[str] = []
    for column in SUBJECT_COLUMNS:
        inconsistent = grouped[column].nunique(dropna=False).gt(1)
        if inconsistent.any():
            examples = inconsistent[inconsistent].index[:5].tolist()
            conflicts.append(f"{column}: {examples}")
    if conflicts:
        raise ValueError(
            "Conflicting subject-level metadata detected (column: example subjects): "
            + "; ".join(conflicts)
        )

    return dataframe


def create_schema(connection: sqlite3.Connection) -> None:
    connection.executescript(SCHEMA_SQL)


def seed_populations(connection: sqlite3.Connection) -> dict[str, int]:
    rows = [(index, name) for index, name in enumerate(POPULATION_NAMES, start=1)]
    connection.executemany(
        "INSERT INTO cell_populations (population_id, population_name) VALUES (?, ?)",
        rows,
    )
    return {name: population_id for population_id, name in rows}


def load_projects(connection: sqlite3.Connection, dataframe: pd.DataFrame) -> None:
    rows = [(project,) for project in dataframe["project"].drop_duplicates()]
    connection.executemany("INSERT INTO projects (project_id) VALUES (?)", rows)


def load_subjects(connection: sqlite3.Connection, dataframe: pd.DataFrame) -> None:
    subject_rows = dataframe.drop_duplicates("subject")
    rows = [
        (
            row.subject,
            row.project,
            row.condition,
            int(row.age),
            row.sex,
            row.treatment,
            None if pd.isna(row.response) else row.response,
        )
        for row in subject_rows.itertuples(index=False)
    ]
    connection.executemany(
        """
        INSERT INTO subjects
            (subject_id, project_id, condition, age, sex, treatment, response)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )


def load_samples(connection: sqlite3.Connection, dataframe: pd.DataFrame) -> None:
    rows = [
        (row.sample, row.subject, row.sample_type, int(row.time_from_treatment_start))
        for row in dataframe.itertuples(index=False)
    ]
    connection.executemany(
        """
        INSERT INTO samples
            (sample_id, subject_id, sample_type, time_from_treatment_start)
        VALUES (?, ?, ?, ?)
        """,
        rows,
    )


def load_cell_counts(
    connection: sqlite3.Connection,
    dataframe: pd.DataFrame,
    population_ids: dict[str, int],
) -> None:
    rows = (
        (row.sample, population_ids[population], int(getattr(row, population)))
        for row in dataframe.itertuples(index=False)
        for population in POPULATION_NAMES
    )
    connection.executemany(
        """
        INSERT INTO cell_counts (sample_id, population_id, cell_count)
        VALUES (?, ?, ?)
        """,
        rows,
    )


def verify_database(connection: sqlite3.Connection) -> dict[str, int]:
    counts = {
        table: connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        for table in EXPECTED_COUNTS
    }
    mismatches = {
        table: (counts[table], expected)
        for table, expected in EXPECTED_COUNTS.items()
        if counts[table] != expected
    }
    if mismatches:
        details = ", ".join(
            f"{table}={actual} (expected {expected})"
            for table, (actual, expected) in mismatches.items()
        )
        raise ValueError(f"Unexpected post-load row counts: {details}")

    orphan_samples = connection.execute(
        """
        SELECT COUNT(*) FROM samples AS s
        LEFT JOIN subjects AS sub ON sub.subject_id = s.subject_id
        WHERE sub.subject_id IS NULL
        """
    ).fetchone()[0]
    orphan_counts = connection.execute(
        """
        SELECT COUNT(*) FROM cell_counts AS cc
        LEFT JOIN samples AS s ON s.sample_id = cc.sample_id
        LEFT JOIN cell_populations AS cp ON cp.population_id = cc.population_id
        WHERE s.sample_id IS NULL OR cp.population_id IS NULL
        """
    ).fetchone()[0]
    samples_without_five = connection.execute(
        """
        SELECT COUNT(*) FROM (
            SELECT s.sample_id
            FROM samples AS s
            LEFT JOIN cell_counts AS cc ON cc.sample_id = s.sample_id
            GROUP BY s.sample_id
            HAVING COUNT(cc.population_id) <> 5
        )
        """
    ).fetchone()[0]
    duplicate_counts = connection.execute(
        """
        SELECT COUNT(*) FROM (
            SELECT sample_id, population_id
            FROM cell_counts
            GROUP BY sample_id, population_id
            HAVING COUNT(*) > 1
        )
        """
    ).fetchone()[0]
    foreign_key_violations = connection.execute("PRAGMA foreign_key_check").fetchall()

    failures = {
        "orphan samples": orphan_samples,
        "orphan cell counts": orphan_counts,
        "samples without exactly five counts": samples_without_five,
        "duplicate sample/population pairs": duplicate_counts,
        "foreign-key violations": len(foreign_key_violations),
    }
    failures = {name: count for name, count in failures.items() if count}
    if failures:
        raise ValueError(f"Database integrity verification failed: {failures}")

    return counts


def build_database(
    dataframe: pd.DataFrame,
    database_path: Path = DATABASE_PATH,
) -> dict[str, int]:
    """Build and atomically replace *database_path* from validated source data."""
    database_path = Path(database_path).resolve()
    temporary_database_path = database_path.with_name(f".{database_path.name}.tmp")
    database_path.parent.mkdir(parents=True, exist_ok=True)
    if temporary_database_path.exists():
        temporary_database_path.unlink()

    connection: sqlite3.Connection | None = None
    try:
        connection = sqlite3.connect(temporary_database_path)
        connection.execute("PRAGMA foreign_keys = ON")
        if connection.execute("PRAGMA foreign_keys").fetchone()[0] != 1:
            raise RuntimeError("SQLite foreign-key enforcement could not be enabled.")

        create_schema(connection)
        with connection:
            population_ids = seed_populations(connection)
            load_projects(connection, dataframe)
            load_subjects(connection, dataframe)
            load_samples(connection, dataframe)
            load_cell_counts(connection, dataframe, population_ids)
            counts = verify_database(connection)
        connection.close()
        connection = None

        # Atomic replacement preserves an existing valid database if rebuilding fails.
        os.replace(temporary_database_path, database_path)
        return counts
    except Exception:
        if connection is not None:
            connection.rollback()
            connection.close()
        if temporary_database_path.exists():
            temporary_database_path.unlink()
        raise


def main() -> None:
    if not CSV_PATH.is_file():
        raise FileNotFoundError(f"Source CSV not found: {CSV_PATH}")

    dataframe = pd.read_csv(CSV_PATH)
    validated = validate_dataframe(dataframe)
    counts = build_database(validated)

    print(f"Database created successfully: {DATABASE_PATH}")
    print()
    print(f"Projects: {counts['projects']:,}")
    print(f"Subjects: {counts['subjects']:,}")
    print(f"Samples: {counts['samples']:,}")
    print(f"Cell populations: {counts['cell_populations']:,}")
    print(f"Cell counts: {counts['cell_counts']:,}")
    print()
    print("Data validation passed.")


if __name__ == "__main__":
    main()
