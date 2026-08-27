# Loblaw Bio Immune Cell Analysis

## Overview

This project analyzes immune-cell population data from a clinical-trial-style dataset. SQLite provides normalized relational storage, Python performs validation and analysis, and Streamlit provides an interactive dashboard. The implementation answers Parts 1–4 of the supplied technical assessment and includes reproducible outputs and automated tests.

## Quick Start

From the repository root in GitHub Codespaces or another Python environment:

```bash
make setup
make pipeline
make dashboard
```

Open the URL printed by Streamlit. In Codespaces, use the forwarded-port link for port 8501. Stop the dashboard with `Ctrl+C`.

Run the automated tests with:

```bash
make test
```

## Live Dashboard

[Open the deployed Streamlit dashboard](https://YOUR-DEPLOYED-DASHBOARD-URL)

> Replace the placeholder URL above after deploying the application.

## Project Structure

```text
.
├── cell-count.csv                 # Source dataset
├── cell_counts.db                 # Generated SQLite database
├── load_data.py                   # Validation, schema creation, and loading
├── analysis.py                    # Parts 2–4 analysis pipeline
├── dashboard.py                   # Streamlit dashboard
├── requirements.txt               # Python dependencies
├── Makefile                       # Reproducible project commands
├── tests/                         # Pytest suite
└── outputs/                       # Generated tables and plot
```

## Dataset

The source file contains 10,500 samples from 3,500 subjects across three projects. Every subject has samples at times 0, 7, and 14. Each sample contains raw counts for:

- `b_cell`
- `cd8_t_cell`
- `cd4_t_cell`
- `nk_cell`
- `monocyte`

The CSV uses `condition` for the assessment's indication concept, `sex` for gender, and `sample` for sample ID. Healthy untreated subjects have a null response.

## Relational Architecture

```text
projects
   └──< subjects
          └──< samples
                 └──< cell_counts >── cell_populations
```

| Table | Responsibility |
|---|---|
| `projects` | Project identifiers |
| `subjects` | Project, condition, age, sex, treatment, and response |
| `samples` | Subject, sample type, and time from treatment start |
| `cell_populations` | Extensible population definitions |
| `cell_counts` | One non-negative count per sample and population |

`sample_type` is stored at the sample level because it describes collected biological material and may vary for a subject in future datasets. Treatment and response are stored at the subject level because they are constant per subject in the supplied data. The age field represents the value supplied by this dataset; a longitudinal real-world system might instead model date of birth or age at a defined event.

The long-form `cell_counts` table permits additional populations without altering the sample schema. Primary keys, foreign keys, uniqueness constraints, checks, and query-oriented indexes protect integrity and support the assessment filters.

## Pipeline

### Database loading

```bash
python3 load_data.py
```

The loader validates required columns, types, identifiers, subject-level consistency, categorical values, ages, times, and non-negative cell counts. It builds a temporary database, verifies referential integrity and expected counts, then atomically replaces `cell_counts.db`. Repeated runs are deterministic and idempotent.

### Analysis

```bash
python3 analysis.py
```

The analysis reads SQLite as its source of truth and performs:

1. Relative-frequency calculation for every sample and population.
2. Responder versus non-responder comparison for melanoma subjects receiving miraclib with PBMC samples.
3. Assessment-defined baseline subset queries.
4. The separate baseline raw B-cell query across all treatments and sample types.

## Analytical Methodology

For each sample:

```text
total_count = sum of the five population counts
percentage = population count / total_count × 100
```

For the response comparison, samples are restricted to melanoma subjects receiving miraclib with PBMC samples and a `yes` or `no` response. Because each subject has repeated measurements, percentages are averaged across eligible samples to one value per subject and population. Responders and non-responders are compared using two-sided Mann–Whitney U tests. Benjamini–Hochberg false-discovery-rate correction is applied across the five populations at α = 0.05.

This is exploratory group-comparison analysis. It does not establish predictive performance or causality.

## Key Results

### Relative frequencies

- 10,500 samples analyzed
- 52,500 sample/population rows
- Five population percentages per sample, summing to approximately 100%

### Response comparison

- Eligible subjects: 656
- Responders: 331
- Non-responders: 325
- Eligible samples: 1,968
- No population was significant after Benjamini–Hochberg correction
- `cd4_t_cell` had the smallest raw p-value (approximately 0.0124), but its adjusted p-value was approximately 0.0621 and therefore above 0.05

### Baseline melanoma + miraclib + PBMC cohort

- Samples: 656
- Subjects: 656
- Project counts: `prj1` = 384, `prj3` = 272
- Response counts: `no` = 325, `yes` = 331
- Sex counts: `F` = 312, `M` = 344

### Baseline B-cell result

For melanoma male responders at time 0 across all treatments and sample types:

- Samples included: 485
- Average raw B-cell count: **10,206.15**

This calculation uses raw B-cell counts, not relative frequencies, and intentionally includes both treatments and both sample types.

## Dashboard

```bash
make dashboard
```

The dashboard includes:

- Database-derived project, subject, sample, and population metrics
- An interactive sample frequency explorer
- Subject-level responder/non-responder boxplots and statistical results
- Fixed baseline cohort summaries
- The separate raw B-cell result and its treatment/sample-type coverage

Dashboard controls do not change the assessment's fixed Part 3 and Part 4 cohorts.

## Generated Outputs

| Output | Description |
|---|---|
| `outputs/summary_table.csv` | Long-form raw counts and relative frequencies |
| `outputs/statistical_results.csv` | Mann–Whitney and adjusted p-values |
| `outputs/responder_vs_nonresponder_boxplot.png` | Subject-level comparison plot |
| `outputs/baseline_melanoma_miraclib_pbmc.csv` | Baseline sample cohort |
| `outputs/baseline_samples_by_project.csv` | Baseline sample counts by project |
| `outputs/baseline_subjects_by_response.csv` | Distinct subjects by response |
| `outputs/baseline_subjects_by_sex.csv` | Distinct subjects by sex |
| `outputs/melanoma_male_responder_baseline_bcell.csv` | Special raw B-cell result |

## Tests

The focused pytest suite validates database creation and idempotency, foreign keys, loader error cases, frequency calculations, cohort filters, repeated-measure aggregation, statistical regression values, Part 4 SQL results, output generation, and dashboard import safety.

```bash
make test
# or
python3 -m pytest -q
```

The current suite contains 15 tests.

## Scalability and Further Development

The normalized project -> subject -> sample hierarchy avoids repeated metadata, while long-form cell counts allow new populations without schema changes. Existing indexes support project joins and the assessment's condition, treatment, response, sex, sample-type, time, and population filters.

For substantially larger workloads, the same model could move to PostgreSQL, loading could use batched or bulk-copy operations, and analytical tables could be materialized or served through an API. Future clinical data may also require treatment histories, dated response assessments, assay metadata, and more explicit age semantics.
