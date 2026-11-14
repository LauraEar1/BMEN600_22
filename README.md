# CAP Sleep Database classification pipeline

This repository contains a reproducible, subject-level pipeline for the CAP
Sleep Database (PhysioNet version 1.0.0). The original verified inventory has
88 selected recordings. Versioned primary-analysis cohorts in
`config/analysis_cohorts/` do not alter the source manifest. The current
proposed cohort is `primary_physiology_84_v2.json`; its exclusions are
documented. The three intended Random Forest experiments use the same
eligible participants and cross-validation splits:

1. EEG
2. EEG + EMG + EOG
3. EEG + EMG + EOG + ECG

The primary experiments use physiological features only. Clinician CAP and
sleep-stage annotations are held out for separate descriptive analyses and a
separate CAP-informed classification experiment. The secondary classifier
compares an EEG-only baseline with EEG plus CAP-derived features on the same
participants and folds. This is a research pipeline, not a clinical diagnostic
tool.

## Cohort verification

`config/cohort_manifest.json` records the selected cohort and the SHA-256 of
the official PhysioNet `RECORDS` inventory. The target is exactly 16 controls
(`n1`–`n16`), 40 NFLE, 22 RBD, and 10 PLM recordings. The remaining 20
database records are intentionally outside this experiment. Run
`python -m capsleep.cli verify-cohort` to compare the checked-in manifest with
the live inventory; verification fails if any selected record is missing or
if the expected class counts change.

## Environment and installation

Use Python 3.13 on Windows. From the repository root in the VS Code terminal:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install -e .
```

The EDF reader is MNE-Python; no compiled `pyedflib` dependency is used.
Recordings are streamed one at a time to an owned temporary file, read in
bounded windows, and removed after successful extraction. Available disk space
is checked against the remote EDF size plus a configurable reserve before each
download. Existing/user-provided EDF files are never overwritten or removed.

## Commands

```powershell
.\.venv\Scripts\python.exe -m capsleep.cli verify-cohort
.\.venv\Scripts\python.exe -m capsleep.cli record-inventory
.\.venv\Scripts\python.exe -m capsleep.cli validate-annotations
.\.venv\Scripts\python.exe -m capsleep.cli diagnose-annotations
.\.venv\Scripts\python.exe -m capsleep.cli extract
.\.venv\Scripts\python.exe -m capsleep.cli validate-schema
.\.venv\Scripts\python.exe -m capsleep.cli validate-analysis-cohort --cohort config\analysis_cohorts\primary_physiology_84_v2.json
.\.venv\Scripts\python.exe -m capsleep.cli train-analysis-cohort --cohort config\analysis_cohorts\primary_physiology_84_v2.json
.\.venv\Scripts\python.exe -m capsleep.cli posthoc-primary
.\.venv\Scripts\python.exe -m capsleep.cli train-primary
.\.venv\Scripts\python.exe -m capsleep.cli analyze-cap
.\.venv\Scripts\python.exe -m capsleep.cli train-cap
```

Extraction checkpoints each recording in `results/checkpoints/` and resumes
from verified completed records. Failures are recorded with available channel
QC evidence, and processing continues sequentially so later recordings are
attempted. The run reports all failures and does not write final feature
datasets unless all 88 recordings pass required extraction checks.
Physiology extraction never downloads or parses CAP annotations and never
modifies CAP validation status. CAP validation and its separate diagnostics
can fail without blocking the primary physiological features. Each successful
recording has a compact subject-level JSON checkpoint and compressed
window-level feature checkpoint. Only pipeline-named temporary EDF files are
deleted, and only after the physiological checkpoint is safely written.
`train-primary` requires an up-to-date successful schema-validation report.
The three models share one deterministic `StratifiedGroupKFold` split plan,
with subject IDs as groups. Imputation is part of each fold's estimator and is
fit on that fold's training data only. No feature selection is performed.
Each analysis cohort is subject to its own checkpoint and modality eligibility
preflight. It must report `eligible` before classification can proceed; a
preflight failure is not permission to drop additional participants.

## Outputs and scientific safeguards

- `results/features/physiology_features.csv.gz`: one row per recording;
  physiology only.
- `results/features/physiology_window_features.csv.gz`: one row per
  recording/channel/30-second window, with feature values and explicit
  flatness, non-finite, saturation, and channel-exclusion flags.
- `results/features/physiology_feature_schema.json`: feature names, normalized
  units, modality-specific calculations, and subject-level aggregation rules.
- `results/features/cap_features.csv.gz`: independent annotation-derived
  features for secondary analyses.
- `results/qc/`: per-recording channel inventory, QC summaries, failures, and
  annotation alignment diagnostics.
- `results/qc/<analysis_id>_preflight.json` and
  `<analysis_id>_record_validation.csv`: provisional cohort
  eligibility, window-count, duration, and aggregation checks.
- `results/features/physiology_primary_<version>_candidate.csv.gz`: one
  checkpoint-aggregated subject row per cohort participant; this is not an
  approved model input unless the matching preflight reports `eligible`.
- `results/checkpoints/records/`: per-recording subject and window checkpoints;
  `results/qc/extraction_failures.csv.gz` records failed attempts.
- `results/models/primary/`: fold assignments, out-of-fold predictions,
  balanced accuracy, macro-F1, per-class metrics, and confusion matrices for
  all three modality sets.
- `results/models/primary_physiology_84_v2/`: versioned v2 fold assignments,
  fold-level metrics and out-of-fold predictions for the approved 84-subject
  cohort, with reproducibility hashes in `metrics.json`.
- `results/analysis/posthoc_primary_84_v2/`: isolated exploratory analyses
  based on the saved v2 folds and out-of-fold predictions. The command checks
  model alignment and reproduces the saved baseline predictions before
  proceeding; it does not retrain or overwrite baseline result files. Outputs
  include paired participant correctness and exact McNemar tables, paired
  class-stratified subject-bootstrap intervals, missingness summaries, detailed
  training-fold all-missing feature/imputation audits, train-only harmonized
  feature sensitivity predictions, held-out grouped modality permutation
  importance, acquisition metadata associations and metadata-only CV, figures,
  and a reproducibility report. These analyses are exploratory, use no CAP
  annotations, and are not a replacement for the prespecified primary results.
- `results/models/cap/`: separate CAP-informed classification results.

Each channel is read at its native EDF sampling rate. Filters are modality
specific; the extraction schema tracks channel labels, units, sampling rates,
missing modalities, and QC exclusions. MNE's physical-unit conversion is
normalized to microvolts for EEG/EOG/EMG features and millivolts for ECG;
header units remain in channel QC metadata. Feature definitions and units are
machine-readable in `physiology_feature_schema.json`. Subject features are
per-channel medians and interquartile ranges of non-flat, complete 30-second
windows. Window rows remain separate for signal/QC review; they are not
treated as independent samples for model evaluation. All three primary models
must use the same participants and subject-level folds. Training-fold
imputation does not replace the required availability of all modalities for
every included recording.

CAP rates are not reported until stage-annotation coverage, 30-second
hypnogram alignment, CAP event bounds and labels, CAP-A durations, sequence
timing, and the NREM denominator checks pass. Missing 30-second stage epochs
are forward-filled from the preceding scored stage only for gaps of up to
three epochs, as in the supplied PhysioNet reader; the imputed count is
reported. The primary CAP rate requires at least three A phases joined by
2-to-under-60-second B phases and uses the sequence's first-to-last A-onset
span; every event in a sequence must align to scored NREM. A separate
PhysioNet-reference rate reproduces the supplied `CAP.m` timing and sequence
threshold over all CAP-A events (without the strict NREM-event filter) and
flags sub-2-second B intervals. A-phase time, orphan
phases, label disagreements, events outside scored NREM, and both sequence
counts are retained for audit. CAP-A events are eligible for a sequence only
when their onset aligns to a scored NREM epoch; the hypnogram is authoritative
for this decision, and the separate CAP event label is checked and reported
rather than silently overriding it. Five- and six-column RemLogic exports are
supported. Since some EDF start clocks differ from the annotation clock,
annotation events are placed on a common timeline relative to the first
scored event; their clock offset from EDF start is retained as a QC feature,
their total span must fit the EDF duration (with a 30-second boundary
tolerance), and at least 80% of EDF duration must have explicit stage scoring.
Annotation-derived columns are never loaded into the primary model.

## Validation status

Automated unit tests cover cohort integrity, EDF header parsing, signal feature
and filtering behavior, annotation timing and CAP sequence calculations,
dataset separation, and shared subject-level model folds. Passing those tests
validates software contracts, not the scientific validity of all database
channels or clinical interpretation. Run the full extraction and review its
QC and annotation reports before interpreting any model results. No model
training should be run on an unvalidated or incomplete feature dataset.
`record-inventory` fetches only byte ranges containing EDF headers, not signal
samples, and writes recording metadata plus channel names, units, and native
rates under `results/qc/`. Unknown EDF record counts are derived from the
remote file size and header sample counts. `validate-annotations` independently
parses all 88 small clinician `.txt` files, checks their event span against EDF
duration, records EDF-to-annotation start-clock offsets, and gates CAP reports
on complete validation. This secondary
annotation pass does not download waveform bodies.
Physiological channels with missing or unsupported EDF physical units are
excluded rather than assigned an assumed scale. If no EEG channel remains with
verified units and acceptable signal QC, that recording is a hard cohort
failure; channel-level evidence is preserved in its failure-QC checkpoint.
`diagnose-annotations` reads the saved CAP QC and validation reports, retrieves
the original annotation text read-only, and writes per-record evidence and
proposed source-level resolutions to `results/qc/annotation_diagnostic.md` and
`.csv`. It does not repair or modify source annotations.

Current workspace status: header metadata is complete for all 88 recordings.
The annotation preflight is incomplete: 72 records pass and 16 remain blocked
in `results/qc/annotation_qc.csv` (11 below the 80% scored-coverage criterion,
two annotation spans longer than the EDF, one invalid 65-second CAP-A phase,
one overlapping CAP event pair, and one truncated annotation row). The CAP
feature dataset is therefore not validated, and CAP analyses or CAP-informed
training must not be run. Physiology extraction has completed `n1`–`n12`;
all 40 NFLE, all 22 RBD, and all 10 PLM records; `n16` also completed. The
three blocked recordings are `n13`, `n14`, and `n15`: each has blank EDF
physical-dimension fields for all 11 physiological channels, making signal
scaling scientifically indeterminate. Resolution requires authoritative
source-unit confirmation or a corrected source header; units are not inferred
from signal amplitudes. Their failed channel-level QC is saved
under `results/checkpoints/records/`, with the summary in
`results/qc/extraction_failures.csv.gz`. The 85 successful recordings have
per-record checkpoints. Provisional cohort v1 identified `n16` as ineligible because it has EEG but
no recognized EMG, EOG, or ECG. The approved v2 cohort explicitly excludes
`n16` for that reason and uses class counts of 12 controls, 40 NFLE, 22 RBD,
and 10 PLM. Its checkpoint preflight passed for all 84 participants; its
primary evaluation is recorded separately from CAP analyses.

## Dataset citation

Terzano MG, Parrino L, Sherieri A, et al. Atlas, rules, and recording
techniques for the scoring of cyclic alternating pattern (CAP) in human sleep.
*Sleep Medicine*. 2001;2(6):537–553. PhysioNet CAP Sleep Database,
version 1.0.0, https://doi.org/10.13026/C2VC7H.

## Existing files

The original course notebook and `GitBasics.md` are retained as exploratory
material. The maintained pipeline and scientific interpretation notebook are
separate from that legacy analysis.
