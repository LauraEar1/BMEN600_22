# Figure captions and interpretation

All primary performance estimates below use the unchanged 84-recording v2 baseline. The post-hoc results are exploratory. Each figure is exported as 300-DPI PNG, SVG, and PDF in this directory.

## Figure 1. Primary model performance

**Caption.** The left panel shows balanced accuracy and macro-F1 calculated from pooled out-of-fold predictions for the three prespecified physiological Random Forest models, with exact values above the bars. The right panel shows the mean and standard deviation of the corresponding five fold-level metric values. Fold SD describes fold-to-fold variability and is not a confidence interval.

**Interpretation.** The figure compares the primary EEG-only, EEG+EMG+EOG, and EEG+EMG+EOG+ECG models using the same participants and folds.

**Limitations.** The cohort is small and class-imbalanced. Fold SD is descriptive and should not be interpreted as sampling uncertainty or a confidence interval.

## Figure 2a. Confusion matrices: counts and row percentages

**Caption.** Each cell shows the number of participants and the percentage of that true diagnosis assigned to the predicted diagnosis. Row percentages use the true-class count as denominator; the count color scale is shared across models.

**Interpretation.** The matrices show class-specific error patterns that are hidden by aggregate metrics.

**Limitations.** Each participant contributes one out-of-fold prediction. Percentages for smaller groups are based on fewer recordings and are consequently coarse.

## Figure 2b. Row-normalized confusion matrices

**Caption.** The same out-of-fold confusion matrices are displayed as row-normalized percentages, with the corresponding count shown in each cell.

**Interpretation.** This view facilitates comparison of class-specific recall across models despite unequal class sizes.

**Limitations.** Row normalization does not change the underlying data and does not remove uncertainty from the small class counts.

## Figure 3. Incremental modality benefit

**Caption.** Paired differences in balanced accuracy and macro-F1 for the EEG-to-EEG+EMG+EOG and EEG+EMG+EOG-to-all-modality model comparisons. Points and percentile intervals come from the saved 10,000-resample paired subject bootstrap stratified by diagnosis.

**Interpretation.** The estimates quantify exploratory changes in paired out-of-fold performance after adding modality blocks.

**Limitations.** These post-hoc comparisons and intervals are exploratory; p-values/intervals are not adjusted for multiple comparisons and do not establish clinical benefit. The lower interval bounds can equal zero.

## Figure 4. Feature-selection sensitivity

**Caption.** The prespecified baseline is compared with the exploratory harmonized-feature sensitivity analysis, which selected features independently within each training fold based only on training-fold observation status. Model colors are consistent; hatching marks the sensitivity results.

**Interpretation.** For EEG+EMG+EOG, balanced accuracy changes from 0.7034 to 0.7492; the baseline remains the primary result.

**Limitations.** This analysis changes the feature set and should not be treated as a replacement or improvement claim for the primary experiment. No test-fold data were used to select features.

## Figure 5. Acquisition-related diagnostics

**Caption.** Left: saved fold-level balanced accuracy for each primary physiological model and the separate acquisition-only diagnostic classifier; points are folds and diamonds/error bars show mean ± fold SD. Right: saved Cramér’s V associations between acquisition metadata and diagnosis.

**Interpretation.** The acquisition-only classifier has mean fold balanced accuracy 0.713 (SD 0.201). EEG channel-label association with diagnosis is 0.797 by Cramér’s V.

**Limitations.** These results indicate possible acquisition-related structure, not causal confounding. The acquisition-only model is a diagnostic analysis, evaluated on the same small folds; association estimates are observational, and listed chi-square p-values are unadjusted.

## Figure 6. Missing-feature patterns

**Caption.** Heatmap of the percentage of missing subject-level feature cells by diagnostic group and modality, recomputed from the saved 84-recording subject feature table and checked against the saved post-hoc missingness table.

**Interpretation.** The figure highlights differences in feature availability across diagnostic groups and physiological modalities.

**Limitations.** A missing feature value reflects the saved feature-generation/channel coverage and does not by itself identify why a value is missing. Availability patterns should not be interpreted as physiological effects.
