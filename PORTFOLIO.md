# Student Depression Survey: Interpretable Logistic Regression

[English](PORTFOLIO.md) | [한국어](PORTFOLIO.ko.md)

## At a glance

Four-variable logistic regression that shows which lifestyle and academic responses go with a depression label in a 502-row student survey. Final project for a Programming Languages course at Incheon National University (November to December 2024); an exploratory analysis, not a diagnostic model.

**Problem.** The sample is a few hundred rows, the variables are sensitive, and the label is a single survey answer rather than a clinical assessment. A high score on such data is easy to over-read, so the path from variables to prediction has to stay inspectable.

**Result.** 5x5 repeated stratified cross-validation gives 0.8259 ± 0.0429 accuracy and 0.8255 ± 0.0431 weighted F1, against 0.5020 ± 0.0024 accuracy for always predicting the majority label; a 101-record holdout (seeded 80/20 split) gives 0.8614 accuracy and 0.8609 weighted F1, with statsmodels coefficients and p-values beside the score.

[README](README.md) · [Training script](src/train.py) · [Model summary](results/model_summary.txt)

## Key decisions

**1. Logistic regression as the model.**
Options: tree ensembles or other flexible classifiers, or logistic regression.
Choice: scikit-learn logistic regression for metrics, plus a statsmodels Logit on the same training rows for coefficients and p-values.
Why: with 502 rows and sensitive variables, the reader should see which responses drive each prediction.
Cost: no interactions or non-linear effects; no other model was run, so the trade-off is unmeasured.

**2. A few variables chosen by simple tests on training rows.**
Options: all ten survey variables, a regularized model that selects variables, or test-based selection.
Choice: categorical fields with chi-square p < 0.05 against the label and numeric or ordinal fields with Spearman |ρ| of at least 0.3 after rounding to two decimals, run on training rows only and refitted inside every CV fold. On the 401 training rows this keeps dietary habits, suicidal-thought history, academic pressure (ρ 0.470), and study satisfaction (ρ −0.305).
Why: a short, explainable model whose evaluation never sees the rows used for selection.
Cost: fields near the cut-off flip between folds (financial stress passes in 14 of 25, study satisfaction in 9), and p-values after selection are still optimistic.

**3. Encoding.**
Options: one-hot encoding for nominal fields, or integer label encoding.
Choice: label encoding for categorical fields and an ordinal encoder for the 1-5 scales.
Why: one coefficient per variable.
Cost: dietary habits becomes 0-2 in alphabetical order (Healthy, Moderate, Unhealthy), which matches a healthy-to-unhealthy trend only by coincidence; one-hot encoding was not compared.

## Results

| Model | Features | Accuracy | Weighted F1 | Setting |
|---|---|---:|---:|---|
| Majority label | none | 0.5020 ± 0.0024 | 0.3355 ± 0.0027 | 5x5 repeated stratified CV, 502 rows |
| Logistic regression | screened per fold | 0.8259 ± 0.0429 | 0.8255 ± 0.0431 | 5x5 repeated stratified CV, 502 rows |
| Majority label (training majority "No") | none | 0.4752 | – | 101-record holdout (53 of 101 positive) |
| Logistic regression | 4 screened on training split | 0.8614 | 0.8609 | Seeded 80/20 split, not stratified |

Confusion matrix on the holdout `[[39, 9], [5, 48]]`: 48 of 53 positive and 39 of 48 negative labels correct. All four Logit coefficients have p < 0.001 in the training summary; suicidal-thought history has the largest (3.5388). Labels are nearly balanced (252 Yes, 250 No), so accuracy is a reasonable headline.

## Limitations

- Not clinical sensitivity, fairness, or individual risk; the coefficients are associations within this sample and variable selection, not causes.
- The 101-record holdout (0.8614) sits above the cross-validation mean (0.8259 ± 0.0429); the CV figure is the steadier estimate.
- Suicidal-thought history overlaps with the labeled construct; its share of the accuracy was not tested.

**Lesson.** An earlier version screened features on all 502 rows before splitting and reported 0.9010 holdout accuracy. Screening now runs on training rows only; on the same split it swaps financial stress for study satisfaction, and holdout accuracy is 0.8614.

## Next steps

- Report ROC/PR and calibration, test the model without suicidal-thought history, and compare one-hot encoding.
- Any use beyond education or exploration would also need external data, measurement-equivalence checks, and clinical review.

## Evidence

- [`results/model_summary.txt`](results/model_summary.txt)
- [`src/train.py`](src/train.py)
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
