# Student Depression-Label Exploration: Choosing Interpretability over Complexity

[English](PORTFOLIO.md) | [한국어](PORTFOLIO.ko.md)

## Summary

A final project for a Programming Languages course at Incheon National
University (November to December 2024). It explores which lifestyle and
academic variables go with a depression label in a 502-row student survey. It
is not a diagnostic model. A four-variable logistic regression reaches 0.9010
accuracy and 0.90 weighted F1 on a 101-record holdout, against 0.5248 for the
majority label, with statsmodels coefficients and p-values kept beside the
score.

## Challenge

The sample is a few hundred rows, the variables are sensitive, and the label
is a single survey answer rather than a clinical assessment. A high score on
such data is easy to over-read, so the work had to keep the path from
variables to prediction inspectable and say plainly what a single split can
and cannot support.

## Decisions

**1. Logistic regression as the model.**
Options: tree ensembles or other flexible classifiers, or a logistic
regression. Choice: scikit-learn logistic regression for metrics, plus a
statsmodels Logit on the same training rows for coefficients and p-values.
Why: with 502 rows and sensitive variables, the reader should be able to see
which responses drive each prediction. Cost: no interactions or non-linear
effects, and no other model was run, so what this choice gave up is unknown.

**2. Four variables chosen by simple tests.**
Options: all ten survey variables, a regularised model that selects
variables, or test-based selection. Choice: keep categorical fields with
chi-square p<0.05 against the label (dietary habits, suicidal-thought history)
and ordinal fields with Spearman |ρ| of at least 0.3 (academic pressure 0.48,
financial stress 0.30). Why: a short, explainable model. Cost: the tests ran
on all 502 rows before the split, so the holdout is not fully independent of
selection, and post-selection p-values are optimistic.

**3. Encoding.**
Options: one-hot encoding for nominal fields, or integer label encoding.
Choice: label encoding for categorical fields and an ordinal encoder for the
1-5 scales. Why: it keeps one coefficient per variable. Cost: dietary habits
becomes 0-2 in alphabetical order (Healthy, Moderate, Unhealthy), which
happens to match a healthy-to-unhealthy trend; that order is a coincidence,
and one-hot encoding was not compared.

## Results

| Model | Features | Accuracy | Weighted F1 | Setting |
|---|---|---:|---:|---|
| Majority label ("Yes") | none | 0.5248 | - | Same 101-record holdout (53 of 101 positive) |
| Logistic regression | 4 selected | 0.9010 | 0.90 | Seeded 80/20 split, not stratified |

The confusion matrix `[[42, 6], [4, 49]]` means 49 of 53 positive and 42 of 48
negative labels were correct on this split. All four Logit coefficients have
p<0.001 in the training summary; suicidal-thought history has the largest
(3.16). The labels are nearly balanced (252 Yes, 250 No), so accuracy is a
reasonable headline here.

## What the result does not show

- Not clinical sensitivity, fairness, or individual risk. Weighted F1 is a
  support-weighted average on one split.
- Not a stable estimate: training accuracy in the notebook is 0.8354, below
  the holdout figure, which suggests a favourable split; there is no repeated
  validation or confidence interval.
- Not causal: coefficients are associations inside this sample and this
  variable selection. Suicidal-thought history overlaps with the construct
  being labelled, and its share of the accuracy was not tested.

## What I learned and would do next

Selecting variables on the full dataset leaks into the holdout even when the
model never sees it. Next: move selection inside the training folds, run
repeated stratified cross-validation with intervals, report ROC/PR and
calibration, test the model without suicidal-thought history, and compare
one-hot encoding. Any use beyond education or exploration would also need
external data, measurement-equivalence checks, and clinical review.

## Evidence

- [`results/model_summary.txt`](results/model_summary.txt)
- [`src/train.py`](src/train.py)
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md)
