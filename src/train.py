"""Train the interpretable logistic model with feature selection inside the training data.

Feature screening (chi-square for categorical fields, Spearman |rho| for numeric
and ordinal fields) is fitted on training rows only: on the 80/20 training split
for the holdout model, and inside every fold for repeated stratified CV.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, spearmanr
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
import statsmodels.api as sm

try:
    from .config import (CATEGORICAL_CANDIDATES, CHI2_ALPHA, CV_REPEATS, CV_SPLITS, DATA_PATH,
                         NUMERIC_CANDIDATES, RANDOM_SEED, RESULTS_DIR, SPEARMAN_MIN_ABS, TEST_SIZE)
    from .preprocessing import load_and_preprocess_data
except ImportError:
    from config import (CATEGORICAL_CANDIDATES, CHI2_ALPHA, CV_REPEATS, CV_SPLITS, DATA_PATH,
                        NUMERIC_CANDIDATES, RANDOM_SEED, RESULTS_DIR, SPEARMAN_MIN_ABS, TEST_SIZE)
    from preprocessing import load_and_preprocess_data


def select_features(X: pd.DataFrame, y: pd.Series) -> list[str]:
    """Return candidate columns that pass the screening tests on (X, y) only."""
    selected = []
    for col in CATEGORICAL_CANDIDATES:
        _, p, _, _ = chi2_contingency(pd.crosstab(X[col], y))
        if p < CHI2_ALPHA:
            selected.append(col)
    for col in NUMERIC_CANDIDATES:
        rho = spearmanr(X[col], y).statistic
        if round(abs(float(rho)), 2) >= SPEARMAN_MIN_ABS:
            selected.append(col)
    return selected


class ScreeningSelector(BaseEstimator, TransformerMixin):
    """Pipeline step that re-runs select_features on each training fold."""

    def fit(self, X, y):
        self.selected_ = select_features(X, pd.Series(np.asarray(y), index=X.index))
        if not self.selected_:
            raise ValueError("No feature passed screening on this fold.")
        return self

    def transform(self, X):
        return X[self.selected_]


def build_pipeline() -> Pipeline:
    return Pipeline([("select", ScreeningSelector()), ("model", LogisticRegression())])


def train_model(data_path: Path = DATA_PATH, results_path: Path = RESULTS_DIR):
    """Fit, evaluate on the holdout and with repeated stratified CV, and save a text report."""
    results_path.mkdir(parents=True, exist_ok=True)
    df, target = load_and_preprocess_data(data_path)
    candidates = CATEGORICAL_CANDIDATES + NUMERIC_CANDIDATES
    X, y = df[candidates], df[target]

    # Holdout: select features on the training split only.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED)
    pipe = build_pipeline().fit(X_train, y_train)
    features = pipe.named_steps["select"].selected_
    y_pred = pipe.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    weighted_f1 = f1_score(y_test, y_pred, average="weighted")
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    majority = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
    majority_acc = accuracy_score(y_test, majority.predict(X_test))

    # Repeated stratified CV: selection is refitted inside every training fold.
    cv = RepeatedStratifiedKFold(n_splits=CV_SPLITS, n_repeats=CV_REPEATS, random_state=RANDOM_SEED)
    scoring = {"accuracy": "accuracy", "f1_weighted": "f1_weighted"}
    cv_model = cross_validate(build_pipeline(), X, y, cv=cv, scoring=scoring, return_estimator=True)
    cv_major = cross_validate(DummyClassifier(strategy="most_frequent"), X, y, cv=cv, scoring=scoring)
    n_folds = CV_SPLITS * CV_REPEATS
    freq = pd.Series(
        [f for est in cv_model["estimator"] for f in est.named_steps["select"].selected_]
    ).value_counts().reindex(candidates, fill_value=0)

    def ms(scores):
        return f"{np.mean(scores):.4f} +/- {np.std(scores):.4f}"

    # Statsmodels Logit on the same training rows and selected features (coefficients, p-values).
    logit = sm.Logit(y_train, sm.add_constant(X_train[features])).fit(disp=0)

    lines = [
        f"Selected features (training split only, n={len(X_train)}): {features}",
        "",
        f"Holdout ({len(X_test)} rows, seed {RANDOM_SEED}, {int((1 - TEST_SIZE) * 100)}/{int(TEST_SIZE * 100)} split)",
        f"Accuracy: {accuracy:.4f}",
        f"Weighted F1: {weighted_f1:.4f}",
        f"Majority-label accuracy: {majority_acc:.4f}",
        "",
        "Confusion Matrix:",
        str(cm),
        "",
        "Classification Report:",
        report,
        f"Repeated stratified CV ({CV_SPLITS}x{CV_REPEATS} = {n_folds} folds, selection inside each fold), mean +/- std",
        f"Logistic regression accuracy:    {ms(cv_model['test_accuracy'])}",
        f"Logistic regression weighted F1: {ms(cv_model['test_f1_weighted'])}",
        f"Majority-label accuracy:         {ms(cv_major['test_accuracy'])}",
        f"Majority-label weighted F1:      {ms(cv_major['test_f1_weighted'])}",
        "",
        f"Feature selected in N of {n_folds} CV folds:",
        freq.to_string(),
        "",
        "Statsmodels Summary (training split, selected features):",
        str(logit.summary()),
    ]
    text = "\n".join(lines)
    print(text)
    with open(results_path / "model_summary.txt", "w", encoding="utf-8") as f:
        f.write(text + "\n")
    return pipe


if __name__ == "__main__":
    train_model()
