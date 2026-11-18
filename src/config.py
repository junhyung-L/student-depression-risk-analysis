"""Shared paths and model defaults."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / "data" / "Depression Student Dataset.csv"
RESULTS_DIR = PROJECT_ROOT / "results"
RANDOM_SEED = 42
TEST_SIZE = 0.20

# Feature screening, fitted on training rows only (src/train.py):
# categorical fields kept when chi-square p < CHI2_ALPHA against the label;
# numeric and ordinal fields kept when Spearman |rho|, rounded to 2 decimals, >= SPEARMAN_MIN_ABS.
CATEGORICAL_CANDIDATES = ["Gender", "Sleep Duration", "Dietary Habits",
                          "Family History of Mental Illness", "Have you ever had suicidal thoughts ?"]
NUMERIC_CANDIDATES = ["Age", "Study Hours", "Academic Pressure", "Study Satisfaction", "Financial Stress"]
CHI2_ALPHA = 0.05
SPEARMAN_MIN_ABS = 0.30

# Repeated stratified cross-validation
CV_SPLITS = 5
CV_REPEATS = 5
