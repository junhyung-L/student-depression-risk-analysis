# Architecture

| Path | Responsibility |
|---|---|
| `src/config.py` | Shared paths, seed, split fraction, candidate fields, screening thresholds, and CV settings. |
| `src/preprocessing.py` | CSV loading and encodings. |
| `src/train.py` | Feature screening on training rows, logistic training, holdout and repeated stratified CV evaluation, p-value summary, and result output. |
| `src/generate_plots.py` | Correlation, countplot, and learning-curve figures. |

Screening runs inside the training data only (the 80/20 training split and every CV fold). No external validation
or clinical deployment path is implemented.
