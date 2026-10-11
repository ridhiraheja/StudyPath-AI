
import io
import json
import zipfile
from pathlib import Path
from urllib.request import Request, urlopen

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

MATH_CSV = DATA_DIR / "student-mat.csv"
PORTUGUESE_CSV = DATA_DIR / "student-por.csv"
MODEL_PATH = MODEL_DIR / "student_risk_model.joblib"
METRICS_PATH = MODEL_DIR / "evaluation_metrics.json"


def download_dataset():
    """Download and locate the two UCI dataset CSV files."""
    if MATH_CSV.exists() and PORTUGUESE_CSV.exists():
        print("Using existing dataset files in data/.")
        return

    urls = [
        "https://archive.ics.uci.edu/static/public/320/student+performance.zip",
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00320/student.zip",
    ]
    found = {}
    errors = []

    def scan_archive(archive, depth=0):
        for member in archive.namelist():
            basename = Path(member.replace("\\", "/")).name.lower()

            if basename in {"student-mat.csv", "student-por.csv"}:
                found[basename] = archive.read(member)
            elif basename.endswith(".zip") and depth < 2:
                try:
                    with zipfile.ZipFile(
                        io.BytesIO(archive.read(member))
                    ) as nested:
                        scan_archive(nested, depth + 1)
                except (zipfile.BadZipFile, KeyError, OSError):
                    continue

    for url in urls:
        try:
            print(f"Trying dataset source: {url}")
            request = Request(
                url, headers={"User-Agent": "StudyPathAI/1.0"}
            )
            with urlopen(request, timeout=45) as response:
                payload = response.read()

            with zipfile.ZipFile(io.BytesIO(payload)) as archive:
                scan_archive(archive)

            if (
                "student-mat.csv" in found
                and "student-por.csv" in found
            ):
                break
            errors.append(f"Required CSVs not found at {url}")
        except Exception as exc:
            errors.append(f"{url}: {exc}")

    required = ["student-mat.csv", "student-por.csv"]
    missing = [name for name in required if name not in found]

    if missing:
        raise RuntimeError(
            f"Missing dataset files: {missing}\n"
            + "\n".join(errors)
            + "\nCheck your internet connection."
        )

    for filename in required:
        (DATA_DIR / filename).write_bytes(found[filename])

    print("Dataset downloaded successfully.")


def build_pipeline(preprocessor, classifier):
    """Create a preprocessing and classification pipeline."""
    return Pipeline([
        ("preprocess", clone(preprocessor)),
        ("classifier", classifier),
    ])


def main():
    download_dataset()

    # Use the mathematics-course dataset for this experiment.
    df = pd.read_csv(MATH_CSV, sep=";")

    if "G3" not in df.columns:
        raise ValueError("Expected target column G3 was not found.")

    # Predict whether final grade is below 10/20.
    y = (df["G3"] < 10).astype(int)

    # Exclude current and final grades to avoid target leakage.
    X = df.drop(columns=["G1", "G2", "G3"])

    # Include text and categorical columns for one-hot encoding.
    categorical = X.select_dtypes(
        include=["object", "str", "category"]
    ).columns.tolist()
    numerical = [
        column for column in X.columns
        if column not in categorical
    ]

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical,
            ),
            ("numerical", "passthrough", numerical),
        ],
        remainder="drop",
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    # Only the training set is used for model selection.
    cv = StratifiedKFold(
        n_splits=5, shuffle=True, random_state=42
    )

    candidates = {
        "Decision Tree": (
            DecisionTreeClassifier(
                class_weight="balanced", random_state=42
            ),
            {
                "classifier__max_depth": [3, 4, 5, None],
                "classifier__min_samples_leaf": [4, 8, 12],
            },
        ),
        "Random Forest": (
            RandomForestClassifier(
                class_weight="balanced",
                random_state=42,
                n_jobs=-1,
            ),
            {
                "classifier__n_estimators": [100, 200],
                "classifier__max_depth": [4, 8, None],
                "classifier__min_samples_leaf": [2, 5],
            },
        ),
        "Logistic Regression": (
            LogisticRegression(
                class_weight="balanced",
                max_iter=2000,
                random_state=42,
            ),
            {
                "classifier__C": [0.1, 1.0, 10.0],
            },
        ),
    }

    comparison = []
    best_score = -1
    best_name = None
    best_model = None
    best_params = None

    print("\nComparing models with 5-fold cross-validation...")
    print("Selection metric: macro F1")

    for name, (classifier, params) in candidates.items():
        pipeline = build_pipeline(preprocessor, classifier)

        search = GridSearchCV(
            estimator=pipeline,
            param_grid=params,
            scoring="f1_macro",
            cv=cv,
            n_jobs=-1,
            refit=True,
            error_score="raise",
        )
        search.fit(X_train, y_train)

        score = float(search.best_score_)
        comparison.append({
            "model": name,
            "best_cv_macro_f1": score,
            "best_parameters": {
                key: (
                    int(value)
                    if hasattr(value, "item")
                    and isinstance(value.item(), int)
                    else value.item()
                    if hasattr(value, "item")
                    else value
                )
                for key, value in search.best_params_.items()
            },
        })

        print(f"{name}: CV macro F1 = {score:.3f}")
        print(f"  Best parameters: {search.best_params_}")

        if score > best_score:
            best_score = score
            best_name = name
            best_model = search.best_estimator_
            best_params = search.best_params_

    # Evaluate the selected model once on the untouched test set.
    predictions = best_model.predict(X_test)

    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train, y_train)
    baseline_predictions = baseline.predict(X_test)

    labels = [0, 1]
    report = classification_report(
        y_test,
        predictions,
        labels=labels,
        target_names=["Not at risk", "At risk"],
        output_dict=True,
        zero_division=0,
    )

    metrics = {
        "dataset": "UCI Student Performance (student-mat.csv)",
        "citation": (
            "Cortez, P. (2008). Student Performance. "
            "UCI Machine Learning Repository. "
            "https://doi.org/10.24432/C5TG7T."
        ),
        "license": "CC BY 4.0",
        "selected_model": best_name,
        "selection_metric": "5-fold cross-validation macro F1",
        "best_cross_validation_macro_f1": best_score,
        "best_parameters": best_params,
        "model_comparison": comparison,
        "target": "At risk = G3 < 10/20",
        "features_used": X.columns.tolist(),
        "excluded_to_avoid_grade_leakage": ["G1", "G2", "G3"],
        "total_records": int(len(df)),
        "train_size": int(len(X_train)),
        "test_size": int(len(X_test)),
        "at_risk_rate": float(y.mean()),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "balanced_accuracy": float(
            balanced_accuracy_score(y_test, predictions)
        ),
        "macro_f1": float(
            f1_score(y_test, predictions, average="macro", zero_division=0)
        ),
        "macro_precision": float(
            precision_score(
                y_test, predictions, average="macro", zero_division=0
            )
        ),
        "macro_recall": float(
            recall_score(
                y_test, predictions, average="macro", zero_division=0
            )
        ),
        "baseline_accuracy": float(
            accuracy_score(y_test, baseline_predictions)
        ),
        "baseline_macro_f1": float(
            f1_score(
                y_test, baseline_predictions,
                average="macro", zero_division=0
            )
        ),
        "confusion_matrix": confusion_matrix(
            y_test, predictions, labels=labels
        ).tolist(),
        "classification_report": report,
        "random_state": 42,
        "caution": (
            "Small historical sample from two Portuguese schools. "
            "Results may not generalize and are for educational "
            "demonstration only, not real student-risk decisions."
        ),
    }

    joblib.dump(best_model, MODEL_PATH)
    METRICS_PATH.write_text(
        json.dumps(metrics, indent=2, default=str),
        encoding="utf-8",
    )

    print("\n========== TRAINING COMPLETE ==========")
    print(f"Selected model:      {best_name}")
    print(f"Best CV macro F1:    {best_score:.3f}")
    print(f"Test accuracy:       {metrics['accuracy']:.3f}")
    print(f"Test balanced acc.:  {metrics['balanced_accuracy']:.3f}")
    print(f"Test macro F1:       {metrics['macro_f1']:.3f}")
    print(f"Baseline accuracy:   {metrics['baseline_accuracy']:.3f}")
    print(f"Baseline macro F1:   {metrics['baseline_macro_f1']:.3f}")
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved metrics: {METRICS_PATH}")
    print("\nModel comparison:")
    for result in comparison:
        print(
            f"  {result['model']}: "
            f"{result['best_cv_macro_f1']:.3f}"
        )

    print(
        "\nNote: this experiment predicts a historical school-course "
        "risk label. It does not predict StudyPath AI's technical quizzes."
    )


if __name__ == "__main__":
    main()
