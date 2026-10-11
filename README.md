# StudyPath AI — Learning Studio

A beginner-friendly Streamlit project for SDG 4 (Quality Education). It combines short topic quizzes, personal progress tracking, a small Q-learning prototype, and a separately evaluated Random Forest experiment using public student-performance data.

## Features

- Polished responsive Streamlit interface
- Python, Data Science, and Machine Learning knowledge checks
- Topic-level attempt history and accuracy
- SQLite quiz history
- Per-student saved Q-learning table
- Four goal-specific study-plan sequences with Python, Data Science, and Machine Learning activities
- Score-aware guidance, weak-topic targeting, and daily time allocations that total the selected budget
- SQLite-persisted activity completion and quiz progress
- Real-data model training and held-out evaluation
- Metrics include accuracy, balanced accuracy, macro precision/recall/F1, confusion matrix, and a majority-class baseline

## Important distinction

The UCI Student Performance dataset describes school-course outcomes in Mathematics and Portuguese for students from two Portuguese schools. It does **not** contain the technical quiz answers from this app. The trained model predicts whether the historical Mathematics final grade (`G3`) is below 10/20. It is shown in the Model Lab as a separate ML experiment and does not score a learner's Python/Data Science/Machine Learning quiz.

The model excludes `G1` and `G2` from the inputs to avoid using earlier grades as shortcut predictors. This is a small, historical dataset and should not be used for high-stakes decisions or assumed to generalize to other student populations.

## Setup on Windows / VS Code

Open this folder in VS Code. In the terminal:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python train_model.py
python -m streamlit run app.py
```

`train_model.py` downloads the official UCI archive into `data/` if the two CSV files are not already there. An internet connection is required for the first training run. The dataset is not bundled in this ZIP because the downloader retrieves the current official archive and keeps the attribution clear.

If PowerShell blocks virtual-environment activation, use:
```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe train_model.py
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## Project structure

```text
StudyPathAI/
├── app.py
├── database.py
├── learner_model.py
├── study_agent.py
├── train_model.py
├── requirements.txt
├── README.md
├── data/
│   └── (dataset CSVs downloaded by train_model.py)
├── models/
│   ├── student_risk_model.joblib
│   └── evaluation_metrics.json
└── studypath.db (created automatically when the app runs)
```

## Dataset citation

Cortez, P. (2008). *Student Performance*. UCI Machine Learning Repository. https://doi.org/10.24432/C5TG7T.

The UCI page lists the dataset under the Creative Commons Attribution 4.0 International (CC BY 4.0) license. Retain attribution when redistributing the data.

## Limitations / responsible use

- Q-learning uses a simple reward based on score improvement. It is a prototype, not evidence that one action causes learning gains. The study-plan generator also applies explicit goal/subject/score rules; the Q-table is not a validated tutoring policy.
- Quiz questions are a small educational demo set, not a validated exam.
- A student's name is a label, not an authenticated identity; do not use this local app for sensitive student records.
- The risk classifier is a classroom ML exercise. Do not use it to make admissions, grading, disciplinary, or other high-stakes decisions.
- Evaluation metrics can vary with the split and should be interpreted alongside class metrics and the baseline.

## Suggested demo flow

1. Enter a learner name in the sidebar.
2. Complete a knowledge check.
3. Open the study plan and note the suggested action.
4. Complete another quiz and show the reward update.
5. Open Progress to see score history and topic accuracy.
6. Open Model Lab to show the separately trained UCI model and held-out metrics.


## Study-plan tests

After installing the requirements, run:

```powershell
python -m unittest test_study_plan.py
```

The tests check that goals produce different sequences, subject tracks change activity content, score bands change guidance, new learners receive a diagnostic path, and allocated minutes exactly match the selected daily budget.
