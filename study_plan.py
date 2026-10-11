"""Goal-, subject-, and performance-aware study-plan generation."""
from __future__ import annotations

from typing import Iterable

GOALS = [
    "Build strong fundamentals",
    "Improve quiz scores",
    "Prepare for exams",
    "Explore advanced concepts",
]

SUBJECT_ACTIVITIES = {
    "Python Programming": {
        "review": ("Concept + code tracing", "Review one Python concept, trace a short example line by line, and predict its output."),
        "practice": ("Coding practice", "Write a small program, run it, and correct any syntax or logic errors."),
        "exam": ("Timed coding drill", "Solve a short coding or output-prediction question under a timer."),
        "project": ("Mini implementation", "Build a small feature using functions, collections, loops, or exception handling."),
        "reflection": ("Explain the code", "Explain why your solution works and note one edge case."),
    },
    "Data Science": {
        "review": ("Concept + data example", "Review one data-analysis concept and connect it to a small table or dataset."),
        "practice": ("Data investigation", "Clean a small dataset, calculate summary statistics, and explain what they mean."),
        "exam": ("Timed analysis questions", "Answer short questions on statistics, preprocessing, pandas, and chart interpretation."),
        "project": ("Mini analysis", "Create one useful visualization and write two evidence-based observations."),
        "reflection": ("Interpret the result", "Record one finding, one limitation, and one follow-up question."),
    },
    "Machine Learning": {
        "review": ("Algorithm and metric review", "Review one ML concept, its assumptions, and when it is appropriate to use."),
        "practice": ("Model reasoning", "Work through a feature, prediction, or evaluation example and justify each step."),
        "exam": ("Timed ML questions", "Answer questions on algorithms, overfitting, train/test splits, and evaluation metrics."),
        "project": ("Mini model challenge", "Design a small experiment: define a target, choose features, and compare a baseline with one model."),
        "reflection": ("Evaluate the model", "Explain the result, identify a limitation, and decide what to test next."),
    },
}

GOAL_TEMPLATES = {
    "Build strong fundamentals": [
        ("Understand", "review", .25),
        ("Study a worked example", "practice", .25),
        ("Guided practice", "practice", .25),
        ("Recall without notes", "reflection", .15),
        ("Check understanding", "exam", .10),
    ],
    "Improve quiz scores": [
        ("Review mistakes", "reflection", .20),
        ("Target weak topics", "practice", .30),
        ("Focused question set", "exam", .25),
        ("Retry missed concepts", "practice", .15),
        ("Compare your result", "reflection", .10),
    ],
    "Prepare for exams": [
        ("Active recall", "review", .20),
        ("Mixed exam practice", "exam", .25),
        ("Timed mini-test", "exam", .25),
        ("Analyse errors", "reflection", .15),
        ("Final topic check", "practice", .15),
    ],
    "Explore advanced concepts": [
        ("Check prerequisites", "review", .15),
        ("Explore an advanced concept", "practice", .20),
        ("Practical challenge", "project", .35),
        ("Debug or analyse results", "reflection", .15),
        ("Choose the next extension", "project", .15),
    ],
}


def _allocate_minutes(total: int, weights: list[float]) -> list[int]:
    """Allocate all available minutes using largest remainders; never exceed total."""
    raw = [total * weight for weight in weights]
    allocated = [int(value) for value in raw]
    remainder = total - sum(allocated)
    order = sorted(range(len(raw)), key=lambda i: raw[i] - allocated[i], reverse=True)
    for i in order[:remainder]:
        allocated[i] += 1
    return allocated


def build_study_plan(
    goal: str,
    subject: str,
    daily_minutes: int,
    score: int | None = None,
    weak_topics: Iterable[str] = (),
) -> dict:
    """Build a distinct plan. Score is the latest subject quiz score, if available."""
    if goal not in GOAL_TEMPLATES:
        goal = GOALS[0]
    if subject not in SUBJECT_ACTIVITIES:
        subject = "Python Programming"
    daily_minutes = max(1, int(daily_minutes))
    score = None if score is None else max(0, min(100, int(score)))
    weak = list(dict.fromkeys(str(topic) for topic in weak_topics if topic))

    if score is None:
        level = "diagnostic"
        adaptation = "No quiz history yet: begin with a short diagnostic and use it to identify a starting point."
    elif score < 40:
        level = "guided"
        adaptation = "Your latest score suggests revisiting prerequisites and using worked examples with hints."
    elif score < 70:
        level = "targeted"
        adaptation = "Build on what you know by targeting missed topics and gradually reducing hints."
    else:
        level = "stretch"
        adaptation = "Your latest score supports a stretch task, timed practice, or deeper application."

    templates = GOAL_TEMPLATES[goal]
    minutes = _allocate_minutes(daily_minutes, [item[2] for item in templates])
    activities = []
    for index, (title, kind, _weight) in enumerate(templates):
        base_title, detail = SUBJECT_ACTIVITIES[subject][kind]
        if score is None and index == 0:
            title = "Quick diagnostic"
            detail = f"Try a few beginner-friendly {subject} questions without worrying about the score. " + detail
        elif score is not None and score < 40 and kind in ("practice", "exam"):
            detail = "Start with hints and a worked example, then try independently. " + detail
        elif score is not None and score >= 70 and kind in ("practice", "project", "exam"):
            detail = "Increase the difficulty and attempt this independently before checking help. " + detail
        if weak and goal in ("Improve quiz scores", "Prepare for exams") and kind in ("practice", "exam", "reflection"):
            detail += " Prioritize: " + ", ".join(weak[:3]) + "."
        activities.append({
            "id": f"{goal}::{subject}::{index}::{title}",
            "title": title,
            "category": base_title,
            "minutes": minutes[index],
            "detail": detail,
            "done_criterion": {
                "Build strong fundamentals": "Explain the concept and solve a basic question without help.",
                "Improve quiz scores": "Correct a previous mistake and explain why the right answer works.",
                "Prepare for exams": "Complete the task within the time limit and review every error.",
                "Explore advanced concepts": "Produce a working result or reasoned analysis and record a next step.",
            }[goal],
        })

    return {
        "goal": goal,
        "subject": subject,
        "daily_minutes": daily_minutes,
        "score": score,
        "level": level,
        "adaptation": adaptation,
        "weak_topics": weak,
        "activities": activities,
        "total_minutes": sum(activity["minutes"] for activity in activities),
        "success_criterion": {
            "Build strong fundamentals": "Explain the concept and solve basic questions independently.",
            "Improve quiz scores": "Improve accuracy and reduce repeated mistakes across attempts.",
            "Prepare for exams": "Recall key concepts and answer mixed questions under time pressure.",
            "Explore advanced concepts": "Apply knowledge to a practical challenge beyond basic quiz questions.",
        }[goal],
    }
