import hashlib
import json
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from database import (
    create_tables,
    save_quiz_result,
    get_quiz_results,
    get_latest_quiz_result,
    save_question_attempts,
    get_topic_performance,
    get_latest_quiz_score,
    get_completed_activities,
    set_activity_completed,
)
from learner_model import load_trained_model, predict_student_risk
from study_agent import StudyAgent
from study_plan import build_study_plan

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"

st.set_page_config(
    page_title="StudyPath AI | Learning Studio",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Polished lightweight CSS; layout remains responsive.
st.markdown("""
<style>
:root {
  --sp-ink: #20283f;
  --sp-muted: #69758b;
  --sp-primary: #635bce;
  --sp-primary-dark: #4942a7;
  --sp-mint: #dff4ec;
  --sp-peach: #fff0e7;
  --sp-lilac: #efedff;
  --sp-border: #e5e8f1;
}
.stApp {
  background: linear-gradient(135deg, #f8f9ff 0%, #f6fbf8 52%, #fff9f5 100%);
  color: var(--sp-ink);
}
.block-container {
  padding-top: 1.35rem;
  padding-bottom: 3rem;
  max-width: 1320px;
}
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #f0efff 0%, #f6fbf8 58%, #fff8f2 100%);
  border-right: 1px solid #e3e4f2;
}
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] label { color: #34365f; }
.hero {
  position: relative;
  overflow: hidden;
  padding: 2rem 2.15rem;
  border-radius: 26px;
  background: linear-gradient(115deg, #34345f 0%, #625bce 53%, #8b83e8 100%);
  color: #fff;
  margin: .2rem 0 1.5rem 0;
  box-shadow: 0 14px 34px rgba(77, 72, 166, .17);
}
.hero:after {
  content: "";
  position: absolute;
  width: 230px;
  height: 230px;
  right: -58px;
  top: -100px;
  border-radius: 50%;
  background: rgba(255,255,255,.10);
  box-shadow: -72px 125px 0 -32px rgba(255,255,255,.08);
}
.hero h1 { color: #fff; font-size: 2.45rem; font-weight: 800; letter-spacing: -.04em; margin: .35rem 0 0 0; }
.hero p { color: #eeedff; margin: .55rem 0 0 0; font-size: 1.04rem; max-width: 650px; }
.eyebrow { font-size: .72rem; letter-spacing: .16em; text-transform: uppercase; font-weight: 800; color: #d9f6e9; }
h1, h2, h3 { color: #292e4c; letter-spacing: -.025em; }
h2, h3 { font-weight: 750; }
p, label, .stMarkdown { line-height: 1.55; }
.small-muted { color: var(--sp-muted); font-size: .88rem; }
/* Gentle card styling for metrics, expanders, and form sections. */
div[data-testid="stMetric"] {
  background: rgba(255,255,255,.82);
  border: 1px solid rgba(223,226,241,.95);
  padding: 1rem 1.1rem;
  border-radius: 18px;
  box-shadow: 0 5px 18px rgba(43,48,85,.045);
}
div[data-testid="stMetricLabel"] { color: #737b91; font-weight: 650; }
div[data-testid="stMetricValue"] { color: #39377f; font-weight: 800; }
div[data-testid="stTabs"] [data-baseweb="tab-list"] {
  gap: .45rem;
  background: rgba(235,236,250,.72);
  border: 1px solid #e5e6f3;
  padding: .42rem;
  border-radius: 17px;
}
div[data-testid="stTabs"] button[data-baseweb="tab"] {
  border-radius: 12px;
  padding: .58rem .95rem;
  color: #656b84;
  font-weight: 700;
}
div[data-testid="stTabs"] button[aria-selected="true"] {
  background: #fff;
  color: #4f49b4;
  box-shadow: 0 3px 10px rgba(55,53,112,.09);
}
div.stButton > button, div[data-testid="stFormSubmitButton"] button {
  border-radius: 13px;
  font-weight: 750;
  border: 1px solid #d9d6fb;
  transition: all .18s ease;
}
div.stButton > button[kind="primary"],
div[data-testid="stFormSubmitButton"] button[kind="primary"] {
  background: linear-gradient(100deg, #625bce, #8178e3);
  color: #fff;
  border: 0;
  box-shadow: 0 5px 14px rgba(98,91,206,.18);
}
div.stButton > button:hover, div[data-testid="stFormSubmitButton"] button:hover {
  border-color: #a9a3ed;
  transform: translateY(-1px);
}
div[data-testid="stAlert"] { border-radius: 15px; }
div[data-testid="stProgressBar"] > div > div { background: linear-gradient(90deg, #65b99a, #8178e3); }
div[data-testid="stDataFrame"], div[data-testid="stTable"] {
  border: 1px solid var(--sp-border);
  border-radius: 15px;
  overflow: hidden;
}
[data-testid="stTextInput"] input,
[data-testid="stSelectbox"] div[data-baseweb="select"],
[data-testid="stNumberInput"] input {
  border-radius: 12px;
}
hr { border-color: #e7e8f2; }
footer { visibility: hidden; }
@media (max-width: 700px) {
  .hero { padding: 1.45rem; border-radius: 20px; }
  .hero h1 { font-size: 2rem; }
  div[data-testid="stTabs"] [data-baseweb="tab-list"] { gap: .15rem; }
  div[data-testid="stTabs"] button[data-baseweb="tab"] { padding: .5rem .55rem; font-size: .82rem; }
}
</style>
""", unsafe_allow_html=True)

# Consistent palette for interactive charts.
px.defaults.template = "plotly_white"
px.defaults.color_discrete_sequence = ["#625bce", "#68b99a", "#f0a77b", "#7d9fe8", "#d77fa8"]

create_tables()

QUIZZES = {
    "Python Programming": [
        {"question":"Which keyword defines a function in Python?","options":["func","def","function","define"],"answer":"def","topic":"Functions"},
        {"question":"Which data type stores key-value pairs?","options":["List","Tuple","Dictionary","Set"],"answer":"Dictionary","topic":"Data Structures"},
        {"question":"What is the result of `3 ** 2`?","options":["6","9","8","5"],"answer":"9","topic":"Operators"},
        {"question":"Which loop iterates over a sequence?","options":["repeat","foreach","for","loop"],"answer":"for","topic":"Loops"},
        {"question":"Which structure is used to handle exceptions?","options":["if-else","try-except","for-while","def-return"],"answer":"try-except","topic":"Exception Handling"},
        {"question":"Which keyword exits a loop immediately?","options":["stop","break","exitloop","return"],"answer":"break","topic":"Loops"},
        {"question":"What does `len([2, 4, 6])` return?","options":["2","3","6","12"],"answer":"3","topic":"Built-in Functions"},
    ],
    "Data Science": [
        {"question":"Which Python library is commonly used for tabular data?","options":["Pandas","Turtle","Pygame","Tkinter"],"answer":"Pandas","topic":"Data Analysis"},
        {"question":"What does CSV stand for?","options":["Computer Stored Values","Comma-Separated Values","Column System Variables","Central Source Version"],"answer":"Comma-Separated Values","topic":"Data Formats"},
        {"question":"Which statistic is the middle value in sorted data?","options":["Mean","Range","Median","Variance"],"answer":"Median","topic":"Statistics"},
        {"question":"Why is data cleaning performed?","options":["To introduce errors","To handle missing or incorrect data","To increase file size","To remove every column"],"answer":"To handle missing or incorrect data","topic":"Data Preprocessing"},
        {"question":"Which chart is useful for showing a numeric distribution?","options":["Histogram","Logo","Flowchart","Network diagram"],"answer":"Histogram","topic":"Data Visualization"},
        {"question":"What is an outlier?","options":["A missing column name","An unusually distant observation","The average value","A duplicate chart"],"answer":"An unusually distant observation","topic":"Statistics"},
        {"question":"Which method returns the first five rows of a pandas DataFrame by default?","options":["head()","top()","first()","show5()"],"answer":"head()","topic":"Pandas"},
    ],
    "Machine Learning": [
        {"question":"Which learning type uses labeled training examples?","options":["Supervised learning","Unsupervised learning","Random search","Manual learning"],"answer":"Supervised learning","topic":"Learning Types"},
        {"question":"Which task predicts a category?","options":["Classification","Regression","Sorting","Compression"],"answer":"Classification","topic":"Classification"},
        {"question":"Which task predicts a continuous number?","options":["Clustering","Classification","Regression","Tokenization"],"answer":"Regression","topic":"Regression"},
        {"question":"Why use a test dataset?","options":["Evaluate on unseen examples","Train every example twice","Rename features","Delete the model"],"answer":"Evaluate on unseen examples","topic":"Model Evaluation"},
        {"question":"Which algorithm can be used for classification?","options":["Decision Tree","HTML","CSS","SQL query only"],"answer":"Decision Tree","topic":"Decision Trees"},
        {"question":"What is overfitting?","options":["Model learns training details too closely and generalizes poorly","Model has no input","Dataset has no labels","Model is always correct"],"answer":"Model learns training details too closely and generalizes poorly","topic":"Model Evaluation"},
        {"question":"What does a confusion matrix summarize?","options":["Predicted versus actual classes","Missing file names","Training time only","Dataset ownership"],"answer":"Predicted versus actual classes","topic":"Model Evaluation"},
    ],
}

def safe_student_key(student_name: str) -> str:
    return hashlib.sha256(student_name.casefold().encode("utf-8")).hexdigest()[:16]

def load_metrics():
    path = MODEL_DIR / "evaluation_metrics.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def persist_activity_checkbox(student_name, subject_name, goal_name, activity_id, widget_key):
    """Persist the current Streamlit checkbox state after a user toggles it."""
    set_activity_completed(
        student_name, subject_name, goal_name, activity_id,
        bool(st.session_state.get(widget_key, False)),
    )

# Header / hero
st.markdown("""
<div class="hero">
  <div class="eyebrow">PERSONALIZED LEARNING • SDG 4</div>
  <h1>StudyPath AI</h1>
  <p>A learning studio that turns quiz attempts into clear next steps.</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## Your learning space")
    name = st.text_input("Student name", placeholder="Enter your name").strip()
    subject = st.selectbox("Learning track", list(QUIZZES.keys()))
    daily_minutes = st.slider("Study time per day", 15, 180, 60, step=15)
    goal = st.selectbox("Current goal", [
        "Build strong fundamentals",
        "Improve quiz scores",
        "Prepare for exams",
        "Explore advanced concepts",
    ])
    st.divider()
    st.caption("Your quiz history is stored locally in SQLite on this computer.")

if name:
    learner_key = safe_student_key(name)
    if st.session_state.get("agent_owner") != learner_key:
        st.session_state.study_agent = StudyAgent(
            file_path=str(DATA_DIR / f"q_table_{learner_key}.json")
        )
        st.session_state.agent_owner = learner_key

for key, default in [
    ("latest_result", None),
    ("latest_subject", None),
    ("latest_topics", []),
    ("last_learning_result", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default

tabs = st.tabs([
    "✨ Overview",
    "📝 Knowledge check",
    "🗺️ Study plan",
    "📈 Progress",
    "🧪 Model lab",
])

with tabs[0]:
    st.subheader("Your learning dashboard")
    c1, c2, c3 = st.columns(3)
    c1.metric("Learning tracks", "03", help="Python, Data Science, Machine Learning")
    c2.metric("Adaptive loop", "Quiz → Learn → Practise")
    c3.metric("Learning history", "Saved locally")
    st.markdown("### How StudyPath AI works")
    cards = st.columns(3)
    cards[0].markdown("#### 01 · Diagnose\nAnswer a short topic-based quiz to surface areas to practise.")
    cards[1].markdown("#### 02 · Personalize\nGet a study action based on your score and recorded learning history.")
    cards[2].markdown("#### 03 · Improve\nRepeat quizzes and monitor topic accuracy over time.")
    st.info("The quiz-based recommendations use your own quiz history. The public UCI model is a separate educational research demo and does not predict your Python/Data Science quiz score.")
    if name:
        history = get_quiz_results(name)
        if history:
            scores = [r[2] for r in history]
            a, b = st.columns(2)
            a.metric("Saved attempts", len(history))
            b.metric("Average quiz score", f"{sum(scores)/len(scores):.1f}%")
        else:
            st.caption("Complete your first quiz to start building your personal progress history.")
    else:
        st.caption("Enter your name in the sidebar to begin.")

with tabs[1]:
    st.subheader("Knowledge check")
    st.write(f"Track: **{subject}** · {len(QUIZZES[subject])} questions")
    questions = QUIZZES[subject]
    if not name:
        st.warning("Enter your name in the sidebar before submitting a quiz.")
    with st.form(f"quiz_{subject}"):
        answers = []
        for i, q in enumerate(questions):
            st.markdown(f"**{i+1}. {q['question']}**")
            answers.append(st.radio(
                "Choose one answer",
                q["options"],
                index=None,
                key=f"answer_{subject}_{i}",
                horizontal=False,
            ))
            if i < len(questions)-1:
                st.divider()
        submitted = st.form_submit_button("Evaluate my answers", type="primary", use_container_width=True, disabled=not bool(name))
    if submitted:
        total = len(questions)
        correct = sum(a == q["answer"] for a, q in zip(answers, questions))
        score = round(correct * 100 / total)
        weak_topics = [q["topic"] for q, a in zip(questions, answers) if a != q["answer"]]
        previous = get_latest_quiz_result(name, subject)
        learning_result = None
        if previous is not None:
            learning_result = st.session_state.study_agent.learn_from_quiz(
                previous_score=int(previous[0]),
                new_score=score,
                action_index=int(previous[1] or 0),
            )
        if score < 40:
            readiness = "Needs fundamentals"
            recommendation = "Review core concepts and practise with worked examples."
        elif score < 70:
            readiness = "Building understanding"
            recommendation = "Review missed questions, then complete a focused practice set."
        else:
            readiness = "Ready to extend"
            recommendation = "Try a harder exercise and explain why each answer is correct."
        action = st.session_state.study_agent.recommend(score)
        save_quiz_result(
            student_name=name, subject=subject, score=score,
            correct_answers=correct, total_questions=total,
            readiness=readiness, recommendation=recommendation,
            agent_action_index=action["action_index"],
        )
        save_question_attempts(name, subject, questions, answers)
        st.session_state.latest_result = {
            "score": score, "correct": correct, "total": total,
            "readiness": readiness, "recommendation": recommendation,
            "agent_action": action["action"], "agent_state": action["state"],
        }
        st.session_state.latest_subject = subject
        st.session_state.latest_topics = weak_topics
        st.session_state.last_learning_result = learning_result
        st.success("Assessment saved. Your learning plan is ready.")
    latest = st.session_state.latest_result
    if latest and st.session_state.latest_subject == subject:
        st.divider()
        a, b, c = st.columns(3)
        a.metric("Your score", f"{latest['score']}%")
        b.metric("Correct answers", f"{latest['correct']}/{latest['total']}")
        c.metric("Assessment", latest["readiness"])
        st.markdown("#### Recommended next action")
        st.info(f"**{latest['agent_action']}** — {latest['recommendation']}")
        if st.session_state.latest_topics:
            st.markdown("#### Topics to revisit")
            st.write(" · ".join(st.session_state.latest_topics))
        else:
            st.success("Perfect score on this attempt. Stretch yourself with advanced practice.")
        lr = st.session_state.last_learning_result
        if lr:
            change = lr["improvement"]
            st.caption(f"Agent feedback from your previous attempt: {change:+d} percentage points; reward {lr['reward']}. The agent is a learning prototype, not a clinically or academically validated tutor.")

with tabs[2]:
    st.subheader("Your study plan")
    st.caption(f"Goal: {goal} · Daily time: {daily_minutes} minutes · Track: {subject}")
    if not name:
        st.info("Enter your name in the sidebar to personalize and save your plan.")
    else:
        # Use persisted quiz data so personalization survives an app restart.
        latest_score = get_latest_quiz_score(name, subject)
        topic_rows_for_plan = get_topic_performance(name, subject)
        weak_topics_for_plan = [
            row[0] for row in topic_rows_for_plan if row[3] < 70
        ]
        if not weak_topics_for_plan and st.session_state.latest_subject == subject:
            weak_topics_for_plan = st.session_state.latest_topics

        plan = build_study_plan(
            goal=goal,
            subject=subject,
            daily_minutes=daily_minutes,
            score=latest_score,
            weak_topics=weak_topics_for_plan,
        )
        st.markdown(f"### {goal}")
        st.write(plan["adaptation"])
        if plan["weak_topics"]:
            st.markdown("**Topics to prioritize:** " + " · ".join(plan["weak_topics"][:5]))
        elif latest_score is None:
            st.info("Complete the Knowledge Check to unlock score-aware recommendations and topic targeting.")

        completed_ids = get_completed_activities(name, subject, goal)
        completed_count = sum(activity["id"] in completed_ids for activity in plan["activities"])
        progress = completed_count / len(plan["activities"]) if plan["activities"] else 0
        c1, c2, c3 = st.columns(3)
        c1.metric("Daily study budget", f"{plan['total_minutes']} / {daily_minutes} min")
        c2.metric("Activities completed", f"{completed_count} / {len(plan['activities'])}")
        c3.metric("Latest track score", f"{latest_score}%" if latest_score is not None else "Not assessed")
        st.progress(progress, text=f"{round(progress * 100)}% of this plan completed")

        st.markdown("#### Today's activity sequence")
        for index, activity in enumerate(plan["activities"], start=1):
            is_done = activity["id"] in completed_ids
            with st.container(border=True):
                left, right = st.columns([5, 1])
                with left:
                    st.markdown(f"**{index:02d}. {activity['title']}** · {activity['minutes']} min")
                    st.caption(activity["category"])
                    st.write(activity["detail"])
                    st.caption(f"Completion check: {activity['done_criterion']}")
                with right:
                    checkbox_key = f"complete_{safe_student_key(name)}_{hashlib.sha256(activity['id'].encode()).hexdigest()[:12]}"
                    st.checkbox(
                        "Done",
                        value=is_done,
                        key=checkbox_key,
                        on_change=persist_activity_checkbox,
                        args=(name, subject, goal, activity["id"], checkbox_key),
                    )

        st.markdown("#### Success criterion")
        st.success(plan["success_criterion"])
        st.caption(
            "The activity sequence changes with the selected goal and track. "
            "Quiz scores adjust the guidance, and completed activities are saved in SQLite."
        )

with tabs[3]:
    st.subheader("Progress and topic insights")
    if not name:
        st.info("Enter your name in the sidebar to view saved progress.")
    else:
        rows = get_quiz_results(name)
        if not rows:
            st.info("No quiz history yet. Submit a quiz to populate this dashboard.")
        else:
            df = pd.DataFrame(rows, columns=["ID","Subject","Score","Correct","Questions","Assessment","Recommendation","Date"])
            df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
            c1, c2, c3 = st.columns(3)
            c1.metric("Attempts", len(df))
            c2.metric("Average score", f"{df['Score'].mean():.1f}%")
            c3.metric("Latest score", f"{df.iloc[-1]['Score']}%")
            chart = px.line(df, x="Date", y="Score", color="Subject", markers=True, range_y=[0,100], title="Quiz score history")
            st.plotly_chart(chart, use_container_width=True)
            topic_rows = get_topic_performance(name, subject)
            st.markdown(f"### Topic accuracy · {subject}")
            if topic_rows:
                topic_df = pd.DataFrame(topic_rows, columns=["Topic","Attempts","Correct answers","Accuracy (%)"])
                st.dataframe(topic_df, use_container_width=True, hide_index=True)
                weak_df = topic_df[topic_df["Accuracy (%)"] < 70]
                if not weak_df.empty:
                    st.warning("Topics below 70% recorded accuracy:")
                    st.write(" · ".join(weak_df["Topic"].astype(str).tolist()))
                else:
                    st.success("All recorded topic accuracies are at least 70%. Keep practising to build confidence.")
                topic_chart = px.bar(topic_df.sort_values("Accuracy (%)"), x="Accuracy (%)", y="Topic", orientation="h", range_x=[0,100], title="Accuracy by topic")
                st.plotly_chart(topic_chart, use_container_width=True)
            else:
                st.caption("Topic history starts with quizzes completed after topic tracking was added.")
            st.markdown("### Saved quiz history")
            view = df.sort_values("ID", ascending=False).copy()
            view["Date"] = view["Date"].dt.strftime("%Y-%m-%d %H:%M")
            st.dataframe(view, use_container_width=True, hide_index=True)

with tabs[4]:
    st.subheader("Model lab · public dataset")
    st.write("This separate ML experiment uses the UCI Student Performance dataset to classify whether a historical school record's final course grade was below 10/20. It is not used to score your technical quizzes.")
    metrics = load_metrics()
    model_path = MODEL_DIR / "student_risk_model.joblib"
    if metrics and model_path.exists():
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Test accuracy", f"{metrics.get('accuracy', 0):.1%}")
        c2.metric("Balanced accuracy", f"{metrics.get('balanced_accuracy', 0):.1%}")
        c3.metric("Macro F1", f"{metrics.get('macro_f1', 0):.1%}")
        c4.metric("Test records", str(metrics.get("test_size", "—")))
        st.caption(f"Dataset: {metrics.get('dataset', 'UCI Student Performance')} · Model: {metrics.get('selected_model', metrics.get('model', 'Unknown'))} · Target: final grade G3 < 10/20.")
        matrix = metrics.get("confusion_matrix")
        if matrix:
            st.markdown("#### Confusion matrix")
            st.dataframe(pd.DataFrame(matrix, index=["Actual: not at risk","Actual: at risk"], columns=["Predicted: not at risk","Predicted: at risk"]), use_container_width=True)
        report = metrics.get("classification_report")
        if report:
            st.markdown("#### Class metrics")
            report_df = pd.DataFrame(report).T
            st.dataframe(report_df.round(3), use_container_width=True)
        st.warning("This dataset describes students from two Portuguese schools. Its performance may not generalize to other populations; use it as an educational experiment, not as a high-stakes decision tool.")
    else:
        st.info("The model has not been trained on this computer yet.")
        st.markdown("Run these commands in the VS Code terminal:")
        st.code("python -m pip install -r requirements.txt\npython train_model.py", language="powershell")
        st.write("The training script downloads the public dataset, evaluates the model on held-out data, and saves metrics locally.")

st.divider()
st.caption("StudyPath AI · SDG 4: Quality Education · Streamlit + scikit-learn + SQLite + Q-learning prototype")
