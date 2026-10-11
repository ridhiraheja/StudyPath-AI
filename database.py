import sqlite3
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "studypath.db"


def get_connection():
    connection = sqlite3.connect(DB_PATH, timeout=15)
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def create_tables():
    with get_connection() as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS quiz_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_name TEXT NOT NULL,
                subject TEXT NOT NULL,
                score INTEGER NOT NULL,
                correct_answers INTEGER NOT NULL,
                total_questions INTEGER NOT NULL,
                readiness TEXT NOT NULL,
                recommendation TEXT NOT NULL,
                created_at TEXT NOT NULL,
                agent_action_index INTEGER DEFAULT 0
            )
        """)
        columns = {
            row[1] for row in connection.execute(
                "PRAGMA table_info(quiz_results)"
            ).fetchall()
        }
        if "agent_action_index" not in columns:
            connection.execute(
                "ALTER TABLE quiz_results ADD COLUMN agent_action_index INTEGER DEFAULT 0"
            )
        connection.execute("""
            CREATE TABLE IF NOT EXISTS question_attempts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_name TEXT NOT NULL,
                subject TEXT NOT NULL,
                topic TEXT NOT NULL,
                is_correct INTEGER NOT NULL CHECK (is_correct IN (0, 1)),
                created_at TEXT NOT NULL
            )
        """)
        connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_quiz_student_subject
            ON quiz_results (student_name, subject, id)
        """)
        connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_question_student_subject
            ON question_attempts (student_name, subject, topic)
        """)
        connection.execute("""
            CREATE TABLE IF NOT EXISTS activity_completions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                student_name TEXT NOT NULL,
                subject TEXT NOT NULL,
                goal TEXT NOT NULL,
                activity_id TEXT NOT NULL,
                completed_at TEXT NOT NULL,
                UNIQUE(student_name, subject, goal, activity_id)
            )
        """)
        connection.execute("""
            CREATE INDEX IF NOT EXISTS idx_activity_student
            ON activity_completions (student_name, subject, goal)
        """)


def save_quiz_result(
    student_name, subject, score, correct_answers, total_questions,
    readiness, recommendation, agent_action_index=0
):
    with get_connection() as connection:
        connection.execute("""
            INSERT INTO quiz_results (
                student_name, subject, score, correct_answers,
                total_questions, readiness, recommendation,
                created_at, agent_action_index
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            student_name.strip(), subject, int(score), int(correct_answers),
            int(total_questions), readiness, recommendation,
            datetime.now().isoformat(timespec="seconds"),
            int(agent_action_index),
        ))


def get_quiz_results(student_name):
    with get_connection() as connection:
        return connection.execute("""
            SELECT id, subject, score, correct_answers, total_questions,
                   readiness, recommendation, created_at
            FROM quiz_results
            WHERE LOWER(TRIM(student_name)) = LOWER(TRIM(?))
            ORDER BY id ASC
        """, (student_name.strip(),)).fetchall()


def get_latest_quiz_result(student_name, subject):
    with get_connection() as connection:
        return connection.execute("""
            SELECT score, agent_action_index
            FROM quiz_results
            WHERE LOWER(TRIM(student_name)) = LOWER(TRIM(?))
              AND subject = ?
            ORDER BY id DESC
            LIMIT 1
        """, (student_name.strip(), subject)).fetchone()


def save_question_attempts(student_name, subject, questions, answers):
    timestamp = datetime.now().isoformat(timespec="seconds")
    records = [
        (
            student_name.strip(), subject, question["topic"],
            int(answer == question["answer"]), timestamp
        )
        for question, answer in zip(questions, answers)
    ]
    with get_connection() as connection:
        connection.executemany("""
            INSERT INTO question_attempts (
                student_name, subject, topic, is_correct, created_at
            ) VALUES (?, ?, ?, ?, ?)
        """, records)


def get_topic_performance(student_name, subject):
    with get_connection() as connection:
        return connection.execute("""
            SELECT topic,
                   COUNT(*) AS attempts,
                   SUM(is_correct) AS correct_answers,
                   ROUND(100.0 * AVG(is_correct), 1) AS accuracy
            FROM question_attempts
            WHERE LOWER(TRIM(student_name)) = LOWER(TRIM(?))
              AND subject = ?
            GROUP BY topic
            ORDER BY accuracy ASC, attempts DESC
        """, (student_name.strip(), subject)).fetchall()


def get_latest_quiz_score(student_name, subject):
    """Return the most recent saved score for a learner and track."""
    with get_connection() as connection:
        row = connection.execute("""
            SELECT score
            FROM quiz_results
            WHERE LOWER(TRIM(student_name)) = LOWER(TRIM(?))
              AND subject = ?
            ORDER BY id DESC
            LIMIT 1
        """, (student_name.strip(), subject)).fetchone()
    return int(row[0]) if row else None


def get_completed_activities(student_name, subject, goal):
    """Return activity IDs completed for this learner, subject, and goal."""
    with get_connection() as connection:
        rows = connection.execute("""
            SELECT activity_id
            FROM activity_completions
            WHERE LOWER(TRIM(student_name)) = LOWER(TRIM(?))
              AND subject = ? AND goal = ?
        """, (student_name.strip(), subject, goal)).fetchall()
    return {row[0] for row in rows}


def set_activity_completed(student_name, subject, goal, activity_id, completed):
    """Persist a completion toggle; unchecking removes the completion record."""
    with get_connection() as connection:
        if completed:
            connection.execute("""
                INSERT OR IGNORE INTO activity_completions
                    (student_name, subject, goal, activity_id, completed_at)
                VALUES (?, ?, ?, ?, ?)
            """, (
                student_name.strip(), subject, goal, activity_id,
                datetime.now().isoformat(timespec="seconds"),
            ))
        else:
            connection.execute("""
                DELETE FROM activity_completions
                WHERE LOWER(TRIM(student_name)) = LOWER(TRIM(?))
                  AND subject = ? AND goal = ? AND activity_id = ?
            """, (student_name.strip(), subject, goal, activity_id))
