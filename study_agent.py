import json
import random
from pathlib import Path


class StudyAgent:
    """Small tabular Q-learning prototype with persistent per-learner values."""

    def __init__(self, file_path="q_table.json", seed=42):
        self.file_path = Path(file_path)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self.actions = [
            "Revise fundamentals",
            "Practise questions",
            "Study advanced topics",
        ]
        self.states = ["beginner", "intermediate", "advanced"]
        self.alpha = 0.1
        self.gamma = 0.9
        self.epsilon = 0.1
        self.rng = random.Random(seed)
        self.q_table = {state: [0.0] * len(self.actions) for state in self.states}
        self.load_q_table()

    @staticmethod
    def get_state(score):
        if score < 40:
            return "beginner"
        if score < 70:
            return "intermediate"
        return "advanced"

    def recommend(self, score):
        state = self.get_state(score)
        if self.rng.random() < self.epsilon:
            action_index = self.rng.randrange(len(self.actions))
        else:
            row = self.q_table[state]
            best = max(row)
            best_indices = [i for i, value in enumerate(row) if value == best]
            action_index = self.rng.choice(best_indices)
        return {
            "state": state,
            "action_index": action_index,
            "action": self.actions[action_index],
        }

    def update(self, state, action_index, reward, next_state):
        if state not in self.q_table or next_state not in self.q_table:
            return
        if not isinstance(action_index, int) or not 0 <= action_index < len(self.actions):
            return
        old = self.q_table[state][action_index]
        future = max(self.q_table[next_state])
        self.q_table[state][action_index] = old + self.alpha * (
            reward + self.gamma * future - old
        )
        self.save_q_table()

    def learn_from_quiz(self, previous_score, new_score, action_index):
        previous_state = self.get_state(previous_score)
        next_state = self.get_state(new_score)
        improvement = int(new_score) - int(previous_score)
        reward = 10 if improvement > 0 else (-5 if improvement < 0 else 0)
        self.update(previous_state, int(action_index), reward, next_state)
        return {
            "previous_score": int(previous_score),
            "new_score": int(new_score),
            "improvement": improvement,
            "reward": reward,
            "previous_state": previous_state,
            "next_state": next_state,
        }

    def save_q_table(self):
        temp = self.file_path.with_suffix(".tmp")
        temp.write_text(json.dumps(self.q_table, indent=2), encoding="utf-8")
        temp.replace(self.file_path)

    def load_q_table(self):
        if not self.file_path.exists():
            return
        try:
            saved = json.loads(self.file_path.read_text(encoding="utf-8"))
            for state in self.states:
                values = saved.get(state)
                if (
                    isinstance(values, list)
                    and len(values) == len(self.actions)
                    and all(isinstance(v, (int, float)) for v in values)
                ):
                    self.q_table[state] = [float(v) for v in values]
        except (OSError, json.JSONDecodeError, TypeError):
            # A malformed or interrupted file should not prevent app startup.
            return
