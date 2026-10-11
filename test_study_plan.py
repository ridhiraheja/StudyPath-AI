import unittest

from study_plan import GOALS, SUBJECT_ACTIVITIES, build_study_plan


class StudyPlanTests(unittest.TestCase):
    def test_each_goal_has_a_distinct_sequence(self):
        plans = [
            build_study_plan(goal, "Python Programming", 30, score=55)
            for goal in GOALS
        ]
        sequences = [
            tuple((a["title"], a["category"]) for a in plan["activities"])
            for plan in plans
        ]
        self.assertEqual(len(sequences), len(set(sequences)))

    def test_subject_changes_activity_details(self):
        plans = [
            build_study_plan("Explore advanced concepts", subject, 45, score=80)
            for subject in SUBJECT_ACTIVITIES
        ]
        details = [tuple(a["detail"] for a in plan["activities"]) for plan in plans]
        self.assertEqual(len(details), len(set(details)))

    def test_daily_budget_is_respected(self):
        for minutes in (15, 30, 45, 60, 75, 180):
            for goal in GOALS:
                for subject in SUBJECT_ACTIVITIES:
                    for score in (None, 20, 55, 90):
                        plan = build_study_plan(goal, subject, minutes, score, ["Loops"])
                        self.assertEqual(plan["total_minutes"], minutes)
                        self.assertTrue(all(a["minutes"] >= 0 for a in plan["activities"]))

    def test_score_changes_guidance(self):
        low = build_study_plan("Improve quiz scores", "Python Programming", 30, score=20)
        high = build_study_plan("Improve quiz scores", "Python Programming", 30, score=90)
        self.assertEqual(low["level"], "guided")
        self.assertEqual(high["level"], "stretch")
        self.assertNotEqual(low["activities"][1]["detail"], high["activities"][1]["detail"])

    def test_no_history_starts_diagnostic(self):
        plan = build_study_plan("Build strong fundamentals", "Data Science", 15)
        self.assertEqual(plan["level"], "diagnostic")
        self.assertEqual(plan["activities"][0]["title"], "Quick diagnostic")


if __name__ == "__main__":
    unittest.main()
