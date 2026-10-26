"""
Unit and integration tests for single-step inference and diagnostics.
"""

import json
import unittest
from tests.common import run_cli


class TestInference(unittest.TestCase):

    def test_01_noun_binary_and_cache(self):
        cmd = [
            "noun",
            "--rubrics", "是否包含讽刺语气",
            "--context", "你可真准时啊，会议都开完了你才来。",
        ]
        # 1. 明确正向用例
        res = run_cli(cmd)
        data = json.loads(res.stdout)
        self.assertEqual(data.get("decision"), "是")
        self.assertTrue(data.get("passed"))
        self.assertGreater(data.get("probability", 0), 0.5)

        # 2. 缓存命中验证
        res_cached = run_cli(cmd)
        data_cached = json.loads(res_cached.stdout)
        self.assertTrue(data_cached.get("cached"))
        self.assertEqual(data_cached.get("decision"), "是")

        # 3. 纯标签输出模式验证
        res_label = run_cli(cmd + ["--format", "label-only"])
        self.assertEqual(res_label.stdout.strip(), "是")

        # 4. 明确负向用例
        res_neg = run_cli([
            "noun",
            "--rubrics", "是否包含讽刺语气",
            "--context", "今天下午两点在三楼会议室召开项目周会，请准时参加。",
        ])
        data_neg = json.loads(res_neg.stdout)
        self.assertEqual(data_neg.get("decision"), "否")
        self.assertFalse(data_neg.get("passed"))
        self.assertLess(data_neg.get("probability", 1), 0.5)

    def test_02_choose_multiclass(self):
        cmd = [
            "choose",
            "--rubrics", "判断诉求类型",
            "--choices", "bug:故障报错,feat:新功能建议,ask:日常咨询",
            "--context", "导出 CSV 按钮点击后白屏，报 500 错误",
        ]
        res = run_cli(cmd)
        data = json.loads(res.stdout)
        self.assertEqual(data.get("primitive"), "choice")
        self.assertEqual(data.get("choice"), "bug")

        probs = data.get("probabilities", {})
        prob_sum = sum(probs.values())
        self.assertAlmostEqual(prob_sum, 1.0, places=3)

        # 开启正逆序去偏置
        res_debias = run_cli(cmd + ["--debias-order"])
        data_debias = json.loads(res_debias.stdout)
        self.assertEqual(data_debias.get("choice"), "bug")

    def test_03_score_rubric(self):
        cmd = [
            "score",
            "--rubrics", "评估情绪激烈程度",
            "--levels", "平静,急躁,严重愤怒",
            "--context", "我都打了五个电话了，到底有没有人管？！",
        ]
        res = run_cli(cmd)
        data = json.loads(res.stdout)
        self.assertEqual(data.get("primitive"), "score")
        self.assertIn(data.get("top_level"), [1, 2])
        self.assertGreaterEqual(data.get("score", 0), 0.5)

        probs = data.get("probabilities", {})
        prob_sum = sum(probs.values())
        self.assertAlmostEqual(prob_sum, 1.0, places=3)

    def test_04_doctor_and_diagnostics(self):
        res_doctor = run_cli(["doctor"])
        self.assertEqual(res_doctor.returncode, 0)
        self.assertIn("HEALTHY", res_doctor.stdout)

        res_stats = run_cli(["cache", "stats"])
        data_stats = json.loads(res_stats.stdout)
        self.assertIn("total_entries", data_stats)

        res_usage = run_cli(["--usage"])
        self.assertEqual(res_usage.returncode, 0)
        self.assertIn("decision-cli", res_usage.stdout)


if __name__ == "__main__":
    unittest.main()
