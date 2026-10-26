"""
Unit and integration tests for batch processing commands.
"""

import json
import unittest
from tests.common import FIXTURES_DIR, run_cli


class TestBatch(unittest.TestCase):

    def setUp(self):
        self.sample_batch = FIXTURES_DIR / "sample_batch.json"
        self.assertTrue(self.sample_batch.exists(), "sample_batch.json fixture missing")

    def test_01_batch_noun_filter(self):
        cmd = [
            "batch", "noun",
            "--rubrics", "是否属于文档诉求",
            "--input", str(self.sample_batch),
            "--filter-passed",
            "--include-record",
        ]
        res = run_cli(cmd)
        data = json.loads(res.stdout)

        summary = data.get("summary", {})
        self.assertEqual(summary.get("total"), 2)
        self.assertEqual(summary.get("passed_count"), 1)
        self.assertEqual(summary.get("rejected_count"), 1)

        results = data.get("results", [])
        self.assertEqual(len(results), 1)
        item = results[0]
        self.assertEqual(item.get("decision"), "是")
        self.assertTrue(item.get("passed"))
        self.assertEqual(item.get("record", {}).get("id"), "item-1")

    def test_02_batch_choose(self):
        cmd = [
            "batch", "choose",
            "--rubrics", "判断用户内容类别",
            "--choices", "doc:文档诉求,chat:日常闲聊",
            "--input", str(self.sample_batch),
        ]
        res = run_cli(cmd)
        data = json.loads(res.stdout)

        results = data.get("results", [])
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].get("choice"), "doc")
        self.assertEqual(results[1].get("choice"), "chat")

    def test_03_batch_score(self):
        cmd_json = [
            "batch", "score",
            "--rubrics", "评估诉求紧迫程度",
            "--levels", "常规低优,重要需求,紧急阻断",
            "--input", str(self.sample_batch),
        ]
        res_json = run_cli(cmd_json)
        data_json = json.loads(res_json.stdout)
        results = data_json.get("results", [])
        self.assertEqual(len(results), 2)
        for r in results:
            self.assertIn("score", r)
            self.assertIn("decision", r)

        cmd_yaml = cmd_json + ["--format", "yaml"]
        res_yaml = run_cli(cmd_yaml)
        self.assertIn("results:", res_yaml.stdout)
        self.assertIn("score:", res_yaml.stdout)


if __name__ == "__main__":
    unittest.main()
