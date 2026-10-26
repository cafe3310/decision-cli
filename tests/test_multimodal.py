"""
Unit and integration tests for multimodal (vision + text) inference.
"""

import json
import unittest
from tests.common import FIXTURES_DIR, is_model_ready, run_cli


class TestMultimodal(unittest.TestCase):

    def setUp(self):
        self.red_alert = FIXTURES_DIR / "red_alert.png"
        self.blue_circle = FIXTURES_DIR / "blue_circle.png"
        self.assertTrue(self.red_alert.exists(), "red_alert.png missing")
        self.assertTrue(self.blue_circle.exists(), "blue_circle.png missing")

    def test_01_text_only_model_rejects_image(self):
        if not is_model_ready("qwen3-reranker-4b"):
            self.skipTest("qwen3-reranker-4b is not ready locally")

        res = run_cli([
            "noun",
            "--model", "qwen3-reranker-4b",
            "--rubrics", "是否为红色",
            "--image", str(self.red_alert),
        ], check=False)
        self.assertNotEqual(res.returncode, 0)
        self.assertTrue(
            "不支持" in res.stderr or "vision" in res.stderr.lower() or "image" in res.stderr.lower()
        )

    def test_02_single_image_inference_and_cache(self):
        model_name = "gemma-4-e4b"
        if not is_model_ready(model_name):
            self.skipTest(f"{model_name} is not ready locally")

        cmd = [
            "noun",
            "--model", model_name,
            "--rubrics", "判断这张图片的背景主色调是否为红色",
            "--image", str(self.red_alert),
            "--image-tokens", "70",
        ]
        res1 = run_cli(cmd)
        data1 = json.loads(res1.stdout)
        self.assertEqual(data1.get("decision"), "是")
        self.assertTrue(data1.get("passed"))

        # 缓存命中验证
        res2 = run_cli(cmd)
        data2 = json.loads(res2.stdout)
        self.assertTrue(data2.get("cached"))
        self.assertEqual(data2.get("decision"), "是")

    def test_03_multi_image_choose_with_debias(self):
        model_name = "gemma-4-e4b"
        if not is_model_ready(model_name):
            self.skipTest(f"{model_name} is not ready locally")

        cmd = [
            "choose",
            "--model", model_name,
            "--rubrics", "对比两张图片，哪一张图片是蓝色背景且中间包含白色圆形？",
            "--choices", "img1:第一张图,img2:第二张图",
            "--image", str(self.red_alert),
            "--image-tokens", "70",
            "--image", str(self.blue_circle),
            "--image-tokens", "70",
            "--context", "第一张图：{{image:1}} 第二张图：{{image:2}}",
            "--debias-order",
        ]
        res = run_cli(cmd)
        data = json.loads(res.stdout)
        self.assertEqual(data.get("primitive"), "choice")
        self.assertEqual(data.get("choice"), "img2")


if __name__ == "__main__":
    unittest.main()
