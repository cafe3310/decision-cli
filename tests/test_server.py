"""
Unit and integration tests for HTTP API server and daemon lifecycle.
"""

import json
import os
import signal
import subprocess
import time
import unittest
import urllib.error
import urllib.request
from tests.common import find_free_port, get_cli_command, run_cli


class TestServerAndLifecycle(unittest.TestCase):

    def test_01_http_serve_and_endpoints(self):
        port = find_free_port()
        cmd = get_cli_command() + ["serve", "--port", str(port)]
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        try:
            # 等待服务就绪
            health_url = f"http://127.0.0.1:{port}/health"
            ready = False
            for _ in range(25):
                try:
                    req = urllib.request.Request(health_url)
                    with urllib.request.urlopen(req, timeout=1) as resp:
                        if resp.status == 200:
                            data = json.loads(resp.read().decode())
                            if data.get("status") == "ok":
                                ready = True
                                break
                except Exception:
                    time.sleep(0.2)

            self.assertTrue(ready, f"Server failed to start on port {port}")

            # 测试 POST /v1/noun
            noun_url = f"http://127.0.0.1:{port}/v1/noun"
            payload_noun = json.dumps({
                "rubrics": "是否包含讽刺语气",
                "context": "你可真准时啊，会议都开完了你才来。",
            }).encode("utf-8")
            req_noun = urllib.request.Request(
                noun_url,
                data=payload_noun,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req_noun, timeout=10) as resp:
                self.assertEqual(resp.status, 200)
                res_data = json.loads(resp.read().decode())
                self.assertEqual(res_data.get("decision"), "是")

            # 测试 POST /v1/systemone
            systemone_url = f"http://127.0.0.1:{port}/v1/systemone"
            payload_so = json.dumps({
                "state": "系统报 500 且无法导出报表",
                "questions": {
                    "q1": {
                        "primitive": "noul",
                        "instruction": "是否发生异常故障？",
                    }
                },
            }).encode("utf-8")
            req_so = urllib.request.Request(
                systemone_url,
                data=payload_so,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req_so, timeout=10) as resp:
                self.assertEqual(resp.status, 200)
                res_so = json.loads(resp.read().decode())
                self.assertIn("answers", res_so)
                self.assertEqual(res_so["answers"]["q1"].get("type"), "noul")

        finally:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
            if proc.stdout:
                proc.stdout.close()
            if proc.stderr:
                proc.stderr.close()

    def test_02_kill_lifecycle(self):
        # 1. dry-run 验证
        res_dry = run_cli(["kill", "--dry-run"])
        self.assertEqual(res_dry.returncode, 0)

        # 2. 执行 kill
        res_kill = run_cli(["kill"])
        self.assertEqual(res_kill.returncode, 0)

        # 3. 再次执行断言整洁
        res_clean = run_cli(["kill"])
        self.assertEqual(res_clean.returncode, 0)
        self.assertTrue(
            "当前环境整洁" in res_clean.stdout or "未发现" in res_clean.stdout
        )


if __name__ == "__main__":
    unittest.main()
