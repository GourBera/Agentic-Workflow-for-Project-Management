import json
import os
import sys
import unittest

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PHASE_DIR = os.path.dirname(TEST_DIR)
sys.path.insert(0, PHASE_DIR)

from workflow_agents import base_agents


class _FakeWorker:
    def respond(self, prompt):
        return "draft answer"


class _FakeChoice:
    def __init__(self, content):
        self.message = type("_Message", (), {"content": content})()


class _FakeResponse:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]


class TestEvaluationAgent(unittest.TestCase):
    def setUp(self):
        self.original_completion = base_agents._completion

        self._responses = iter(
            [
                _FakeResponse('{"passes": false, "reason": "missing format"}'),
                _FakeResponse('{"instructions": "use the required format"}'),
                _FakeResponse('{"passes": true, "reason": "ok"}'),
            ]
        )

        def fake_completion(*args, **kwargs):
            return next(self._responses)

        base_agents._completion = fake_completion

    def tearDown(self):
        base_agents._completion = self.original_completion

    def test_evaluation_flow(self):
        worker = _FakeWorker()
        evaluator = base_agents.EvaluationAgent(
            "dummy-key",
            "evaluation persona",
            "must follow format",
            worker,
            5,
        )
        result = evaluator.evaluate("write a formatted answer")

        golden_path = os.path.join(os.path.dirname(__file__), "golden", "evaluation_result.json")
        with open(golden_path, "r", encoding="utf-8") as golden_file:
            golden = json.load(golden_file)

        self.assertEqual(result["iterations"], golden["iterations"])
        self.assertEqual(result["evaluation"], golden["evaluation"])
        self.assertEqual(result["final_response"], golden["final_response"])


if __name__ == "__main__":
    unittest.main()
