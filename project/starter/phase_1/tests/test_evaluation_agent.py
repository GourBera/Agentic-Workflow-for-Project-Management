import json
import os
import sys
import unittest
from unittest.mock import patch, MagicMock

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PHASE_DIR = os.path.dirname(TEST_DIR)
sys.path.insert(0, PHASE_DIR)

from workflow_agents import base_agents


class _FakeWorker:
    """Fake worker agent that returns a fixed response."""
    def respond(self, prompt):
        return "draft answer"


class TestEvaluationAgent(unittest.TestCase):
    """Unit tests for the EvaluationAgent using mocked LLM responses."""

    @patch("workflow_agents.base_agents.OpenAI")
    def test_evaluation_flow(self, mock_openai_cls):
        """Evaluator iterates until the response passes criteria."""
        worker = _FakeWorker()
        evaluator = base_agents.EvaluationAgent(
            "dummy-key",
            "evaluation persona",
            "must follow format",
            worker,
            5,
        )

        # Create mock responses for the evaluator's LLM calls:
        # Iteration 1: evaluation = "No, missing format", then correction instructions
        # Iteration 2: evaluation = "Yes, looks good"
        mock_eval_fail = MagicMock()
        mock_eval_fail.choices = [MagicMock(message=MagicMock(content="No, missing format"))]

        mock_instructions = MagicMock()
        mock_instructions.choices = [MagicMock(message=MagicMock(content="Fix: use the required format"))]

        mock_eval_pass = MagicMock()
        mock_eval_pass.choices = [MagicMock(message=MagicMock(content="Yes, looks good"))]

        evaluator.client = MagicMock()
        evaluator.client.chat.completions.create.side_effect = [
            mock_eval_fail,       # Iteration 1 - evaluation
            mock_instructions,    # Iteration 1 - correction instructions
            mock_eval_pass,       # Iteration 2 - evaluation
        ]

        result = evaluator.evaluate("write a formatted answer")

        evaluation_path = os.path.join(os.path.dirname(__file__), "evaluation", "evaluation_result.json")
        with open(evaluation_path, "r", encoding="utf-8") as evaluation_file:
            evaluation = json.load(evaluation_file)

        self.assertEqual(result["iterations"], evaluation["iterations"])
        self.assertEqual(result["evaluation"], evaluation["evaluation"])
        self.assertEqual(result["final_response"], evaluation["final_response"])


if __name__ == "__main__":
    unittest.main()
