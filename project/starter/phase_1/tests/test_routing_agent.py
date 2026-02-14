import json
import os
import sys
import unittest
from unittest.mock import patch, MagicMock

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PHASE_DIR = os.path.dirname(TEST_DIR)
sys.path.insert(0, PHASE_DIR)

from workflow_agents import base_agents


class TestRoutingAgent(unittest.TestCase):
    """Unit tests for the RoutingAgent using mocked embeddings."""

    @patch("workflow_agents.base_agents.OpenAI")
    def test_route_selects_best_agent(self, mock_openai_cls):
        """Routing selects the agent whose description is most similar."""
        routing_agent = base_agents.RoutingAgent("dummy-key", [])

        # Mock get_embedding on the instance to return deterministic vectors
        def fake_embedding(text):
            text_lower = text.lower()
            if "texas" in text_lower:
                return [1.0, 0.0]
            if "europe" in text_lower or "italy" in text_lower:
                return [0.0, 1.0]
            return [0.1, 0.1]

        routing_agent.get_embedding = fake_embedding

        agents = [
            {
                "name": "texas agent",
                "description": "Answer a question about Texas",
                "func": lambda x: "texas",
            },
            {
                "name": "europe agent",
                "description": "Answer a question about Europe",
                "func": lambda x: "europe",
            },
        ]
        routing_agent.agents = agents

        prompt = "Tell me about the history of Rome, Texas"
        response = routing_agent.route(prompt)

        evaluation_path = os.path.join(os.path.dirname(__file__), "evaluation", "routing_result.json")
        with open(evaluation_path, "r", encoding="utf-8") as evaluation_file:
            evaluation = json.load(evaluation_file)

        self.assertEqual(response, evaluation["response"])


if __name__ == "__main__":
    unittest.main()
