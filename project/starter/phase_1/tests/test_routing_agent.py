import json
import os
import sys
import unittest

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
PHASE_DIR = os.path.dirname(TEST_DIR)
sys.path.insert(0, PHASE_DIR)

from workflow_agents import base_agents


class TestRoutingAgent(unittest.TestCase):
    def setUp(self):
        self.original_embedding = base_agents._embedding

        def fake_embedding(text):
            text_lower = text.lower()
            if "texas" in text_lower:
                return [1.0, 0.0]
            if "europe" in text_lower or "italy" in text_lower:
                return [0.0, 1.0]
            return [0.1, 0.1]

        base_agents._embedding = fake_embedding

    def tearDown(self):
        base_agents._embedding = self.original_embedding

    def test_route_selects_best_agent(self):
        routing_agent = base_agents.RoutingAgent("dummy-key", [])
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

        golden_path = os.path.join(os.path.dirname(__file__), "golden", "routing_result.json")
        with open(golden_path, "r", encoding="utf-8") as golden_file:
            golden = json.load(golden_file)

        self.assertEqual(response, golden["response"])


if __name__ == "__main__":
    unittest.main()
