# Test script for DirectPromptAgent class

import sys
import os
from dotenv import load_dotenv

# Add the parent directory to the path to import the base agents
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from workflow_agents.base_agents import DirectPromptAgent

# Load environment variables from .env file
load_dotenv()

# Load the API key
api_key = os.getenv("OPENAI_API_KEY")

prompt = "What is the Capital of France?"

# Instantiate the DirectPromptAgent
direct_agent = DirectPromptAgent(api_key)

# Use direct_agent to send the prompt and store the response
direct_agent_response = direct_agent.respond(prompt)

# Print the response from the agent
print(direct_agent_response)

# Print an explanatory message describing the knowledge source
# The DirectPromptAgent uses general knowledge from the gpt-3.5-turbo LLM model.
# No system prompt or additional context is provided — the model relies entirely
# on its pre-trained knowledge to answer the question.
print("\nKnowledge source: The agent uses the general knowledge of the gpt-3.5-turbo model.")
print("No system prompt or additional context is provided.")
