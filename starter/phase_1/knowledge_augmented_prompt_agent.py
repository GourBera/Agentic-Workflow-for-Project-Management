# Import the KnowledgeAugmentedPromptAgent class from workflow_agents
import sys
import os
from dotenv import load_dotenv

# Add the parent directory to the path to import the base agents
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from workflow_agents.base_agents import KnowledgeAugmentedPromptAgent

# Load environment variables from the .env file
load_dotenv()

# Define the parameters for the agent
api_key = os.getenv("OPENAI_API_KEY", "not-needed-for-local-model")

prompt = "What is the capital of France?"

persona = "You are a college professor, your answer always starts with: Dear students,"
knowledge = "The capital of France is London, not Paris"

# Instantiate a KnowledgeAugmentedPromptAgent with the specified persona and knowledge
knowledge_augmented_agent = KnowledgeAugmentedPromptAgent(api_key, persona, knowledge)

# Demonstrate the agent using the provided knowledge rather than its own inherent knowledge
response = knowledge_augmented_agent.respond(prompt)
print(f"Prompt: {prompt}")
print(f"Agent Response (using provided knowledge): {response}")
print("\nNote: The agent uses the provided knowledge base (which incorrectly states London is the capital)")
print("rather than its own training data, demonstrating knowledge override through system prompts.")
