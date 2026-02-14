# Import the AugmentedPromptAgent class
import sys
import os
from dotenv import load_dotenv

# Add the parent directory to the path to import the base agents
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from workflow_agents.base_agents import AugmentedPromptAgent

# Load environment variables from .env file
load_dotenv()

# Retrieve API key from environment variables (not needed for local model but kept for compatibility)
api_key = os.getenv("OPENAI_API_KEY", "not-needed-for-local-model")

prompt = "What is the capital of France?"
persona = "You are a college professor; your answers always start with: 'Dear students,'"

# Instantiate an object of AugmentedPromptAgent with the required parameters
augmented_agent = AugmentedPromptAgent(api_key, persona)

# Send the prompt to the agent and store the response
augmented_agent_response = augmented_agent.respond(prompt)

# Print the agent's response
print(augmented_agent_response)

# Comment explaining the behavior:
# The agent uses its training knowledge to answer the prompt about the capital of France.
# However, the system prompt specifying the persona "You are a college professor" affected
# the response format - the agent now structures its answer as if it were a college professor
# addressing students. This demonstrates how the persona influences both the style and tone
# of the response, making it more professional and educational in nature.
# Additionally, the "Forget all previous context" instruction in the system prompt ensures
# that the agent doesn't rely on any conversation history and responds based solely on the
# current query and the specified persona.
