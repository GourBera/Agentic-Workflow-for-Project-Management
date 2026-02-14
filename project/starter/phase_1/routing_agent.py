
# Import the KnowledgeAugmentedPromptAgent and RoutingAgent
import sys
import os
from dotenv import load_dotenv

# Add the parent directory to the path to import the base agents
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from workflow_agents.base_agents import KnowledgeAugmentedPromptAgent, RoutingAgent

# Load environment variables from .env file
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

persona = "You are a college professor"

knowledge = "You know everything about Texas"
# Define the Texas Knowledge Augmented Prompt Agent
texas_agent = KnowledgeAugmentedPromptAgent(api_key, persona, knowledge)

knowledge = "You know everything about Europe"
# Define the Europe Knowledge Augmented Prompt Agent
europe_agent = KnowledgeAugmentedPromptAgent(api_key, persona, knowledge)

persona = "You are a college math professor"
knowledge = "You know everything about math, you take prompts with numbers, extract math formulas, and show the answer without explanation"
# Define the Math Knowledge Augmented Prompt Agent
math_agent = KnowledgeAugmentedPromptAgent(api_key, persona, knowledge)

routing_agent = RoutingAgent(api_key, [])
agents = [
    {
        "name": "texas agent",
        "description": "Answer a question about Texas",
        "func": lambda x: texas_agent.respond(x)
    },
    {
        "name": "europe agent",
        "description": "Answer a question about Europe",
        "func": lambda x: europe_agent.respond(x)
    },
    {
        "name": "math agent",
        "description": "When a prompt contains numbers, respond with a math formula",
        "func": lambda x: math_agent.respond(x)
    }
]

routing_agent.agents = agents

# Print the RoutingAgent responses to the following prompts
prompts = [
    "Tell me about the history of Rome, Texas",
    "Tell me about the history of Rome, Italy",
    "One story takes 2 days, and there are 20 stories"
]

for prompt in prompts:
    print(f"\nPrompt: {prompt}")
    response = routing_agent.route(prompt)
    print(f"Response: {response}")
