# agentic_workflow.py

# Import the following agents: ActionPlanningAgent, KnowledgeAugmentedPromptAgent, EvaluationAgent, RoutingAgent
import sys
import os
from dotenv import load_dotenv

# Add the parent directory to the path to import the base agents
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'phase_1'))
from workflow_agents.base_agents import ActionPlanningAgent, KnowledgeAugmentedPromptAgent, EvaluationAgent, RoutingAgent

# Load environment variables
load_dotenv()

# Load the OpenAI key (not needed for local model but kept for compatibility)
api_key = os.getenv("OPENAI_API_KEY", "not-needed-for-local-model")

# Load the product spec document
script_dir = os.path.dirname(os.path.abspath(__file__))
product_spec_path = os.path.join(script_dir, 'Product-Spec-Email-Router.txt')
with open(product_spec_path, 'r') as f:
    product_spec = f.read()

# Instantiate all the agents

# Action Planning Agent
knowledge_action_planning = (
    "Stories are defined from a product spec by identifying a "
    "persona, an action, and a desired outcome for each story. "
    "Each story represents a specific functionality of the product "
    "described in the specification. \n"
    "Features are defined by grouping related user stories. \n"
    "Tasks are defined for each story and represent the engineering "
    "work required to develop the product. \n"
    "A development Plan for a product contains all these components"
)
# Instantiate an action_planning_agent using the 'knowledge_action_planning'
action_planning_agent = ActionPlanningAgent(api_key, knowledge_action_planning)

# Product Manager - Knowledge Augmented Prompt Agent
persona_product_manager = "You are a Product Manager, you are responsible for defining the user stories for a product."
knowledge_product_manager = (
    "Stories are defined by writing sentences with a persona, an action, and a desired outcome. "
    "The sentences always start with: As a "
    "Write several stories for the product spec below, where the personas are the different users of the product. "
    + product_spec
)
# Instantiate a product_manager_knowledge_agent using 'persona_product_manager' and the completed 'knowledge_product_manager'
product_manager_knowledge_agent = KnowledgeAugmentedPromptAgent(api_key, persona_product_manager, knowledge_product_manager)

# Product Manager - Evaluation Agent
persona_product_manager_eval = "You are one of the expert Product Manager evaluation agent that checks the answers of other worker agents."
evaluation_criteria_pm = "The answer should be product user stories that follow the following structure: As a [type of user], I want [an action or feature] so that [benefit/value]."
product_manager_evaluation_agent = EvaluationAgent(api_key, persona_product_manager_eval, evaluation_criteria_pm, product_manager_knowledge_agent, 3)

# Program Manager - Knowledge Augmented Prompt Agent
persona_program_manager = "You are a Program Manager, you are responsible for defining the features for a product."
knowledge_program_manager = "Features of a product are defined by organizing similar user stories into cohesive groups."
# Instantiate a program_manager_knowledge_agent using 'persona_program_manager' and 'knowledge_program_manager'
program_manager_knowledge_agent = KnowledgeAugmentedPromptAgent(api_key, persona_program_manager, knowledge_program_manager)

# Program Manager - Evaluation Agent
persona_program_manager_eval = "You are an evaluation agent that checks the answers of other worker agents."
evaluation_criteria_progm = ("The answer should be product features that follow the following structure: " 
                              "Feature Name: A clear, concise title that identifies the capability\n"
                              "Description: A brief explanation of what the feature does and its purpose\n"
                              "Key Functionality: The specific capabilities or actions the feature provides\n"
                              "User Benefit: How this feature creates value for the user")
program_manager_evaluation_agent = EvaluationAgent(api_key, persona_program_manager_eval, evaluation_criteria_progm, program_manager_knowledge_agent, 3)

# Development Engineer - Knowledge Augmented Prompt Agent
persona_dev_engineer = "You are a Development Engineer, you are responsible for defining the development tasks for a product."
knowledge_dev_engineer = "Development tasks are defined by identifying what needs to be built to implement each user story."
# Instantiate a development_engineer_knowledge_agent using 'persona_dev_engineer' and 'knowledge_dev_engineer'
development_engineer_knowledge_agent = KnowledgeAugmentedPromptAgent(api_key, persona_dev_engineer, knowledge_dev_engineer)

# Development Engineer - Evaluation Agent
persona_dev_engineer_eval = "You are an evaluation agent that checks the answers of other worker agents."
evaluation_criteria_dev = ("The answer should be tasks following this exact structure: "
                           "Task ID: A unique identifier for tracking purposes\n"
                           "Task Title: Brief description of the specific development work\n"
                           "Related User Story: Reference to the parent user story\n"
                           "Description: Detailed explanation of the technical work required\n"
                           "Acceptance Criteria: Specific requirements that must be met for completion\n"
                           "Estimated Effort: Time or complexity estimation\n"
                           "Dependencies: Any tasks that must be completed first")
development_engineer_evaluation_agent = EvaluationAgent(api_key, persona_dev_engineer_eval, evaluation_criteria_dev, development_engineer_knowledge_agent, 3)


# Define the support functions for the routes of the routing agent
def product_manager_support_function(query):
    """Support function for Product Manager - gets response and evaluates it."""
    print(f"  [Product Manager] Processing: {query[:80]}...")
    response = product_manager_knowledge_agent.respond(query)
    return response

def program_manager_support_function(query):
    """Support function for Program Manager - gets response and evaluates it."""
    print(f"  [Program Manager] Processing: {query[:80]}...")
    response = program_manager_knowledge_agent.respond(query)
    return response

def development_engineer_support_function(query):
    """Support function for Development Engineer - gets response and evaluates it."""
    print(f"  [Development Engineer] Processing: {query[:80]}...")
    response = development_engineer_knowledge_agent.respond(query)
    return response

# Simple routing function that routes to the most appropriate agent
def simple_routing_function(step):
    """Route the step to the most appropriate agent based on keywords."""
    step_lower = step.lower()
    
    # Route to Development Engineer for task-related steps
    if any(word in step_lower for word in ['task', 'develop', 'build', 'implement', 'code', 'estimate', 'effort']):
        return development_engineer_support_function(step)
    # Route to Program Manager for feature-related steps
    elif any(word in step_lower for word in ['feature', 'group', 'organize', 'component']):
        return program_manager_support_function(step)
    # Default to Product Manager for story-related steps
    else:
        return product_manager_support_function(step)

# Run the workflow

print("\n*** Workflow execution started ***\n")
# Workflow Prompt
# ****
workflow_prompt = "What would the development tasks for this product be?"
# ****
print(f"Task to complete in this workflow, workflow prompt = {workflow_prompt}")

print("\nDefining workflow steps from the workflow prompt")
# Implement the workflow:
# 1. Use the 'action_planning_agent' to extract steps from the 'workflow_prompt'
workflow_steps = action_planning_agent.extract_steps_from_prompt(workflow_prompt)
print(f"Extracted workflow steps: {len(workflow_steps)} steps\n")

# 2. Initialize an empty list to store 'completed_steps'
completed_steps = []

# 3. Loop through the extracted workflow steps
for i, step in enumerate(workflow_steps, 1):
    print(f"--- Step {i}: {step[:100]}{'...' if len(step) > 100 else ''} ---")
    
    # a. For each step, use simple routing to route the step to the appropriate support function
    try:
        result = simple_routing_function(step)
        print(f"Response: {result[:150]}{'...' if len(result) > 150 else ''}\n")
        
        # b. Append the result to 'completed_steps'
        completed_steps.append(result)
    except Exception as e:
        print(f"Error processing step: {e}\n")

# 4. After the loop, print the final output of the workflow
print("\n*** Workflow execution completed ***\n")
if completed_steps:
    print("=== Final Workflow Output ===")
    print(completed_steps[-1])
else:
    print("No steps were successfully completed.")