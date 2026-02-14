"""Reusable agent library for agentic workflows.

Contains seven agent classes for building AI-powered workflows:
DirectPromptAgent, AugmentedPromptAgent, KnowledgeAugmentedPromptAgent,
RAGKnowledgePromptAgent, EvaluationAgent, RoutingAgent, ActionPlanningAgent.
"""
import ast
import csv
import json
import re
import time
import uuid
import warnings
from datetime import datetime
from typing import Any, Callable, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
from openai import OpenAI

# Default models as specified by project requirements
COMPLETION_MODEL = "gpt-3.5-turbo"
EMBEDDING_MODEL = "text-embedding-3-large"

# Vocareum proxy base URL (routes requests through Vocareum's OpenAI gateway)
OPENAI_BASE_URL = "https://openai.vocareum.com/v1"


def _retry_with_backoff(
    func: Callable[..., Any],
    *args: Any,
    max_retries: int = 3,
    base_delay: float = 0.5,
    **kwargs: Any,
) -> Any:
    """Retry a callable with exponential backoff on transient failures.

    Args:
        func: The callable to retry.
        max_retries: Maximum number of retry attempts (default 3).
        base_delay: Initial delay in seconds between retries (default 1.0).

    Returns:
        The result of the successful function call.

    Raises:
        Exception: The last exception if all retries are exhausted.
    """
    last_exc: Optional[Exception] = None
    for attempt in range(max_retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            last_exc = exc
            if attempt >= max_retries:
                break
            time.sleep(base_delay * (2 ** attempt))
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("Retry failed without an exception.")


# DirectPromptAgent class definition
class DirectPromptAgent:
    """A direct prompt agent that forwards user input to the LLM without a system prompt.

    This is the simplest agent type. It relays the user's input directly to the
    gpt-3.5-turbo model and returns the response without additional context,
    memory, or specialized tools.
    """

    def __init__(self, openai_api_key: str) -> None:
        """Initialize the DirectPromptAgent.

        Args:
            openai_api_key: The OpenAI API key for authentication.
        """
        self.openai_api_key = openai_api_key
        self.client = OpenAI(api_key=openai_api_key, base_url=OPENAI_BASE_URL)

    def respond(self, prompt: str) -> str:
        """Send a prompt directly to the LLM and return the text response.

        Args:
            prompt: The user's input prompt.

        Returns:
            The text content of the LLM's response.
        """
        response = _retry_with_backoff(
            self.client.chat.completions.create,
            model=COMPLETION_MODEL,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content
        
# AugmentedPromptAgent class definition
class AugmentedPromptAgent:
    """A prompt agent that responds according to a predefined persona.

    Uses a system prompt to instruct the LLM to adopt a specific persona,
    producing more targeted and contextually relevant outputs.
    """

    def __init__(self, openai_api_key: str, persona: str) -> None:
        """Initialize the AugmentedPromptAgent.

        Args:
            openai_api_key: The OpenAI API key for authentication.
            persona: A description of the persona the agent should adopt.
        """
        self.openai_api_key = openai_api_key
        self.persona = persona
        self.client = OpenAI(api_key=openai_api_key, base_url=OPENAI_BASE_URL)

    def respond(self, input_text: str) -> str:
        """Generate a persona-based response to the given input.

        Args:
            input_text: The user's input prompt.

        Returns:
            The text content of the LLM's response.
        """
        response = _retry_with_backoff(
            self.client.chat.completions.create,
            model=COMPLETION_MODEL,
            messages=[
                {"role": "system", "content": f"You are {self.persona}. Forget all previous context."},
                {"role": "user", "content": input_text},
            ],
            temperature=0,
        )

        return response.choices[0].message.content

# KnowledgeAugmentedPromptAgent class definition
class KnowledgeAugmentedPromptAgent:
    """A prompt agent that answers using only explicitly provided knowledge.

    Combines a persona with specific knowledge so responses are grounded in the
    provided information rather than the model's general training data.
    """

    def __init__(self, openai_api_key: str, persona: str, knowledge: str) -> None:
        """Initialize the KnowledgeAugmentedPromptAgent.

        Args:
            openai_api_key: The OpenAI API key for authentication.
            persona: A description of the persona the agent should adopt.
            knowledge: The specific knowledge the agent should use to answer.
        """
        self.openai_api_key = openai_api_key
        self.persona = persona
        self.knowledge = knowledge
        self.client = OpenAI(api_key=openai_api_key, base_url=OPENAI_BASE_URL)

    def respond(self, input_text: str) -> str:
        """Generate a knowledge-grounded response to the given input.

        Args:
            input_text: The user's input prompt.

        Returns:
            The text content of the LLM's response.
        """
        system_message = (
            f"You are {self.persona} knowledge-based assistant. Forget all previous context. "
            f"Use only the following knowledge to answer, do not use your own knowledge: {self.knowledge} "
            "Answer the prompt based on this knowledge, not your own."
        )
        response = _retry_with_backoff(
            self.client.chat.completions.create,
            model=COMPLETION_MODEL,
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": input_text},
            ],
        )
        return response.choices[0].message.content

# RAGKnowledgePromptAgent class definition
class RAGKnowledgePromptAgent:
    """
    An agent that uses Retrieval-Augmented Generation (RAG) to find knowledge from a large corpus
    and leverages embeddings to respond to prompts based solely on retrieved information.
    """

    def __init__(self, openai_api_key: str, persona: str, chunk_size: int = 2000, chunk_overlap: int = 100) -> None:
        """
        Initializes the RAGKnowledgePromptAgent with API credentials and configuration settings.

        Parameters:
        openai_api_key (str): API key for accessing OpenAI.
        persona (str): Persona description for the agent.
        chunk_size (int): The size of text chunks for embedding. Defaults to 2000.
        chunk_overlap (int): Overlap between consecutive chunks. Defaults to 100.
        """
        self.persona = persona
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.openai_api_key = openai_api_key
        self.client = OpenAI(api_key=openai_api_key, base_url=OPENAI_BASE_URL)
        self.unique_filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.csv"

    def get_embedding(self, text: str) -> List[float]:
        """
        Fetches the embedding vector for given text using the text-embedding-3-large model.

        Parameters:
            text (str): Text to embed.

        Returns:
            list: The embedding vector.
        """
        response = _retry_with_backoff(
            self.client.embeddings.create,
            model=EMBEDDING_MODEL,
            input=text,
        )
        return response.data[0].embedding

    def calculate_similarity(self, vector_one: Iterable[float], vector_two: Iterable[float]) -> float:
        """
        Calculates cosine similarity between two vectors.

        Parameters:
        vector_one (list): First embedding vector.
        vector_two (list): Second embedding vector.

        Returns:
        float: Cosine similarity between vectors.
        """
        vec1, vec2 = np.array(vector_one), np.array(vector_two)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def chunk_text(self, text: str) -> List[Dict[str, Any]]:
        """
        Splits text into manageable chunks, attempting natural breaks.

        Parameters:
        text (str): Text to split into chunks.

        Returns:
        list: List of dictionaries containing chunk metadata.
        """
        separator = "\n"
        text = re.sub(r'\s+', ' ', text).strip()

        if len(text) <= self.chunk_size:
            return [{"chunk_id": 0, "text": text, "chunk_size": len(text)}]

        chunks, start, chunk_id = [], 0, 0

        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            if separator in text[start:end]:
                end = start + text[start:end].rindex(separator) + len(separator)

            chunks.append({
                "chunk_id": chunk_id,
                "text": text[start:end],
                "chunk_size": end - start,
                "start_char": start,
                "end_char": end
            })

            if end >= len(text):
                break
            start = end - self.chunk_overlap
            chunk_id += 1

        with open(f"chunks-{self.unique_filename}", 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=["text", "chunk_size"])
            writer.writeheader()
            for chunk in chunks:
                writer.writerow({k: chunk[k] for k in ["text", "chunk_size"]})

        return chunks

    def calculate_embeddings(self) -> pd.DataFrame:
        """
        Calculates embeddings for each chunk and stores them in a CSV file.

        Returns:
        DataFrame: DataFrame containing text chunks and their embeddings.
        """
        df = pd.read_csv(f"chunks-{self.unique_filename}", encoding='utf-8')
        df.loc[:, 'embeddings'] = df['text'].apply(self.get_embedding)
        df.to_csv(f"embeddings-{self.unique_filename}", encoding='utf-8', index=False)
        return df

    def find_prompt_in_knowledge(self, prompt: str) -> str:
        """
        Finds and responds to a prompt based on similarity with embedded knowledge.

        Parameters:
        prompt (str): User input prompt.

        Returns:
        str: Response derived from the most similar chunk in knowledge.
        """
        prompt_embedding = self.get_embedding(prompt)
        df = pd.read_csv(f"embeddings-{self.unique_filename}", encoding='utf-8')
        df.loc[:, 'embeddings'] = df['embeddings'].apply(lambda x: np.array(ast.literal_eval(x)))
        df.loc[:, 'similarity'] = df['embeddings'].apply(lambda emb: self.calculate_similarity(prompt_embedding, emb))

        best_chunk = df.loc[df['similarity'].idxmax(), 'text']

        response = _retry_with_backoff(
            self.client.chat.completions.create,
            model=COMPLETION_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": f"You are {self.persona}, a knowledge-based assistant. Forget previous context.",
                },
                {"role": "user", "content": f"Answer based only on this information: {best_chunk}. Prompt: {prompt}"},
            ],
            temperature=0,
        )

        return response.choices[0].message.content

class EvaluationAgent:
    """An agent that evaluates worker agent responses against defined criteria.

    Implements an iterative evaluation loop: gets a response from a worker agent,
    evaluates it against criteria, and if needed generates correction instructions
    for refinement.  Uses temperature=0 for deterministic evaluations.

    Attributes:
        openai_api_key: The OpenAI API key.
        persona: The evaluator persona description.
        evaluation_criteria: The criteria to evaluate responses against.
        worker_agent: The worker agent whose responses are evaluated.
        max_interactions: Maximum evaluation-refinement iterations.
    """

    def __init__(
        self,
        openai_api_key: str,
        persona: str,
        evaluation_criteria: str,
        worker_agent: Any,
        max_interactions: int,
    ) -> None:
        """Initialize the EvaluationAgent.

        Args:
            openai_api_key: The OpenAI API key for authentication.
            persona: The evaluator persona description.
            evaluation_criteria: Criteria to evaluate worker responses against.
            worker_agent: The agent whose responses are evaluated and refined.
            max_interactions: Maximum number of evaluation-refinement loops.
        """
        self.openai_api_key = openai_api_key
        self.persona = persona
        self.evaluation_criteria = evaluation_criteria
        self.worker_agent = worker_agent
        self.max_interactions = max_interactions
        self.client = OpenAI(api_key=openai_api_key, base_url=OPENAI_BASE_URL)

    def evaluate(self, initial_prompt: str) -> Dict[str, Any]:
        """Evaluate and iteratively refine a worker agent's response.

        Runs an evaluation loop up to *max_interactions* times.  In each
        iteration the worker agent generates a response, the evaluator judges
        it against criteria, and if it fails, correction instructions are
        generated and fed back to the worker.

        Args:
            initial_prompt: The initial prompt to send to the worker agent.

        Returns:
            A dictionary with keys:
                - final_response (str): The last response from the worker agent.
                - evaluation (str): The last evaluation result text.
                - iterations (int): Number of iterations performed.
        """
        prompt_to_evaluate = initial_prompt
        iteration_count = 0
        final_response: Optional[str] = None
        final_evaluation: Optional[str] = None

        for i in range(self.max_interactions):
            iteration_count = i + 1
            print(f"\n--- Interaction {iteration_count} ---")

            # Step 1: Worker agent generates a response
            print(" Step 1: Worker agent generates a response to the prompt")
            print(f"Prompt:\n{prompt_to_evaluate}")
            response_from_worker = self.worker_agent.respond(prompt_to_evaluate)
            print(f"Worker Agent Response:\n{response_from_worker}")

            # Step 2: Evaluate the response against criteria
            print(" Step 2: Evaluator agent judges the response")
            eval_prompt = (
                f"Does the following answer: {response_from_worker}\n"
                f"Meet this criteria: {self.evaluation_criteria}\n"
                "Respond Yes or No, and the reason why it does or doesn't meet the criteria."
            )
            response = _retry_with_backoff(
                self.client.chat.completions.create,
                model=COMPLETION_MODEL,
                messages=[{"role": "user", "content": eval_prompt}],
                temperature=0,
            )
            evaluation = response.choices[0].message.content.strip()
            final_evaluation = evaluation
            final_response = response_from_worker
            print(f"Evaluator Agent Evaluation:\n{evaluation}")

            # Step 3: Check if evaluation passes
            print(" Step 3: Check if evaluation is positive")
            if evaluation.lower().startswith("yes"):
                print("Final solution accepted.")
                break

            # Step 4: Generate correction instructions
            print(" Step 4: Generate instructions to correct the response")
            instruction_prompt = (
                f"Provide instructions to fix an answer based on these reasons "
                f"why it is incorrect: {evaluation}"
            )
            response = _retry_with_backoff(
                self.client.chat.completions.create,
                model=COMPLETION_MODEL,
                messages=[{"role": "user", "content": instruction_prompt}],
                temperature=0,
            )
            instructions = response.choices[0].message.content.strip()
            print(f"Instructions to fix:\n{instructions}")

            # Step 5: Feed corrections back to the worker
            print(" Step 5: Send feedback to worker agent for refinement")
            prompt_to_evaluate = (
                f"The original prompt was: {initial_prompt}\n"
                f"The response to that prompt was: {response_from_worker}\n"
                "It has been evaluated as incorrect.\n"
                f"Make only these corrections, do not alter content validity: {instructions}"
            )
        return {
            "final_response": final_response,
            "evaluation": final_evaluation,
            "iterations": iteration_count,
        }


class RoutingAgent:
    """Routes user prompts to the best-matching agent using embedding similarity.

    Computes embeddings for the user prompt and each agent's description
    using the text-embedding-3-large model, then selects the agent with the
    highest cosine similarity score.

    Attributes:
        openai_api_key: The OpenAI API key.
        agents: A list of agent route dicts with name, description, and func.
    """

    def __init__(self, openai_api_key: str, agents: List[Dict[str, Any]]) -> None:
        """Initialize the RoutingAgent.

        Args:
            openai_api_key: The OpenAI API key for authentication.
            agents: A list of dicts, each with 'name', 'description', and 'func' keys.
        """
        self.openai_api_key = openai_api_key
        self.agents = agents
        self.client = OpenAI(api_key=openai_api_key, base_url=OPENAI_BASE_URL)

    def get_embedding(self, text: str) -> List[float]:
        """Compute an embedding for the given text using text-embedding-3-large.

        Args:
            text: The text to embed.

        Returns:
            A list of floats representing the embedding vector.
        """
        response = _retry_with_backoff(
            self.client.embeddings.create,
            model=EMBEDDING_MODEL,
            input=text,
        )
        return response.data[0].embedding

    def route(self, user_input: str) -> str:
        """Route a user prompt to the most appropriate agent.

        Computes cosine similarity between the prompt embedding and each
        agent description embedding, then calls the best-matching agent's
        function.

        Args:
            user_input: The user's input prompt.

        Returns:
            The selected agent's response string.
        """
        input_emb = np.array(self.get_embedding(user_input))
        best_agent: Optional[Dict[str, Any]] = None
        best_score = -1.0

        for agent in self.agents:
            agent_emb = np.array(self.get_embedding(agent["description"]))
            similarity = float(
                np.dot(input_emb, agent_emb)
                / (np.linalg.norm(input_emb) * np.linalg.norm(agent_emb))
            )
            print(f"Similarity with {agent['name']}: {similarity}")

            if similarity > best_score:
                best_score = similarity
                best_agent = agent

        if best_agent is None:
            return "Sorry, no suitable agent could be selected."

        print(f"[Router] Best agent: {best_agent['name']} (score={best_score:.3f})")
        return best_agent["func"](user_input)


class ActionPlanningAgent:
    """An agent that extracts actionable steps from a user prompt.

    Uses its provided knowledge to dynamically identify and list the steps
    required to execute a task described in the user's prompt.

    Attributes:
        openai_api_key: The OpenAI API key.
        knowledge: Domain knowledge the agent uses to plan actions.
    """

    def __init__(self, openai_api_key: str, knowledge: str) -> None:
        """Initialize the ActionPlanningAgent.

        Args:
            openai_api_key: The OpenAI API key for authentication.
            knowledge: The domain knowledge used for action planning.
        """
        self.openai_api_key = openai_api_key
        self.knowledge = knowledge
        self.client = OpenAI(api_key=openai_api_key, base_url=OPENAI_BASE_URL)

    def extract_steps_from_prompt(self, prompt: str) -> List[str]:
        """Extract a list of action steps from a user prompt.

        Sends the prompt to the LLM along with the agent's knowledge and
        processes the response into a clean list of steps.

        Args:
            prompt: The user's input describing the desired action.

        Returns:
            A list of step strings extracted from the LLM response.
        """
        system_prompt = (
            "You are an action planning agent. Using your knowledge, you extract "
            "from the user prompt the steps requested to complete the action the "
            "user is asking for. You return the steps as a list. Only return the "
            "steps in your knowledge. Forget any previous context. "
            f"This is your knowledge: {self.knowledge}"
        )

        response = _retry_with_backoff(
            self.client.chat.completions.create,
            model=COMPLETION_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
        )

        response_text = response.choices[0].message.content.strip()

        # Try JSON array first (structured output), fall back to line-splitting
        try:
            parsed = json.loads(response_text)
            if isinstance(parsed, list):
                return [str(step).strip() for step in parsed if str(step).strip()]
        except json.JSONDecodeError:
            pass

        return [step.strip() for step in response_text.split("\n") if step.strip()]