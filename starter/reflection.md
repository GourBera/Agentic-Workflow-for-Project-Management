# Reflection: AI-Powered Agentic Workflow for Project Management

## 1. Architecture Strengths

### Modular Agent Design
Each of the seven agent classes in `base_agents.py` follows a clear single-responsibility pattern. The progression from `DirectPromptAgent` to `RAGKnowledgePromptAgent` demonstrates increasing sophistication, with each layer adding a new capability (persona → knowledge injection → embedding-based retrieval) while sharing the same `respond()` interface.

### Evaluation-Driven Quality
The `EvaluationAgent` creates a feedback loop that mirrors human review workflows. Instead of blindly trusting the first LLM output, the system iteratively evaluates and refines responses against explicit criteria, capping iterations to prevent runaway loops.

### Semantic Routing
The `RoutingAgent` uses cosine similarity on `text-embedding-3-large` embeddings to select the best agent for each sub-task. This is flexible — adding a new role requires only a new route dictionary, no code changes.

### Reliability Layer
The `_retry_with_backoff` helper wraps all OpenAI API calls with exponential backoff (3 retries, base delay 1s). This makes the system resilient to transient API errors and rate limits without cluttering each agent's logic.

## 2. Limitations & Known Issues

| Limitation | Impact | Potential Fix |
|---|---|---|
| Single-model dependency (`gpt-3.5-turbo`) | Cost and capability ceiling | Support model selection per agent via config |
| No persistent state | Agents start fresh each run; no memory of past interactions | Add a conversation/session store (e.g., SQLite or Redis) |
| RAG: CSV-based embedding deserialization uses `ast.literal_eval()` | Safer than `eval()` but still parses string representations of lists | Serialize embeddings in a binary format (e.g., pickle or numpy `.npy`) |
| CSV-based embedding storage | Not scalable for large knowledge bases | Use a vector database (Chroma, FAISS, Pinecone) |
| Fixed evaluation criteria | Criteria are hardcoded strings per agent | Load criteria from config files or a criteria registry |
| Chunking uses regex whitespace normalization | Removes all newlines, preventing natural paragraph breaks | Chunk on double-newline (paragraph) boundaries first |

## 3. Prompt Engineering Insights

### What Worked
- **Role-based system prompts**: Giving agents a clear persona (e.g., "You are a college professor") significantly improved answer style consistency.
- **Temperature=0 for evaluation**: Deterministic outputs for the evaluator gave reproducible pass/fail decisions.
- **Explicit "Yes/No" instruction**: Telling the evaluator to start with "Yes" or "No" made parsing reliable without needing structured JSON for that role.

### What Didn't Work Initially
- **Generic evaluation criteria**: Vague criteria like "check if good" caused the evaluator to almost always say "Yes". We fixed this by writing domain-specific criteria (e.g., "As a [user], I want [feature] so that [benefit]").
- **Overly long knowledge contexts**: Providing the full product spec in every prompt token-limited the LLM's response window. RAG chunking helped solve this.

## 4. Testing Strategy

### Unit Tests (Mocked)
Two test suites (`test_routing_agent.py`, `test_evaluation_agent.py`) use `unittest.mock` to patch the `OpenAI` client with deterministic responses. This tests routing logic and evaluation iteration without API calls.

### Evaluation Outputs
Regression tests compare against evaluation JSON files (`tests/evaluation/`), catching unintentional changes to agent behavior during refactoring.

### Live Integration Tests
Each of the 7 Phase 1 scripts serves as a live integration test against the Vocareum-proxied OpenAI API, verifying end-to-end functionality including prompt formatting, API communication, and response parsing.

## 5. Alternative `workflow_prompt` Values Explored

The Phase 2 `agentic_workflow.py` uses a `workflow_prompt` to drive the `ActionPlanningAgent`. Different prompt formulations yield different workflow decompositions:

| Prompt Variation | Effect |
|---|---|
| *"Take the product spec and extract user stories, then features, then tasks"* (original) | 3 broad steps → each agent handles bulk work |
| *"Analyze the product spec step by step: first identify personas, then write user stories for each persona, then group stories into features, then create engineering tasks for each feature"* | 4 granular steps → more routing decisions, finer output |
| *"As a TPM, plan the development of this product from spec to sprint-ready tasks"* | LLM generates its own sub-task decomposition → more creative but less predictable |

The structured, explicit prompt (variation 2) produced the most consistent and reviewable output.

## 6. Potential Improvements

1. **Structured JSON outputs for all agents**: Currently only the evaluator and action planner attempt JSON. Making all agents return structured data would enable downstream tooling.
2. **Async API calls**: The RAG agent calls embeddings sequentially per chunk. `asyncio` + batched embedding calls would speed this up 5-10x on larger documents.
3. **Configurable models**: Allow each agent to specify its model, enabling cheaper models for simple tasks and more capable models for complex reasoning.
4. **Logging framework**: Replace `print()` statements with Python `logging` at DEBUG/INFO levels for production readiness.
5. **Vector database integration**: Replace CSV-based embedding storage with Chroma or FAISS for scalable knowledge retrieval.
