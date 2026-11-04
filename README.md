# ArXiv Research Agent

A conversational LLM agent that searches and summarises arXiv papers in real time. Built from scratch with no agent framework: a custom tool-calling dispatcher on top of the arXiv Atom API, with Llama 3.3 70B (via Groq) deciding when to search, when to fetch, and when to answer directly.

## What it does

- **Topic search**: ask for recent papers on any subject and get results sorted by date.
- **Paper summarisation**: give it an arXiv ID and get the title, authors, and abstract.
- **Multi-turn conversation**: context is kept across turns, so follow-up questions work naturally.
- **Grounded answers**: the model reasons over retrieved data, not its training memory. It is explicitly prevented from inventing paper titles.
- **Session-level caching**: duplicate API calls within a session are avoided.

## Architecture

```
User input
    │
    ▼
LLM (Llama 3.3 70B via Groq)
    │
    ├── decides: search_arxiv(query, max_results)
    │       │
    │       ▼
    │   arXiv Atom API → results added to conversation
    │
    ├── decides: fetch_arxiv_paper(arxiv_id)
    │       │
    │       ▼
    │   arXiv Atom API → metadata + abstract added to conversation
    │
    └── decides: reply directly from context
            │
            ▼
        Response to user
```

The agent uses a custom dispatcher that maps tool names to methods. The LLM generates structured tool calls (function name + arguments as JSON), the dispatcher executes them against the arXiv API, and the results are fed back into the conversation for the model to synthesise into a final answer. If the model chains multiple tool calls, it loops up to 3 iterations before forcing a text reply.

## Example

```
You: Find the 5 most recent papers on diffusion models in astrophysics
Agent: [calls search_arxiv] Here are the 5 most recent papers...

You: Summarise 2603.26613
Agent: [calls fetch_arxiv_paper] Title: ... Authors: ... Abstract: ...

You: How does this compare to the second paper from the search?
Agent: [answers from conversation context, no new API call needed]
```

## Setup

```bash
pip install -r requirements.txt
```

Set your Groq API key, then:

```python
from arxiv_agent import run_agent
run_agent(api_key="your-key-here")
```

Or run the notebook `arxiv_agent.ipynb` interactively.

## Stack

Python 3.9+ · Llama 3.3 70B (Groq) · arXiv Atom API · no agent framework

## Author

Ippocratis Saltas
