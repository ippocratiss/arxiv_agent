# ArXiv Research Agent

An LLM agent that reads arXiv for you: ask for the latest papers on a topic, or hand it an arXiv ID and get a summary. It decides for itself when to search, when to fetch, and when to just answer.

Built from scratch (no agent framework) on **Llama 3.3 70B via Groq**, as a compact demonstration of tool calling, external API integration and prompt design.

## Example prompts

```
Find the 5 most recent papers on diffusion models in astrophysics
Summarise 2603.26613
```

## How it works

```
User input → LLM decides → Tool runs → Result fed back → LLM replies
```

- **The model picks the tool.** Search arXiv by topic, fetch a paper by ID, or reply directly.
- **Grounded answers.** Tool results go back into the conversation, so the model reasons over retrieved papers instead of its training memory.
- **Memory.** Full context is kept across turns.
- **No wasted calls.** Duplicate API calls within a session are avoided.
- **Small and readable.** A custom tool dispatcher on the arXiv Atom API, with the whole agent in one notebook.

## Run it

1. Install the dependency: `pip install -r requirements.txt`
2. Add your Groq API key as a variable named `GROQ_KEY`.
3. Run `arxiv_agent.ipynb` and start chatting. Type `exit` to quit.

## Stack

Python 3.9+ · Llama 3.3 70B (Groq) · arXiv Atom API

## Author

Ippocratis Saltas



