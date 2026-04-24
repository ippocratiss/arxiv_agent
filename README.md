# ArXiv Research Agent

---

A conversational LLM-based research assistant that searches and summarises academic papers from arXiv in real time.

Built from scratch using the Groq API and Llama 3.3, this agent demonstrates multi-tool calling, external API integration, and prompt engineering.

---

## Author
Ippocratis Saltas.

---

## Features

- **Paper search & summarisation** — find recent papers by topic using arXiv's API, fetch and summarise any paper by arXiv ID. Avoids duplicate API calls within a session
- **Conversation memory** — maintains full context across turns

- **LLM** — Llama 3.3 70B via [Groq](https://groq.com)
- **Tools** — arXiv Atom API, custom dispatcher
- **Language** — Python 3.9+

---

## How It Works

The agent uses a tool-calling loop where the LLM decides whether to search arXiv, fetch a specific paper, or reply directly. Tool results are fed back into the conversation so the model can reason over real data rather than its training memory.

```
User input → LLM decides → Tool executes → Result fed back → LLM replies
```

---


## Setup

```bash
pip install groq
```

Add your Groq API key:

```python
GROQ_KEY = "your-key-here"
```

Run the notebook and start chatting:

```
Agent ready. Type 'exit' to quit.

You: Find the 5 most recent papers on diffusion models in astrophysics
You: Summarise 2603.26613
```

---

## Project Structure

```
├── arxiv_agent.ipynb   # Main notebook
└── README.md
```

---




