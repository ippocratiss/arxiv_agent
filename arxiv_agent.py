"""
ArXiv Research Agent
A conversational LLM agent that searches and summarises arXiv papers
using tool calling over the arXiv Atom API.

Author: Ippocratis Saltas
"""

import json
import urllib.request
import xml.etree.ElementTree as ET
from typing import Optional
from groq import Groq
from IPython.display import display, Markdown


SYSTEM_PROMPT = """You are a research assistant that helps find and summarise academic papers.
CRITICAL RULES:
- You ALWAYS have access to search_arxiv and fetch_arxiv_paper tools
- You NEVER generate paper titles from memory
- You NEVER say you don't have access to tools
- If search returns no results, tell the user honestly
- ONLY present papers returned by the search_arxiv tool
- Inventing paper titles is strictly forbidden"""

DEFAULT_MODEL = "llama-3.3-70b-versatile"
MAX_TOOL_ITERATIONS = 3


class ArxivTools:
    """Manages arXiv API interactions and tool definitions for the agent."""

    def __init__(self):
        self.cache: dict[str, str] = {}
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": "fetch_arxiv_paper",
                    "description": "Fetch title, authors and abstract of a paper given its arxiv ID",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "arxiv_id": {
                                "type": "string",
                                "description": "The arxiv ID e.g. 2301.07041",
                            }
                        },
                        "required": ["arxiv_id"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "search_arxiv",
                    "description": (
                        "Search arxiv for academic papers by topic or keyword. "
                        "The query parameter must use arxiv API syntax: "
                        'Use abs:"phrase" for abstract search, ti:"phrase" for titles. '
                        "Combine with AND. Examples: "
                        'abs:"neural networks" AND abs:astronomy, '
                        'abs:"transformer" AND abs:"protein folding"'
                    ),
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query in arXiv API syntax",
                            },
                            "max_results": {
                                "type": "integer",
                                "description": "Number of results (default 5)",
                            },
                        },
                        "required": ["query"],
                    },
                },
            },
        ]

    def fetch_arxiv_paper(self, arxiv_id: str) -> str:
        """Fetch metadata and abstract for a single paper by arXiv ID."""
        if arxiv_id in self.cache:
            return self.cache[arxiv_id]

        url = f"http://export.arxiv.org/api/query?id_list={arxiv_id}"
        try:
            response = urllib.request.urlopen(url)
            xml_data = response.read().decode("utf-8")
            root = ET.fromstring(xml_data)
            ns = {"atom": "http://www.w3.org/2005/Atom"}

            entry = root.find("atom:entry", ns)
            if entry is None:
                return f"No paper found with ID: {arxiv_id}"

            title = entry.find("atom:title", ns).text.strip()
            summary = entry.find("atom:summary", ns).text.strip()
            authors = [
                a.find("atom:name", ns).text
                for a in entry.findall("atom:author", ns)
            ]
            author_str = f"{authors[0]} et al." if len(authors) > 1 else authors[0]

            result = (
                f"Title: {title}\n"
                f"Authors: {author_str}\n"
                f"Link: https://arxiv.org/abs/{arxiv_id}\n"
                f"{summary}"
            )
            self.cache[arxiv_id] = result
            return result
        except Exception as e:
            return f"Error fetching {arxiv_id}: {e}"

    def search_arxiv(self, query: str, max_results: int = 5) -> str:
        """Search arXiv by topic, sorted by submission date."""
        query_clean = query.replace('"', "").replace("'", "")
        query_encoded = query_clean.replace(" AND ", "+AND+").replace(" ", "+")
        url = (
            f"https://export.arxiv.org/api/query?"
            f"search_query={query_encoded}"
            f"&max_results={max_results}"
            f"&sortBy=submittedDate&sortOrder=descending"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "ArxivAgent/1.0"})

        try:
            response = urllib.request.urlopen(req)
            xml_data = response.read().decode("utf-8")
            root = ET.fromstring(xml_data)
            ns = {"atom": "http://www.w3.org/2005/Atom"}

            results = []
            for entry in root.findall("atom:entry", ns):
                title = entry.find("atom:title", ns).text.strip()
                arxiv_id = entry.find("atom:id", ns).text.strip().split("/")[-1]
                authors = [
                    a.find("atom:name", ns).text
                    for a in entry.findall("atom:author", ns)
                ]
                author_str = (
                    f"{authors[0]} et al." if len(authors) > 1 else authors[0]
                )
                results.append(
                    f"ID: {arxiv_id} | {title} | {author_str} | "
                    f"https://arxiv.org/abs/{arxiv_id}"
                )

            if not results:
                return f"No results found for: {query}"
            return "\n".join(results)
        except Exception as e:
            return f"Error searching arXiv: {e}"

    def get_dispatcher(self) -> dict:
        """Return a mapping of tool names to callable methods."""
        return {
            "search_arxiv": self.search_arxiv,
            "fetch_arxiv_paper": self.fetch_arxiv_paper,
        }


def _execute_tool_calls(tool_calls, dispatcher: dict, messages: list) -> None:
    """Run tool calls and append results to the message history."""
    for tool_call in tool_calls:
        args = json.loads(tool_call.function.arguments)
        result = dispatcher[tool_call.function.name](**args)
        messages.append({
            "role": "tool",
            "tool_call_id": tool_call.id,
            "content": result,
        })


def run_agent(api_key: Optional[str] = None, model: str = DEFAULT_MODEL) -> None:
    """
    Run the interactive agent loop.

    Args:
        api_key: Groq API key. If None, reads from GROQ_KEY variable.
        model: Model identifier for the Groq API.
    """
    client = Groq(api_key=api_key or GROQ_KEY)
    arxiv = ArxivTools()
    dispatcher = arxiv.get_dispatcher()
    tools = arxiv.tools

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    print("ArXiv Research Agent ready. Type 'exit' to quit.\n")

    while True:
        user_input = input("You: ")
        if user_input.strip().lower() == "exit":
            break

        messages.append({"role": "user", "content": user_input})

        # Initial LLM call
        response = client.chat.completions.create(
            model=model, messages=messages, tools=tools, temperature=0
        )
        message = response.choices[0].message

        # If the model wants to call tools, execute and loop until text reply
        if message.tool_calls:
            messages.append({
                "role": "assistant",
                "content": None,
                "tool_calls": message.tool_calls,
            })
            _execute_tool_calls(message.tool_calls, dispatcher, messages)

            reply = None
            for _ in range(MAX_TOOL_ITERATIONS):
                followup = client.chat.completions.create(
                    model=model, messages=messages, tools=tools, temperature=0
                )
                followup_msg = followup.choices[0].message

                if followup_msg.tool_calls:
                    messages.append({
                        "role": "assistant",
                        "content": None,
                        "tool_calls": followup_msg.tool_calls,
                    })
                    _execute_tool_calls(
                        followup_msg.tool_calls, dispatcher, messages
                    )
                else:
                    reply = followup_msg.content
                    break

            # Force a final reply if the model kept calling tools
            if not reply:
                forced = client.chat.completions.create(
                    model=model,
                    messages=messages + [
                        {"role": "user", "content": "Present the results you found."}
                    ],
                    temperature=0,
                )
                reply = forced.choices[0].message.content
        else:
            reply = message.content or "No response generated."

        messages.append({"role": "assistant", "content": reply})
        display(Markdown(str(reply)))


if __name__ == "__main__":
    GROQ_KEY = input("Enter your Groq API key: ")
    run_agent(api_key=GROQ_KEY)
