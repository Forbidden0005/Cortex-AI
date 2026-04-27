"""
Web Agent - Handles web search and URL fetch operations.

Uses DuckDuckGo for search (no API key required) and httpx for
fetching page content. Results are summarized by the LLM when one
is available.

task.parameters can contain:
  "query"     — search terms (falls back to task.description if absent)
  "url"       — fetch a specific URL instead of searching
  "max_results" — how many search results to return (default 5)
"""

import re
from typing import Optional

from agents.base_agent import BaseAgent
from models import Task


def _strip_tags(html: str) -> str:
    """Very light HTML tag stripper — good enough for readable page text."""
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


class WebAgent(BaseAgent):
    """
    Performs web searches and fetches remote content.

    Search backend: DuckDuckGo (via duckduckgo-search, no key needed).
    Fetch backend:  httpx.
    Summarization:  optional LLM passed at construction time.

    Install the search library if not already present:
        pip install duckduckgo-search httpx
    """

    def __init__(self, llm=None):
        super().__init__(
            name="WebAgent",
            agent_type="web",
            description="Searches the web and fetches URLs using DuckDuckGo",
        )
        self.llm = llm

    # ── helpers ──────────────────────────────────────────────────────────────

    def _search(self, query: str, max_results: int = 5) -> list[dict]:
        """Run a DuckDuckGo text search. Returns list of result dicts."""
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            raise RuntimeError(
                "duckduckgo-search is not installed. "
                "Run: pip install duckduckgo-search"
            )

        results = []
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append({
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "snippet": r.get("body", ""),
                })
        return results

    def _fetch(self, url: str, char_limit: int = 4000) -> str:
        """Fetch a URL and return readable plain text (truncated)."""
        try:
            import httpx
        except ImportError:
            raise RuntimeError(
                "httpx is not installed. Run: pip install httpx"
            )

        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0 Safari/537.36"
            )
        }
        r = httpx.get(url, headers=headers, timeout=15, follow_redirects=True)
        r.raise_for_status()
        return _strip_tags(r.text)[:char_limit]

    def _summarize(self, query: str, content: str) -> str:
        """Ask the LLM to summarize search/fetch results for the user."""
        if self.llm is None:
            return content

        system = (
            "You are Cortex, a helpful AI assistant. "
            "The user asked you to search the web. "
            "Summarize the search results below concisely and helpfully, "
            "answering the user's question directly."
        )
        prompt = (
            f"User's request: {query}\n\n"
            f"Search results:\n{content}\n\n"
            "Please provide a clear, helpful summary."
        )
        try:
            return self.llm.ask(prompt, system_prompt=system, max_tokens=600).strip()
        except Exception as exc:
            self._log(f"LLM summarization failed: {exc}", "warning")
            return content

    # ── main work ────────────────────────────────────────────────────────────

    def _do_work(self, task: Task) -> dict:
        params = task.parameters or {}

        # Determine what to do
        url = params.get("url")
        query = params.get("query") or task.description
        max_results = int(params.get("max_results", 5))

        # ── URL fetch mode ────────────────────────────────────────────────
        if url:
            try:
                self._track_tool_used("httpx")
                page_text = self._fetch(url)
                summary = self._summarize(query, page_text)
                return {
                    "response": summary,
                    "url": url,
                    "raw_length": len(page_text),
                }
            except Exception as exc:
                if self.llm:
                    # Graceful fallback — tell the user what happened
                    return {
                        "response": (
                            f"I couldn't fetch {url} ({exc}). "
                            "The site may be blocking automated requests. "
                            "Try opening it in your browser."
                        )
                    }
                raise

        # ── Search mode ───────────────────────────────────────────────────
        try:
            self._track_tool_used("duckduckgo_search")
            results = self._search(query, max_results=max_results)

            if not results:
                return {"response": f"No results found for: {query}"}

            # Build a readable block from snippets
            snippets = []
            for i, r in enumerate(results, 1):
                snippets.append(
                    f"[{i}] {r['title']}\n"
                    f"    {r['url']}\n"
                    f"    {r['snippet']}"
                )
            raw_text = "\n\n".join(snippets)

            summary = self._summarize(query, raw_text)

            return {
                "response": summary,
                "results": results,
                "query": query,
            }

        except RuntimeError as exc:
            # Library not installed
            return {
                "response": str(exc),
            }
        except Exception as exc:
            if self.llm:
                return {
                    "response": (
                        f"Web search failed ({exc}). "
                        "Check your internet connection and try again."
                    )
                }
            raise
