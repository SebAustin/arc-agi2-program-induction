"""Pluggable LLM clients.

The proposer depends only on the :class:`LLMClient` protocol, so the provider is
swappable and — crucially — tests inject :class:`MockClient`, exercising the full
proposal/refinement path with zero network calls. Real clients lazily import
their SDK and read the API key from the environment; no secret is ever embedded.
"""

from __future__ import annotations

import os
from typing import Protocol


class LLMClient(Protocol):
    def complete(
        self, system: str, user: str, *, n: int, temperature: float
    ) -> list[str]:
        """Return ``n`` completions for a (system, user) prompt."""
        ...


class MockClient:
    """Deterministic client returning canned completions (for tests/offline)."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls: list[tuple[str, str]] = []

    def complete(
        self, system: str, user: str, *, n: int, temperature: float
    ) -> list[str]:
        self.calls.append((system, user))
        return list(self._responses[:n]) if self._responses else []


class OpenAIClient:
    """OpenAI-backed client. Requires the ``openai`` package and an API key."""

    def __init__(self, model: str | None = None) -> None:
        import openai  # lazy import; only needed when actually used

        key = os.environ.get("ARC_LLM_KEY") or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise RuntimeError("no OpenAI key in ARC_LLM_KEY / OPENAI_API_KEY")
        self._client = openai.OpenAI(api_key=key)
        self._model = model or os.environ.get("ARC_LLM_MODEL", "gpt-5.2")

    def complete(
        self, system: str, user: str, *, n: int, temperature: float
    ) -> list[str]:
        resp = self._client.chat.completions.create(
            model=self._model,
            n=n,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return [c.message.content or "" for c in resp.choices]


class AnthropicClient:
    """Anthropic-backed client. Requires ``anthropic`` and an API key."""

    def __init__(self, model: str | None = None) -> None:
        import anthropic  # lazy import

        key = os.environ.get("ARC_LLM_KEY") or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise RuntimeError("no Anthropic key in ARC_LLM_KEY / ANTHROPIC_API_KEY")
        self._client = anthropic.Anthropic(api_key=key)
        self._model = model or os.environ.get("ARC_LLM_MODEL", "claude-opus-4-8")

    def complete(
        self, system: str, user: str, *, n: int, temperature: float
    ) -> list[str]:
        # The Messages API returns one completion per call; sample n times.
        out: list[str] = []
        for _ in range(n):
            msg = self._client.messages.create(
                model=self._model,
                max_tokens=1536,
                temperature=temperature,
                system=system,
                messages=[{"role": "user", "content": user}],
            )
            out.append("".join(blk.text for blk in msg.content if blk.type == "text"))
        return out


def build_client(provider: str, model: str | None = None) -> LLMClient:
    """Construct a client by provider name ('openai' | 'anthropic')."""
    if provider == "openai":
        return OpenAIClient(model)
    if provider == "anthropic":
        return AnthropicClient(model)
    raise ValueError(f"unknown LLM provider: {provider}")
