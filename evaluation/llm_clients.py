"""LLM clients for the Level 2 (answer accuracy) evaluation.

One small interface over the Claude API and a local Ollama server, so every
model gets byte-identical prompts. Thinking is minimized (it is billed as
output and would add variance), and no server-side model fallback is used:
an answer must come from the model being evaluated.
"""

from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class Completion:
    text: str
    input_tokens: int
    output_tokens: int
    refused: bool = False


class LLMClient(Protocol):
    name: str
    # USD per million tokens (input, output); local models are free.
    price_per_mtok: tuple[float, float]

    def complete(self, prompt: str) -> Completion: ...


@dataclass(frozen=True, slots=True)
class ClaudeModel:
    model_id: str
    price_per_mtok: tuple[float, float]
    # Haiku 5.5 accepts thinking "disabled" at effort <= high; Sonnet 5.5
    # rejects it and turns thinking off with "between_tools".
    thinking_off: str


# Prices from the Claude API pricing table (cached 2026-10-06).
CLAUDE_MODELS = {
    "haiku": ClaudeModel("claude-haiku-5-5", (0.10, 0.50), "disabled"),
    "sonnet": ClaudeModel("claude-sonnet-5-5", (2.00, 10.00), "between_tools"),
}


class ClaudeClient:
    def __init__(self, model_key: str, max_tokens: int = 256) -> None:
        import anthropic

        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RuntimeError(
                "ANTHROPIC_API_KEY is not set. Store it as a Windows user environment "
                "variable and restart VS Code (never paste it into code or chat)."
            )
        spec = CLAUDE_MODELS[model_key]
        self.name = model_key
        self.price_per_mtok = spec.price_per_mtok
        self._spec = spec
        self._max_tokens = max_tokens
        self._client = anthropic.Anthropic(max_retries=5)

    def complete(self, prompt: str) -> Completion:
        from anthropic.types import ThinkingConfigParam

        thinking: ThinkingConfigParam = (
            {"type": "disabled"} if self._spec.thinking_off == "disabled" else {"type": "between_tools"}
        )
        response = self._client.messages.create(
            model=self._spec.model_id,
            max_tokens=self._max_tokens,
            thinking=thinking,
            output_config={"effort": "low"},
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(block.text for block in response.content if block.type == "text")
        return Completion(
            text=text.strip(),
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            refused=response.stop_reason == "refusal",
        )


class OllamaClient:
    """A model served by a local Ollama (http://localhost:11434)."""

    def __init__(self, model: str, name: str = "qwen", host: str = "http://localhost:11434") -> None:
        self.name = name
        self.price_per_mtok = (0.0, 0.0)
        self._model = model
        self._url = f"{host}/api/chat"

    def complete(self, prompt: str) -> Completion:
        body: dict[str, Any] = {
            "model": self._model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "think": False,  # Qwen3-style models: answer directly
            "options": {"temperature": 0, "num_predict": 256},
        }
        request = urllib.request.Request(
            self._url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(request, timeout=600) as response:
            data = json.loads(response.read())
        return Completion(
            text=str(data["message"]["content"]).strip(),
            input_tokens=int(data.get("prompt_eval_count", 0)),
            output_tokens=int(data.get("eval_count", 0)),
        )
