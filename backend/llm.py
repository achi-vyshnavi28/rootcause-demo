"""One small interface to any LLM provider (Gemini, Groq, Ollama, Claude, OpenAI) via LiteLLM.

The agent only ever calls `complete_json`, which returns a validated Pydantic object.
Tests use `ScriptedLLM`, so the agent can be tested without an API key.
"""

import json
import re
import time
from dataclasses import dataclass, field
from typing import Protocol, TypeVar

from pydantic import BaseModel, ValidationError

from backend import config

T = TypeVar("T", bound=BaseModel)


@dataclass
class Usage:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    latency_s: float = 0.0
    models: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cost_usd": round(self.cost_usd, 6),
            "latency_s": round(self.latency_s, 2),
            "models": sorted(set(self.models)),
        }


class LLM(Protocol):
    usage: Usage

    def complete_json(self, system: str, user: str, schema: type[T]) -> T: ...


def extract_json(text: str) -> dict:
    """Pull the first JSON object out of a model reply (handles ```json fences and chatter)."""
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else text
    start, end = candidate.find("{"), candidate.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("No JSON object found in model reply.")
    return json.loads(candidate[start : end + 1])


class LiteLLMClient:
    """Calls the configured model, falls back to backup models, and validates the JSON reply."""

    def __init__(self, model: str | None = None, fallbacks: list[str] | None = None):
        self.models = [model or config.LLM_MODEL, *(fallbacks if fallbacks is not None else config.LLM_FALLBACK_MODELS)]
        self.usage = Usage()

    def _call(self, messages: list[dict]) -> str:
        import litellm  # imported lazily: it is slow to import and not needed in tests

        litellm.suppress_debug_info = True
        extra = {"reasoning_effort": config.LLM_REASONING_EFFORT} if config.LLM_REASONING_EFFORT else {}
        last_error: Exception | None = None
        for model in self.models:
            try:
                started = time.perf_counter()
                response = litellm.completion(
                    model=model,
                    messages=messages,
                    num_retries=3,  # retries with backoff on rate limits / transient errors
                    **extra,
                )
                self.usage.latency_s += time.perf_counter() - started
                self.usage.calls += 1
                self.usage.models.append(model)
                if response.usage:
                    self.usage.prompt_tokens += response.usage.prompt_tokens or 0
                    self.usage.completion_tokens += response.usage.completion_tokens or 0
                try:
                    self.usage.cost_usd += litellm.completion_cost(completion_response=response) or 0.0
                except Exception:
                    pass  # free-tier / local models may have no price entry
                return response.choices[0].message.content or ""
            except Exception as e:  # try the next provider
                last_error = e
        raise RuntimeError(f"All LLM providers failed. Last error: {last_error}")

    def complete_json(self, system: str, user: str, schema: type[T]) -> T:
        instructions = (
            f"{system}\n\nReply with ONE JSON object only, matching this JSON schema:\n"
            f"{json.dumps(schema.model_json_schema())}"
        )
        messages = [{"role": "system", "content": instructions}, {"role": "user", "content": user}]
        for attempt in range(2):
            reply = self._call(messages)
            try:
                return schema.model_validate(extract_json(reply))
            except (ValueError, ValidationError) as e:
                if attempt == 1:
                    raise
                messages += [
                    {"role": "assistant", "content": reply},
                    {"role": "user", "content": f"That reply was invalid: {e}. Return only the corrected JSON object."},
                ]
        raise AssertionError("unreachable")


class ScriptedLLM:
    """Test double: returns pre-written replies in order, keyed by schema name."""

    def __init__(self, replies: dict[str, list[dict]]):
        self.replies = {k: list(v) for k, v in replies.items()}
        self.usage = Usage()
        self.prompts: list[tuple[str, str]] = []

    def complete_json(self, system: str, user: str, schema: type[T]) -> T:
        self.prompts.append((schema.__name__, user))
        self.usage.calls += 1
        queue = self.replies.get(schema.__name__)
        if not queue:
            raise AssertionError(f"No scripted reply left for {schema.__name__}")
        return schema.model_validate(queue.pop(0))
