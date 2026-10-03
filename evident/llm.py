"""LLM access with response caching and token/cost accounting.

Providers
  ollama   local models via Ollama's native /api/chat (thinking disabled), no key needed
  groq     https://api.groq.com/openai/v1                        GROQ_API_KEY
  gemini   https://generativelanguage.googleapis.com/v1beta/openai GEMINI_API_KEY
  openai   https://api.openai.com/v1                             OPENAI_API_KEY
The three hosted providers all speak the OpenAI chat-completions protocol, so one client
covers them. Responses are cached in SQLite keyed by the full request, which makes
re-running an evaluation free and deterministic.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import threading
import time
from dataclasses import dataclass

import httpx

from . import config

PROVIDERS = {
    "ollama": {"base_url": os.environ.get("OLLAMA_HOST", "http://localhost:11434"), "key_env": None},
    "groq": {"base_url": "https://api.groq.com/openai/v1", "key_env": "GROQ_API_KEY"},
    "gemini": {"base_url": "https://generativelanguage.googleapis.com/v1beta/openai", "key_env": "GEMINI_API_KEY"},
    "openai": {"base_url": "https://api.openai.com/v1", "key_env": "OPENAI_API_KEY"},
}

# USD per 1M tokens (input, output). Edit evident/prices.json to update; values there win.
DEFAULT_PRICES = {
    "groq/llama-3.1-8b-instant": (0.05, 0.08),
    "groq/llama-3.3-70b-versatile": (0.59, 0.79),
}


def load_prices() -> dict[str, tuple[float, float]]:
    prices = dict(DEFAULT_PRICES)
    path = config.ROOT / "evident" / "prices.json"
    if path.exists():
        for k, v in json.loads(path.read_text()).get("usd_per_million_tokens", {}).items():
            prices[k] = (float(v["input"]), float(v["output"]))
    return prices


@dataclass
class LLMResponse:
    text: str
    input_tokens: int
    output_tokens: int
    latency_s: float
    model: str
    cached: bool = False


def extract_json(text: str):
    """Parses the first JSON object in `text` (models sometimes wrap it in prose or fences)."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return None


class LLM:
    def __init__(self, provider: str = config.LLM_PROVIDER, model: str = config.LLM_MODEL,
                 base_url: str | None = None, api_key: str | None = None, temperature: float = 0.0,
                 cache: bool = True, timeout: float = 300.0, num_ctx: int = 8192):
        if provider not in PROVIDERS:
            raise ValueError(f"provider must be one of {sorted(PROVIDERS)}")
        self.provider, self.model, self.temperature, self.num_ctx = provider, model, temperature, num_ctx
        self.base_url = (base_url or PROVIDERS[provider]["base_url"]).rstrip("/")
        key_env = PROVIDERS[provider]["key_env"]
        self.api_key = api_key or (os.environ.get(key_env) if key_env else None)
        if key_env and not self.api_key:
            raise RuntimeError(f"set {key_env} to use the {provider} provider")
        self.client = httpx.Client(timeout=timeout)
        self._db = None
        self._lock = threading.Lock()
        if cache:
            path = config.CACHE_DIR / "llm_cache.sqlite"
            path.parent.mkdir(parents=True, exist_ok=True)
            self._db = sqlite3.connect(path, check_same_thread=False)
            self._db.execute("CREATE TABLE IF NOT EXISTS responses (key TEXT PRIMARY KEY, value TEXT)")
        self.price = load_prices().get(f"{provider}/{model}")

    @property
    def name(self) -> str:
        return f"{self.provider}/{self.model}"

    def cost_usd(self, input_tokens: int, output_tokens: int) -> float | None:
        """None when the model has no price (e.g. local models, which cost $0 in API fees)."""
        if self.provider == "ollama":
            return 0.0
        if self.price is None:
            return None
        return (input_tokens * self.price[0] + output_tokens * self.price[1]) / 1e6

    def chat(self, messages: list[dict], max_tokens: int = 512, json_mode: bool = False) -> LLMResponse:
        key = hashlib.sha256(json.dumps([self.provider, self.model, self.temperature, max_tokens, json_mode,
                                         messages], sort_keys=True).encode()).hexdigest()
        if self._db is not None:
            with self._lock:
                row = self._db.execute("SELECT value FROM responses WHERE key = ?", (key,)).fetchone()
            if row:
                data = json.loads(row[0])
                return LLMResponse(**data, cached=True)

        t0 = time.perf_counter()
        for attempt in range(5):
            try:
                text, n_in, n_out = (self._ollama if self.provider == "ollama" else self._openai)(messages, max_tokens, json_mode)
                break
            except httpx.HTTPStatusError as e:
                if e.response.status_code in (429, 500, 502, 503) and attempt < 4:
                    time.sleep(2 ** attempt)
                    continue
                raise
        resp = LLMResponse(text=text, input_tokens=n_in, output_tokens=n_out,
                           latency_s=time.perf_counter() - t0, model=self.name)
        if self._db is not None:
            payload = {k: v for k, v in resp.__dict__.items() if k != "cached"}
            with self._lock:
                self._db.execute("INSERT OR REPLACE INTO responses VALUES (?, ?)", (key, json.dumps(payload)))
                self._db.commit()
        return resp

    def _ollama(self, messages, max_tokens, json_mode):
        body = {"model": self.model, "messages": messages, "stream": False, "think": False,
                "options": {"temperature": self.temperature, "num_predict": max_tokens, "num_ctx": self.num_ctx, "seed": 0}}
        if json_mode:
            body["format"] = "json"
        r = self.client.post(f"{self.base_url}/api/chat", json=body)
        r.raise_for_status()
        data = r.json()
        text = re.sub(r"<think>.*?</think>", "", data["message"]["content"], flags=re.DOTALL).strip()
        return text, data.get("prompt_eval_count", 0), data.get("eval_count", 0)

    def _openai(self, messages, max_tokens, json_mode):
        body = {"model": self.model, "messages": messages, "temperature": self.temperature, "max_tokens": max_tokens}
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        r = self.client.post(f"{self.base_url}/chat/completions", json=body,
                             headers={"Authorization": f"Bearer {self.api_key}"})
        r.raise_for_status()
        data = r.json()
        usage = data.get("usage", {})
        return data["choices"][0]["message"]["content"].strip(), usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)


class FakeLLM:
    """Deterministic stand-in for tests: returns canned responses in order (or a function's output)."""

    def __init__(self, responses, name: str = "fake/model"):
        self.responses = responses
        self.name = name
        self.calls: list[list[dict]] = []

    def cost_usd(self, input_tokens: int, output_tokens: int) -> float:
        return 0.0

    def chat(self, messages, max_tokens: int = 512, json_mode: bool = False) -> LLMResponse:
        self.calls.append(messages)
        text = self.responses(messages) if callable(self.responses) else self.responses[len(self.calls) - 1]
        return LLMResponse(text=text, input_tokens=100, output_tokens=20, latency_s=0.0, model=self.name)
