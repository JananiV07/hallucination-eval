"""OpenAI-compatible model client so any model can be plugged in.

The framework talks to models through the OpenAI Chat Completions API, which is
the de-facto standard implemented by OpenAI, Ollama, vLLM, LM Studio, Together,
Groq, OpenRouter and many more. That means "plug in any model" reduces to
"point at the right ``base_url`` with the right model id".

Presets
-------
``--model`` accepts a friendly preset name or an arbitrary model id:

============  =================================  ==================
preset        base_url                           api key
============  =================================  ==================
gpt-4o-mini   https://api.openai.com/v1 (default) ``OPENAI_API_KEY``
gemma2        http://localhost:11434/v1 (Ollama)  none (local)
mistral       http://localhost:11434/v1 (Ollama)  none (local)
<any id>      ``--base-url`` (else OpenAI)         ``--api-key``/``OPENAI_API_KEY``
============  =================================  ==================

Any preset value can be overridden with ``--base-url`` / ``--api-key``.
"""
from __future__ import annotations

import os
import time

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful, knowledgeable assistant. Answer the question as "
    "accurately and concisely as you can. If a context passage is provided, "
    "base your answer only on that passage and do not invent details."
)

# base_url=None means "use the OpenAI default endpoint".
OLLAMA_BASE = "http://localhost:11434/v1"
GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/openai/"

PRESETS: dict[str, dict] = {
    "gpt-4o-mini": {"base_url": None, "model": "gpt-4o-mini", "api_key_env": "OPENAI_API_KEY"},
    "gemma2": {"base_url": OLLAMA_BASE, "model": "gemma2", "api_key_env": None},
    "mistral": {"base_url": OLLAMA_BASE, "model": "mistral", "api_key_env": None},
    # Google Gemini via its OpenAI-compatible endpoint (reads GEMINI_API_KEY).
    "gemini-2.5-flash": {"base_url": GEMINI_BASE, "model": "gemini-2.5-flash", "api_key_env": "GEMINI_API_KEY"},
    "gemini-flash-latest": {"base_url": GEMINI_BASE, "model": "gemini-flash-latest", "api_key_env": "GEMINI_API_KEY"},
    "gemini-flash-lite-latest": {"base_url": GEMINI_BASE, "model": "gemini-flash-lite-latest", "api_key_env": "GEMINI_API_KEY"},
}


class ModelClient:
    """A minimal wrapper over the OpenAI Chat Completions API.

    The ``openai`` SDK is imported lazily and the underlying client is created
    on first use, so constructing a :class:`ModelClient` never touches the
    network.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        base_url: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
        timeout: float = 60.0,
        max_retries: int = 3,
        system_prompt: str | None = None,
    ) -> None:
        preset = PRESETS.get(model)
        if preset:
            self.model_id = preset["model"]
            self.base_url = base_url or preset["base_url"]
            key_env = preset["api_key_env"]
            self.api_key = api_key or (os.environ.get(key_env) if key_env else None)
        else:
            self.model_id = model
            self.base_url = base_url
            self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        # Local OpenAI-compatible servers (e.g. Ollama) accept any key; the SDK
        # still requires a non-empty string.
        if not self.api_key:
            self.api_key = "not-needed"

        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.timeout = float(timeout)
        self.max_retries = max(1, int(max_retries))
        self.system_prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
        self._client = None

    @property
    def name(self) -> str:
        return self.model_id

    def _ensure_client(self):
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover - import guard
                raise ImportError(
                    "The 'openai' package is required for generation. "
                    "Install it with `pip install openai`."
                ) from exc
            kwargs = {"api_key": self.api_key, "timeout": self.timeout}
            if self.base_url:
                kwargs["base_url"] = self.base_url
            self._client = OpenAI(**kwargs)
        return self._client

    @staticmethod
    def _build_prompt(question: str, context: str | None = None) -> str:
        if context and str(context).strip():
            return (
                f"Context:\n{context}\n\n"
                f"Question: {question}\n\n"
                "Answer the question using only the context above."
            )
        return f"Question: {question}\n\nAnswer:"

    def generate(self, question: str, context: str | None = None, system_prompt: str | None = None) -> str:
        """Generate an answer for ``question`` (optionally grounded in ``context``).

        Retries transient failures with exponential backoff and raises a clear
        :class:`RuntimeError` if generation ultimately fails (e.g. missing API
        key or an unreachable local server).
        """
        client = self._ensure_client()
        messages = [
            {"role": "system", "content": system_prompt or self.system_prompt},
            {"role": "user", "content": self._build_prompt(question, context)},
        ]
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                response = client.chat.completions.create(
                    model=self.model_id,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                content = response.choices[0].message.content
                return (content or "").strip()
            except Exception as exc:  # noqa: BLE001 - re-raised with context below
                last_error = exc
                if attempt < self.max_retries - 1:
                    time.sleep(min(2**attempt, 8))
        raise RuntimeError(
            f"Generation failed for model '{self.model_id}' "
            f"(base_url={self.base_url or 'https://api.openai.com/v1'}) after "
            f"{self.max_retries} attempt(s): {last_error}. "
            "Check that your API key is set (e.g. OPENAI_API_KEY) or that the "
            "local endpoint is running (e.g. `ollama serve`)."
        ) from last_error
