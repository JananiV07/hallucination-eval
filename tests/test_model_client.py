"""Tests for ModelClient preset resolution and generation (network mocked)."""
from types import SimpleNamespace

import pytest

from hallucination_eval import model_client
from hallucination_eval.model_client import ModelClient


def test_preset_openai_resolution(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    client = ModelClient("gpt-4o-mini")
    assert client.model_id == "gpt-4o-mini"
    assert client.base_url is None  # OpenAI default endpoint
    assert client.api_key == "sk-test"


def test_preset_local_resolution(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    client = ModelClient("mistral")
    assert client.model_id == "mistral"
    assert client.base_url == "http://localhost:11434/v1"
    assert client.api_key == "not-needed"  # local servers need no key


def test_custom_model_and_base_url():
    client = ModelClient("my-llama", base_url="http://host:8000/v1", api_key="abc")
    assert client.model_id == "my-llama"
    assert client.base_url == "http://host:8000/v1"
    assert client.api_key == "abc"


def _fake_client(responder):
    completions = SimpleNamespace(create=responder)
    return SimpleNamespace(chat=SimpleNamespace(completions=completions))


def test_generate_returns_stripped_content():
    client = ModelClient("gpt-4o-mini", api_key="x")
    client._client = _fake_client(
        lambda **kw: SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="  hi there  "))])
    )
    assert client.generate("question") == "hi there"


def test_generate_retries_then_succeeds(monkeypatch):
    monkeypatch.setattr(model_client.time, "sleep", lambda *_: None)
    calls = {"n": 0}

    def responder(**kw):
        calls["n"] += 1
        if calls["n"] < 2:
            raise RuntimeError("transient")
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))])

    client = ModelClient("gpt-4o-mini", api_key="x", max_retries=3)
    client._client = _fake_client(responder)
    assert client.generate("q") == "ok"
    assert calls["n"] == 2


def test_generate_raises_clear_error_after_exhausting_retries(monkeypatch):
    monkeypatch.setattr(model_client.time, "sleep", lambda *_: None)

    def responder(**kw):
        raise RuntimeError("boom")

    client = ModelClient("gpt-4o-mini", api_key="x", max_retries=2)
    client._client = _fake_client(responder)
    with pytest.raises(RuntimeError, match="Generation failed"):
        client.generate("q")


def test_preset_gemini_resolution(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "g-test")
    client = ModelClient("gemini-2.5-flash")
    assert client.model_id == "gemini-2.5-flash"
    assert client.base_url == "https://generativelanguage.googleapis.com/v1beta/openai/"
    assert client.api_key == "g-test"
