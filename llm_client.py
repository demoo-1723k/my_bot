"""Unified LLM client — one interface, many backends, graceful fallback.

Resolution order (first available wins):
  1. OpenAI        (OPENAI_API_KEY)
  2. Google Gemini (GEMINI_API_KEY / GOOGLE_API_KEY)
  3. Groq          (GROQ_API_KEY)            — fast & cheap
  4. Ollama local  (OLLAMA_HOST or localhost:11434)
  5. None          — caller must fall back to rule-based generation

No hard dependency: if no key is set, every call returns None instantly
so the rule-based engine keeps working with zero config.

Environment:
  LLM_PROVIDER  = auto | openai | gemini | groq | ollama | none
  LLM_MODEL     = model name override
  LLM_BASE_URL  = custom base URL (for proxies / Ollama)
  LLM_TIMEOUT   = seconds (default 25)
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    provider: str   # auto | openai | gemini | groq | ollama | none
    model: str | None
    base_url: str | None
    api_key: str | None
    timeout: float


def _config() -> LLMConfig:
    provider = os.getenv("LLM_PROVIDER", "auto").strip().lower() or "auto"
    model = os.getenv("LLM_MODEL", "").strip() or None
    base_url = os.getenv("LLM_BASE_URL", "").strip() or None
    timeout = float(os.getenv("LLM_TIMEOUT", "25"))
    # api_key resolved per-provider below
    return LLMConfig(provider=provider, model=model, base_url=base_url,
                     api_key=None, timeout=timeout)


# ---------------------------------------------------------------------------
# Detection — is a provider actually configured?
# ---------------------------------------------------------------------------

def _has_openai() -> bool:
    return bool(os.getenv("OPENAI_API_KEY", "").strip())

def _has_gemini() -> bool:
    return bool(os.getenv("GEMINI_API_KEY", "").strip() or os.getenv("GOOGLE_API_KEY", "").strip())

def _has_groq() -> bool:
    return bool(os.getenv("GROQ_API_KEY", "").strip())

def _has_ollama() -> bool:
    # Ollama is "available" if explicitly requested or host is set
    return bool(os.getenv("OLLAMA_HOST", "").strip() or os.getenv("LLM_PROVIDER", "").strip().lower() == "ollama")


def is_llm_available() -> bool:
    """True if any LLM backend is configured and reachable."""
    cfg = _config()
    if cfg.provider == "none":
        return False
    if cfg.provider == "openai":
        return _has_openai()
    if cfg.provider == "gemini":
        return _has_gemini()
    if cfg.provider == "groq":
        return _has_groq()
    if cfg.provider == "ollama":
        return True  # will try localhost
    # auto
    return _has_openai() or _has_gemini() or _has_groq() or _has_ollama()


def active_provider() -> str | None:
    """Name of the provider that would be used, or None."""
    cfg = _config()
    if cfg.provider == "none":
        return None
    if cfg.provider != "auto":
        return cfg.provider if is_llm_available() else None
    if _has_openai():
        return "openai"
    if _has_groq():
        return "groq"
    if _has_gemini():
        return "gemini"
    if _has_ollama():
        return "ollama"
    return None


# ---------------------------------------------------------------------------
# Low-level HTTP helpers (no extra deps beyond stdlib + httpx if available)
# ---------------------------------------------------------------------------

async def _http_post(url: str, headers: dict, body: dict, timeout: float) -> dict | None:
    try:
        import httpx  # already a transitive dep via python-telegram-bot
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(url, headers=headers, json=body)
            resp.raise_for_status()
            return resp.json()
    except Exception as exc:
        logger.info("LLM http post failed (%s): %s", url, exc)
        return None


def _http_post_sync(url: str, headers: dict, body: dict, timeout: float) -> dict | None:
    try:
        import httpx
        with httpx.Client(timeout=timeout) as client:
            resp = client.post(url, headers=headers, json=body)
            resp.raise_for_status()
            return resp.json()
    except Exception as exc:
        logger.info("LLM http post failed (%s): %s", url, exc)
        return None


# ---------------------------------------------------------------------------
# Provider implementations
# ---------------------------------------------------------------------------

async def _call_openai(prompt: str, system: str, cfg: LLMConfig) -> str | None:
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        return None
    model = cfg.model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    base = (cfg.base_url or "https://api.openai.com/v1").rstrip("/")
    url = f"{base}/chat/completions"
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.7,
        "max_tokens": 2000,
    }
    data = await _http_post(url, {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}, body, cfg.timeout)
    if not data:
        return None
    try:
        return data["choices"][0]["message"]["content"].strip()
    except Exception:
        return None


async def _call_groq(prompt: str, system: str, cfg: LLMConfig) -> str | None:
    key = os.getenv("GROQ_API_KEY", "").strip()
    if not key:
        return None
    model = cfg.model or "llama-3.3-70b-versatile"
    url = "https://api.groq.com/openai/v1/chat/completions"
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.7,
        "max_tokens": 2000,
    }
    data = await _http_post(url, {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}, body, cfg.timeout)
    if not data:
        return None
    try:
        return data["choices"][0]["message"]["content"].strip()
    except Exception:
        return None


async def _call_gemini(prompt: str, system: str, cfg: LLMConfig) -> str | None:
    key = os.getenv("GEMINI_API_KEY", "").strip() or os.getenv("GOOGLE_API_KEY", "").strip()
    if not key:
        return None
    model = cfg.model or "gemini-2.0-flash"
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
    body = {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": 0.7, "maxOutputTokens": 2000},
    }
    data = await _http_post(url, {"Content-Type": "application/json"}, body, cfg.timeout)
    if not data:
        return None
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception:
        return None


async def _call_ollama(prompt: str, system: str, cfg: LLMConfig) -> str | None:
    host = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    if cfg.base_url:
        host = cfg.base_url.rstrip("/")
    model = cfg.model or os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    url = f"{host}/api/generate"
    body = {
        "model": model,
        "prompt": prompt,
        "system": system,
        "stream": False,
        "options": {"temperature": 0.7, "num_predict": 2000},
    }
    data = await _http_post(url, {"Content-Type": "application/json"}, body, cfg.timeout)
    if not data:
        return None
    return (data.get("response") or "").strip() or None


# Sync variants for non-async callers (generator.py)
def _call_openai_sync(prompt: str, system: str, cfg: LLMConfig) -> str | None:
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        return None
    model = cfg.model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    base = (cfg.base_url or "https://api.openai.com/v1").rstrip("/")
    url = f"{base}/chat/completions"
    body = {"model": model, "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}], "temperature": 0.7, "max_tokens": 2000}
    data = _http_post_sync(url, {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}, body, cfg.timeout)
    if not data:
        return None
    try:
        return data["choices"][0]["message"]["content"].strip()
    except Exception:
        return None

def _call_groq_sync(prompt: str, system: str, cfg: LLMConfig) -> str | None:
    key = os.getenv("GROQ_API_KEY", "").strip()
    if not key:
        return None
    model = cfg.model or "llama-3.3-70b-versatile"
    url = "https://api.groq.com/openai/v1/chat/completions"
    body = {"model": model, "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}], "temperature": 0.7, "max_tokens": 2000}
    data = _http_post_sync(url, {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}, body, cfg.timeout)
    if not data:
        return None
    try:
        return data["choices"][0]["message"]["content"].strip()
    except Exception:
        return None

def _call_gemini_sync(prompt: str, system: str, cfg: LLMConfig) -> str | None:
    key = os.getenv("GEMINI_API_KEY", "").strip() or os.getenv("GOOGLE_API_KEY", "").strip()
    if not key:
        return None
    model = cfg.model or "gemini-2.0-flash"
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
    body = {"systemInstruction": {"parts": [{"text": system}]}, "contents": [{"role": "user", "parts": [{"text": prompt}]}], "generationConfig": {"temperature": 0.7, "maxOutputTokens": 2000}}
    data = _http_post_sync(url, {"Content-Type": "application/json"}, body, cfg.timeout)
    if not data:
        return None
    try:
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception:
        return None

def _call_ollama_sync(prompt: str, system: str, cfg: LLMConfig) -> str | None:
    host = os.getenv("OLLAMA_HOST", "http://localhost:11434").rstrip("/")
    if cfg.base_url:
        host = cfg.base_url.rstrip("/")
    model = cfg.model or os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    url = f"{host}/api/generate"
    body = {"model": model, "prompt": prompt, "system": system, "stream": False, "options": {"temperature": 0.7, "num_predict": 2000}}
    data = _http_post_sync(url, {"Content-Type": "application/json"}, body, cfg.timeout)
    if not data:
        return None
    return (data.get("response") or "").strip() or None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def generate(prompt: str, system: str = "You are a helpful assistant.", timeout: float | None = None) -> str | None:
    """Generate text via the best available LLM. Returns None if unavailable/failed."""
    cfg = _config()
    if timeout is not None:
        cfg.timeout = timeout
    if cfg.provider == "none":
        return None

    order: list[str]
    if cfg.provider == "auto":
        # priority: openai > groq > gemini > ollama
        order = []
        if _has_openai():
            order.append("openai")
        if _has_groq():
            order.append("groq")
        if _has_gemini():
            order.append("gemini")
        if _has_ollama() or not order:
            order.append("ollama")
    else:
        order = [cfg.provider]

    mapping = {"openai": _call_openai, "groq": _call_groq, "gemini": _call_gemini, "ollama": _call_ollama}
    for name in order:
        fn = mapping.get(name)
        if not fn:
            continue
        try:
            result = await fn(prompt, system, cfg)
            if result:
                logger.info("LLM generate via %s: %d chars", name, len(result))
                return result
        except Exception as exc:
            logger.info("LLM %s failed: %s", name, exc)
    return None


def generate_sync(prompt: str, system: str = "You are a helpful assistant.", timeout: float | None = None) -> str | None:
    """Synchronous version."""
    cfg = _config()
    if timeout is not None:
        cfg.timeout = timeout
    if cfg.provider == "none":
        return None
    order: list[str] = []
    if cfg.provider == "auto":
        if _has_openai():
            order.append("openai")
        if _has_groq():
            order.append("groq")
        if _has_gemini():
            order.append("gemini")
        if _has_ollama() or not order:
            order.append("ollama")
    else:
        order = [cfg.provider]
    mapping_sync = {"openai": _call_openai_sync, "groq": _call_groq_sync, "gemini": _call_gemini_sync, "ollama": _call_ollama_sync}
    for name in order:
        fn = mapping_sync.get(name)
        if not fn:
            continue
        try:
            result = fn(prompt, system, cfg)
            if result:
                logger.info("LLM generate_sync via %s: %d chars", name, len(result))
                return result
        except Exception as exc:
            logger.info("LLM sync %s failed: %s", name, exc)
    return None


def generate_json(prompt: str, system: str = "You are a helpful assistant. Reply with valid JSON only.", timeout: float | None = None) -> Any | None:
    """Generate and parse JSON. Returns None on failure."""
    raw = generate_sync(prompt, system, timeout)
    if not raw:
        return None
    # strip markdown fences
    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    raw = re.sub(r"\s*```\s*$", "", raw)
    # extract first JSON object/array
    m = re.search(r"(\{.*\}|\[.*\])", raw, re.DOTALL)
    if m:
        raw = m.group(1)
    try:
        return json.loads(raw)
    except Exception as exc:
        logger.info("LLM JSON parse failed: %s | raw=%.300s", exc, raw)
        return None


async def generate_json_async(prompt: str, system: str = "You are a helpful assistant. Reply with valid JSON only.", timeout: float | None = None) -> Any | None:
    raw = await generate(prompt, system, timeout)
    if not raw:
        return None
    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    raw = re.sub(r"\s*```\s*$", "", raw)
    m = re.search(r"(\{.*\}|\[.*\])", raw, re.DOTALL)
    if m:
        raw = m.group(1)
    try:
        return json.loads(raw)
    except Exception as exc:
        logger.info("LLM JSON parse failed: %s | raw=%.300s", exc, raw)
        return None
