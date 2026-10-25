"""Backends de visão: Ollama (local) e API da Anthropic (opcional)."""
from __future__ import annotations

import base64
import os
from typing import Any, Protocol

import requests

from .tile_prompt import TILE_SCHEMA


def detect_physical_cores() -> int:
    """Núcleos físicos (melhor para inferência em CPU que threads lógicas)."""
    try:
        import psutil

        cores = psutil.cpu_count(logical=False)
        if cores:
            return cores
    except ImportError:
        pass
    return max(1, (os.cpu_count() or 2) // 2)


class VisionBackend(Protocol):
    name: str

    def extract(self, jpeg_bytes: bytes, prompt: str) -> str:
        """Devolve o texto bruto (JSON) gerado pelo modelo."""
        ...


class OllamaBackend:
    def __init__(
        self,
        model: str = "qwen2.5vl:latest",
        host: str = "http://localhost:11434",
        num_ctx: int = 16384,
        num_predict: int = 8192,
        num_thread: int | None = None,
        num_batch: int | None = None,
        timeout: int = 1800,
        keep_alive: str = "10m",
    ) -> None:
        self.model = model
        self.host = host.rstrip("/")
        self.num_ctx = num_ctx
        self.num_predict = num_predict
        self.num_thread = num_thread or detect_physical_cores()
        self.num_batch = num_batch
        self.timeout = timeout
        self.keep_alive = keep_alive
        self.name = f"ollama:{model}"

    def extract(self, jpeg_bytes: bytes, prompt: str) -> str:
        payload: dict[str, Any] = {
            "model": self.model,
            "stream": False,
            "format": TILE_SCHEMA,  # saída estruturada: obriga JSON válido
            "keep_alive": self.keep_alive,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                    "images": [base64.b64encode(jpeg_bytes).decode("ascii")],
                }
            ],
            "options": {
                "temperature": 0,
                "num_ctx": self.num_ctx,
                "num_predict": self.num_predict,
                "num_thread": self.num_thread,
            },
        }
        if self.num_batch:
            payload["options"]["num_batch"] = self.num_batch
        response = requests.post(f"{self.host}/api/chat", json=payload, timeout=self.timeout)
        response.raise_for_status()
        return response.json()["message"]["content"]


class AnthropicBackend:
    """Usa a API da Anthropic. Requer a variável de ambiente ANTHROPIC_API_KEY."""

    def __init__(
        self,
        model: str = "claude-sonnet-5-5",
        api_key: str | None = None,
        max_tokens: int = 4096,
        timeout: int = 300,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not self.api_key:
            raise RuntimeError("Defina ANTHROPIC_API_KEY para usar o backend anthropic.")
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.name = f"anthropic:{model}"

    def extract(self, jpeg_bytes: bytes, prompt: str) -> str:
        body = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": 0,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": base64.b64encode(jpeg_bytes).decode("ascii"),
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        }
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            json=body,
            timeout=self.timeout,
        )
        response.raise_for_status()
        blocks = response.json().get("content", [])
        return "".join(b.get("text", "") for b in blocks if b.get("type") == "text")


def build_backend(kind: str, model: str | None = None, **kwargs: Any) -> VisionBackend:
    if kind == "ollama":
        return OllamaBackend(model=model or "qwen2.5vl:latest", **kwargs)
    if kind == "anthropic":
        return AnthropicBackend(model=model or "claude-sonnet-5-5", **kwargs)
    raise ValueError(f"backend desconhecido: {kind}")