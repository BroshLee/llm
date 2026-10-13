
import os
from dataclasses import dataclass

import httpx


@dataclass
class GenerationResult:
    answer: str
    prompt_tokens: int
    output_tokens: int
    tokens_per_second: float


class AIClient:
    def __init__(self):
        self.base_url = os.getenv(
            "OLLAMA_BASE_URL",
            "http://localhost:11434",
        )
        self.model = os.getenv(
            "OLLAMA_MODEL",
            "gemma4:e4b",
            # "deepseek-coder:latest",
        )

        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(180.0, connect=5.0),
        )

    async def generate(self, prompt: str) -> GenerationResult:
        response = await self.client.post(
            "/api/chat",
            json={
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                "stream": False,
            },
        )

        response.raise_for_status()
        data = response.json()

        # Ollama reports token counts directly; eval_duration is in nanoseconds.
        output_tokens = data.get("eval_count", 0)
        eval_seconds = data.get("eval_duration", 0) / 1e9

        return GenerationResult(
            answer=data["message"]["content"],
            prompt_tokens=data.get("prompt_eval_count", 0),
            output_tokens=output_tokens,
            tokens_per_second=round(output_tokens / eval_seconds, 2) if eval_seconds else 0.0,
        )

    async def close(self):
        await self.client.aclose()
