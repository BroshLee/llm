
import os
import httpx


class AIClient:
    def __init__(self):
        self.base_url = os.getenv(
            "OLLAMA_BASE_URL",
            "http://localhost:11434",
        )
        self.model = os.getenv(
            "OLLAMA_MODEL",
            "llama3:latest",
            # "deepseek-coder:latest",
        )

        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(
                180.0,
                connect=5.0,
            ),
        )

    async def chat(self, messages: list[dict]) -> dict:
        response = await self.client.post(
            "/api/chat",
            json={
                "model": self.model,
                "messages": messages,
                "stream": False,
            },
        )

        response.raise_for_status()
        data = response.json()

        # Extract token statistics from Ollama
        input_tokens = data.get("prompt_eval_count") or 0
        output_tokens = data.get("eval_count") or 0

        # Ollama reports durations in nanoseconds
        prompt_duration_ns = (
            data.get("prompt_eval_duration") or 0
        )
        generation_duration_ns = (
            data.get("eval_duration") or 0
        )
        total_duration_ns = (
            data.get("total_duration") or 0
        )

        # Calculate generation speed
        tokens_per_second = (
            output_tokens / (generation_duration_ns / 1e9)
            if generation_duration_ns > 0
            else 0.0
        )

        return {
            "model": data.get("model", self.model),
            "answer": data["message"]["content"],
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            },
            "performance": {
                "prompt_eval_duration_ns": prompt_duration_ns,
                "generation_duration_ns": generation_duration_ns,
                "total_duration_ns": total_duration_ns,
                "tokens_per_second": round(
                    tokens_per_second, 2
                ),
            },
        }

    async def generate(self, prompt: str) -> dict:
        # Preserve the existing single-prompt API.
        return await self.chat([
            {
                "role": "user",
                "content": prompt,
            }
        ])

    async def close(self):
        await self.client.aclose()
