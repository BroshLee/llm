import json
import os
import re

import httpx


# Kept short: small models like phi3 copy the prompt's style (lists, headings)
# and ignore the user's requested length when the instructions are long.
SYSTEM_PROMPT = (
    "You are a helpful technical assistant. Answer the user's question "
    "directly and concisely, respecting any requested length or format. "
    "Use Markdown and fenced code blocks only when the answer needs code "
    "or structure."
)

# Cheap keyword/arithmetic check so routing needs no extra LLM call.
MATH_PATTERN = re.compile(
    r"\d\s*[+*/^=x]\s*\d|\d\s+-\s+\d"
    r"|\b(solve|equation|calculate|compute|integral|integrate|derivative|"
    r"differentiate|algebra|geometry|trigonometry|probability|matrix|"
    r"factorial|sqrt|square root|logarithm|percentage|prime number)\b",
    re.IGNORECASE,
)


class AIClient:
    def __init__(self):
        self.base_url = os.getenv(
            "OLLAMA_BASE_URL",
            "http://localhost:11434",
        ).rstrip("/")

        self.model = os.getenv(
            "OLLAMA_MODEL",
            "llama3:latest",
        )

        self.math_model = os.getenv("OLLAMA_MATH_MODEL", self.model)

        # Keeps the model in RAM between requests (Ollama's default is 5m).
        self.keep_alive = os.getenv("OLLAMA_KEEP_ALIVE", "30m")

        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(
                180.0,
                connect=5.0,
            ),
        )

        self.num_ctx = int(
            os.getenv("OLLAMA_NUM_CTX", "2048")
        )

        self.num_predict = int(
            os.getenv("OLLAMA_NUM_PREDICT", "256")
        )

        # Unset means Ollama's default (0.8); 0 gives repeatable answers.
        temperature = os.getenv("OLLAMA_TEMPERATURE")
        self.temperature = (
            float(temperature) if temperature else None
        )

        self.embedding_model = os.getenv(
            "OLLAMA_EMBED_MODEL",
            "nomic-embed-text",
        )

    def model_for(self, text: str, task: str = "auto") -> str:
        if task == "math":
            return self.math_model

        if task == "auto" and MATH_PATTERN.search(text):
            return self.math_model

        return self.model

    def _options(self, limit_output: bool = True) -> dict:
        # Same num_ctx on every call, otherwise Ollama reloads the model.
        options = {"num_ctx": self.num_ctx}

        if limit_output:
            options["num_predict"] = self.num_predict

        if self.temperature is not None:
            options["temperature"] = self.temperature

        return options

    async def _post(
        self,
        path: str,
        body: dict,
        model: str | None = None,
    ) -> dict:
        response = await self.client.post(
            path,
            json={
                "model": model or self.model,
                "stream": False,
                "keep_alive": self.keep_alive,
                **body,
            },
        )
        response.raise_for_status()
        return response.json()

    def _stats(self, data: dict) -> dict:
        input_tokens = data.get("prompt_eval_count") or 0
        output_tokens = data.get("eval_count") or 0

        # Ollama reports durations in nanoseconds
        prompt_duration_ns = data.get("prompt_eval_duration") or 0
        generation_duration_ns = data.get("eval_duration") or 0
        total_duration_ns = data.get("total_duration") or 0

        tokens_per_second = (
            output_tokens / (generation_duration_ns / 1e9)
            if generation_duration_ns > 0
            else 0.0
        )

        return {
            "model": data.get("model", self.model),
            "usage": {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            },
            "performance": {
                "prompt_eval_duration_ns": prompt_duration_ns,
                "generation_duration_ns": generation_duration_ns,
                "total_duration_ns": total_duration_ns,
                "tokens_per_second": round(tokens_per_second, 2),
            },
        }

    def _with_system_prompt(self, messages: list[dict]) -> list[dict]:
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            *messages,
        ]

    async def chat(
        self,
        messages: list[dict],
        model: str | None = None,
    ) -> dict:
        data = await self._post(
            "/api/chat",
            {
                "messages": self._with_system_prompt(messages),
                "options": self._options(),
            },
            model,
        )

        return {
            **self._stats(data),
            "answer": data["message"]["content"],
        }

    async def open_chat_stream(
        self,
        messages: list[dict],
        model: str | None = None,
    ) -> httpx.Response:
        # Returns once Ollama starts answering, so connection/model errors
        # surface here before the response to the caller has started.
        request = self.client.build_request(
            "POST",
            "/api/chat",
            json={
                "model": model or self.model,
                "stream": True,
                "keep_alive": self.keep_alive,
                "messages": self._with_system_prompt(messages),
                "options": self._options(),
            },
        )
        response = await self.client.send(request, stream=True)

        if response.is_error:
            await response.aread()
            await response.aclose()
            response.raise_for_status()

        return response

    async def iter_chat_stream(self, response: httpx.Response):
        # Yields each text chunk (str), then one final stats dict.
        answer_parts = []

        try:
            async for line in response.aiter_lines():
                if not line:
                    continue

                data = json.loads(line)

                if "error" in data:
                    raise ValueError(data["error"])

                chunk = data.get("message", {}).get("content", "")

                if chunk:
                    answer_parts.append(chunk)
                    yield chunk

                if data.get("done"):
                    yield {
                        **self._stats(data),
                        "answer": "".join(answer_parts),
                    }

        finally:
            await response.aclose()

    async def generate(
        self,
        prompt: str,
        model: str | None = None,
    ) -> dict:
        return await self.chat(
            [{"role": "user", "content": prompt}],
            model,
        )

    async def generate_json(
        self,
        prompt: str,
        schema: dict,
    ) -> dict:
        data = await self._post(
            "/api/generate",
            {
                "prompt": prompt,
                "format": schema,
                # No num_predict: a cut-off answer would be invalid JSON.
                "options": self._options(limit_output=False),
            },
        )

        # Ollama returns the generated JSON as a string in "response".
        return {
            **self._stats(data),
            "data": json.loads(data.get("response", "")),
        }

    async def close(self):
        await self.client.aclose()

    async def embed(self, text: str) -> list[float]:
        """
        Convert text into a vector using Ollama's embedding API.
        """
        response = await self.client.post(
            "/api/embed",
            json={
                "model": self.embedding_model,
                "input": text,
                "keep_alive": self.keep_alive,
            },
        )

        response.raise_for_status()
        data = response.json()

        embeddings = data.get("embeddings")

        if not embeddings or not embeddings[0]:
            raise ValueError(
                "Ollama returned an empty embedding."
            )

        return embeddings[0]