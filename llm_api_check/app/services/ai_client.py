import json
import os

import httpx


SYSTEM_PROMPT = """
You are a helpful technical assistant.

Format your answers using clean, readable Markdown.

Follow these rules:
1. Start with a short, descriptive heading when appropriate.
2. Use clear section headings for longer answers.
3. Use bullet points or numbered steps for explanations.
4. Put Python code inside fenced code blocks with the python language label.
5. Use inline code formatting for variables, functions, commands,
   filenames, and API endpoints.
6. Use tables only when comparing multiple items.
7. Explain code examples in simple, step-by-step language.
8. Highlight important warnings and key takeaways.
9. Avoid unnecessary repetition and excessively long paragraphs.
10. Adapt the structure to the question. For simple questions,
    answer concisely without unnecessary headings.
11. Do not invent facts or claim that code was executed when it was not.
"""


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

        # Keeps the model in RAM between requests (Ollama's default is 5m).
        self.keep_alive = os.getenv("OLLAMA_KEEP_ALIVE", "30m")

        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(
                180.0,
                connect=5.0,
            ),
        )

    async def _post(self, path: str, body: dict) -> dict:
        response = await self.client.post(
            path,
            json={
                "model": self.model,
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

    async def chat(self, messages: list[dict]) -> dict:
        data = await self._post(
            "/api/chat",
            {"messages": self._with_system_prompt(messages)},
        )

        return {
            **self._stats(data),
            "answer": data["message"]["content"],
        }

    async def open_chat_stream(
        self,
        messages: list[dict],
    ) -> httpx.Response:
        # Returns once Ollama starts answering, so connection/model errors
        # surface here, before the HTTP response to the caller has started.
        request = self.client.build_request(
            "POST",
            "/api/chat",
            json={
                "model": self.model,
                "stream": True,
                "keep_alive": self.keep_alive,
                "messages": self._with_system_prompt(messages),
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

    async def generate(self, prompt: str) -> dict:
        return await self.chat([
            {
                "role": "user",
                "content": prompt,
            }
        ])

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
            },
        )

        # Ollama returns the generated JSON as a string in "response".
        return {
            **self._stats(data),
            "data": json.loads(data.get("response", "")),
        }

    async def close(self):
        await self.client.aclose()
