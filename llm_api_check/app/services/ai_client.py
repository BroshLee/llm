
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
        # Add formatting instructions before the conversation.
        formatted_messages = [
            {
                "role": "system",
                "content": SYSTEM_PROMPT,
            },
            *messages,
        ]
        # formatted_messages = messages

        response = await self.client.post(
            "/api/chat",
            json={
                "model": self.model,
                "messages": formatted_messages,
                "stream": False,
            },
        )

        response.raise_for_status()
        data = response.json()

        input_tokens = data.get("prompt_eval_count") or 0
        output_tokens = data.get("eval_count") or 0

        prompt_duration_ns = (
            data.get("prompt_eval_duration") or 0
        )
        generation_duration_ns = (
            data.get("eval_duration") or 0
        )
        total_duration_ns = (
            data.get("total_duration") or 0
        )

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
        return await self.chat([
            {
                "role": "user",
                "content": prompt,
            }
        ])

    async def close(self):
        await self.client.aclose()
