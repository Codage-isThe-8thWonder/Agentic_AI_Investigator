from __future__ import annotations

import os
from typing import Optional, Type, TypeVar

from dotenv import load_dotenv
from google import genai
from pydantic import BaseModel


load_dotenv()


T = TypeVar(
    "T",
    bound=BaseModel,
)


class GeminiClient:

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ):

        self.api_key = (
            api_key
            or os.getenv("GEMINI_API_KEY")
        )

        self.model = (
            model
            or os.getenv(
                "GEMINI_MODEL",
                "gemini-3.6-flash",
            )
        )

        if not self.api_key:

            raise RuntimeError(
                "GEMINI_API_KEY is missing. "
                "Add it to .env."
            )

        self.client = genai.Client(
            api_key=self.api_key
        )

        print(
            f"Gemini client ready: {self.model}"
        )

    # ========================================================
    # TEXT
    # ========================================================

    def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
    ) -> str:

        final_prompt = prompt

        if system_instruction:

            final_prompt = (
                "SYSTEM INSTRUCTION:\n"
                f"{system_instruction}\n\n"
                "TASK:\n"
                f"{prompt}"
            )

        try:

            response = (
                self.client
                .interactions
                .create(
                    model=self.model,
                    input=final_prompt,
                    store=False,
                )
            )

            output = response.output_text

            if not output:

                raise RuntimeError(
                    "Gemini returned empty output."
                )

            return output.strip()

        except Exception as exc:

            message = str(exc)

            if (
                "429" in message
                or "quota" in message.lower()
                or "too_many_requests"
                in message.lower()
            ):

                raise RuntimeError(
                    "GEMINI_QUOTA_EXCEEDED: "
                    f"{message}"
                ) from exc

            raise RuntimeError(
                "Gemini API request failed: "
                f"{message}"
            ) from exc

    # ========================================================
    # STRUCTURED
    # ========================================================

    def generate_structured(
        self,
        prompt: str,
        schema: Type[T],
        system_instruction: Optional[str] = None,
    ) -> T:

        if not isinstance(
            schema,
            type,
        ):
            raise TypeError(
                "schema must be a Pydantic model class."
            )

        if not issubclass(
            schema,
            BaseModel,
        ):
            raise TypeError(
                "schema must inherit from BaseModel."
            )

        final_prompt = prompt

        if system_instruction:

            final_prompt = (
                "SYSTEM INSTRUCTION:\n"
                f"{system_instruction}\n\n"
                "TASK:\n"
                f"{prompt}"
            )

        try:

            response = (
                self.client
                .interactions
                .create(
                    model=self.model,
                    input=final_prompt,
                    store=False,

                    response_format={
                        "type": "text",
                        "mime_type": (
                            "application/json"
                        ),
                        "schema": (
                            schema
                            .model_json_schema()
                        ),
                    },
                )
            )

            output = response.output_text

            if not output:

                raise RuntimeError(
                    "Gemini returned empty structured output."
                )

            return schema.model_validate_json(
                output
            )

        except Exception as exc:

            message = str(exc)

            if (
                "429" in message
                or "quota" in message.lower()
                or "too_many_requests"
                in message.lower()
            ):

                raise RuntimeError(
                    "GEMINI_QUOTA_EXCEEDED: "
                    f"{message}"
                ) from exc

            if isinstance(
                exc,
                RuntimeError,
            ):
                raise

            raise RuntimeError(
                "Gemini structured request failed: "
                f"{message}"
            ) from exc