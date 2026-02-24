import logging
import os
from typing import List

from google import genai
from google.genai import errors, types

from config import GENERATION_MODEL, MAX_OUTPUT_TOKENS, NO_ANSWER_EN, NO_ANSWER_SV

logger = logging.getLogger(__name__)

# One client per process is enough.
_client: genai.Client | None = None


def _get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=os.getenv("API_KEY"))
    return _client


def generate_response(
    query: str,
    context: str | List[str],
    model_name: str = GENERATION_MODEL,
    answer_language: str = "English",
) -> str:
    """Generate a grounded answer from context chunks using Gemini."""
    context_text = "\n\n".join(context) if isinstance(context, list) else context
    no_answer = NO_ANSWER_EN if answer_language == "English" else NO_ANSWER_SV

    if answer_language == "English":
        lang = "Always respond in English, regardless of the language of the question."
    else:
        lang = "Always respond in Swedish, regardless of the language of the question."

    system_prompt = (
        f"You answer questions about Ableton Live 12 and MIDI. {lang} "
        "Base your answer on the context below. "
        "If the context directly answers the question, give a short, direct answer. "
        "If the context is only partly relevant, use whatever is useful and fill in with general Ableton Live knowledge to give a helpful answer. "
        "Point toward the relevant manual section when it helps. "
        f"Only say '{no_answer}' if the context has absolutely nothing to do with the question."
    )

    try:
        response = _get_client().models.generate_content(
            model=model_name,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                max_output_tokens=MAX_OUTPUT_TOKENS,
            ),
            contents=f"Context:\n{context_text}\n\nQuestion:\n{query}",
        )
        return response.text
    except errors.ClientError as e:
        # 429 = quota exhausted; surface a readable message instead of crashing.
        if e.code == 429:
            logger.warning("Gemini quota exceeded: %s", e)
            return (
                "⚠️ Gemini API quota exceeded. The free tier resets daily — "
                "try again later or enable billing at https://ai.dev/rate-limit"
            )
        logger.error("Gemini API error: %s", e)
        raise
