"""
src/chatbot_engine.py

Orchestrates the full RAG pipeline:
  1. Retrieve context (semantic or aggregate)
  2. Build prompt with history
  3. Call Anthropic API via the official Python SDK
  4. Return answer + metadata

Uses the Anthropic Python SDK — no API key management needed.
When deployed on Streamlit Cloud the SDK picks up the key automatically
from the ANTHROPIC_API_KEY secret you set in the dashboard.
For local dev: export ANTHROPIC_API_KEY=sk-ant-...
"""

import anthropic

from src.prompts import SYSTEM_PROMPT, build_rag_prompt
from src.retrieval import retrieve_context

MODEL = "claude-sonnet-4-20250514"
MAX_TOKENS = 800


def _call_llm(system: str, user_message: str) -> str:
    """
    Call Claude via the official Anthropic Python SDK.
    The SDK reads ANTHROPIC_API_KEY from the environment automatically.
    """
    client = anthropic.Anthropic()   # picks up key from env
    message = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=system,
        messages=[{"role": "user", "content": user_message}],
    )
    return message.content[0].text


def chat(
    question: str,
    history: list[dict],
    df,
    index=None,
    k: int = 6,
) -> tuple[str, str, str]:
    """
    Full RAG + LLM pipeline.

    Parameters
    ----------
    question : user's latest message
    history  : list of {"role": ..., "content": ...} dicts
    df       : unsold cars DataFrame (with Predicted Price, Deal Label, etc.)
    index    : FAISS index (or None — falls back to aggregate retrieval)
    k        : number of cars to retrieve

    Returns
    -------
    answer   : LLM response string
    context  : retrieved context string (for transparency)
    mode     : 'semantic' or 'aggregate'
    """
    context, mode = retrieve_context(question, df, index=index, k=k)
    user_message  = build_rag_prompt(context, history, question)
    answer        = _call_llm(SYSTEM_PROMPT, user_message)
    return answer, context, mode
