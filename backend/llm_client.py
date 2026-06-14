"""
Lantern — LLM Client
Single entry point for all Azure OpenAI calls in this codebase.
"""
import os
import time

from openai import AzureOpenAI


def call_llm(messages: list[dict], role: str = "answerer", temperature: float = 0.3) -> dict:
    """
    Call Azure OpenAI and return a normalized response dict.

    role: "answerer" → AZURE_ANSWERER_DEPLOYMENT
          "evaluator" → AZURE_EVALUATOR_DEPLOYMENT
    """
    client = AzureOpenAI(
        azure_endpoint=os.getenv("AZURE_AI_PROJECT_ENDPOINT"),
        api_key=os.getenv("AZURE_AI_API_KEY"),
        api_version="2024-08-01-preview",
    )
    deployment = (
        os.getenv("AZURE_ANSWERER_DEPLOYMENT")
        if role == "answerer"
        else os.getenv("AZURE_EVALUATOR_DEPLOYMENT")
    )

    t0 = time.perf_counter()
    response = client.chat.completions.create(
        model=deployment,
        messages=messages,
        temperature=temperature,
    )
    latency_ms = int((time.perf_counter() - t0) * 1000)

    usage = response.usage
    return {
        "content":           response.choices[0].message.content or "",
        "tokens_used":       usage.total_tokens,
        "prompt_tokens":     usage.prompt_tokens,
        "completion_tokens": usage.completion_tokens,
        "model":             response.model,
        "finish_reason":     response.choices[0].finish_reason,
        "latency_ms":        latency_ms,
    }
