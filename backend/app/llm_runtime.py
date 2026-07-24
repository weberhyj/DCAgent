from __future__ import annotations

from collections.abc import Mapping

NON_GENERATING_PROVIDERS = frozenset({"", "template", "mock"})


def normalize_llm_provider(environ: Mapping[str, str]) -> str:
    return environ.get("LLM_PROVIDER", "template").strip().lower().replace("-", "_")


def validate_production_llm_provider(environ: Mapping[str, str]) -> str:
    provider = normalize_llm_provider(environ)
    if provider in NON_GENERATING_PROVIDERS:
        raise ValueError(
            "Production runtime requires a real LLM provider; "
            "set LLM_PROVIDER=physoc_deepseek for the internal deployment"
        )
    return provider
