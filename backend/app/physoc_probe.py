from __future__ import annotations

import argparse
import json
import os
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import perf_counter
from typing import Any

from .llm import (
    LLMProvider,
    LLMRequest,
    PhysocDeepSeekLLMProvider,
    create_llm_provider,
)
from .models import (
    KnowledgeChunkModel,
    KnowledgeSearchHitModel,
    KnowledgeSourceModel,
)
from .runtime_env import load_runtime_environment

_REPORT_FIELDS = (
    "answerChars",
    "citationCount",
    "elapsedMs",
    "model",
    "passed",
    "provider",
    "streamPath",
)


def _probe_hit() -> KnowledgeSearchHitModel:
    source = KnowledgeSourceModel(
        id="physoc-probe-source",
        name="physoc-probe.txt",
        source_type="文档",
        records=1,
        status="已索引",
        updated_at="probe",
        classification="内部",
    )
    chunk = KnowledgeChunkModel(
        id="physoc-probe-chunk",
        source_id=source.id,
        chunk_index=0,
        text="Physoc 链路正常",
        token_count=8,
    )
    return KnowledgeSearchHitModel(
        source=source,
        chunk=chunk,
        score=10.0,
        keyword_score=10.0,
        vector_score=10.0,
        rank=1,
        matched_terms=["Physoc", "链路"],
    )


def run_physoc_probe(
    environ: Mapping[str, str],
    provider_factory: Callable[[Mapping[str, str] | None], LLMProvider] = create_llm_provider,
    clock_values: Iterable[float] | None = None,
) -> dict[str, Any]:
    provider = provider_factory(environ)
    if not isinstance(provider, PhysocDeepSeekLLMProvider):
        raise ValueError("Physoc probe requires the physoc_deepseek provider")

    values = iter(clock_values) if clock_values is not None else None

    def clock() -> float:
        if values is None:
            return perf_counter()
        return next(values)

    request = LLMRequest(
        content="请仅根据证据说明 Physoc 链路是否正常",
        mode="source",
        knowledge_hits=[_probe_hit()],
        previous_messages=[],
        agent_context="目标服务器 Physoc POST/SSE 互操作探测",
    )
    started_at = clock()
    message = provider.generate_reply(request)
    elapsed_ms = round((clock() - started_at) * 1000, 3)

    answer = " ".join(paragraph.text for paragraph in message.paragraphs).strip()
    citations = [citation for paragraph in message.paragraphs for citation in paragraph.citations]
    if not answer:
        raise ValueError("Physoc probe returned an empty answer")
    if not citations:
        raise ValueError("Physoc probe returned no citation")

    return {
        "passed": True,
        "provider": "physoc_deepseek",
        "model": provider.model,
        "streamPath": provider.stream_path,
        "elapsedMs": elapsed_ms,
        "answerChars": len(answer),
        "citationCount": len(citations),
    }


def write_probe_report(path: str | Path, report: Mapping[str, Any]) -> None:
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    safe_report = {field: report[field] for field in _REPORT_FIELDS}
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=target.parent,
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            json.dump(
                safe_report,
                temporary_file,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            temporary_file.write("\n")
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        temporary_path.replace(target)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Probe the configured Physoc LLM route")
    parser.add_argument(
        "--report",
        default="artifacts/benchmarks/physoc-probe.json",
        help="Path for the sanitized JSON probe report",
    )
    args = parser.parse_args(argv)

    load_runtime_environment()
    report = run_physoc_probe(os.environ)
    write_probe_report(args.report, report)
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
