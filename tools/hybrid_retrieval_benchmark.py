"""CPU-only private-network capacity gate for Qwen3 hybrid retrieval."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Protocol


REPO_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = (REPO_ROOT / "backend").resolve()
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.retrieval_models import RetrievalRequest, RetrievalScope  # noqa: E402


_FALLBACK_REASONS = frozenset(
    {
        "alias_mismatch",
        "circuit_open",
        "embedding_unavailable",
        "hybrid_unavailable",
        "qdrant_timeout",
        "qdrant_unavailable",
        "qwen_empty_legacy_nonempty",
        "qwen_timeout",
        "reranker_unavailable",
        "retrieval_scope_unavailable",
        "shadow_queue_full",
    }
)
_IDENTIFIER_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_GATE_NAMES = frozenset({"requests", "p95_seconds", "error_rate", "fallback_rate"})


class Retriever(Protocol):
    def retrieve(self, request: RetrievalRequest) -> object: ...


@dataclass(frozen=True, slots=True)
class BenchmarkQuestion:
    case_id: str
    question: str

    def __post_init__(self) -> None:
        case_id = self.case_id.strip()
        question = self.question.strip()
        if _IDENTIFIER_PATTERN.fullmatch(case_id) is None or not question:
            raise ValueError("benchmark case_id and question must not be empty")
        object.__setattr__(self, "case_id", case_id)
        object.__setattr__(self, "question", question)


@dataclass(frozen=True, slots=True)
class BenchmarkSummary:
    passed: bool
    requests: int
    p95_seconds: float
    error_rate: float
    fallback_rate: float
    passed_gates: tuple[str, ...]
    failed_gates: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "requests": self.requests,
            "p95Seconds": self.p95_seconds,
            "errorRate": self.error_rate,
            "fallbackRate": self.fallback_rate,
            "passedGates": list(self.passed_gates),
            "failedGates": list(self.failed_gates),
        }


@dataclass(slots=True)
class ProductionRuntime:
    retriever: Retriever
    scope: RetrievalScope
    resources: tuple[object, ...]

    def close(self) -> None:
        closed: set[int] = set()
        for resource in reversed(self.resources):
            if id(resource) in closed:
                continue
            closed.add(id(resource))
            close = getattr(resource, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    pass


def summarize_results(
    *,
    latencies: Sequence[float],
    errors: int,
    fallbacks: int,
    requests: int,
    p95_limit: float,
    error_rate_limit: float,
    fallback_rate_limit: float,
) -> BenchmarkSummary:
    p95_valid = bool(latencies) and all(
        _finite_nonnegative(value) for value in latencies
    )
    p95_seconds = _percentile(latencies, 0.95) if p95_valid else 0.0
    requests_valid = (
        not isinstance(requests, bool) and isinstance(requests, int) and requests > 0
    )
    errors_valid = _valid_count(errors, requests)
    fallbacks_valid = _valid_count(fallbacks, requests)
    error_rate = round(errors / requests, 6) if requests_valid and errors_valid else 0.0
    fallback_rate = (
        round(fallbacks / requests, 6) if requests_valid and fallbacks_valid else 0.0
    )
    limits_valid = all(
        _finite_nonnegative(value)
        for value in (p95_limit, error_rate_limit, fallback_rate_limit)
    )
    gate_results = (
        ("p95_seconds", p95_valid and limits_valid and p95_seconds <= p95_limit),
        (
            "error_rate",
            requests_valid
            and errors_valid
            and limits_valid
            and error_rate <= error_rate_limit,
        ),
        (
            "fallback_rate",
            requests_valid
            and fallbacks_valid
            and limits_valid
            and fallback_rate <= fallback_rate_limit,
        ),
    )
    passed_gates = tuple(name for name, passed in gate_results if passed)
    failed = [name for name, passed in gate_results if not passed]
    if not requests_valid:
        failed.insert(0, "requests")
    return BenchmarkSummary(
        passed=not failed,
        requests=requests if requests_valid else 0,
        p95_seconds=p95_seconds,
        error_rate=error_rate,
        fallback_rate=fallback_rate,
        passed_gates=passed_gates,
        failed_gates=tuple(failed),
    )


def run_benchmark(
    *,
    retriever: Retriever,
    scope: RetrievalScope,
    questions: Sequence[BenchmarkQuestion],
    concurrency: int,
    requests: int,
    p95_limit: float,
    error_rate_limit: float,
    fallback_rate_limit: float,
) -> dict[str, object]:
    if (
        isinstance(concurrency, bool)
        or not isinstance(concurrency, int)
        or concurrency <= 0
    ):
        raise ValueError("concurrency must be a positive integer")
    if isinstance(requests, bool) or not isinstance(requests, int) or requests < 0:
        raise ValueError("requests must be a non-negative integer")
    if requests and not questions:
        raise ValueError("questions JSONL must contain at least one case")
    next_request = 0
    index_lock = Lock()
    result_lock = Lock()
    records: list[tuple[int, dict[str, object]]] = []
    latencies: list[float] = []
    errors = 0
    fallbacks = 0

    def worker() -> None:
        nonlocal next_request, errors, fallbacks
        while True:
            with index_lock:
                if next_request >= requests:
                    return
                request_index = next_request
                next_request += 1
            question = questions[request_index % len(questions)]
            try:
                outcome = retriever.retrieve(
                    RetrievalRequest(
                        query=question.question,
                        limit=8,
                        routing_key=question.case_id,
                        scope=scope,
                    )
                )
                latency = _outcome_latency_seconds(outcome)
                fallback_reason = _sanitize_fallback(
                    getattr(outcome, "fallback_reason", None)
                )
                record = {
                    "caseId": question.case_id,
                    "chunkIds": _outcome_chunk_ids(outcome),
                    "mode": _outcome_mode(outcome),
                    "latencySeconds": latency,
                    "fallbackReason": fallback_reason,
                    "error": False,
                }
                with result_lock:
                    latencies.append(latency)
                    if fallback_reason is not None:
                        fallbacks += 1
                    records.append((request_index, record))
            except Exception:
                with result_lock:
                    errors += 1
                    records.append(
                        (
                            request_index,
                            {
                                "caseId": question.case_id,
                                "chunkIds": [],
                                "mode": "error",
                                "latencySeconds": 0.0,
                                "fallbackReason": None,
                                "error": True,
                            },
                        )
                    )

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(worker) for _ in range(concurrency)]
        for future in futures:
            future.result()
    summary = summarize_results(
        latencies=latencies,
        errors=errors,
        fallbacks=fallbacks,
        requests=requests,
        p95_limit=p95_limit,
        error_rate_limit=error_rate_limit,
        fallback_rate_limit=fallback_rate_limit,
    )
    records.sort(key=lambda item: item[0])
    return {
        "summary": summary.to_dict(),
        "records": [record for _, record in records],
    }


def load_questions(path: Path) -> list[BenchmarkQuestion]:
    questions: list[BenchmarkQuestion] = []
    with path.resolve().open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid questions JSONL at line {line_number}"
                ) from error
            if not isinstance(payload, Mapping):
                raise ValueError(
                    f"questions JSONL line {line_number} must be an object"
                )
            case_id = payload.get("caseId", payload.get("case_id"))
            question = payload.get("question")
            if not isinstance(case_id, str) or not isinstance(question, str):
                raise ValueError(
                    f"questions JSONL line {line_number} requires string caseId and question"
                )
            questions.append(BenchmarkQuestion(case_id, question))
    return questions


def write_report(path: Path, report: Mapping[str, object]) -> None:
    safe_report = _safe_report(report)
    destination = path.resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(
            safe_report, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
        )
        + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)


def build_production_runtime(
    environ: Mapping[str, str] | None = None,
) -> ProductionRuntime:
    from app.main import _DefaultRetrievalResourceFactory
    from app.retrieval_models import RetrievalMode
    from app.retrieval_publication import collection_publication_version
    from app.retrieval_settings import RetrievalSettings

    environment = dict(os.environ if environ is None else environ)
    settings = RetrievalSettings.from_environ(environment)
    if settings.mode is RetrievalMode.LEGACY:
        raise ValueError("hybrid benchmark requires RETRIEVAL_MODE=shadow or qwen3")
    factory = _DefaultRetrievalResourceFactory()
    qdrant = factory.create_qdrant_client(settings)
    gateway = factory.create_gateway(qdrant, settings)
    embedding = factory.create_embedding_client(settings)
    reranker = factory.create_reranker_client(settings)
    sparse = factory.create_sparse_encoder(environment)
    retriever = factory.create_hybrid_retriever(
        settings=settings,
        embedding=embedding,
        sparse=sparse,
        gateway=gateway,
        reranker=reranker,
    )
    try:
        collection_name = gateway.resolve_alias()  # type: ignore[attr-defined]
        scope = RetrievalScope(
            settings.knowledge_base_id,
            settings.permission_tags,
            collection_publication_version(collection_name),
        )
    except Exception:
        ProductionRuntime(
            retriever=retriever,  # type: ignore[arg-type]
            scope=RetrievalScope("unavailable", ("unavailable",), "unavailable"),
            resources=(qdrant, embedding, reranker, retriever),
        ).close()
        raise
    return ProductionRuntime(
        retriever=retriever,  # type: ignore[arg-type]
        scope=scope,
        resources=(qdrant, embedding, reranker, retriever),
    )


def _outcome_latency_seconds(outcome: object) -> float:
    stage_ms = getattr(outcome, "stage_ms", None)
    if not isinstance(stage_ms, Mapping) or not stage_ms:
        raise ValueError("retrieval outcome has no stage timings")
    values = list(stage_ms.values())
    if not all(_finite_nonnegative(value) for value in values):
        raise ValueError("retrieval outcome has invalid stage timings")
    return round(sum(float(value) for value in values) / 1000.0, 6)


def _outcome_chunk_ids(outcome: object) -> list[str]:
    candidates = getattr(outcome, "candidates", ())
    return [str(item.chunk_id) for item in candidates if hasattr(item, "chunk_id")]


def _outcome_mode(outcome: object) -> str:
    mode = getattr(outcome, "mode", "qwen3")
    value = getattr(mode, "value", mode)
    return str(value) if str(value) in {"legacy", "shadow", "qwen3"} else "qwen3"


def _sanitize_fallback(value: object) -> str | None:
    if value is None:
        return None
    normalized = str(value)
    return normalized if normalized in _FALLBACK_REASONS else "hybrid_unavailable"


def _finite_nonnegative(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and float(value) >= 0.0
    )


def _valid_count(value: object, requests: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, int)
        and value >= 0
        and isinstance(requests, int)
        and value <= requests
    )


def _percentile(values: Sequence[float], quantile: float) -> float:
    ordered = sorted(float(value) for value in values)
    index = max(0, math.ceil(len(ordered) * quantile) - 1)
    return round(ordered[index], 6)


def _safe_report(report: Mapping[str, object]) -> dict[str, object]:
    summary = report.get("summary", {})
    records = report.get("records", ())
    safe_summary: dict[str, object] = {
        "passed": False,
        "requests": 0,
        "p95Seconds": 0.0,
        "errorRate": 0.0,
        "fallbackRate": 0.0,
        "passedGates": [],
        "failedGates": [],
    }
    if isinstance(summary, Mapping):
        safe_summary["passed"] = summary.get("passed") is True
        requests = summary.get("requests")
        if (
            not isinstance(requests, bool)
            and isinstance(requests, int)
            and requests >= 0
        ):
            safe_summary["requests"] = requests
        for key in ("p95Seconds", "errorRate", "fallbackRate"):
            value = summary.get(key)
            if _finite_nonnegative(value):
                safe_summary[key] = float(value)
        for key in ("passedGates", "failedGates"):
            values = summary.get(key)
            if isinstance(values, Sequence) and not isinstance(
                values, (str, bytes, bytearray)
            ):
                safe_summary[key] = [
                    str(value) for value in values if str(value) in _GATE_NAMES
                ]
    safe_records: list[dict[str, object]] = []
    if isinstance(records, Sequence) and not isinstance(
        records, (str, bytes, bytearray)
    ):
        for record in records:
            if not isinstance(record, Mapping):
                continue
            safe_records.append(
                {
                    "caseId": _safe_identifier(record.get("caseId")),
                    "chunkIds": _safe_identifiers(record.get("chunkIds")),
                    "mode": _safe_mode(record.get("mode")),
                    "latencySeconds": (
                        float(record["latencySeconds"])
                        if _finite_nonnegative(record.get("latencySeconds"))
                        else 0.0
                    ),
                    "fallbackReason": _sanitize_fallback(record.get("fallbackReason")),
                    "error": record.get("error") is True,
                }
            )
    return {"summary": safe_summary, "records": safe_records}


def _safe_identifier(value: object) -> str:
    normalized = str(value)
    return normalized if _IDENTIFIER_PATTERN.fullmatch(normalized) else "redacted"


def _safe_identifiers(value: object) -> list[str]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        return []
    return [
        identifier
        for item in value
        if (identifier := _safe_identifier(item)) != "redacted"
    ]


def _safe_mode(value: object) -> str:
    normalized = str(value)
    return (
        normalized if normalized in {"legacy", "shadow", "qwen3", "error"} else "error"
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--concurrency", type=int, default=15)
    parser.add_argument("--requests", type=int, default=150)
    parser.add_argument("--p95-seconds", type=float, default=5.0)
    parser.add_argument("--max-error-rate", type=float, default=0.01)
    parser.add_argument("--max-fallback-rate", type=float, default=0.01)
    parser.add_argument("--questions-jsonl", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    runtime: ProductionRuntime | None = None
    try:
        questions = load_questions(arguments.questions_jsonl)
        runtime = build_production_runtime()
        report = run_benchmark(
            retriever=runtime.retriever,
            scope=runtime.scope,
            questions=questions,
            concurrency=arguments.concurrency,
            requests=arguments.requests,
            p95_limit=arguments.p95_seconds,
            error_rate_limit=arguments.max_error_rate,
            fallback_rate_limit=arguments.max_fallback_rate,
        )
        write_report(arguments.output_json, report)
        return 0 if report["summary"]["passed"] else 1  # type: ignore[index]
    except Exception:
        return 2
    finally:
        if runtime is not None:
            runtime.close()


if __name__ == "__main__":
    raise SystemExit(main())
