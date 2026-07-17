"""Fail-closed orchestration for fully offline local-model candidate probes."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
import errno
import hashlib
import hmac
import json
import math
import os
from pathlib import Path
import re
import stat
import struct
import subprocess
import tempfile
import textwrap
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from tools.benchmarks.report import CapacityResult


EMBEDDING_CONCURRENCIES = (1, 5, 15)
CONTEXT_LENGTHS = (512, 1024, 2048)
MAX_OUTPUT_TOKENS = 256
RERANKER_P95_GATE_MS = 1500
SUPPORTED_KINDS = ("embedding-model", "reranker-model", "generation-model")
ARTIFACT_FIELDS = ("name", "kind", "version", "sha256", "license", "localPath")
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
SAFE_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
URI_SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")
WINDOWS_ABSOLUTE = re.compile(r"^[A-Za-z]:[\\/]")
MAX_ARTIFACT_TREE_ENTRIES = 1_000_000
LOCK_FILE_NAME = ".model-probe.lock"
HELPER_IDLE_SCRIPT = "import time; time.sleep(86400)"
PUBLIC_NETWORK_MARKER = "/tmp/model-probe-public-network-attempted"

EMBEDDING_SERVER_SCRIPT = textwrap.dedent(
    f"""
    import ipaddress, pathlib, sys
    marker = pathlib.Path({PUBLIC_NETWORK_MARKER!r})
    marker.unlink(missing_ok=True)
    def public_host(value):
        host = str(value).strip().strip("[]").casefold()
        if host in {{"localhost", "embedding-service", "llama"}}:
            return False
        try:
            address = ipaddress.ip_address(host)
        except ValueError:
            return True
        return not (address.is_private or address.is_loopback or address.is_link_local)
    def audit(event, arguments):
        if event != "socket.connect" or len(arguments) < 2:
            return
        destination = arguments[1]
        host = destination[0] if isinstance(destination, tuple) and destination else destination
        if public_host(host):
            marker.write_text(str(host), encoding="utf-8")
            raise RuntimeError("public network access is forbidden during model loading")
    sys.addaudithook(audit)
    import uvicorn
    uvicorn.run("app.embedding_service:create_production_app", factory=True, host="0.0.0.0", port=8081)
    """
).strip()

EMBEDDING_METADATA_SCRIPT = textwrap.dedent(
    """
    import json, os, pathlib, urllib.request
    with urllib.request.urlopen("http://127.0.0.1:8081/v1/metadata", timeout=10) as response:
        service = json.load(response)
    print(json.dumps({
        "name": service.get("modelName"),
        "version": service.get("modelVersion"),
        "modelChecksum": service.get("modelChecksum"),
        "serviceMetadata": service,
        "publicNetworkAttempted": pathlib.Path("/tmp/model-probe-public-network-attempted").exists(),
    }, sort_keys=True))
    """
).strip()

EMBEDDING_BENCHMARK_SCRIPT = textwrap.dedent(
    """
    import concurrent.futures, json, math, os, statistics, time, urllib.request
    import psutil
    from transformers import AutoTokenizer
    def get_json(url):
        with urllib.request.urlopen(url, timeout=30) as response:
            return json.load(response)
    def post(texts, purpose):
        body = json.dumps({"texts": texts, "purpose": purpose}).encode()
        request = urllib.request.Request(
            "http://127.0.0.1:8081/v1/embeddings",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            json.load(response)
    def timed_post():
        started = time.perf_counter()
        post(["offline query"], "query")
        return (time.perf_counter() - started) * 1000
    service = get_json("http://127.0.0.1:8081/v1/metadata")
    model_name = str(service.get("modelName", ""))
    lowered_name = model_name.casefold()
    family = "BGE" if "bge" in lowered_name else "not_available"
    variant = "small" if "small" in lowered_name else "base" if "base" in lowered_name else "not_available"
    tokenizer = AutoTokenizer.from_pretrained("/probe-candidate", local_files_only=True)
    seed_ids = tokenizer.encode("offline representative document " * 2000, add_special_tokens=False)
    if len(seed_ids) < 800:
        raise RuntimeError("candidate tokenizer could not construct 800-token document")
    documents = []
    actual_counts = []
    for target in (300, 800):
        document = tokenizer.decode(seed_ids[:target], skip_special_tokens=True)
        actual = len(tokenizer.encode(document, add_special_tokens=False))
        if actual < 300 or actual > 800:
            raise RuntimeError("candidate tokenizer produced document outside 300-800 tokens")
        documents.append(document)
        actual_counts.append(actual)
    batch_start = time.perf_counter()
    post(documents * 4, "document")
    batch_docs_per_second = 8 / max(time.perf_counter() - batch_start, 1e-9)
    query = {}
    for concurrency in (1, 5, 15):
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [pool.submit(timed_post) for _ in range(max(15, concurrency * 3))]
            samples = [future.result() for future in futures]
        samples.sort()
        query[str(concurrency)] = {"p50Ms": statistics.median(samples), "p95Ms": samples[math.ceil(len(samples) * .95) - 1]}
    print(json.dumps({
        "family": family,
        "variant": variant,
        "actualTokenRange": [min(actual_counts), max(actual_counts)],
        "batchDocumentsPerSecond": batch_docs_per_second,
        "queryEmbedding": query,
        "residentMemoryBytes": psutil.Process(1).memory_info().rss,
        "modelMetadata": service,
        "publicNetworkAttempted": False,
    }, sort_keys=True))
    """
).strip()

RERANKER_METADATA_SCRIPT = textwrap.dedent(
    """
    import json, os
    print(json.dumps({
        "name": os.environ["MODEL_PROBE_CANDIDATE_NAME"],
        "version": os.environ["MODEL_PROBE_CANDIDATE_VERSION"],
        "modelChecksum": os.environ["MODEL_PROBE_CANDIDATE_SHA256"],
        "publicNetworkAttempted": False,
    }, sort_keys=True))
    """
).strip()

RERANKER_BENCHMARK_SCRIPT = textwrap.dedent(
    """
    import concurrent.futures, ipaddress, json, math, os, pathlib, statistics, sys, threading, time
    import psutil
    from FlagEmbedding import FlagReranker
    marker = pathlib.Path("/tmp/model-probe-public-network-attempted")
    marker.unlink(missing_ok=True)
    def audit(event, arguments):
        if event != "socket.connect" or len(arguments) < 2:
            return
        destination = arguments[1]
        host = destination[0] if isinstance(destination, tuple) and destination else destination
        try:
            address = ipaddress.ip_address(str(host).strip("[]"))
            public = not (address.is_private or address.is_loopback or address.is_link_local)
        except ValueError:
            public = str(host).casefold() not in {"localhost", "embedding-service", "llama"}
        if public:
            marker.write_text(str(host), encoding="utf-8")
            raise RuntimeError("public network access is forbidden during model loading")
    sys.addaudithook(audit)
    model = FlagReranker("/probe-candidate", use_fp16=False)
    pairs = [["offline query", "offline candidate"] for _ in range(20)]
    slot = threading.BoundedSemaphore(1)
    def run_once(_):
        if not slot.acquire(timeout=0.05):
            return None
        started = time.perf_counter()
        try:
            model.compute_score(pairs, normalize=True)
            return (time.perf_counter() - started) * 1000
        finally:
            slot.release()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        outcomes = list(pool.map(run_once, range(20)))
    samples = [value for value in outcomes if value is not None]
    busy = len(outcomes) - len(samples)
    samples.sort()
    print(json.dumps({
        "family": "BGE",
        "top20To10P50Ms": statistics.median(samples) if samples else "not_available",
        "top20To10P95Ms": samples[math.ceil(len(samples) * .95) - 1] if samples else "not_available",
        "residentMemoryBytes": psutil.Process().memory_info().rss,
        "modelChecksum": os.environ["MODEL_PROBE_CANDIDATE_SHA256"],
        "busyTimeoutRate": busy / len(outcomes),
        "publicNetworkAttempted": marker.exists(),
    }, sort_keys=True))
    """
).strip()

GENERATION_METADATA_SCRIPT = textwrap.dedent(
    """
    import json, os, urllib.request
    with urllib.request.urlopen("http://llama:8080/props", timeout=10) as response:
        props = json.load(response)
    model_path = str(props.get("model_path", props.get("modelPath", "")))
    if not model_path:
        raise SystemExit("llama model path is missing")
    if model_path != "/probe-candidate/model.gguf":
        raise SystemExit("llama model path is not the locked candidate")
    loader_audit = props.get("model_loader_network_audit")
    public_attempt = props.get("public_network_attempted")
    audit_enabled = loader_audit is True or str(loader_audit).casefold() in {"complete", "enabled", "audited"}
    loader_audit_available = audit_enabled and isinstance(public_attempt, bool)
    print(json.dumps({
        "name": os.environ["MODEL_PROBE_CANDIDATE_NAME"],
        "version": os.environ["MODEL_PROBE_CANDIDATE_VERSION"],
        "modelChecksum": os.environ["MODEL_PROBE_CANDIDATE_SHA256"],
        "serviceMetadata": props,
        "modelMetadata": {
            "architecture": os.environ["MODEL_PROBE_ARCHITECTURE"],
            "parameterSize": os.environ["MODEL_PROBE_PARAMETER_SIZE"],
            "quantization": os.environ["MODEL_PROBE_QUANTIZATION"],
        },
        "loaderAuditAvailable": loader_audit_available,
        "publicNetworkAttempted": public_attempt if loader_audit_available else "not_available",
    }, sort_keys=True))
    """
).strip()

GENERATION_BENCHMARK_SCRIPT = textwrap.dedent(
    """
    import concurrent.futures, json, math, os, statistics, time, urllib.request
    def json_request(path, payload):
        body = json.dumps(payload).encode()
        request = urllib.request.Request("http://llama:8080" + path, data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=300) as response:
            return json.load(response)
    seed = "offline context token " * 3000
    seed_tokens = json_request("/tokenize", {"content": seed, "add_special": False}).get("tokens", [])
    if len(seed_tokens) < 2048:
        raise RuntimeError("llama tokenizer could not construct 2048 context tokens")
    contexts = {}
    prompts = {}
    for target in (512, 1024, 2048):
        tokens = seed_tokens[:target]
        prompt = json_request("/detokenize", {"tokens": tokens}).get("content", "")
        actual = len(json_request("/tokenize", {"content": prompt, "add_special": False}).get("tokens", []))
        if actual != target:
            raise RuntimeError("llama tokenize/detokenize did not preserve context count")
        contexts[str(target)] = actual
        prompts[str(target)] = prompt
    def sample(prompt):
        body = json.dumps({"model": os.environ["MODEL_PROBE_CANDIDATE_NAME"], "prompt": prompt, "n_predict": 256, "stream": True}).encode()
        request = urllib.request.Request("http://llama:8080/completion", data=body, headers={"Content-Type": "application/json"})
        started = time.perf_counter()
        with urllib.request.urlopen(request, timeout=300) as response:
            first = None
            content = []
            timings_rate = None
            terminal_timings = None
            for chunk in response:
                line = chunk.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                if line.startswith("data:"):
                    line = line[5:].strip()
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = event.get("content", "")
                if text and first is None:
                    first = (time.perf_counter() - started) * 1000
                if text:
                    content.append(str(text))
                timings = event.get("timings")
                if isinstance(timings, dict):
                    terminal_timings = timings
                    if timings.get("predicted_per_second") is not None:
                        timings_rate = float(timings["predicted_per_second"])
        terminal = time.perf_counter()
        if first is None:
            raise RuntimeError("llama stream did not emit a content token")
        if (
            not isinstance(terminal_timings, dict)
            or terminal_timings.get("prompt_ms") is None
            or terminal_timings.get("predicted_ms") is None
        ):
            raise RuntimeError("llama terminal timings do not expose prompt_ms/predicted_ms")
        total_ms = (terminal - started) * 1000
        prompt_ms = float(terminal_timings["prompt_ms"])
        predicted_ms = float(terminal_timings["predicted_ms"])
        queue_wait_ms = max(0, total_ms - prompt_ms - predicted_ms)
        available_first = max(0, first - queue_wait_ms)
        if timings_rate is None:
            generated_tokens = len(json_request("/tokenize", {"content": "".join(content), "add_special": False}).get("tokens", []))
            generated_seconds = max(terminal - (started + first / 1000), 1e-9)
            timings_rate = generated_tokens / generated_seconds
        return queue_wait_ms, first, available_first, timings_rate
    output = {}
    for context in (512, 1024, 2048):
        values = []
        failures = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=15) as pool:
            futures = [pool.submit(sample, prompts[str(context)]) for _ in range(15)]
            for future in futures:
                try:
                    values.append(future.result())
                except Exception:
                    failures += 1
        queue = sorted(value[0] for value in values)
        first = sorted(value[1] for value in values)
        available = sorted(value[2] for value in values)
        throughput = sorted(value[3] for value in values)
        output[str(context)] = {
            "actualContextTokens": contexts[str(context)],
            "queueWaitP50Ms": statistics.median(queue) if queue else "not_available",
            "queueWaitP95Ms": queue[math.ceil(len(queue) * .95) - 1] if queue else "not_available",
            "firstTokenP50Ms": statistics.median(first) if first else "not_available",
            "firstTokenP95Ms": first[math.ceil(len(first) * .95) - 1] if first else "not_available",
            "availableFirstTokenP50Ms": statistics.median(available) if available else "not_available",
            "availableFirstTokenP95Ms": available[math.ceil(len(available) * .95) - 1] if available else "not_available",
            "outputTokensPerSecond": statistics.median(throughput) if throughput else "not_available",
            "failureRate": failures / 15,
        }
    print(json.dumps({"family": os.environ.get("MODEL_PROBE_ARCHITECTURE", "not_available"), "parameterSize": os.environ.get("MODEL_PROBE_PARAMETER_SIZE", "not_available"), "quantization": os.environ.get("MODEL_PROBE_QUANTIZATION", "not_available"), "modelMetadata": {"architecture": os.environ.get("MODEL_PROBE_ARCHITECTURE", "not_available"), "parameterSize": os.environ.get("MODEL_PROBE_PARAMETER_SIZE", "not_available"), "quantization": os.environ.get("MODEL_PROBE_QUANTIZATION", "not_available")}, "contexts": output, "maxOutputTokens": 256, "publicNetworkAttempted": "not_available"}, sort_keys=True))
    """
).strip()


def _finite_number(value: object) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, ValueError):
        return False


def available_first_token_ms(
    first_from_start_ms: float, queue_wait_ms: float
) -> float:
    """Remove measured queue wait from request-start first-token latency."""

    if not _finite_number(first_from_start_ms) or not _finite_number(queue_wait_ms):
        raise ValueError("first-token and queue wait must be finite numbers")
    return max(0.0, float(first_from_start_ms) - float(queue_wait_ms))


@dataclass(frozen=True, slots=True)
class ModelGate:
    max_query_embedding_p95_ms: float
    max_queue_feedback_p95_ms: float
    max_first_token_p95_ms: float

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if not _finite_number(value) or float(value) < 0:
                raise ValueError(f"{name} must be a finite non-negative number")


DEFAULT_MODEL_GATE = ModelGate(
    max_query_embedding_p95_ms=1500,
    max_queue_feedback_p95_ms=2000,
    max_first_token_p95_ms=10000,
)


def evaluate_model_probe(
    gate: ModelGate,
    metrics: Mapping[str, object],
) -> CapacityResult:
    """Evaluate the three latency gates in stable order and fail closed."""

    limits = (
        ("query_embedding_p95_ms", gate.max_query_embedding_p95_ms),
        ("queue_feedback_p95_ms", gate.max_queue_feedback_p95_ms),
        ("first_token_p95_ms", gate.max_first_token_p95_ms),
    )
    failures = [
        name
        for name, limit in limits
        if not _finite_number(metrics.get(name))
        or float(metrics[name]) > float(limit)
    ]
    return CapacityResult(passed=not failures, failures=failures)


def evaluate_candidate_gate(
    kind: str,
    gate: ModelGate,
    metrics: Mapping[str, object],
) -> CandidateGateEvaluation:
    """Evaluate only metrics that apply to this candidate kind."""

    applicable = {
        "embedding-model": (
            ("query_embedding_p95_ms", gate.max_query_embedding_p95_ms),
        ),
        "generation-model": (
            ("queue_feedback_p95_ms", gate.max_queue_feedback_p95_ms),
            ("first_token_p95_ms", gate.max_first_token_p95_ms),
        ),
        "reranker-model": (),
    }
    try:
        selected = applicable[kind]
    except KeyError as exc:
        raise ValueError("unsupported candidate kind") from exc
    report_metrics: dict[str, object] = {
        "query_embedding_p95_ms": "not_applicable",
        "queue_feedback_p95_ms": "not_applicable",
        "first_token_p95_ms": "not_applicable",
    }
    failures: list[str] = []
    for name, limit in selected:
        value = metrics.get(name)
        report_metrics[name] = value if _finite_number(value) else "not_available"
        if not _finite_number(value) or float(value) > float(limit):
            failures.append(name)
    return CandidateGateEvaluation(
        result=CapacityResult(passed=not failures, failures=failures),
        metrics=report_metrics,
    )


@dataclass(frozen=True, slots=True)
class CommandResult:
    exit_code: int
    stdout: str

    def __post_init__(self) -> None:
        if isinstance(self.exit_code, bool) or not isinstance(self.exit_code, int):
            raise ValueError("command exit code must be an integer")
        if not isinstance(self.stdout, str):
            raise ValueError("command stdout must be text")


@dataclass(frozen=True, slots=True)
class ArtifactMemberFingerprint:
    relative_path: str
    device: int
    inode: int
    mode: int
    size: int
    mtime_ns: int


@dataclass(frozen=True, slots=True)
class ArtifactFingerprint:
    device: int
    inode: int
    mode: int
    size: int
    mtime_ns: int
    members: tuple[ArtifactMemberFingerprint, ...] = ()


@dataclass(frozen=True, slots=True)
class CandidateArtifact:
    name: str
    kind: str
    version: str
    sha256: str
    license: str
    path: Path
    metadata: Mapping[str, object]
    fingerprint: ArtifactFingerprint


@dataclass(frozen=True, slots=True)
class CandidateGateEvaluation:
    result: CapacityResult
    metrics: Mapping[str, object]


def probe_project_name(compose_file: Path) -> str:
    """Derive a stable Compose project name unique to this probe directory."""

    resolved = os.path.normcase(str(Path(compose_file).resolve()))
    digest = hashlib.sha256(os.fsencode(resolved)).hexdigest()[:12]
    return f"dc-agent-probe-{digest}"


class ProbeMutex:
    """Non-blocking cross-process lock for the isolated Compose probe project."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._handle = None
        self._identity: tuple[int, int] | None = None

    def __enter__(self) -> "ProbeMutex":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            existing = self.path.lstat()
        except FileNotFoundError:
            existing = None
        if existing is not None:
            if stat.S_ISLNK(existing.st_mode):
                raise ValueError("model probe lock must not be a symbolic link")
            if not stat.S_ISREG(existing.st_mode):
                raise ValueError("model probe lock must be a regular file")

        flags = os.O_RDWR | os.O_CREAT
        flags |= getattr(os, "O_CLOEXEC", 0)
        if os.name != "nt":
            flags |= getattr(os, "O_NOFOLLOW", 0)
        try:
            descriptor = os.open(self.path, flags, 0o600)
        except OSError as exc:
            if getattr(exc, "errno", None) == errno.ELOOP:
                raise ValueError("model probe lock must not be a symbolic link") from exc
            raise
        try:
            opened = os.fstat(descriptor)
            if not stat.S_ISREG(opened.st_mode):
                raise ValueError("model probe lock must be a regular file")
            if hasattr(os, "fchmod"):
                os.fchmod(descriptor, 0o600)
            handle = os.fdopen(descriptor, "r+b")
        except BaseException:
            os.close(descriptor)
            raise
        try:
            handle.seek(0)
            if handle.read(1) != b"1":
                handle.seek(0)
                handle.write(b"1")
                handle.flush()
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            handle.close()
            raise RuntimeError("another model candidate probe is already running") from exc
        self._handle = handle
        self._identity = (opened.st_dev, opened.st_ino)
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        handle = self._handle
        self._handle = None
        identity = self._identity
        self._identity = None
        if handle is None:
            return
        try:
            handle.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()
            try:
                current = self.path.lstat()
            except FileNotFoundError:
                current = None
            if current is not None and (
                identity == (current.st_dev, current.st_ino)
                and stat.S_ISREG(current.st_mode)
            ):
                self.path.unlink(missing_ok=True)


Runner = Callable[..., CommandResult]
ArtifactHasher = Callable[[Path], str]


def _artifact_fingerprint(path: Path) -> ArtifactFingerprint:
    artifact = Path(path)
    try:
        details = artifact.lstat()
    except OSError as exc:
        raise ValueError("candidate artifact is no longer available") from exc
    if stat.S_ISLNK(details.st_mode):
        raise ValueError("candidate artifact changed to a symbolic link")
    if not (stat.S_ISREG(details.st_mode) or stat.S_ISDIR(details.st_mode)):
        raise ValueError("candidate artifact must remain a regular file or directory")
    members: list[ArtifactMemberFingerprint] = []
    if stat.S_ISDIR(details.st_mode):
        entries_seen = 0

        def reject_walk_error(error: OSError) -> None:
            raise ValueError("candidate artifact tree could not be fully inspected") from error

        for root, directory_names, file_names in os.walk(
            artifact, followlinks=False, onerror=reject_walk_error
        ):
            directory = Path(root)
            directory_names.sort()
            file_names.sort()
            entries_seen += len(directory_names) + len(file_names)
            if entries_seen > MAX_ARTIFACT_TREE_ENTRIES:
                raise ValueError("candidate artifact tree entry budget exceeded")
            for name in directory_names:
                child = directory / name
                try:
                    child_details = child.lstat()
                except OSError as exc:
                    raise ValueError("candidate artifact tree changed during validation") from exc
                if stat.S_ISLNK(child_details.st_mode):
                    raise ValueError(
                        "candidate artifact tree must not contain symbolic links"
                    )
                if not stat.S_ISDIR(child_details.st_mode):
                    raise ValueError(
                        "candidate artifact tree must contain only directories and regular files"
                    )
            for name in file_names:
                child = directory / name
                try:
                    child_details = child.lstat()
                except OSError as exc:
                    raise ValueError("candidate artifact tree changed during validation") from exc
                if stat.S_ISLNK(child_details.st_mode) or not stat.S_ISREG(
                    child_details.st_mode
                ):
                    raise ValueError(
                        "candidate artifact tree must contain only regular files"
                    )
                members.append(
                    ArtifactMemberFingerprint(
                        relative_path=child.relative_to(artifact).as_posix(),
                        device=child_details.st_dev,
                        inode=child_details.st_ino,
                        mode=child_details.st_mode,
                        size=child_details.st_size,
                        mtime_ns=child_details.st_mtime_ns,
                    )
                )
        members.sort(key=lambda item: item.relative_path.encode("utf-8"))
    return ArtifactFingerprint(
        device=details.st_dev,
        inode=details.st_ino,
        mode=details.st_mode,
        size=details.st_size,
        mtime_ns=details.st_mtime_ns,
        members=tuple(members),
    )


def _verify_artifact_stable(
    path: Path,
    expected_checksum: str,
    expected_fingerprint: ArtifactFingerprint,
    artifact_hasher: ArtifactHasher,
) -> None:
    before = _artifact_fingerprint(path)
    if before != expected_fingerprint:
        raise ValueError("candidate artifact changed after validation")
    actual_checksum = artifact_hasher(path)
    after = _artifact_fingerprint(path)
    if after != before or after != expected_fingerprint:
        raise ValueError("candidate artifact changed while hashing")
    if not isinstance(actual_checksum, str) or SHA256_PATTERN.fullmatch(actual_checksum) is None:
        raise ValueError("artifact hasher returned an invalid sha256")
    if not hmac.compare_digest(actual_checksum, expected_checksum):
        raise ValueError("candidate artifact checksum mismatch")


def _verify_candidate_stable(
    candidate: CandidateArtifact, artifact_hasher: ArtifactHasher
) -> None:
    _verify_artifact_stable(
        candidate.path,
        candidate.sha256,
        candidate.fingerprint,
        artifact_hasher,
    )


def sha256_artifact(path: Path) -> str:
    """Hash an artifact file or a symlink-free model tree deterministically."""

    artifact = Path(path)
    if artifact.is_symlink():
        raise ValueError("candidate artifact must not be a symbolic link")
    if artifact.is_file():
        digest = hashlib.sha256()
        with artifact.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    if not artifact.is_dir():
        raise ValueError("candidate artifact must be an existing file or directory")

    files: list[Path] = []
    for root, directory_names, file_names in os.walk(artifact, followlinks=False):
        directory = Path(root)
        directory_names.sort()
        file_names.sort()
        for name in directory_names:
            child = directory / name
            if child.is_symlink():
                raise ValueError("candidate artifact tree must not contain symbolic links")
        for name in file_names:
            child = directory / name
            if child.is_symlink() or not stat.S_ISREG(child.stat(follow_symlinks=False).st_mode):
                raise ValueError("candidate artifact tree must contain only regular files")
            files.append(child)

    files.sort(key=lambda item: item.relative_to(artifact).as_posix().encode("utf-8"))
    digest = hashlib.sha256()
    digest.update(b"dc-agent-embedding-model-tree-v1\0")
    for file_path in files:
        relative = file_path.relative_to(artifact).as_posix().encode("utf-8")
        before = file_path.stat()
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        digest.update(before.st_size.to_bytes(8, "big"))
        bytes_read = 0
        with file_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
                bytes_read += len(chunk)
        after = file_path.stat()
        if (
            bytes_read != before.st_size
            or after.st_size != before.st_size
            or after.st_mtime_ns != before.st_mtime_ns
        ):
            raise ValueError(f"candidate artifact changed while hashing: {file_path}")
    return digest.hexdigest()


GGUF_FILE_TYPES: Mapping[int, str] = {
    2: "Q4_0",
    3: "Q4_1",
    14: "Q4_K_S",
    15: "Q4_K_M",
}


GGUF_MAX_METADATA_DEPTH = 64
GGUF_MAX_METADATA_VALUES = 1_000_000
GGUF_MAX_METADATA_BYTES = 64 * 1024 * 1024


@dataclass(slots=True)
class _GgufBudget:
    remaining_values: int = GGUF_MAX_METADATA_VALUES
    remaining_bytes: int = GGUF_MAX_METADATA_BYTES

    def consume_values(self, count: int = 1) -> None:
        if count < 0 or count > self.remaining_values:
            raise ValueError("GGUF metadata value budget exceeded")
        self.remaining_values -= count

    def consume_bytes(self, count: int) -> None:
        if count < 0 or count > self.remaining_bytes:
            raise ValueError("GGUF metadata byte budget exceeded")
        self.remaining_bytes -= count


def _read_exact(handle, size: int, budget: _GgufBudget) -> bytes:
    budget.consume_bytes(size)
    data = handle.read(size)
    if len(data) != size:
        raise ValueError("truncated GGUF metadata")
    return data


def _read_gguf_string(handle, budget: _GgufBudget) -> str:
    length = struct.unpack("<Q", _read_exact(handle, 8, budget))[0]
    if length > 16 * 1024 * 1024:
        raise ValueError("GGUF metadata string is too large")
    try:
        return _read_exact(handle, length, budget).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError("GGUF metadata must be valid UTF-8") from exc


def _read_gguf_value(
    handle,
    value_type: int,
    *,
    budget: _GgufBudget,
    depth: int = 0,
    materialize: bool = True,
) -> object:
    if depth > GGUF_MAX_METADATA_DEPTH:
        raise ValueError("GGUF metadata nesting depth exceeded")
    budget.consume_values()
    formats = {
        0: "<B",
        1: "<b",
        2: "<H",
        3: "<h",
        4: "<I",
        5: "<i",
        6: "<f",
        7: "<?",
        10: "<Q",
        11: "<q",
        12: "<d",
    }
    if value_type == 8:
        value = _read_gguf_string(handle, budget)
        return value if materialize else None
    if value_type == 9:
        item_type = struct.unpack("<I", _read_exact(handle, 4, budget))[0]
        length = struct.unpack("<Q", _read_exact(handle, 8, budget))[0]
        if length > 1_000_000:
            raise ValueError("GGUF metadata array is too large")
        if length > budget.remaining_values:
            raise ValueError("GGUF metadata value budget exceeded")
        if materialize:
            return [
                _read_gguf_value(
                    handle,
                    item_type,
                    budget=budget,
                    depth=depth + 1,
                )
                for _ in range(length)
            ]
        for _ in range(length):
            _read_gguf_value(
                handle,
                item_type,
                budget=budget,
                depth=depth + 1,
                materialize=False,
            )
        return None
    try:
        format_string = formats[value_type]
    except KeyError as exc:
        raise ValueError(f"unsupported GGUF metadata value type: {value_type}") from exc
    size = struct.calcsize(format_string)
    value = struct.unpack(format_string, _read_exact(handle, size, budget))[0]
    return value if materialize else None


def read_gguf_metadata(path: Path) -> dict[str, object]:
    """Read bounded GGUF key/value metadata without loading tensor data."""

    with Path(path).open("rb") as handle:
        budget = _GgufBudget()
        if _read_exact(handle, 4, budget) != b"GGUF":
            raise ValueError("generation candidate must be a GGUF file")
        version, _tensor_count, metadata_count = struct.unpack(
            "<IQQ", _read_exact(handle, 20, budget)
        )
        if version not in (2, 3):
            raise ValueError("unsupported GGUF version")
        if metadata_count > 100_000:
            raise ValueError("GGUF metadata entry count is too large")
        relevant_keys = {
            "general.architecture",
            "general.name",
            "general.version",
            "general.size_label",
            "general.file_type",
        }
        seen_keys: set[str] = set()
        metadata: dict[str, object] = {}
        for _ in range(metadata_count):
            key = _read_gguf_string(handle, budget)
            if key in seen_keys:
                raise ValueError("GGUF metadata keys must be unique")
            seen_keys.add(key)
            value_type = struct.unpack("<I", _read_exact(handle, 4, budget))[0]
            value = _read_gguf_value(
                handle,
                value_type,
                budget=budget,
                materialize=key in relevant_keys,
            )
            if key in relevant_keys:
                metadata[key] = value
    return metadata


def _load_embedding_metadata(path: Path) -> dict[str, object]:
    metadata_path = path / "embedding-metadata.json"
    try:
        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("embedding candidate requires valid embedding-metadata.json") from exc
    if not isinstance(payload, dict):
        raise ValueError("embedding-metadata.json must be an object")
    for field in ("modelName", "modelVersion", "dimensions"):
        if field not in payload:
            raise ValueError(f"embedding-metadata.json is missing {field}")
    if (
        not isinstance(payload["dimensions"], int)
        or isinstance(payload["dimensions"], bool)
        or payload["dimensions"] <= 0
    ):
        raise ValueError("embedding metadata dimensions must be positive")
    return payload


def load_candidate_artifact(
    entry: Mapping[str, object],
    *,
    artifact_root: Path,
    artifact_hasher: ArtifactHasher = sha256_artifact,
) -> CandidateArtifact:
    """Validate one exact artifact-lock candidate and its local checksum."""

    if not isinstance(entry, Mapping):
        raise ValueError("candidate artifact-lock entry must be an object")
    if set(entry) != set(ARTIFACT_FIELDS):
        raise ValueError("candidate artifact-lock entry must contain exactly the lock fields")
    values: dict[str, str] = {}
    for field in ARTIFACT_FIELDS:
        value = entry[field]
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"candidate artifact-lock entry is missing {field}")
        values[field] = value.strip()
    if values["kind"] not in SUPPORTED_KINDS:
        raise ValueError("candidate artifact kind is not probeable")
    if SHA256_PATTERN.fullmatch(values["sha256"]) is None:
        raise ValueError("candidate sha256 must be exactly 64 lowercase hexadecimal characters")

    raw_path = values["localPath"]
    if (
        raw_path.startswith(("\\\\", "//"))
        or (URI_SCHEME.match(raw_path) and not WINDOWS_ABSOLUTE.match(raw_path))
    ):
        raise ValueError("candidate localPath must be a local filesystem path")
    root_input = Path(artifact_root)
    if _path_contains_symlink(root_input):
        raise ValueError("artifact_root must not contain symbolic links")
    root = root_input.resolve(strict=True)
    unresolved = Path(raw_path)
    is_absolute = unresolved.is_absolute() or WINDOWS_ABSOLUTE.match(raw_path) is not None
    path = unresolved if is_absolute else root / unresolved
    if _path_contains_symlink(path):
        raise ValueError("candidate localPath must not contain symbolic links")
    try:
        path = path.resolve(strict=True)
    except OSError as exc:
        raise ValueError("candidate localPath must point to an existing artifact") from exc
    if not is_absolute:
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError("relative candidate localPath must remain inside artifact_root") from exc
    if values["kind"] in ("embedding-model", "reranker-model") and not path.is_dir():
        raise ValueError(f"{values['kind']} candidate must be a directory")
    if values["kind"] == "generation-model" and (
        not path.is_file() or path.suffix.casefold() != ".gguf"
    ):
        raise ValueError("generation-model candidate must be a regular GGUF file")
    fingerprint = _artifact_fingerprint(path)
    _verify_artifact_stable(
        path,
        values["sha256"],
        fingerprint,
        artifact_hasher,
    )
    candidate_metadata: dict[str, object] = {}
    if values["kind"] == "embedding-model":
        candidate_metadata = _load_embedding_metadata(path)
        if (
            candidate_metadata.get("modelName") != values["name"]
            or candidate_metadata.get("modelVersion") != values["version"]
        ):
            raise ValueError("embedding lock identity does not match embedding metadata")
    elif values["kind"] == "generation-model":
        candidate_metadata = read_gguf_metadata(path)
        architecture = candidate_metadata.get("general.architecture")
        size_label = candidate_metadata.get("general.size_label")
        file_type = candidate_metadata.get("general.file_type")
        quantization = GGUF_FILE_TYPES.get(file_type) if isinstance(file_type, int) else None
        if not isinstance(architecture, str) or not architecture.casefold().startswith("qwen"):
            raise ValueError("generation GGUF architecture must be Qwen-family")
        if size_label not in ("1.5B", "3B"):
            raise ValueError("generation GGUF size label must be 1.5B or 3B")
        if quantization is None:
            raise ValueError("generation GGUF quantization must be Q4")
        if (
            candidate_metadata.get("general.name") != values["name"]
            or candidate_metadata.get("general.version") != values["version"]
        ):
            raise ValueError("generation lock identity does not match GGUF metadata")
        candidate_metadata["probe.quantization"] = quantization
    _verify_artifact_stable(
        path,
        values["sha256"],
        fingerprint,
        artifact_hasher,
    )
    return CandidateArtifact(
        name=values["name"],
        kind=values["kind"],
        version=values["version"],
        sha256=values["sha256"],
        license=values["license"],
        path=path,
        metadata=candidate_metadata,
        fingerprint=fingerprint,
    )


def _path_contains_symlink(path: Path) -> bool:
    current = Path(path)
    while True:
        try:
            if current.exists() and current.is_symlink():
                return True
        except OSError:
            return True
        parent = current.parent
        if parent == current:
            return False
        current = parent


def _default_runner(
    command: Sequence[str],
    *,
    shell: bool,
    environment: Mapping[str, str] | None = None,
) -> CommandResult:
    if shell is not False:
        raise ValueError("model probe commands must not use a shell")
    arguments = _validated_argv(command)
    try:
        completed = subprocess.run(
            arguments,
            shell=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=3600,
            check=False,
            env=None if environment is None else dict(environment),
        )
    except FileNotFoundError:
        return CommandResult(127, "")
    except subprocess.TimeoutExpired as error:
        output = error.stdout if isinstance(error.stdout, str) else ""
        return CommandResult(124, output)
    return CommandResult(int(completed.returncode), completed.stdout or "")


def _validated_argv(command: Sequence[str]) -> list[str]:
    if isinstance(command, (str, bytes)) or not command:
        raise ValueError("command must be a non-empty argument vector")
    arguments = list(command)
    if any(not isinstance(value, str) or not value or "\x00" in value for value in arguments):
        raise ValueError("command arguments must be non-empty strings")
    return arguments


def _validate_identifier(value: str, label: str) -> str:
    if not isinstance(value, str) or SAFE_IDENTIFIER.fullmatch(value) is None:
        raise ValueError(f"{label} must be a safe Compose identifier")
    return value


def _json_object(stdout: str, label: str) -> dict[str, object]:
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} output must be JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} output must be a JSON object")
    return payload


def _public_network_attempted(payload: Mapping[str, object]) -> bool:
    if payload.get("publicNetworkAttempted") is True:
        return True
    attempts = payload.get("publicNetworkAttempts")
    if _finite_number(attempts) and float(attempts) > 0:
        return True
    return False


def _required_number(
    payload: Mapping[str, object],
    key: str,
    failure_name: str,
    failures: list[str],
    *,
    minimum: float = 0,
    maximum: float | None = None,
) -> float | None:
    value = payload.get(key)
    if not _finite_number(value) or float(value) < minimum:
        failures.append(failure_name)
        return None
    number = float(value)
    if maximum is not None and number > maximum:
        failures.append(failure_name)
        return None
    return number


def _embedding_metrics(payload: Mapping[str, object]) -> tuple[dict[str, object], list[str]]:
    failures: list[str] = []
    model_metadata = payload.get("modelMetadata")
    model_name = (
        str(model_metadata.get("modelName", ""))
        if isinstance(model_metadata, Mapping)
        else ""
    )
    if payload.get("family") != "BGE" or "bge" not in model_name.casefold():
        failures.append("embedding_family")
    variant = payload.get("variant")
    if variant not in ("small", "base") or str(variant) not in model_name.casefold():
        failures.append("embedding_variant")
    token_range = payload.get("actualTokenRange")
    if (
        not isinstance(token_range, list)
        or len(token_range) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) for value in token_range)
        or token_range[0] < 300
        or token_range[1] > 800
        or token_range[0] > token_range[1]
    ):
        failures.append("embedding_token_range")
    if (
        not isinstance(model_metadata, Mapping)
        or not isinstance(model_metadata.get("modelName"), str)
        or not isinstance(model_metadata.get("modelVersion"), str)
        or isinstance(model_metadata.get("dimensions"), bool)
        or not isinstance(model_metadata.get("dimensions"), int)
        or int(model_metadata.get("dimensions", 0)) <= 0
    ):
        failures.append("embedding_model_metadata")
    _required_number(
        payload,
        "batchDocumentsPerSecond",
        "batch_documents_per_second",
        failures,
        minimum=0.0000001,
    )
    _required_number(
        payload,
        "residentMemoryBytes",
        "resident_memory_bytes",
        failures,
        minimum=1,
    )
    concurrency_payload = payload.get("queryEmbedding")
    p95_values: list[float] = []
    if not isinstance(concurrency_payload, Mapping):
        failures.append("query_embedding")
    else:
        if set(concurrency_payload) != {str(value) for value in EMBEDDING_CONCURRENCIES}:
            failures.append("query_embedding_concurrencies")
        for concurrency in EMBEDDING_CONCURRENCIES:
            timings = concurrency_payload.get(str(concurrency))
            if not isinstance(timings, Mapping):
                failures.append(f"query_embedding_{concurrency}")
                continue
            p50 = _required_number(
                timings,
                "p50Ms",
                f"query_embedding_{concurrency}_p50_ms",
                failures,
            )
            p95 = _required_number(
                timings,
                "p95Ms",
                f"query_embedding_{concurrency}_p95_ms",
                failures,
            )
            if p50 is not None and p95 is not None and p95 < p50:
                failures.append(f"query_embedding_{concurrency}_percentiles")
            if p95 is not None:
                p95_values.append(p95)
    gate_metrics: dict[str, object] = {}
    if len(p95_values) == len(EMBEDDING_CONCURRENCIES):
        gate_metrics["query_embedding_p95_ms"] = max(p95_values)
    return gate_metrics, failures


def _generation_metrics(payload: Mapping[str, object]) -> tuple[dict[str, object], list[str]]:
    failures: list[str] = []
    family = payload.get("family")
    if not isinstance(family, str) or not family.casefold().startswith("qwen"):
        failures.append("generation_family")
    if payload.get("parameterSize") not in ("1.5B", "3B"):
        failures.append("generation_parameter_size")
    quantization = payload.get("quantization")
    if not isinstance(quantization, str) or not quantization.upper().startswith("Q4"):
        failures.append("generation_quantization")
    output_tokens = payload.get("maxOutputTokens")
    if (
        isinstance(output_tokens, bool)
        or not isinstance(output_tokens, int)
        or output_tokens <= 0
        or output_tokens > MAX_OUTPUT_TOKENS
    ):
        failures.append("max_output_tokens")

    contexts = payload.get("contexts")
    model_metadata = payload.get("modelMetadata")
    if (
        not isinstance(model_metadata, Mapping)
        or model_metadata.get("architecture") != family
        or model_metadata.get("parameterSize") != payload.get("parameterSize")
        or model_metadata.get("quantization") != quantization
    ):
        failures.append("generation_model_metadata")
    queue_p95_values: list[float] = []
    first_token_p95_values: list[float] = []
    if not isinstance(contexts, Mapping):
        failures.append("contexts")
    else:
        if set(contexts) != {str(value) for value in CONTEXT_LENGTHS}:
            failures.append("context_lengths")
        fields = (
            ("queueWaitP50Ms", "queue_wait_p50_ms", 0, None),
            ("queueWaitP95Ms", "queue_wait_p95_ms", 0, None),
            ("firstTokenP50Ms", "first_token_p50_ms", 0, None),
            ("firstTokenP95Ms", "first_token_p95_ms", 0, None),
            (
                "availableFirstTokenP50Ms",
                "available_first_token_p50_ms",
                0,
                None,
            ),
            (
                "availableFirstTokenP95Ms",
                "available_first_token_p95_ms",
                0,
                None,
            ),
            ("outputTokensPerSecond", "output_tokens_per_second", 0.0000001, None),
            ("failureRate", "failure_rate", 0, 1),
        )
        for context in CONTEXT_LENGTHS:
            metrics = contexts.get(str(context))
            if not isinstance(metrics, Mapping):
                failures.append(f"context_{context}")
                continue
            if metrics.get("actualContextTokens") != context:
                failures.append(f"context_{context}_actual_tokens")
            numbers: dict[str, float] = {}
            for source, failure, minimum, maximum in fields:
                number = _required_number(
                    metrics,
                    source,
                    f"context_{context}_{failure}",
                    failures,
                    minimum=minimum,
                    maximum=maximum,
                )
                if number is not None:
                    numbers[source] = number
            if (
                "queueWaitP50Ms" in numbers
                and "queueWaitP95Ms" in numbers
                and numbers["queueWaitP95Ms"] < numbers["queueWaitP50Ms"]
            ):
                failures.append(f"context_{context}_queue_percentiles")
            if (
                "firstTokenP50Ms" in numbers
                and "firstTokenP95Ms" in numbers
                and numbers["firstTokenP95Ms"] < numbers["firstTokenP50Ms"]
            ):
                failures.append(f"context_{context}_first_token_percentiles")
            if (
                "availableFirstTokenP50Ms" in numbers
                and "availableFirstTokenP95Ms" in numbers
                and numbers["availableFirstTokenP95Ms"]
                < numbers["availableFirstTokenP50Ms"]
            ):
                failures.append(
                    f"context_{context}_available_first_token_percentiles"
                )
            if (
                "availableFirstTokenP95Ms" in numbers
                and "firstTokenP95Ms" in numbers
                and numbers["availableFirstTokenP95Ms"]
                > numbers["firstTokenP95Ms"]
            ):
                failures.append(f"context_{context}_available_first_token")
            if "queueWaitP95Ms" in numbers:
                queue_p95_values.append(numbers["queueWaitP95Ms"])
            if "availableFirstTokenP95Ms" in numbers:
                first_token_p95_values.append(numbers["availableFirstTokenP95Ms"])
    gate_metrics: dict[str, object] = {}
    if len(queue_p95_values) == len(CONTEXT_LENGTHS):
        gate_metrics["queue_feedback_p95_ms"] = max(queue_p95_values)
    if len(first_token_p95_values) == len(CONTEXT_LENGTHS):
        gate_metrics["first_token_p95_ms"] = max(first_token_p95_values)
    return gate_metrics, failures


def _reranker_metrics(
    payload: Mapping[str, object], expected_checksum: str
) -> tuple[dict[str, object], list[str], dict[str, object]]:
    failures: list[str] = []
    if payload.get("family") != "BGE":
        failures.append("reranker_family")
    p50 = _required_number(payload, "top20To10P50Ms", "top20_to10_p50_ms", failures)
    p95 = _required_number(payload, "top20To10P95Ms", "top20_to10_p95_ms", failures)
    _required_number(
        payload,
        "residentMemoryBytes",
        "resident_memory_bytes",
        failures,
        minimum=1,
    )
    _required_number(
        payload,
        "busyTimeoutRate",
        "busy_timeout_rate",
        failures,
        maximum=1,
    )
    checksum = payload.get("modelChecksum")
    if (
        not isinstance(checksum, str)
        or SHA256_PATTERN.fullmatch(checksum) is None
        or not hmac.compare_digest(checksum, expected_checksum)
    ):
        failures.append("reranker_checksum")
    if p50 is not None and p95 is not None and p95 < p50:
        failures.append("reranker_percentiles")
    disabled = p95 is not None and p95 > RERANKER_P95_GATE_MS
    reranker = {
        "status": "disabled" if disabled else "enabled",
        "disabledReason": (
            "top20_to10_p95_ms_exceeds_1500" if disabled else None
        ),
    }
    return {}, failures, reranker


def _sanitize(value: object) -> object:
    if isinstance(value, float) and not math.isfinite(value):
        return "not_available"
    if isinstance(value, Mapping):
        return {str(key): _sanitize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(item) for item in value]
    return value


def _write_report(path: Path, payload: Mapping[str, object]) -> None:
    encoded = (
        json.dumps(
            _sanitize(payload),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="\n",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, destination)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _candidate_environment(
    candidate: CandidateArtifact, discovery_label: str
) -> dict[str, str]:
    return {
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "NO_PROXY": "*",
        "no_proxy": "*",
        "HTTP_PROXY": "",
        "HTTPS_PROXY": "",
        "ALL_PROXY": "",
        "MODEL_PROBE_CANDIDATE_NAME": candidate.name,
        "MODEL_PROBE_CANDIDATE_VERSION": candidate.version,
        "MODEL_PROBE_CANDIDATE_SHA256": candidate.sha256,
        "MODEL_PROBE_DISCOVERY_LABEL": discovery_label,
        "MODEL_PROBE_ARCHITECTURE": str(
            candidate.metadata.get("general.architecture", "not_available")
        ),
        "MODEL_PROBE_PARAMETER_SIZE": str(
            candidate.metadata.get("general.size_label", "not_available")
        ),
        "MODEL_PROBE_QUANTIZATION": str(
            candidate.metadata.get("probe.quantization", "not_available")
        ),
    }


def _bind(source: Path, target: str) -> dict[str, object]:
    return {
        "type": "bind",
        "source": str(source),
        "target": target,
        "read_only": True,
        "bind": {"create_host_path": False},
    }


def _build_candidate_override(
    candidate: CandidateArtifact,
    *,
    embedding_service: str,
    llama_service: str,
    discovery_label: str,
) -> dict[str, object]:
    environment = _candidate_environment(candidate, discovery_label)
    if candidate.kind == "embedding-model":
        return {
            "services": {
                embedding_service: {
                    "command": ["python", "-c", EMBEDDING_SERVER_SCRIPT],
                    "environment": {
                        **environment,
                        "EMBEDDING_MODEL_ROOT": "/probe-candidate",
                        "EMBEDDING_MODEL_SHA256": candidate.sha256,
                    },
                    "volumes": [_bind(candidate.path, "/probe-candidate")],
                }
            }
        }
    if candidate.kind == "reranker-model":
        return {
            "services": {
                embedding_service: {
                    "command": ["python", "-c", HELPER_IDLE_SCRIPT],
                    "environment": environment,
                    "healthcheck": {"disable": True},
                    "volumes": [_bind(candidate.path, "/probe-candidate")],
                }
            }
        }
    return {
        "services": {
            embedding_service: {
                "command": ["python", "-c", HELPER_IDLE_SCRIPT],
                "environment": environment,
                "healthcheck": {"disable": True},
                "volumes": [],
            },
            llama_service: {
                "command": [
                    "--model",
                    "/probe-candidate/model.gguf",
                    "--host",
                    "0.0.0.0",
                    "--port",
                    "8080",
                    "--parallel",
                    "1",
                ],
                "volumes": [_bind(candidate.path, "/probe-candidate/model.gguf")],
            },
        }
    }


def _write_candidate_override(directory: Path, payload: Mapping[str, object]) -> Path:
    handle = tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        newline="\n",
        dir=directory,
        prefix=".model-probe.",
        suffix=".override.json",
        delete=False,
    )
    path = Path(handle.name)
    try:
        with handle:
            json.dump(payload, handle, ensure_ascii=False, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(path, 0o600)
        return path
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _read_env_names(path: Path) -> set[str]:
    names: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*=", stripped)
        if match is None:
            raise ValueError("offline .env contains an invalid entry")
        if match.group(1) in names:
            raise ValueError("offline .env contains a duplicate key")
        names.add(match.group(1))
    return names


def _clean_compose_environment(env_path: Path) -> dict[str, str]:
    blocked = _read_env_names(env_path) | {
        "COMPOSE_ENV_FILES",
        "COMPOSE_FILE",
        "COMPOSE_PROFILES",
        "COMPOSE_PROJECT_NAME",
    }
    return {name: value for name, value in os.environ.items() if name not in blocked}


def _validate_local_docker_environment() -> None:
    docker_host = os.environ.get("DOCKER_HOST", "").strip()
    if docker_host and docker_host != "unix:///var/run/docker.sock":
        raise ValueError("model probes require the local Docker host")
    docker_context = os.environ.get("DOCKER_CONTEXT", "").strip()
    if docker_context and docker_context != "default":
        raise ValueError("model probes require the local Docker default context")


def _service_network_names(service: Mapping[str, object]) -> set[str]:
    networks = service.get("networks")
    if isinstance(networks, Mapping):
        return {str(name) for name in networks}
    if isinstance(networks, list):
        return {str(name) for name in networks}
    return set()


def _rendered_environment(service: Mapping[str, object]) -> dict[str, str]:
    environment = service.get("environment")
    if isinstance(environment, Mapping):
        return {str(name): str(value) for name, value in environment.items()}
    if isinstance(environment, list):
        result: dict[str, str] = {}
        for item in environment:
            if isinstance(item, str) and "=" in item:
                name, value = item.split("=", 1)
                result[name] = value
        return result
    return {}


def _rendered_volumes(service: Mapping[str, object]) -> list[Mapping[str, object]]:
    volumes = service.get("volumes")
    if not isinstance(volumes, list):
        return []
    return [item for item in volumes if isinstance(item, Mapping)]


def _has_candidate_bind(
    service: Mapping[str, object], candidate: CandidateArtifact, target: str
) -> bool:
    expected_source = os.path.normcase(os.path.normpath(str(candidate.path)))
    for volume in _rendered_volumes(service):
        source = volume.get("source")
        if not isinstance(source, str):
            continue
        if (
            os.path.normcase(os.path.normpath(source)) == expected_source
            and volume.get("target") == target
        ):
            return True
    return False


def _python_helper_command(service: Mapping[str, object]) -> bool:
    command = service.get("command")
    return (
        isinstance(command, list)
        and len(command) >= 3
        and command[0] == "python"
        and command[1] == "-c"
    )


def _validate_candidate_injection(
    rendered: Mapping[str, object],
    candidate: CandidateArtifact,
    *,
    embedding_service: str,
    llama_service: str,
) -> bool:
    services = rendered.get("services")
    if not isinstance(services, Mapping):
        return False
    helper = services.get(embedding_service)
    if not isinstance(helper, Mapping):
        return False
    environment = _rendered_environment(helper)
    expected_environment = {
        "MODEL_PROBE_CANDIDATE_NAME": candidate.name,
        "MODEL_PROBE_CANDIDATE_VERSION": candidate.version,
        "MODEL_PROBE_CANDIDATE_SHA256": candidate.sha256,
    }
    if any(environment.get(name) != value for name, value in expected_environment.items()):
        return False
    if not _python_helper_command(helper):
        return False
    if candidate.kind == "generation-model":
        llama = services.get(llama_service)
        if not isinstance(llama, Mapping):
            return False
        command = llama.get("command")
        return (
            isinstance(command, list)
            and "/probe-candidate/model.gguf" in command
            and _has_candidate_bind(
                llama, candidate, "/probe-candidate/model.gguf"
            )
        )
    return _has_candidate_bind(helper, candidate, "/probe-candidate")


def _validate_private_compose(
    rendered: Mapping[str, object],
    required_services: Sequence[str],
    expected_project_name: str,
) -> bool:
    if rendered.get("name") != expected_project_name:
        return False
    networks = rendered.get("networks")
    services = rendered.get("services")
    if not isinstance(networks, Mapping) or not isinstance(services, Mapping):
        return False
    internal_networks = {
        str(name)
        for name, definition in networks.items()
        if isinstance(definition, Mapping) and definition.get("internal") is True
    }
    if not internal_networks:
        return False
    for service_name in required_services:
        service = services.get(service_name)
        if not isinstance(service, Mapping):
            return False
        attached = _service_network_names(service)
        if not attached or not attached.issubset(internal_networks):
            return False
        if service.get("ports") or service.get("network_mode") == "host":
            return False
    return True


def _probe_scripts(kind: str) -> tuple[str, str]:
    if kind == "embedding-model":
        return EMBEDDING_METADATA_SCRIPT, EMBEDDING_BENCHMARK_SCRIPT
    if kind == "reranker-model":
        return RERANKER_METADATA_SCRIPT, RERANKER_BENCHMARK_SCRIPT
    return GENERATION_METADATA_SCRIPT, GENERATION_BENCHMARK_SCRIPT


def run_model_probe(
    *,
    compose_file: Path,
    embedding_service: str,
    llama_service: str,
    discovery_label: str,
    candidate_entry: Mapping[str, object],
    artifact_root: Path,
    report_path: Path,
    runner: Runner = _default_runner,
    artifact_hasher: ArtifactHasher = sha256_artifact,
    gate: ModelGate = DEFAULT_MODEL_GATE,
) -> CapacityResult:
    """Probe exactly one configured candidate and write a deterministic report."""

    compose = Path(compose_file).resolve(strict=True)
    if not compose.is_file():
        raise ValueError("compose_file must be an existing file")
    env_path = compose.parent / ".env"
    if not env_path.is_file():
        raise ValueError("the Compose file directory must contain the offline .env")
    embedding = _validate_identifier(embedding_service, "embedding_service")
    llama = _validate_identifier(llama_service, "llama_service")
    label = _validate_identifier(discovery_label, "discovery_label")
    candidate = load_candidate_artifact(
        candidate_entry,
        artifact_root=artifact_root,
        artifact_hasher=artifact_hasher,
    )
    project_name = probe_project_name(compose)
    required_services = (
        (llama, embedding) if candidate.kind == "generation-model" else (embedding,)
    )
    helper_service = embedding
    profile_arguments = (
        ["--profile", "generation"] if candidate.kind == "generation-model" else []
    )
    metadata_script, benchmark_script = _probe_scripts(candidate.kind)
    _validate_local_docker_environment()
    clean_environment = _clean_compose_environment(env_path)
    command_exit_codes: dict[str, int | None] = {
        "config": None,
        "up": None,
        "metadata": None,
        "benchmark": None,
        "down": None,
    }
    metadata: dict[str, object] = {}
    raw_metrics: dict[str, object] = {}
    operational_failures: list[str] = []
    override_path: Path | None = None
    config_is_private = False

    with ProbeMutex(compose.parent / LOCK_FILE_NAME):
        try:
            override = _build_candidate_override(
                candidate,
                embedding_service=embedding,
                llama_service=llama,
                discovery_label=label,
            )
            override_path = _write_candidate_override(compose.parent, override)
            # This direct Compose project is intentionally isolated from production.
            # It uses the production .env and base file, a controlled candidate-only
            # override, a fixed local context and path-specific project, config validation before up,
            # and lifecycle flags that prohibit image pulls and builds.
            compose_prefix = [
                "docker",
                "--context",
                "default",
                "compose",
                "--env-file",
                str(env_path),
                "-f",
                str(compose),
                "-f",
                str(override_path),
                "--project-name",
                project_name,
                *profile_arguments,
            ]
            commands = {
                "config": compose_prefix + ["config", "--format", "json"],
                "up": compose_prefix
                + [
                    "up",
                    "-d",
                    "--wait",
                    "--no-build",
                    "--pull",
                    "never",
                    *required_services,
                ],
                "metadata": compose_prefix
                + [
                    "exec",
                    "-T",
                    helper_service,
                    "python",
                    "-c",
                    metadata_script,
                    label,
                ],
                "benchmark": compose_prefix
                + [
                    "exec",
                    "-T",
                    helper_service,
                    "python",
                    "-c",
                    benchmark_script,
                    label,
                    ",".join(str(value) for value in EMBEDDING_CONCURRENCIES),
                    ",".join(str(value) for value in CONTEXT_LENGTHS),
                    str(MAX_OUTPUT_TOKENS),
                ],
                "down": compose_prefix + ["down", "--remove-orphans", "--volumes"],
            }
            config_result = runner(
                _validated_argv(commands["config"]),
                shell=False,
                environment=clean_environment,
            )
            command_exit_codes["config"] = config_result.exit_code
            if config_result.exit_code != 0:
                operational_failures.append("command:config")
            else:
                try:
                    rendered = _json_object(config_result.stdout, "compose config")
                except ValueError:
                    operational_failures.append("compose_config_payload")
                else:
                    private_network_valid = _validate_private_compose(
                        rendered, required_services, project_name
                    )
                    injection_valid = _validate_candidate_injection(
                        rendered,
                        candidate,
                        embedding_service=embedding,
                        llama_service=llama,
                    )
                    config_is_private = private_network_valid and injection_valid
                    if not private_network_valid:
                        operational_failures.append("compose_private_network")
                    elif not injection_valid:
                        operational_failures.append("candidate_injection")

            if config_is_private:
                _verify_candidate_stable(candidate, artifact_hasher)
                try:
                    up_result = runner(
                        _validated_argv(commands["up"]),
                        shell=False,
                        environment=clean_environment,
                    )
                    command_exit_codes["up"] = up_result.exit_code
                    if up_result.exit_code != 0:
                        operational_failures.append("command:up")
                    else:
                        _verify_candidate_stable(candidate, artifact_hasher)
                        metadata_result = runner(
                            _validated_argv(commands["metadata"]),
                            shell=False,
                            environment=clean_environment,
                        )
                        command_exit_codes["metadata"] = metadata_result.exit_code
                        if metadata_result.exit_code != 0:
                            operational_failures.append("command:metadata")
                        else:
                            try:
                                metadata = _json_object(
                                    metadata_result.stdout, "metadata"
                                )
                            except ValueError:
                                operational_failures.append("metadata_payload")

                        benchmark_result = runner(
                            _validated_argv(commands["benchmark"]),
                            shell=False,
                            environment=clean_environment,
                        )
                        command_exit_codes["benchmark"] = benchmark_result.exit_code
                        if benchmark_result.exit_code != 0:
                            operational_failures.append("command:benchmark")
                        try:
                            raw_metrics = _json_object(
                                benchmark_result.stdout, "benchmark"
                            )
                        except ValueError:
                            operational_failures.append("benchmark_payload")
                        _verify_candidate_stable(candidate, artifact_hasher)
                finally:
                    active_exception = sys.exception()
                    try:
                        down_result = runner(
                            _validated_argv(commands["down"]),
                            shell=False,
                            environment=clean_environment,
                        )
                    except BaseException as cleanup_exception:
                        if active_exception is None:
                            raise
                        if hasattr(active_exception, "add_note"):
                            active_exception.add_note(
                                f"model probe cleanup also failed: {cleanup_exception}"
                            )
                    else:
                        command_exit_codes["down"] = down_result.exit_code
                        if down_result.exit_code != 0:
                            operational_failures.append("command:down")
        finally:
            if override_path is not None:
                active_exception = sys.exception()
                try:
                    override_path.unlink(missing_ok=True)
                except BaseException as cleanup_exception:
                    if active_exception is None:
                        raise
                    if hasattr(active_exception, "add_note"):
                        active_exception.add_note(
                            "model probe override cleanup also failed: "
                            f"{cleanup_exception}"
                        )

    metadata_checksum = metadata.get("modelChecksum")
    if (
        not isinstance(metadata_checksum, str)
        or SHA256_PATTERN.fullmatch(metadata_checksum) is None
        or not hmac.compare_digest(metadata_checksum, candidate.sha256)
    ):
        operational_failures.append("metadata_checksum")
    if (
        metadata.get("name") != candidate.name
        or metadata.get("version") != candidate.version
    ):
        operational_failures.append("metadata_identity")
    if _public_network_attempted(metadata) or _public_network_attempted(raw_metrics):
        operational_failures.append("public_network_attempt")
    if candidate.kind == "generation-model":
        loader_audit = (
            "llama_props"
            if metadata.get("loaderAuditAvailable") is True
            and isinstance(metadata.get("publicNetworkAttempted"), bool)
            else "unavailable"
        )
    else:
        loader_audit = "python_audit_marker"
    if loader_audit == "unavailable":
        operational_failures.append("public_network_audit_unavailable")

    reranker: dict[str, object] = {
        "status": "not_applicable",
        "disabledReason": None,
    }
    if candidate.kind == "embedding-model":
        gate_metrics, metric_failures = _embedding_metrics(raw_metrics)
        actual_metadata = raw_metrics.get("modelMetadata")
        expected_metadata = candidate.metadata
        if (
            not isinstance(actual_metadata, Mapping)
            or actual_metadata.get("modelName") != expected_metadata.get("modelName")
            or actual_metadata.get("modelVersion") != expected_metadata.get("modelVersion")
            or actual_metadata.get("dimensions") != expected_metadata.get("dimensions")
        ):
            metric_failures.append("embedding_metadata_identity")
    elif candidate.kind == "generation-model":
        gate_metrics, metric_failures = _generation_metrics(raw_metrics)
        actual_metadata = raw_metrics.get("modelMetadata")
        expected_metadata = {
            "architecture": candidate.metadata.get("general.architecture"),
            "parameterSize": candidate.metadata.get("general.size_label"),
            "quantization": candidate.metadata.get("probe.quantization"),
        }
        if actual_metadata != expected_metadata:
            metric_failures.append("generation_metadata_identity")
    else:
        gate_metrics, metric_failures, reranker = _reranker_metrics(
            raw_metrics, candidate.sha256
        )
    gate_evaluation = evaluate_candidate_gate(candidate.kind, gate, gate_metrics)
    failures = list(
        dict.fromkeys(
            [
                *operational_failures,
                *metric_failures,
                *gate_evaluation.result.failures,
            ]
        )
    )
    result = CapacityResult(passed=not failures, failures=failures)
    report = {
        "candidate": {
            "artifactMetadata": dict(candidate.metadata),
            "kind": candidate.kind,
            "license": candidate.license,
            "localPath": str(candidate.path),
            "name": candidate.name,
            "sha256": candidate.sha256,
            "version": candidate.version,
        },
        "commandExitCodes": command_exit_codes,
        "discoveryLabel": label,
        "gate": asdict(gate),
        "gateMetrics": dict(gate_evaluation.metrics),
        "gateResult": {"passed": result.passed, "failures": list(result.failures)},
        "metadata": metadata,
        "networkPolicy": {
            "buildDisabled": True,
            "internalNetworkValidated": config_is_private,
            "loaderAudit": loader_audit,
            "pullPolicy": "never",
        },
        "offlineOnly": True,
        "probeMatrix": {
            "contextLengths": list(CONTEXT_LENGTHS),
            "embeddingConcurrencies": list(EMBEDDING_CONCURRENCIES),
            "maxOutputTokens": MAX_OUTPUT_TOKENS,
        },
        "rawMetrics": raw_metrics,
        "reranker": reranker,
    }
    _write_report(report_path, report)
    return result


def _load_candidate_lock(
    path: Path, *, candidate_name: str | None = None
) -> tuple[Mapping[str, object], Path]:
    lock_path = Path(path).resolve(strict=True)
    payload = json.loads(lock_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping) or set(payload) != {"artifacts"}:
        raise ValueError("candidate lock must contain only an artifacts list")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, list):
        raise ValueError("candidate lock artifacts must be a list")
    candidates = [
        item
        for item in artifacts
        if isinstance(item, Mapping) and item.get("kind") in SUPPORTED_KINDS
    ]
    if candidate_name is not None:
        selected = [item for item in candidates if item.get("name") == candidate_name]
        if len(selected) != 1:
            raise ValueError("candidate-name must select exactly one probeable candidate")
        return selected[0], lock_path.parent
    if len(candidates) != 1:
        raise ValueError(
            "candidate lock contains multiple probeable candidates; pass --candidate-name"
        )
    return candidates[0], lock_path.parent


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compose", type=Path, required=True)
    parser.add_argument("--embedding-service", required=True)
    parser.add_argument("--llama-service", required=True)
    parser.add_argument("--candidate-lock", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--candidate-name")
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("artifacts/benchmarks/model-probe-report.json"),
    )
    arguments = parser.parse_args(argv)
    entry, artifact_root = _load_candidate_lock(
        arguments.candidate_lock, candidate_name=arguments.candidate_name
    )
    result = run_model_probe(
        compose_file=arguments.compose,
        embedding_service=arguments.embedding_service,
        llama_service=arguments.llama_service,
        discovery_label=arguments.label,
        candidate_entry=entry,
        artifact_root=artifact_root,
        report_path=arguments.report,
    )
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
