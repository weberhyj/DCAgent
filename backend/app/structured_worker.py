from __future__ import annotations

import os
import signal
import socket
from collections.abc import Callable, Mapping
from hashlib import sha256
from pathlib import Path
from threading import Event, Lock, Thread, current_thread, main_thread
from time import monotonic, sleep
from typing import Any, Protocol

from .clickhouse_gateway import ClickHouseGateway
from .database import Database
from .offline_settings import OfflineSettings, require_secret_file
from .structured_ingestion import SpreadsheetPublisher
from .structured_models import StructuredPublicationResult
from .structured_repository import (
    StructuredLeaseError,
    StructuredPublicationJob,
    StructuredRepository,
)


class StructuredPublisher(Protocol):
    def publish(
        self,
        path: Path,
        schema: object,
        publication_id: str,
        *,
        lease_guard: Callable[[], None] | None = None,
        staging_token: str | None = None,
        staging_generation: int | None = None,
    ) -> StructuredPublicationResult: ...


class StructuredIngestionWorker:
    def __init__(
        self,
        repository: StructuredRepository,
        publisher: StructuredPublisher,
        *,
        worker_id: str,
        lease_seconds: int = 60,
        retry_delay_seconds: int = 60,
        poll_interval_seconds: float = 1.0,
        sleeper: Callable[[float], None] = sleep,
    ) -> None:
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        if retry_delay_seconds < 0:
            raise ValueError("retry_delay_seconds must be non-negative")
        if poll_interval_seconds <= 0:
            raise ValueError("poll_interval_seconds must be positive")
        self._repository = repository
        self._publisher = publisher
        self._worker_id = worker_id
        self._lease_seconds = lease_seconds
        self._retry_delay_seconds = retry_delay_seconds
        self._poll_interval_seconds = poll_interval_seconds
        self._sleeper = sleeper
        self._stop = Event()
        self._run_lock = Lock()

    def stop(self) -> None:
        self._stop.set()

    def run_once(self) -> bool:
        if not self._run_lock.acquire(blocking=False):
            return False
        try:
            job = self._repository.claim_publication(
                self._worker_id,
                self._lease_seconds,
            )
            if job is None:
                return False
            self._run_claimed(job)
            return True
        finally:
            self._run_lock.release()

    def _run_claimed(self, job: StructuredPublicationJob) -> None:
        lease_token = job.lease_token
        if lease_token is None:
            raise StructuredLeaseError("Claimed structured publication has no lease token")
        heartbeat_stop = Event()
        lease_lost = Event()
        renew_lock = Lock()
        next_renewal_at = [0.0]

        def renew(*, force: bool = False, checkpoint_row: int | None = None) -> None:
            if lease_lost.is_set():
                raise StructuredLeaseError("Structured publication lease was lost")
            with renew_lock:
                current = monotonic()
                if not force and current < next_renewal_at[0]:
                    return
                try:
                    self._repository.renew_publication_lease(
                        job.id,
                        lease_token,
                        self._lease_seconds,
                        checkpoint_row=checkpoint_row,
                    )
                except Exception as error:
                    lease_lost.set()
                    if isinstance(error, StructuredLeaseError):
                        raise
                    raise StructuredLeaseError(
                        "Structured publication lease could not be renewed"
                    ) from error
                next_renewal_at[0] = current + max(0.1, self._lease_seconds / 3)

        heartbeat = Thread(
            target=self._heartbeat,
            args=(heartbeat_stop, lease_lost, renew),
            name=f"structured-lease-{job.id}",
            daemon=True,
        )
        heartbeat.start()
        try:
            publication_input = self._repository.get_publication_input(job)
            renew(force=True)
            result = self._publisher.publish(
                Path(publication_input.path),
                publication_input.schema,
                job.publication_id,
                lease_guard=renew,
                staging_token=_staging_token(lease_token),
                staging_generation=job.attempt,
            )
            _validate_publication_result(result, job, len(publication_input.schema.columns))
            renew(force=True, checkpoint_row=result.row_count)
            self._repository.complete_publication(job.id, lease_token, result)
        except Exception as error:
            if not lease_lost.is_set():
                try:
                    self._repository.fail_publication(
                        job.id,
                        lease_token,
                        str(error) or error.__class__.__name__,
                        retry_delay_seconds=self._retry_delay_seconds,
                    )
                except StructuredLeaseError:
                    pass
        finally:
            heartbeat_stop.set()
            heartbeat.join(timeout=max(1.0, self._lease_seconds / 2))

    def _heartbeat(
        self,
        heartbeat_stop: Event,
        lease_lost: Event,
        renew: Callable[..., None],
    ) -> None:
        interval = max(0.1, self._lease_seconds / 3)
        while not heartbeat_stop.wait(interval):
            try:
                renew(force=True)
            except StructuredLeaseError:
                lease_lost.set()
                return

    def run_forever(self) -> None:
        previous_handler: Any = None
        can_install_handler = current_thread() is main_thread()
        if can_install_handler:
            previous_handler = signal.getsignal(signal.SIGTERM)
            signal.signal(signal.SIGTERM, lambda _signum, _frame: self.stop())
        try:
            while not self._stop.is_set():
                if not self.run_once():
                    self._stop.wait(self._poll_interval_seconds)
        finally:
            if can_install_handler and previous_handler is not None:
                signal.signal(signal.SIGTERM, previous_handler)


def _validate_publication_result(
    result: StructuredPublicationResult,
    job: StructuredPublicationJob,
    expected_column_count: int,
) -> None:
    if result.publication_id != job.publication_id:
        raise ValueError("Publisher returned a different publication id")
    if not result.physical_table_name.strip():
        raise ValueError("Publisher returned an empty physical table name")
    if result.row_count < 0:
        raise ValueError("Publisher returned a negative row count")
    if result.column_count != expected_column_count:
        raise ValueError("Publisher returned an unexpected column count")
    if len(result.content_hash) != 64 or any(
        character not in "0123456789abcdef" for character in result.content_hash
    ):
        raise ValueError("Publisher returned an invalid content hash")


def _staging_token(lease_token: str) -> str:
    return sha256(lease_token.encode("utf-8")).hexdigest()[:24]


def build_structured_worker(
    environ: Mapping[str, str] | None = None,
    *,
    database_factory: Callable[[str], Database] = Database,
    clickhouse_client_factory: Callable[..., Any] | None = None,
) -> StructuredIngestionWorker:
    source = os.environ if environ is None else environ
    settings = OfflineSettings.from_environ(source)
    if not settings.structured_query_enabled:
        raise ValueError("structured worker requires STRUCTURED_QUERY_ENABLED=true")
    ingest_password = require_secret_file(
        settings.clickhouse_ingest_password_file,
        "CLICKHOUSE_INGEST_PASSWORD_FILE",
    )
    database = database_factory(settings.database_url)
    repository = StructuredRepository(database)
    if clickhouse_client_factory is None:
        import clickhouse_connect

        clickhouse_client_factory = clickhouse_connect.get_client
    client_kwargs = {
        "dsn": settings.clickhouse_url,
        "username": settings.clickhouse_ingest_user,
        "password": ingest_password,
        "send_receive_timeout": settings.structured_query_timeout_seconds,
    }
    clients: tuple[object, ...] = ()
    try:
        ingest_client = clickhouse_client_factory(**client_kwargs)
        clients = (ingest_client,)
        query_client = clickhouse_client_factory(
            **client_kwargs,
            autogenerate_session_id=False,
        )
        clients = (ingest_client, query_client)
        publisher = SpreadsheetPublisher(
            clickhouse=ClickHouseGateway(
                ingest_client,
                query_client=query_client,
                max_execution_time=settings.structured_query_timeout_seconds,
            ),
            parquet_root=settings.parquet_root,
            batch_rows=settings.structured_ingest_batch_rows,
        )
    except Exception:
        _close_clients(clients)
        raise
    worker_id = source.get("STRUCTURED_WORKER_ID", f"{socket.gethostname()}:{os.getpid()}")
    return StructuredIngestionWorker(
        repository,
        publisher,
        worker_id=worker_id,
        lease_seconds=int(source.get("STRUCTURED_JOB_LEASE_SECONDS", "60")),
        retry_delay_seconds=int(source.get("STRUCTURED_JOB_RETRY_SECONDS", "60")),
        poll_interval_seconds=float(source.get("STRUCTURED_JOB_POLL_SECONDS", "1")),
    )


def _close_clients(clients: tuple[object, ...]) -> None:
    closed_ids: set[int] = set()
    for client in reversed(clients):
        if id(client) in closed_ids:
            continue
        closed_ids.add(id(client))
        close = getattr(client, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                pass


def main() -> None:
    worker = build_structured_worker()
    worker.run_forever()


if __name__ == "__main__":
    main()
