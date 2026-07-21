from __future__ import annotations

import secrets
from typing import Annotated

from fastapi import (
    APIRouter,
    BackgroundTasks,
    Depends,
    File,
    Form,
    HTTPException,
    Query,
    Request,
    UploadFile,
)
from fastapi.responses import JSONResponse
from pydantic import BeforeValidator

from .evaluation import (
    EvaluationCaseDuplicateError,
    EvaluationCaseFilterStatus,
    parse_evaluation_category_filter,
    parse_evaluation_expect_answer_filter,
    parse_evaluation_status_filter,
    parse_evaluation_tag_filter,
)
from .evaluation_batches import compare_evaluation_batches, summarize_evaluation_runs
from .evaluation_import import (
    MAX_IMPORT_BYTES,
    EvaluationImportFileError,
    EvaluationImportService,
    EvaluationImportTokenBusyError,
    EvaluationImportTokenError,
)
from .ingestion import KnowledgeIngestionQueue
from .llm import LLMProviderError
from .repository import ChatRepository
from .schemas import (
    AgentRunAudit,
    ChatMessage,
    ConversationBundle,
    EvaluationBatch,
    EvaluationBatchComparison,
    EvaluationBatchDetail,
    EvaluationBatchRequest,
    EvaluationCaseCollection,
    EvaluationCaseRequest,
    EvaluationDashboard,
    EvaluationImportConfirmRequest,
    EvaluationImportConfirmResponse,
    EvaluationImportPreviewResponse,
    EvaluationRunRequest,
    KnowledgeChunk,
    KnowledgeSource,
    KnowledgeSourceRequest,
    SendMessageRequest,
    SpreadsheetPreview,
    StructuredSchemaConfirmationRequest,
    StructuredSchemaConfirmationResponse,
)
from .storage import KnowledgeFileStorage
from .structured_repository import (
    StructuredColumnConfirmation,
    StructuredConflictError,
    StructuredDatasetConfirmation,
    StructuredNotFoundError,
    StructuredRepository,
    StructuredValidationError,
)

router = APIRouter(prefix="/api")


def get_repository(request: Request) -> ChatRepository:
    return request.app.state.repository


def get_storage(request: Request) -> KnowledgeFileStorage:
    return request.app.state.knowledge_file_storage


def get_ingestion_queue(request: Request) -> KnowledgeIngestionQueue:
    return request.app.state.knowledge_ingestion_queue


def get_structured_repository(request: Request) -> StructuredRepository:
    if not getattr(request.app.state, "structured_query_enabled", False):
        raise HTTPException(status_code=404, detail="Structured query feature is disabled")
    repository = getattr(request.app.state, "structured_repository", None)
    if repository is None:
        raise HTTPException(status_code=503, detail="Structured repository is unavailable")
    return repository


def get_evaluation_import_service(request: Request) -> EvaluationImportService:
    return request.app.state.evaluation_import_service


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/healthz")
async def healthz() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/readyz")
def readyz(request: Request) -> JSONResponse:
    if not getattr(request.app.state, "health_checks_active", False):
        report: dict[str, dict[str, bool | str]] = {
            "startup": {"ok": False, "detail": "not initialized"}
        }
    else:
        registry = getattr(request.app.state, "health_registry", None)
        if registry is None:
            report = {"startup": {"ok": False, "detail": "not initialized"}}
        else:
            try:
                report = registry.report()
            except Exception:
                report = {
                    "health_registry": {
                        "ok": False,
                        "detail": "check failed",
                    }
                }

    ready = all(bool(item["ok"]) for item in report.values())
    return JSONResponse(
        status_code=200 if ready else 503,
        content={
            "status": "ready" if ready else "not_ready",
            "dependencies": report,
        },
    )


@router.get("/admin/agent/runs", response_model=list[AgentRunAudit])
def list_agent_runs(repository: ChatRepository = Depends(get_repository)) -> list[AgentRunAudit]:
    return [AgentRunAudit.from_model(run) for run in repository.list_agent_runs()]


def evaluation_dashboard(repository: ChatRepository) -> EvaluationDashboard:
    return EvaluationDashboard.from_models(
        repository.list_evaluation_cases(),
        repository.list_evaluation_runs(),
    )


@router.get("/admin/evaluations", response_model=EvaluationDashboard)
def list_evaluations(
    repository: ChatRepository = Depends(get_repository),
) -> EvaluationDashboard:
    return evaluation_dashboard(repository)


@router.get("/admin/evaluations/cases", response_model=EvaluationCaseCollection)
def list_evaluation_cases(
    category: Annotated[
        str | None,
        Query(max_length=80),
        BeforeValidator(parse_evaluation_category_filter),
    ] = None,
    tag: Annotated[
        str | None,
        Query(max_length=80),
        BeforeValidator(parse_evaluation_tag_filter),
    ] = None,
    expect_answer: Annotated[
        bool | None,
        BeforeValidator(parse_evaluation_expect_answer_filter),
        Query(alias="expectAnswer"),
    ] = None,
    status: Annotated[
        EvaluationCaseFilterStatus | None,
        BeforeValidator(parse_evaluation_status_filter),
        Query(),
    ] = None,
    repository: ChatRepository = Depends(get_repository),
) -> EvaluationCaseCollection:
    facets = repository.get_evaluation_case_facets()
    filtered_cases = repository.list_evaluation_cases(
        category=category,
        tag=tag,
        expect_answer=expect_answer,
        status=status,
    )
    return EvaluationCaseCollection.from_models(filtered_cases, facets)


@router.post(
    "/admin/evaluations/import/preview",
    response_model=EvaluationImportPreviewResponse,
)
async def preview_evaluation_import(
    file: UploadFile = File(...),
    repository: ChatRepository = Depends(get_repository),
    service: EvaluationImportService = Depends(get_evaluation_import_service),
) -> EvaluationImportPreviewResponse:
    try:
        preview = service.preview(
            file.filename or "",
            await file.read(MAX_IMPORT_BYTES + 1),
            repository.list_knowledge_sources(),
            repository.list_evaluation_cases(),
        )
        if preview.valid_rows == 0:
            service.consume(preview.token)
            raise EvaluationImportFileError("文件中没有可导入的有效数据")
        return EvaluationImportPreviewResponse.from_model(preview)
    except EvaluationImportFileError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=f"导入文件无效：{error}") from error


@router.post(
    "/admin/evaluations/import/confirm",
    response_model=EvaluationImportConfirmResponse,
)
def confirm_evaluation_import(
    request: EvaluationImportConfirmRequest,
    repository: ChatRepository = Depends(get_repository),
    service: EvaluationImportService = Depends(get_evaluation_import_service),
) -> EvaluationImportConfirmResponse:
    try:
        preview = service.reserve(request.preview_token)
    except EvaluationImportTokenBusyError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except EvaluationImportTokenError as error:
        raise HTTPException(status_code=410, detail=str(error)) from error

    try:
        import_batch_id = f"eval-import-{secrets.token_hex(6)}"
        result = repository.create_evaluation_cases(
            rows=preview.rows,
            import_batch_id=import_batch_id,
            file_name=preview.file_name,
            total_rows=preview.total_rows,
            valid_rows=preview.valid_rows,
            invalid_rows=preview.invalid_rows,
        )
    except ValueError as error:
        service.release(request.preview_token)
        raise HTTPException(status_code=400, detail=f"导入数据无效：{error}") from error
    except Exception:
        service.release(request.preview_token)
        raise

    service.complete(request.preview_token)

    return EvaluationImportConfirmResponse(
        importBatchId=result.batch.id,
        createdCount=result.created_count,
        duplicateCount=result.duplicate_count,
        dashboard=evaluation_dashboard(repository),
    )


@router.post("/admin/evaluations/cases", response_model=EvaluationDashboard)
def create_evaluation_case(
    request: EvaluationCaseRequest,
    repository: ChatRepository = Depends(get_repository),
) -> EvaluationDashboard:
    try:
        repository.create_evaluation_case(
            question=request.question,
            expected_source_ids=request.expected_source_ids,
            expected_terms=request.expected_terms,
            top_k=request.top_k,
            expect_answer=request.expect_answer,
            category=request.category,
            tags=request.tags,
            external_key=request.external_key,
            import_batch_id=request.import_batch_id,
        )
    except EvaluationCaseDuplicateError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    return evaluation_dashboard(repository)


@router.delete("/admin/evaluations/cases/{case_id}", response_model=EvaluationDashboard)
def delete_evaluation_case(
    case_id: str,
    repository: ChatRepository = Depends(get_repository),
) -> EvaluationDashboard:
    repository.delete_evaluation_case(case_id)
    return evaluation_dashboard(repository)


@router.post("/admin/evaluations/run", response_model=EvaluationDashboard)
def run_evaluations(
    request: EvaluationRunRequest,
    repository: ChatRepository = Depends(get_repository),
) -> EvaluationDashboard:
    repository.run_evaluation_cases(request.case_ids or None)
    return evaluation_dashboard(repository)


@router.post("/admin/evaluations/batches", response_model=EvaluationBatch)
def create_evaluation_batch(
    request: EvaluationBatchRequest,
    background_tasks: BackgroundTasks,
    repository: ChatRepository = Depends(get_repository),
) -> EvaluationBatch:
    try:
        batch = repository.create_evaluation_batch(
            name=request.name,
            case_ids=request.case_ids,
            retrieval_min_score=request.retrieval_min_score,
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    background_tasks.add_task(repository.run_evaluation_batch, batch.id)
    return EvaluationBatch.from_model(batch)


@router.get("/admin/evaluations/batches", response_model=list[EvaluationBatch])
def list_evaluation_batches(
    repository: ChatRepository = Depends(get_repository),
) -> list[EvaluationBatch]:
    return [EvaluationBatch.from_model(batch) for batch in repository.list_evaluation_batches()]


@router.get(
    "/admin/evaluations/batches/compare",
    response_model=EvaluationBatchComparison,
)
def compare_evaluation_batch_results(
    left: str,
    right: str,
    repository: ChatRepository = Depends(get_repository),
) -> EvaluationBatchComparison:
    left_batch = repository.get_evaluation_batch(left)
    right_batch = repository.get_evaluation_batch(right)
    if left_batch.status != "completed" or right_batch.status != "completed":
        raise HTTPException(
            status_code=409,
            detail="仅支持比较已完成的评测批次",
        )

    comparison = compare_evaluation_batches(
        left_batch,
        repository.list_evaluation_runs_for_batch(left_batch.id),
        right_batch,
        repository.list_evaluation_runs_for_batch(right_batch.id),
    )
    return EvaluationBatchComparison.from_model(comparison)


# Keep literal batch routes above this parameterized route for Task7 extensions.
@router.get(
    "/admin/evaluations/batches/{batch_id}",
    response_model=EvaluationBatchDetail,
)
def get_evaluation_batch_detail(
    batch_id: str,
    repository: ChatRepository = Depends(get_repository),
) -> EvaluationBatchDetail:
    batch = repository.get_evaluation_batch(batch_id)
    runs = repository.list_evaluation_runs_for_batch(batch_id)
    cases_by_id = {case.id: case for case in repository.list_evaluation_cases()}
    cases = [cases_by_id[case_id] for case_id in batch.case_ids if case_id in cases_by_id]
    summary = summarize_evaluation_runs(runs, cases_by_id)
    return EvaluationBatchDetail.from_models(
        batch=batch,
        summary=summary,
        runs=runs,
        cases=cases,
    )


@router.get("/conversations", response_model=ConversationBundle)
def list_conversations(repository: ChatRepository = Depends(get_repository)) -> ConversationBundle:
    conversations = repository.list_conversations()
    if not conversations:
        conversations, active_id, messages = repository.create_conversation()
        return ConversationBundle.from_models(conversations, active_id, messages)
    active_id = conversations[0].id
    messages = repository.get_messages(active_id)
    return ConversationBundle.from_models(conversations, active_id, messages)


@router.post("/conversations", response_model=ConversationBundle)
def create_conversation(repository: ChatRepository = Depends(get_repository)) -> ConversationBundle:
    conversations, active_id, messages = repository.create_conversation()
    return ConversationBundle.from_models(conversations, active_id, messages)


@router.delete("/conversations/{conversation_id}", response_model=ConversationBundle)
def delete_conversation(
    conversation_id: str,
    repository: ChatRepository = Depends(get_repository),
) -> ConversationBundle:
    conversations, active_id, messages = repository.delete_conversation(conversation_id)
    return ConversationBundle.from_models(conversations, active_id, messages)


@router.get("/conversations/{conversation_id}/messages", response_model=list[ChatMessage])
def list_messages(
    conversation_id: str,
    repository: ChatRepository = Depends(get_repository),
) -> list[ChatMessage]:
    return [ChatMessage.from_model(message) for message in repository.get_messages(conversation_id)]


@router.post("/conversations/{conversation_id}/messages", response_model=ConversationBundle)
def send_message(
    conversation_id: str,
    request: SendMessageRequest,
    repository: ChatRepository = Depends(get_repository),
) -> ConversationBundle:
    try:
        conversations, active_id, messages = repository.send_message(
            conversation_id,
            request.content,
            request.mode,
        )
    except LLMProviderError as error:
        raise HTTPException(status_code=502, detail=str(error)) from error
    return ConversationBundle.from_models(conversations, active_id, messages)


@router.get("/knowledge/sources", response_model=list[KnowledgeSource])
def list_knowledge_sources(
    repository: ChatRepository = Depends(get_repository),
    ingestion_queue: KnowledgeIngestionQueue = Depends(get_ingestion_queue),
) -> list[KnowledgeSource]:
    ingestion_queue.drain()
    return [KnowledgeSource.from_model(source) for source in repository.list_knowledge_sources()]


@router.post("/knowledge/sources", response_model=list[KnowledgeSource])
def add_knowledge_source(
    request: KnowledgeSourceRequest,
    repository: ChatRepository = Depends(get_repository),
) -> list[KnowledgeSource]:
    return [
        KnowledgeSource.from_model(source)
        for source in repository.add_knowledge_source(
            request.name,
            request.source_type,
            request.classification,
        )
    ]


@router.delete("/knowledge/sources/{source_id}", response_model=list[KnowledgeSource])
def delete_knowledge_source(
    source_id: str,
    repository: ChatRepository = Depends(get_repository),
    storage: KnowledgeFileStorage = Depends(get_storage),
    ingestion_queue: KnowledgeIngestionQueue = Depends(get_ingestion_queue),
) -> list[KnowledgeSource]:
    ingestion_queue.discard_source(source_id)
    sources, deleted = repository.delete_knowledge_source(source_id)
    storage.delete(deleted.file_path)
    return [KnowledgeSource.from_model(source) for source in sources]


@router.post("/knowledge/sources/{source_id}/reindex", response_model=list[KnowledgeSource])
def reindex_knowledge_source(
    source_id: str,
    repository: ChatRepository = Depends(get_repository),
    ingestion_queue: KnowledgeIngestionQueue = Depends(get_ingestion_queue),
) -> list[KnowledgeSource]:
    source = next(
        (item for item in repository.list_knowledge_sources() if item.id == source_id), None
    )
    if source is None:
        raise HTTPException(status_code=404, detail="Knowledge source not found")
    if not source.file_path:
        raise HTTPException(
            status_code=400, detail="Knowledge source has no uploaded file to reindex"
        )

    retried = repository.reindex_knowledge_source(source_id)
    ingestion_queue.discard_source(source_id)
    ingestion_queue.enqueue(retried.id, retried.file_path, retried.source_type)
    return [KnowledgeSource.from_model(item) for item in repository.list_knowledge_sources()]


@router.post("/knowledge/uploads", response_model=list[KnowledgeSource])
async def upload_knowledge_file(
    files: list[UploadFile] | None = File(default=None),
    file: UploadFile | None = File(default=None),
    classification: str = Form(default="内部·机密"),
    repository: ChatRepository = Depends(get_repository),
    storage: KnowledgeFileStorage = Depends(get_storage),
    ingestion_queue: KnowledgeIngestionQueue = Depends(get_ingestion_queue),
) -> list[KnowledgeSource]:
    selected_files = files or ([file] if file is not None else [])
    if not selected_files:
        raise HTTPException(status_code=400, detail="Upload at least one knowledge file")

    sources = repository.list_knowledge_sources()
    for upload in selected_files:
        content = await upload.read()
        stored = storage.save(upload.filename or "", content)
        sources = repository.add_uploaded_knowledge_source(
            source_id=stored.source_id,
            name=stored.original_name,
            source_type=stored.source_type,
            classification=classification,
            records=stored.records,
            file_path=str(stored.path),
            file_size=stored.size,
            mime_type=upload.content_type,
        )
        ingestion_queue.enqueue(stored.source_id, stored.path, stored.source_type)

    return [KnowledgeSource.from_model(source) for source in sources]


@router.get(
    "/knowledge/sources/{source_id}/structured-preview",
    response_model=SpreadsheetPreview,
)
def get_structured_preview(
    source_id: str,
    structured_repository: StructuredRepository = Depends(get_structured_repository),
) -> SpreadsheetPreview:
    try:
        preview = structured_repository.get_preview(source_id)
    except StructuredNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except StructuredConflictError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    return SpreadsheetPreview.from_model(preview)


@router.put(
    "/knowledge/sources/{source_id}/structured-schema",
    response_model=StructuredSchemaConfirmationResponse,
)
def confirm_structured_schema(
    source_id: str,
    request: StructuredSchemaConfirmationRequest,
    structured_repository: StructuredRepository = Depends(get_structured_repository),
) -> StructuredSchemaConfirmationResponse:
    submissions = tuple(
        StructuredDatasetConfirmation(
            dataset_id=dataset.dataset_id,
            columns=tuple(
                StructuredColumnConfirmation(
                    physical_name=column.physical_name,
                    display_name=column.display_name,
                    data_type=column.data_type,
                    aliases=tuple(column.aliases),
                    allow_aggregate=column.allow_aggregate,
                    allow_filter=column.allow_filter,
                    null_policy=column.null_policy,
                )
                for column in dataset.columns
            ),
        )
        for dataset in request.datasets
    )
    try:
        result = structured_repository.confirm_schema(source_id, submissions)
    except StructuredNotFoundError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except StructuredValidationError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    except StructuredConflictError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    return StructuredSchemaConfirmationResponse.from_model(result)


@router.get("/knowledge/sources/{source_id}/chunks", response_model=list[KnowledgeChunk])
def list_knowledge_chunks(
    source_id: str,
    repository: ChatRepository = Depends(get_repository),
    ingestion_queue: KnowledgeIngestionQueue = Depends(get_ingestion_queue),
) -> list[KnowledgeChunk]:
    ingestion_queue.drain()
    return [
        KnowledgeChunk.from_model(chunk) for chunk in repository.list_knowledge_chunks(source_id)
    ]
