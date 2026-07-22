from __future__ import annotations

import math
from datetime import datetime
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StringConstraints,
    field_validator,
    model_validator,
)

from .agent import AgentRunAudit as AgentRunAuditModel
from .agent import AgentStep as AgentStepModel
from .answer_text import remove_inline_citation_markers
from .evaluation import (
    EvaluationBatchModel,
    EvaluationCaseFacets,
    EvaluationCaseModel,
    EvaluationHitModel,
    EvaluationRunModel,
    normalize_evaluation_case_metadata,
    normalized_unique,
)
from .evaluation_batches import (
    EvaluationBatchComparisonModel,
    EvaluationBatchMetricDeltaModel,
    EvaluationBatchSummaryModel,
    EvaluationFailureReason,
    EvaluationMetricGroupModel,
    evaluation_failure_reasons,
)
from .evaluation_import import (
    EvaluationImportError as EvaluationImportErrorModel,
)
from .evaluation_import import (
    EvaluationImportPreview,
    EvaluationImportRow,
)
from .models import (
    ChatMessageModel,
    CitationModel,
    ConversationModel,
    ImageArtifactModel,
    KnowledgeChunkModel,
    KnowledgeSourceModel,
    ResponseParagraphModel,
    SummaryArtifactModel,
    TableArtifactModel,
    VideoArtifactModel,
)
from .spreadsheet_schema import MAX_COLUMNS_PER_DATASET, MAX_WORKSHEETS
from .structured_models import (
    MAX_STRUCTURED_ALIAS_LENGTH,
    MAX_STRUCTURED_ALIASES_PER_COLUMN,
    StructuredColumnType,
    StructuredConfirmationResult,
)
from .structured_models import (
    SpreadsheetPreview as SpreadsheetPreviewModel,
)
from .structured_models import (
    StructuredColumnPreview as StructuredColumnPreviewModel,
)
from .structured_models import (
    StructuredColumnSchema as StructuredColumnSchemaModel,
)
from .structured_models import (
    StructuredDatasetPreview as StructuredDatasetPreviewModel,
)
from .structured_models import (
    StructuredDatasetSchema as StructuredDatasetSchemaModel,
)
from .structured_models import StructuredDiagnostic as StructuredDiagnosticModel
from .structured_models import StructuredPublication as StructuredPublicationModel
from .structured_repository import (
    StructuredPublicationJob as StructuredPublicationJobModel,
)
from .structured_repository import StructuredStatus as StructuredStatusModel
from .time_utils import normalize_display_timestamp

StructuredAlias = Annotated[
    str,
    StringConstraints(
        strip_whitespace=True,
        min_length=1,
        max_length=MAX_STRUCTURED_ALIAS_LENGTH,
    ),
]


class ApiModel(BaseModel):
    model_config = ConfigDict(populate_by_name=True)


class Citation(ApiModel):
    label: str
    classification: str
    source_id: str = Field(alias="sourceId")
    source_name: str | None = Field(default=None, alias="sourceName")
    chunk_id: str | None = Field(default=None, alias="chunkId")
    chunk_index: int | None = Field(default=None, alias="chunkIndex")
    excerpt: str | None = None
    score: float | None = None
    rank: int | None = None
    matched_terms: list[str] = Field(default_factory=list, alias="matchedTerms")

    @classmethod
    def from_model(cls, citation: CitationModel) -> Citation:
        return cls(
            label=citation.label,
            classification=citation.classification,
            sourceId=citation.source_id,
            sourceName=citation.source_name,
            chunkId=citation.chunk_id,
            chunkIndex=citation.chunk_index,
            excerpt=citation.excerpt,
            score=citation.score,
            rank=citation.rank,
            matchedTerms=citation.matched_terms,
        )


class ResponseParagraph(ApiModel):
    text: str
    citations: list[Citation] = Field(default_factory=list)

    @classmethod
    def from_model(
        cls, paragraph: ResponseParagraphModel, expose_citations: bool = False
    ) -> ResponseParagraph:
        return cls(
            text=(
                paragraph.text
                if expose_citations
                else remove_inline_citation_markers(paragraph.text)
            ),
            citations=[Citation.from_model(citation) for citation in paragraph.citations]
            if expose_citations
            else [],
        )


class SummaryArtifact(ApiModel):
    type: Literal["summary"]
    title: str
    source: str
    bullets: list[str]

    @classmethod
    def from_model(
        cls, artifact: SummaryArtifactModel, expose_source: bool = False
    ) -> SummaryArtifact:
        return cls(
            type=artifact.type,
            title=artifact.title,
            source=artifact.source if expose_source else "",
            bullets=artifact.bullets,
        )


class ImageArtifact(ApiModel):
    type: Literal["image"]
    title: str
    source: str
    asset_key: Literal["city", "analysis"] = Field(alias="assetKey")

    @classmethod
    def from_model(cls, artifact: ImageArtifactModel, expose_source: bool = False) -> ImageArtifact:
        return cls(
            type=artifact.type,
            title=artifact.title,
            source=artifact.source if expose_source else "",
            assetKey=artifact.asset_key,
        )


class VideoArtifact(ApiModel):
    type: Literal["video"]
    title: str
    source: str
    duration: str
    asset_key: Literal["city", "analysis"] = Field(alias="assetKey")

    @classmethod
    def from_model(cls, artifact: VideoArtifactModel, expose_source: bool = False) -> VideoArtifact:
        return cls(
            type=artifact.type,
            title=artifact.title,
            source=artifact.source if expose_source else "",
            duration=artifact.duration,
            assetKey=artifact.asset_key,
        )


class TableArtifact(ApiModel):
    type: Literal["table"]
    title: str
    source: str
    columns: list[str]
    rows: list[list[str]]

    @classmethod
    def from_model(cls, artifact: TableArtifactModel, expose_source: bool = False) -> TableArtifact:
        return cls(
            type=artifact.type,
            title=artifact.title,
            source=artifact.source if expose_source else "",
            columns=artifact.columns,
            rows=artifact.rows,
        )


Artifact = Annotated[
    SummaryArtifact | ImageArtifact | VideoArtifact | TableArtifact,
    Field(discriminator="type"),
]


def artifact_from_model(
    artifact: SummaryArtifactModel | ImageArtifactModel | VideoArtifactModel | TableArtifactModel,
    expose_source: bool = False,
) -> Artifact:
    if isinstance(artifact, SummaryArtifactModel):
        return SummaryArtifact.from_model(artifact, expose_source=expose_source)
    if isinstance(artifact, ImageArtifactModel):
        return ImageArtifact.from_model(artifact, expose_source=expose_source)
    if isinstance(artifact, VideoArtifactModel):
        return VideoArtifact.from_model(artifact, expose_source=expose_source)
    return TableArtifact.from_model(artifact, expose_source=expose_source)


class ChatMessage(ApiModel):
    id: str
    role: Literal["user", "assistant"]
    time: str
    content: str | None = None
    paragraphs: list[ResponseParagraph] = Field(default_factory=list)
    artifacts: list[Artifact] = Field(default_factory=list)

    @classmethod
    def from_model(
        cls,
        message: ChatMessageModel,
        expose_citations: bool = False,
        expose_artifact_sources: bool = False,
    ) -> ChatMessage:
        return cls(
            id=message.id,
            role=message.role,
            time=normalize_display_timestamp(message.time),
            content=message.content,
            paragraphs=[
                ResponseParagraph.from_model(paragraph, expose_citations=expose_citations)
                for paragraph in message.paragraphs
            ],
            artifacts=[
                artifact_from_model(artifact, expose_source=expose_artifact_sources)
                for artifact in message.artifacts
            ],
        )


class Conversation(ApiModel):
    id: str
    title: str
    topic: str
    group: str
    updated_at: str = Field(alias="updatedAt")
    pinned: bool = False

    @classmethod
    def from_model(cls, conversation: ConversationModel) -> Conversation:
        return cls(
            id=conversation.id,
            title=conversation.title,
            topic=conversation.topic,
            group=conversation.group,
            updatedAt=normalize_display_timestamp(conversation.updated_at),
            pinned=conversation.pinned,
        )


class ConversationBundle(ApiModel):
    conversations: list[Conversation]
    active_conversation_id: str = Field(alias="activeConversationId")
    messages: list[ChatMessage]

    @classmethod
    def from_models(
        cls,
        conversations: list[ConversationModel],
        active_conversation_id: str,
        messages: list[ChatMessageModel],
    ) -> ConversationBundle:
        return cls(
            conversations=[Conversation.from_model(conversation) for conversation in conversations],
            activeConversationId=active_conversation_id,
            messages=[ChatMessage.from_model(message) for message in messages],
        )


class SendMessageRequest(ApiModel):
    content: str = Field(min_length=1, max_length=4000)
    mode: Literal["quick", "deep", "source"] = "deep"

    @field_validator("content")
    @classmethod
    def content_must_not_be_blank(cls, value: str) -> str:
        trimmed = value.strip()
        if not trimmed:
            raise ValueError("content must not be blank")
        return trimmed


class KnowledgeSource(ApiModel):
    id: str
    name: str
    source_type: str = Field(alias="sourceType")
    records: int
    status: Literal[
        "已索引",
        "解析中",
        "待复核",
        "解析失败",
        "待确认表结构",
        "结构化导入中",
    ]
    updated_at: str = Field(alias="updatedAt")
    classification: str
    file_size: int | None = Field(default=None, alias="fileSize")
    mime_type: str | None = Field(default=None, alias="mimeType")
    error_message: str | None = Field(default=None, alias="errorMessage")

    @classmethod
    def from_model(cls, source: KnowledgeSourceModel) -> KnowledgeSource:
        return cls(
            id=source.id,
            name=source.name,
            sourceType=source.source_type,
            records=source.records,
            status=source.status,
            updatedAt=normalize_display_timestamp(source.updated_at),
            classification=source.classification,
            fileSize=source.file_size,
            mimeType=source.mime_type,
            errorMessage=source.error_message,
        )


class KnowledgeChunk(ApiModel):
    id: str
    source_id: str = Field(alias="sourceId")
    chunk_index: int = Field(alias="chunkIndex")
    text: str
    token_count: int = Field(alias="tokenCount")

    @classmethod
    def from_model(cls, chunk: KnowledgeChunkModel) -> KnowledgeChunk:
        return cls(
            id=chunk.id,
            sourceId=chunk.source_id,
            chunkIndex=chunk.chunk_index,
            text=chunk.text,
            tokenCount=chunk.token_count,
        )


class StructuredDiagnostic(ApiModel):
    code: str
    message: str
    worksheet_name: str = Field(alias="worksheetName")
    column_name: str | None = Field(default=None, alias="columnName")
    row_number: int | None = Field(default=None, alias="rowNumber")

    @classmethod
    def from_model(cls, diagnostic: StructuredDiagnosticModel) -> StructuredDiagnostic:
        return cls(
            code=diagnostic.code,
            message=diagnostic.message,
            worksheetName=diagnostic.worksheet_name,
            columnName=diagnostic.column_name,
            rowNumber=diagnostic.row_number,
        )


class StructuredColumnPreview(ApiModel):
    physical_name: str = Field(alias="physicalName")
    original_name: str = Field(alias="originalName")
    display_name: str = Field(alias="displayName")
    data_type: StructuredColumnType = Field(alias="dataType")
    aliases: list[str]
    examples: list[str]
    sampled_rows: int = Field(alias="sampledRows")
    null_count: int = Field(alias="nullCount")

    @classmethod
    def from_model(cls, column: StructuredColumnPreviewModel) -> StructuredColumnPreview:
        return cls(
            physicalName=column.physical_name,
            originalName=column.original_name,
            displayName=column.display_name,
            dataType=column.data_type,
            aliases=list(column.aliases),
            examples=list(column.examples),
            sampledRows=column.sampled_rows,
            nullCount=column.null_count,
        )


class StructuredDatasetPreview(ApiModel):
    dataset_id: str = Field(alias="datasetId")
    source_id: str = Field(alias="sourceId")
    worksheet_name: str = Field(alias="worksheetName")
    columns: list[StructuredColumnPreview]
    sampled_rows: int = Field(alias="sampledRows")
    schema_hash: str = Field(alias="schemaHash")

    @classmethod
    def from_model(cls, dataset: StructuredDatasetPreviewModel) -> StructuredDatasetPreview:
        return cls(
            datasetId=dataset.dataset_id,
            sourceId=dataset.source_id,
            worksheetName=dataset.worksheet_name,
            columns=[StructuredColumnPreview.from_model(column) for column in dataset.columns],
            sampledRows=dataset.sampled_rows,
            schemaHash=dataset.schema_hash,
        )


class SpreadsheetPreview(ApiModel):
    source_id: str = Field(alias="sourceId")
    datasets: list[StructuredDatasetPreview]
    diagnostics: list[StructuredDiagnostic]

    @classmethod
    def from_model(cls, preview: SpreadsheetPreviewModel) -> SpreadsheetPreview:
        return cls(
            sourceId=preview.source_id,
            datasets=[StructuredDatasetPreview.from_model(item) for item in preview.datasets],
            diagnostics=[StructuredDiagnostic.from_model(item) for item in preview.diagnostics],
        )


class StructuredColumnConfirmationRequest(ApiModel):
    physical_name: str = Field(alias="physicalName", min_length=1, max_length=160)
    display_name: str = Field(alias="displayName", max_length=240)
    data_type: StructuredColumnType = Field(alias="dataType")
    aliases: list[StructuredAlias] = Field(max_length=MAX_STRUCTURED_ALIASES_PER_COLUMN)
    allow_aggregate: StrictBool = Field(alias="allowAggregate")
    allow_filter: StrictBool = Field(alias="allowFilter")
    null_policy: str = Field(alias="nullPolicy", min_length=1, max_length=40)


class StructuredDatasetConfirmationRequest(ApiModel):
    dataset_id: str = Field(alias="datasetId", min_length=1, max_length=128)
    columns: list[StructuredColumnConfirmationRequest] = Field(
        min_length=1,
        max_length=MAX_COLUMNS_PER_DATASET,
    )


class StructuredSchemaConfirmationRequest(ApiModel):
    datasets: list[StructuredDatasetConfirmationRequest] = Field(
        min_length=1,
        max_length=MAX_WORKSHEETS,
    )


class StructuredColumnSchema(ApiModel):
    physical_name: str = Field(alias="physicalName")
    original_name: str = Field(alias="originalName")
    display_name: str = Field(alias="displayName")
    data_type: StructuredColumnType = Field(alias="dataType")
    aliases: list[str]
    allow_aggregate: bool = Field(alias="allowAggregate")
    allow_filter: bool = Field(alias="allowFilter")
    null_policy: str = Field(alias="nullPolicy")

    @classmethod
    def from_model(cls, column: StructuredColumnSchemaModel) -> StructuredColumnSchema:
        return cls(
            physicalName=column.physical_name,
            originalName=column.original_name,
            displayName=column.display_name,
            dataType=column.data_type,
            aliases=list(column.aliases),
            allowAggregate=column.allow_aggregate,
            allowFilter=column.allow_filter,
            nullPolicy=column.null_policy,
        )


class StructuredDatasetSchema(ApiModel):
    dataset_id: str = Field(alias="datasetId")
    source_id: str = Field(alias="sourceId")
    worksheet_name: str = Field(alias="worksheetName")
    schema_version: int = Field(alias="schemaVersion")
    columns: list[StructuredColumnSchema]
    schema_hash: str = Field(alias="schemaHash")

    @classmethod
    def from_model(cls, dataset: StructuredDatasetSchemaModel) -> StructuredDatasetSchema:
        return cls(
            datasetId=dataset.dataset_id,
            sourceId=dataset.source_id,
            worksheetName=dataset.worksheet_name,
            schemaVersion=dataset.schema_version,
            columns=[StructuredColumnSchema.from_model(column) for column in dataset.columns],
            schemaHash=dataset.schema_hash,
        )


class StructuredSchemaConfirmationResponse(ApiModel):
    status: str
    datasets: list[StructuredDatasetSchema]

    @classmethod
    def from_model(
        cls, result: StructuredConfirmationResult
    ) -> StructuredSchemaConfirmationResponse:
        return cls(
            status=result.status,
            datasets=[StructuredDatasetSchema.from_model(item) for item in result.datasets],
        )


class StructuredPublication(ApiModel):
    publication_id: str = Field(alias="publicationId")
    dataset_id: str = Field(alias="datasetId")
    schema_version: int = Field(alias="schemaVersion")
    physical_table_name: str = Field(alias="physicalTableName")
    row_count: int = Field(alias="rowCount")
    content_hash: str = Field(alias="contentHash")

    @classmethod
    def from_model(cls, publication: StructuredPublicationModel) -> StructuredPublication:
        return cls(
            publicationId=publication.publication_id,
            datasetId=publication.dataset_id,
            schemaVersion=publication.schema_version,
            physicalTableName=publication.physical_table_name,
            rowCount=publication.row_count,
            contentHash=publication.content_hash,
        )


class StructuredPublicationJob(ApiModel):
    id: str
    source_id: str = Field(alias="sourceId")
    dataset_id: str = Field(alias="datasetId")
    schema_version: int = Field(alias="schemaVersion")
    publication_id: str = Field(alias="publicationId")
    status: Literal["queued", "running", "published", "failed"]
    lease_expires_at: datetime | None = Field(alias="leaseExpiresAt")
    checkpoint_row: int = Field(alias="checkpointRow")
    attempt: int
    next_attempt_at: datetime | None = Field(alias="nextAttemptAt")
    error_message: str | None = Field(alias="errorMessage")

    @classmethod
    def from_model(cls, job: StructuredPublicationJobModel) -> StructuredPublicationJob:
        return cls(
            id=job.id,
            sourceId=job.source_id,
            datasetId=job.dataset_id,
            schemaVersion=job.schema_version,
            publicationId=job.publication_id,
            status=job.status,
            leaseExpiresAt=job.lease_expires_at,
            checkpointRow=job.checkpoint_row,
            attempt=job.attempt,
            nextAttemptAt=job.next_attempt_at,
            errorMessage=job.error_message,
        )


class StructuredPublicationEnqueueResponse(ApiModel):
    job_id: str = Field(alias="jobId")
    status: Literal["queued"]

    @classmethod
    def from_model(cls, job: StructuredPublicationJobModel) -> StructuredPublicationEnqueueResponse:
        return cls(jobId=job.id, status="queued")


class StructuredStatusResponse(ApiModel):
    source_id: str = Field(alias="sourceId")
    source_status: str = Field(alias="sourceStatus")
    job: StructuredPublicationJob
    active_publication: StructuredPublication | None = Field(alias="activePublication")

    @classmethod
    def from_model(cls, status: StructuredStatusModel) -> StructuredStatusResponse:
        return cls(
            sourceId=status.source_id,
            sourceStatus=status.source_status,
            job=StructuredPublicationJob.from_model(status.job),
            activePublication=(
                None
                if status.active_publication is None
                else StructuredPublication.from_model(status.active_publication)
            ),
        )


class KnowledgeSourceRequest(ApiModel):
    name: str = Field(min_length=1, max_length=200)
    source_type: str = Field(alias="sourceType", min_length=1, max_length=80)
    classification: str = Field(default="内部·机密", min_length=1, max_length=80)

    @field_validator("name", "source_type", "classification")
    @classmethod
    def text_fields_must_not_be_blank(cls, value: str) -> str:
        trimmed = value.strip()
        if not trimmed:
            raise ValueError("field must not be blank")
        return trimmed


class AgentStepAudit(ApiModel):
    id: str
    step_index: int = Field(alias="stepIndex")
    tool_name: str = Field(alias="toolName")
    status: Literal["completed", "failed"]
    input_summary: str = Field(alias="inputSummary")
    output_summary: str = Field(alias="outputSummary")
    source_ids: list[str] = Field(default_factory=list, alias="sourceIds")
    read_only: bool = Field(alias="readOnly")
    started_at: str = Field(alias="startedAt")
    completed_at: str = Field(alias="completedAt")

    @classmethod
    def from_model(cls, step: AgentStepModel) -> AgentStepAudit:
        return cls(
            id=step.id,
            stepIndex=step.step_index,
            toolName=step.tool_name,
            status=step.status,
            inputSummary=step.input_summary,
            outputSummary=step.output_summary,
            sourceIds=step.source_ids,
            readOnly=step.read_only,
            startedAt=normalize_display_timestamp(step.started_at),
            completedAt=normalize_display_timestamp(step.completed_at),
        )


class AgentRunAudit(ApiModel):
    id: str
    conversation_id: str = Field(alias="conversationId")
    query: str
    mode: Literal["quick", "deep", "source"]
    status: Literal["completed", "failed"]
    started_at: str = Field(alias="startedAt")
    completed_at: str = Field(alias="completedAt")
    answer_message_id: str = Field(alias="answerMessageId")
    evidence_count: int = Field(alias="evidenceCount")
    source_count: int = Field(alias="sourceCount")
    steps: list[AgentStepAudit] = Field(default_factory=list)

    @classmethod
    def from_model(cls, run: AgentRunAuditModel) -> AgentRunAudit:
        return cls(
            id=run.id,
            conversationId=run.conversation_id,
            query=run.query,
            mode=run.mode,
            status=run.status,
            startedAt=normalize_display_timestamp(run.started_at),
            completedAt=normalize_display_timestamp(run.completed_at),
            answerMessageId=run.answer_message_id,
            evidenceCount=run.evidence_count,
            sourceCount=run.source_count,
            steps=[AgentStepAudit.from_model(step) for step in run.steps],
        )


class EvaluationCaseRequest(ApiModel):
    question: str = Field(min_length=1, max_length=1000)
    expect_answer: bool = Field(default=True, alias="expectAnswer")
    expected_source_ids: list[str] = Field(default_factory=list, alias="expectedSourceIds")
    expected_terms: list[str] = Field(default_factory=list, alias="expectedTerms")
    category: str | None = None
    tags: list[str] = Field(default_factory=list)
    external_key: str | None = Field(default=None, alias="externalKey")
    import_batch_id: str | None = Field(default=None, alias="importBatchId")
    top_k: int = Field(default=5, alias="topK", ge=1, le=10)

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        trimmed = value.strip()
        if not trimmed:
            raise ValueError("question must not be blank")
        return trimmed

    @field_validator("expected_source_ids", "expected_terms")
    @classmethod
    def normalize_expected_values(cls, value: list[str]) -> list[str]:
        return normalized_unique(value)

    @model_validator(mode="after")
    def normalize_metadata(self) -> EvaluationCaseRequest:
        self.category, self.tags, self.external_key, self.import_batch_id = (
            normalize_evaluation_case_metadata(
                category=self.category,
                tags=self.tags,
                external_key=self.external_key,
                import_batch_id=self.import_batch_id,
            )
        )
        return self

    @model_validator(mode="after")
    def require_expected_evidence(self) -> EvaluationCaseRequest:
        if self.expect_answer and not self.expected_source_ids and not self.expected_terms:
            raise ValueError("at least one expected source or term is required")
        return self


class EvaluationRunRequest(ApiModel):
    case_ids: list[str] = Field(default_factory=list, alias="caseIds")

    @field_validator("case_ids")
    @classmethod
    def normalize_case_ids(cls, value: list[str]) -> list[str]:
        return normalized_unique(value)


class EvaluationBatchRequest(ApiModel):
    name: str
    case_ids: list[str] = Field(alias="caseIds")
    retrieval_min_score: float | None = Field(
        default=None,
        alias="retrievalMinScore",
    )

    @model_validator(mode="before")
    @classmethod
    def require_name_and_cases(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        name = value.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ValueError("批次名称不能为空")
        case_ids = value.get(
            "caseIds",
            value.get("case_ids"),
        )
        if not isinstance(case_ids, list) or not case_ids:
            raise ValueError("评测用例不能为空")
        if any(not isinstance(case_id, str) for case_id in case_ids):
            raise ValueError("评测用例 ID 必须是字符串")
        if not any(case_id.strip() for case_id in case_ids):
            raise ValueError("评测用例不能为空")

        retrieval_min_score = value.get(
            "retrievalMinScore",
            value.get("retrieval_min_score"),
        )
        if retrieval_min_score is not None:
            if isinstance(retrieval_min_score, bool):
                raise ValueError("检索阈值必须是数值")
            try:
                float(retrieval_min_score)
            except (TypeError, ValueError):
                raise ValueError("检索阈值必须是数值") from None
        return value

    @field_validator("name")
    @classmethod
    def normalize_name(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("批次名称不能为空")
        if len(normalized) > 120:
            raise ValueError("批次名称不能超过 120 个字符")
        return normalized

    @field_validator("case_ids")
    @classmethod
    def normalize_batch_case_ids(cls, value: list[str]) -> list[str]:
        normalized = normalized_unique(value)
        if not normalized:
            raise ValueError("评测用例不能为空")
        return normalized

    @field_validator("retrieval_min_score")
    @classmethod
    def validate_retrieval_min_score(cls, value: float | None) -> float | None:
        if value is not None and (not math.isfinite(value) or value < 0):
            raise ValueError("检索阈值必须是大于等于 0 的有限数")
        return value


class EvaluationCase(ApiModel):
    id: str
    question: str
    expected_source_ids: list[str] = Field(alias="expectedSourceIds")
    expected_terms: list[str] = Field(alias="expectedTerms")
    category: str | None = None
    tags: list[str] = Field(default_factory=list)
    external_key: str | None = Field(default=None, alias="externalKey")
    import_batch_id: str | None = Field(default=None, alias="importBatchId")
    expect_answer: bool = Field(alias="expectAnswer")
    top_k: int = Field(alias="topK")
    created_at: str = Field(alias="createdAt")
    updated_at: str = Field(alias="updatedAt")

    @classmethod
    def from_model(cls, case: EvaluationCaseModel) -> EvaluationCase:
        return cls(
            id=case.id,
            question=case.question,
            expectedSourceIds=case.expected_source_ids,
            expectedTerms=case.expected_terms,
            category=case.category,
            tags=case.tags,
            externalKey=case.external_key,
            importBatchId=case.import_batch_id,
            expectAnswer=case.expect_answer,
            topK=case.top_k,
            createdAt=normalize_display_timestamp(case.created_at),
            updatedAt=normalize_display_timestamp(case.updated_at),
        )


class EvaluationCaseCollection(ApiModel):
    items: list[EvaluationCase]
    categories: list[str]
    tags: list[str]
    total: int

    @classmethod
    def from_models(
        cls,
        items: list[EvaluationCaseModel],
        facets: EvaluationCaseFacets,
    ) -> EvaluationCaseCollection:
        return cls(
            items=[EvaluationCase.from_model(case) for case in items],
            categories=facets.categories,
            tags=facets.tags,
            total=len(items),
        )


class EvaluationHit(ApiModel):
    rank: int
    source_id: str = Field(alias="sourceId")
    source_name: str = Field(alias="sourceName")
    chunk_id: str = Field(alias="chunkId")
    chunk_index: int = Field(alias="chunkIndex")
    score: float
    keyword_score: float = Field(alias="keywordScore")
    vector_score: float = Field(alias="vectorScore")
    matched_terms: list[str] = Field(default_factory=list, alias="matchedTerms")
    excerpt: str

    @classmethod
    def from_model(cls, hit: EvaluationHitModel) -> EvaluationHit:
        return cls(
            rank=hit.rank,
            sourceId=hit.source_id,
            sourceName=hit.source_name,
            chunkId=hit.chunk_id,
            chunkIndex=hit.chunk_index,
            score=hit.score,
            keywordScore=hit.keyword_score,
            vectorScore=hit.vector_score,
            matchedTerms=hit.matched_terms,
            excerpt=hit.excerpt,
        )


class EvaluationRun(ApiModel):
    id: str
    case_id: str = Field(alias="caseId")
    batch_id: str | None = Field(default=None, alias="batchId")
    question: str
    status: Literal["passed", "failed"]
    expect_answer: bool = Field(alias="expectAnswer")
    answerable: bool
    false_positive: bool = Field(alias="falsePositive")
    expected_source_ids: list[str] = Field(alias="expectedSourceIds")
    matched_source_ids: list[str] = Field(alias="matchedSourceIds")
    missing_source_ids: list[str] = Field(alias="missingSourceIds")
    expected_terms: list[str] = Field(alias="expectedTerms")
    found_terms: list[str] = Field(alias="foundTerms")
    missing_terms: list[str] = Field(alias="missingTerms")
    source_recall: float = Field(alias="sourceRecall")
    term_recall: float = Field(alias="termRecall")
    top_score: float = Field(alias="topScore")
    hit_count: int = Field(alias="hitCount")
    failure_reasons: list[EvaluationFailureReason] = Field(
        default_factory=list,
        alias="failureReasons",
    )
    started_at: str = Field(alias="startedAt")
    completed_at: str = Field(alias="completedAt")
    hits: list[EvaluationHit] = Field(default_factory=list)

    @classmethod
    def from_model(cls, run: EvaluationRunModel) -> EvaluationRun:
        return cls(
            id=run.id,
            caseId=run.case_id,
            batchId=run.batch_id,
            question=run.question,
            status=run.status,
            expectAnswer=run.expect_answer,
            answerable=run.answerable,
            falsePositive=run.false_positive,
            expectedSourceIds=run.expected_source_ids,
            matchedSourceIds=run.matched_source_ids,
            missingSourceIds=run.missing_source_ids,
            expectedTerms=run.expected_terms,
            foundTerms=run.found_terms,
            missingTerms=run.missing_terms,
            sourceRecall=run.source_recall,
            termRecall=run.term_recall,
            topScore=run.top_score,
            hitCount=run.hit_count,
            failureReasons=evaluation_failure_reasons(run),
            startedAt=normalize_display_timestamp(run.started_at),
            completedAt=normalize_display_timestamp(run.completed_at),
            hits=[EvaluationHit.from_model(hit) for hit in run.hits],
        )


class EvaluationBatch(ApiModel):
    id: str
    name: str
    status: Literal["queued", "running", "completed", "failed"]
    case_ids: list[str] = Field(alias="caseIds")
    retrieval_min_score: float = Field(alias="retrievalMinScore")
    case_count: int = Field(alias="caseCount")
    completed_count: int = Field(alias="completedCount")
    passed_count: int = Field(alias="passedCount")
    failed_count: int = Field(alias="failedCount")
    false_positive_count: int = Field(alias="falsePositiveCount")
    started_at: str = Field(alias="startedAt")
    completed_at: str | None = Field(default=None, alias="completedAt")
    error_message: str | None = Field(default=None, alias="errorMessage")

    @classmethod
    def from_model(cls, batch: EvaluationBatchModel) -> EvaluationBatch:
        return cls(
            id=batch.id,
            name=batch.name,
            status=batch.status,
            caseIds=batch.case_ids,
            retrievalMinScore=batch.retrieval_min_score,
            caseCount=batch.case_count,
            completedCount=batch.completed_count,
            passedCount=batch.passed_count,
            failedCount=batch.failed_count,
            falsePositiveCount=batch.false_positive_count,
            startedAt=normalize_display_timestamp(batch.started_at),
            completedAt=(
                normalize_display_timestamp(batch.completed_at)
                if batch.completed_at is not None
                else None
            ),
            errorMessage=batch.error_message,
        )


class EvaluationMetricGroup(ApiModel):
    name: str
    total: int
    passed: int
    pass_rate: float = Field(alias="passRate")

    @classmethod
    def from_model(cls, group: EvaluationMetricGroupModel) -> EvaluationMetricGroup:
        return cls(
            name=group.name,
            total=group.total,
            passed=group.passed,
            passRate=group.pass_rate,
        )


class EvaluationBatchSummary(ApiModel):
    total: int
    passed: int
    failed: int
    pass_rate: float = Field(alias="passRate")
    answer_pass_rate: float = Field(alias="answerPassRate")
    no_answer_accuracy: float = Field(alias="noAnswerAccuracy")
    false_positive_count: int = Field(alias="falsePositiveCount")
    false_positive_rate: float = Field(alias="falsePositiveRate")
    average_source_recall: float = Field(alias="averageSourceRecall")
    average_term_recall: float = Field(alias="averageTermRecall")
    average_top_score: float = Field(alias="averageTopScore")
    maximum_top_score: float = Field(alias="maximumTopScore")
    category_breakdown: list[EvaluationMetricGroup] = Field(alias="categoryBreakdown")
    tag_breakdown: list[EvaluationMetricGroup] = Field(alias="tagBreakdown")

    @classmethod
    def from_model(
        cls,
        summary: EvaluationBatchSummaryModel,
    ) -> EvaluationBatchSummary:
        return cls(
            total=summary.total,
            passed=summary.passed,
            failed=summary.failed,
            passRate=summary.pass_rate,
            answerPassRate=summary.answer_pass_rate,
            noAnswerAccuracy=summary.no_answer_accuracy,
            falsePositiveCount=summary.false_positive_count,
            falsePositiveRate=summary.false_positive_rate,
            averageSourceRecall=summary.average_source_recall,
            averageTermRecall=summary.average_term_recall,
            averageTopScore=summary.average_top_score,
            maximumTopScore=summary.maximum_top_score,
            categoryBreakdown=[
                EvaluationMetricGroup.from_model(group) for group in summary.category_breakdown
            ],
            tagBreakdown=[
                EvaluationMetricGroup.from_model(group) for group in summary.tag_breakdown
            ],
        )


class EvaluationBatchMetricDelta(ApiModel):
    total: int
    passed: int
    failed: int
    pass_rate: float = Field(alias="passRate")
    answer_pass_rate: float = Field(alias="answerPassRate")
    no_answer_accuracy: float = Field(alias="noAnswerAccuracy")
    false_positive_count: int = Field(alias="falsePositiveCount")
    false_positive_rate: float = Field(alias="falsePositiveRate")
    average_source_recall: float = Field(alias="averageSourceRecall")
    average_term_recall: float = Field(alias="averageTermRecall")
    average_top_score: float = Field(alias="averageTopScore")
    maximum_top_score: float = Field(alias="maximumTopScore")

    @classmethod
    def from_model(
        cls,
        delta: EvaluationBatchMetricDeltaModel,
    ) -> EvaluationBatchMetricDelta:
        return cls(
            total=delta.total,
            passed=delta.passed,
            failed=delta.failed,
            passRate=delta.pass_rate,
            answerPassRate=delta.answer_pass_rate,
            noAnswerAccuracy=delta.no_answer_accuracy,
            falsePositiveCount=delta.false_positive_count,
            falsePositiveRate=delta.false_positive_rate,
            averageSourceRecall=delta.average_source_recall,
            averageTermRecall=delta.average_term_recall,
            averageTopScore=delta.average_top_score,
            maximumTopScore=delta.maximum_top_score,
        )


class EvaluationBatchComparison(ApiModel):
    left_batch_id: str = Field(alias="leftBatchId")
    right_batch_id: str = Field(alias="rightBatchId")
    metric_delta: EvaluationBatchMetricDelta = Field(alias="metricDelta")
    shared_case_count: int = Field(alias="sharedCaseCount")
    improved_case_ids: list[str] = Field(alias="improvedCaseIds")
    regressed_case_ids: list[str] = Field(alias="regressedCaseIds")
    left_only_case_ids: list[str] = Field(alias="leftOnlyCaseIds")
    right_only_case_ids: list[str] = Field(alias="rightOnlyCaseIds")

    @classmethod
    def from_model(
        cls,
        comparison: EvaluationBatchComparisonModel,
    ) -> EvaluationBatchComparison:
        return cls(
            leftBatchId=comparison.left_batch_id,
            rightBatchId=comparison.right_batch_id,
            metricDelta=EvaluationBatchMetricDelta.from_model(comparison.metric_delta),
            sharedCaseCount=comparison.shared_case_count,
            improvedCaseIds=comparison.improved_case_ids,
            regressedCaseIds=comparison.regressed_case_ids,
            leftOnlyCaseIds=comparison.left_only_case_ids,
            rightOnlyCaseIds=comparison.right_only_case_ids,
        )


class EvaluationBatchDetail(EvaluationBatch):
    summary: EvaluationBatchSummary
    runs: list[EvaluationRun]
    cases: list[EvaluationCase]

    @classmethod
    def from_models(
        cls,
        batch: EvaluationBatchModel,
        summary: EvaluationBatchSummaryModel,
        runs: list[EvaluationRunModel],
        cases: list[EvaluationCaseModel],
    ) -> EvaluationBatchDetail:
        return cls(
            **EvaluationBatch.from_model(batch).model_dump(by_alias=True),
            summary=EvaluationBatchSummary.from_model(summary),
            runs=[EvaluationRun.from_model(run) for run in runs],
            cases=[EvaluationCase.from_model(case) for case in cases],
        )


class EvaluationDashboard(ApiModel):
    cases: list[EvaluationCase]
    runs: list[EvaluationRun]

    @classmethod
    def from_models(
        cls,
        cases: list[EvaluationCaseModel],
        runs: list[EvaluationRunModel],
    ) -> EvaluationDashboard:
        return cls(
            cases=[EvaluationCase.from_model(case) for case in cases],
            runs=[EvaluationRun.from_model(run) for run in runs],
        )


class EvaluationImportRowResponse(ApiModel):
    row_number: int = Field(alias="rowNumber")
    question: str
    expect_answer: bool = Field(alias="expectAnswer")
    expected_source_ids: list[str] = Field(alias="expectedSourceIds")
    expected_terms: list[str] = Field(alias="expectedTerms")
    category: str | None = None
    tags: list[str] = Field(default_factory=list)
    top_k: int = Field(alias="topK")
    external_key: str | None = Field(default=None, alias="externalKey")

    @classmethod
    def from_model(cls, row: EvaluationImportRow) -> EvaluationImportRowResponse:
        return cls(
            rowNumber=row.row_number,
            question=row.question,
            expectAnswer=row.expect_answer,
            expectedSourceIds=row.expected_source_ids,
            expectedTerms=row.expected_terms,
            category=row.category,
            tags=row.tags,
            topK=row.top_k,
            externalKey=row.external_key,
        )


class EvaluationImportErrorResponse(ApiModel):
    row_number: int = Field(alias="rowNumber")
    field: str
    message: str

    @classmethod
    def from_model(
        cls,
        error: EvaluationImportErrorModel,
    ) -> EvaluationImportErrorResponse:
        return cls(
            rowNumber=error.row_number,
            field=error.field,
            message=error.message,
        )


class EvaluationImportPreviewResponse(ApiModel):
    preview_token: str = Field(alias="previewToken")
    file_name: str = Field(alias="fileName")
    total_rows: int = Field(alias="totalRows")
    valid_rows: int = Field(alias="validRows")
    invalid_rows: int = Field(alias="invalidRows")
    duplicate_rows: int = Field(alias="duplicateRows")
    rows: list[EvaluationImportRowResponse]
    errors: list[EvaluationImportErrorResponse]
    duplicate_keys: list[str] = Field(alias="duplicateKeys")

    @classmethod
    def from_model(
        cls,
        preview: EvaluationImportPreview,
    ) -> EvaluationImportPreviewResponse:
        return cls(
            previewToken=preview.token,
            fileName=preview.file_name,
            totalRows=preview.total_rows,
            validRows=preview.valid_rows,
            invalidRows=preview.invalid_rows,
            duplicateRows=preview.duplicate_rows,
            rows=[EvaluationImportRowResponse.from_model(row) for row in preview.rows],
            errors=[EvaluationImportErrorResponse.from_model(error) for error in preview.errors],
            duplicateKeys=preview.duplicate_keys,
        )


class EvaluationImportConfirmRequest(ApiModel):
    preview_token: str = Field(alias="previewToken")

    @model_validator(mode="before")
    @classmethod
    def preview_token_must_not_be_blank(cls, value: object) -> object:
        if not isinstance(value, dict):
            raise ValueError("预览令牌不能为空")
        token_key = (
            "previewToken"
            if "previewToken" in value
            else "preview_token"
            if "preview_token" in value
            else None
        )
        token = value.get(token_key) if token_key is not None else None
        if not isinstance(token, str) or not token.strip():
            raise ValueError("预览令牌不能为空")
        normalized = dict(value)
        normalized[token_key] = token.strip()
        return normalized


class EvaluationImportConfirmResponse(ApiModel):
    import_batch_id: str = Field(alias="importBatchId")
    created_count: int = Field(alias="createdCount")
    duplicate_count: int = Field(alias="duplicateCount")
    dashboard: EvaluationDashboard
