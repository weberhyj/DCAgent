# Word Factual QA Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract high-confidence field facts from DOCX files and answer exact questions such as “张三几岁” with only the requested field, while open-ended Word questions continue to use the existing Embedding + BGE Reranker + LLM path.

**Architecture:** DOCX ingestion produces ordinary retrieval chunks plus a small source-scoped `knowledge_facts` index. A deterministic intent parser resolves one entity and one canonical field; a permission-filtered repository query returns only exact fact matches; a template answer validator prevents adjacent fields from leaking into the response. Recognized factual questions that are missing or conflicting terminate inside this path and never fall back to unrelated RAG evidence.

**Tech Stack:** Python 3.12, `python-docx`, SQLAlchemy/Alembic, PostgreSQL and SQLite tests, FastAPI repository injection, Python `unittest`, uv.

## Global Constraints

- Preserve `parse_knowledge_file(path, source_id, source_type) -> list[KnowledgeChunkModel]` for existing callers.
- Extract facts only from explicit key/value records, heading-plus-field records, and tables with one recognized entity column.
- Narrative prose remains a normal RAG chunk and is never promoted to a fact by guessing.
- Canonical field aliases include `几岁/多大/岁数 -> 年龄`, `男女 -> 性别`, and `什么职务/职位/岗位/担任什么 -> 职务`.
- A factual question must resolve exactly one entity and one field; ambiguity returns clarification and never selects the first candidate.
- A factual answer may contain the requested field and its value only. It may not include gender, job, age, or another field that the user did not request.
- Factual answers do not call Embedding, BGE Reranker, `inspect_document`, or the LLM.
- `介绍张三` and other open questions return `None` from the factual service and continue to normal document RAG.
- Facts are queryable only from indexed sources within the configured permission tags.
- Reindexing replaces a source's chunks and facts together. Existing Word sources must be re-parsed after deployment; Excel publications are unaffected.
- Duplicate identical facts inside one source are deduplicated. Matches from more than one source for the same entity/field are treated as a possible same-name entity and require source clarification, even when values happen to match.

---

## File map

- `backend/app/word_facts.py`: fact contracts, aliases, normalization, factual intent parsing, conflict rules, and answer validation.
- `backend/app/docx_parser.py`: block-aware DOCX parsing and high-confidence fact extraction.
- `backend/app/text_parser.py`: parse-bundle dispatch while retaining the legacy chunk-only function.
- `backend/app/models.py`: in-memory fact storage on `ChatState`.
- `backend/app/database.py`: `KnowledgeFactRecord`.
- `backend/alembic/versions/20260811_07_word_facts.py`: fact table and lookup indexes.
- `backend/app/repository.py` and `backend/app/sql_repository.py`: fact replacement/query methods.
- `backend/app/ingestion.py`: pass parsed facts into source completion.
- `backend/app/word_fact_answer.py`: exact field-answer service.
- `backend/app/main.py`: production service wiring.
- `backend/tests/test_docx_parser.py`: DOCX structure and extraction coverage.
- `backend/tests/test_word_facts.py`: intent, normalization, conflict, and validator coverage.
- `backend/tests/test_word_fact_answer.py`: route and answer coverage.
- `backend/tests/test_sql_repository.py`: persistence and permission coverage.
- `backend/tests/test_knowledge_ingestion_pipeline.py`: ingestion/reindex replacement coverage.
- `backend/tests/test_alembic_baseline.py`, `backend/tests/test_migration_entrypoint.py`, `backend/tests/test_lazy_startup.py`: migration-head coverage.

### Task 1: Add fact contracts and the durable index

**Files:**
- Create: `backend/app/word_facts.py`
- Modify: `backend/app/models.py`
- Modify: `backend/app/database.py`
- Create: `backend/alembic/versions/20260811_07_word_facts.py`
- Create: `backend/tests/test_word_facts.py`
- Modify: `backend/tests/test_alembic_baseline.py`
- Modify: `backend/tests/test_migration_entrypoint.py`
- Modify: `backend/tests/test_lazy_startup.py`

**Interfaces:**
- Consumes: `KnowledgeSourceRecord`, `KnowledgeChunkRecord`, `KnowledgeChunkModel`, and existing permission-tag conventions.
- Produces: `KnowledgeFactModel`, `WordFactMatch`, `WordFactualIntent`, `WordFactRepository`, and `knowledge_facts`.

- [ ] **Step 1: Write failing contracts and migration tests**

```python
def test_fact_normalization_keeps_display_values(self) -> None:
    fact = KnowledgeFactModel.create(
        id="fact-1",
        source_id="kb-people",
        chunk_id="chunk-1",
        entity=" 张三 ",
        field="年龄",
        value="28岁",
        confidence=0.98,
        locator={"paragraph": 3},
    )
    self.assertEqual(fact.entity, "张三")
    self.assertEqual(fact.entity_normalized, "张三")
    self.assertEqual(fact.field_normalized, "年龄")

def test_head_migration_creates_fact_lookup_indexes(self) -> None:
    database = Database("sqlite+pysqlite:///:memory:")
    database.create_schema()
    inspector = inspect(database.engine)
    self.assertIn("knowledge_facts", inspector.get_table_names())
    indexes = {item["name"] for item in inspector.get_indexes("knowledge_facts")}
    self.assertIn("ix_knowledge_facts_entity_field", indexes)
    self.assertIn("ix_knowledge_facts_source_id", indexes)
    self.assertIn("ix_knowledge_facts_chunk_id", indexes)

def test_fact_foreign_keys_cascade_to_source_and_chunk(self) -> None:
    database = Database("sqlite+pysqlite:///:memory:")
    database.create_schema()
    foreign_keys = {
        (
            tuple(item["constrained_columns"]),
            item["referred_table"],
            tuple(item["referred_columns"]),
            item["options"].get("ondelete"),
        )
        for item in inspect(database.engine).get_foreign_keys("knowledge_facts")
    }
    self.assertEqual(
        {
            (("source_id",), "knowledge_sources", ("id",), "CASCADE"),
            (("chunk_id",), "knowledge_chunks", ("id",), "CASCADE"),
        },
        foreign_keys,
    )
```

- [ ] **Step 2: Run tests and verify RED**

```powershell
Push-Location backend
uv run --project . --group dev python -m unittest tests.test_word_facts tests.test_alembic_baseline -v
Pop-Location
```

Expected: FAIL because the contracts, ORM record, and migration do not exist.

- [ ] **Step 3: Implement exact contracts and schema**

Define:

```python
@dataclass(frozen=True, slots=True)
class KnowledgeFactModel:
    id: str
    source_id: str
    chunk_id: str
    entity: str
    entity_normalized: str
    field: str
    field_normalized: str
    value: str
    confidence: float
    locator: Mapping[str, object]

    @classmethod
    def create(
        cls,
        *,
        id: str,
        source_id: str,
        chunk_id: str,
        entity: str,
        field: str,
        value: str,
        confidence: float,
        locator: Mapping[str, object],
    ) -> KnowledgeFactModel:
        clean_entity = _bounded_display(entity, "entity", 240)
        clean_field = canonical_fact_field(field)
        clean_value = _bounded_display(value, "value", 2000)
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")
        return cls(
            id=id,
            source_id=_required_identifier(source_id),
            chunk_id=_required_identifier(chunk_id),
            entity=clean_entity,
            entity_normalized=normalize_fact_key(clean_entity),
            field=clean_field,
            field_normalized=normalize_fact_key(clean_field),
            value=clean_value,
            confidence=float(confidence),
            locator=dict(locator),
        )

@dataclass(frozen=True, slots=True)
class WordFactMatch:
    fact: KnowledgeFactModel
    source_name: str
    classification: str

@dataclass(frozen=True, slots=True)
class WordFactualIntent:
    entity: str
    entity_normalized: str
    field: str
    field_normalized: str

class WordFactRepository(Protocol):
    def replace_knowledge_facts(
        self, source_id: str, facts: Sequence[KnowledgeFactModel]
    ) -> None:
        raise NotImplementedError

    def find_knowledge_facts(
        self,
        intent: WordFactualIntent,
        *,
        permission_tags: Sequence[str] = (),
    ) -> list[WordFactMatch]:
        raise NotImplementedError
```

`normalize_fact_key()` uses Unicode NFKC normalization, case folding, and removal of whitespace/punctuation. `canonical_fact_field()` resolves only configured aliases and rejects unknown fields.

Add `KnowledgeFactRecord` with `String(64)` ID/source/chunk keys, `String(240)` entity keys, `String(120)` field keys, `Text` value, `Float` confidence, and non-null `JSON` locator. Add source/chunk cascade foreign keys and a composite non-unique index on `(entity_normalized, field_normalized)`.

Create migration `20260811_07` with `down_revision = "20260730_06"`. Update the expected migration head from `20260730_06` to `20260811_07` in the three migration/startup test files.

- [ ] **Step 4: Run migration tests and commit**

```powershell
Push-Location backend
uv run --project . --group dev python -m unittest tests.test_word_facts tests.test_alembic_baseline tests.test_migration_entrypoint tests.test_lazy_startup -v
Pop-Location
git add backend/app/word_facts.py backend/app/models.py backend/app/database.py backend/alembic/versions/20260811_07_word_facts.py backend/tests/test_word_facts.py backend/tests/test_alembic_baseline.py backend/tests/test_migration_entrypoint.py backend/tests/test_lazy_startup.py
git commit -m "feat: add durable Word fact index"
```

### Task 2: Parse DOCX blocks and extract high-confidence facts

**Files:**
- Create: `backend/app/docx_parser.py`
- Modify: `backend/app/text_parser.py`
- Create: `backend/tests/test_docx_parser.py`
- Modify: `backend/tests/test_knowledge_ingestion_pipeline.py`

**Interfaces:**
- Consumes: DOCX `Path`, source ID, `KnowledgeChunkModel`, and `KnowledgeFactModel`.
- Produces: `KnowledgeParseResult`, `parse_docx_knowledge_file()`, and `parse_knowledge_file_result()`.

- [ ] **Step 1: Write failing DOCX extraction tests**

```python
def test_inline_record_extracts_only_known_fields(self) -> None:
    path = write_docx(
        self.temp_dir / "people.docx",
        paragraphs=["姓名：张三，年龄：28岁，性别：女，职务：工程师"],
    )
    result = parse_docx_knowledge_file(path, source_id="kb-people")
    self.assertEqual(
        {(item.entity, item.field, item.value) for item in result.facts},
        {
            ("张三", "年龄", "28岁"),
            ("张三", "性别", "女"),
            ("张三", "职务", "工程师"),
        },
    )

def test_table_row_extracts_fields_and_locator(self) -> None:
    path = write_docx_table(
        self.temp_dir / "staff.docx",
        headers=["姓名", "年龄", "性别", "职务"],
        rows=[["李四", "31岁", "男", "产品经理"]],
    )
    result = parse_docx_knowledge_file(path, source_id="kb-staff")
    age = next(item for item in result.facts if item.field == "年龄")
    self.assertEqual(age.entity, "李四")
    self.assertEqual(age.locator, {"table": 0, "row": 1, "column": 1})
    self.assertEqual(age.confidence, 0.99)

def test_heading_entity_applies_to_following_key_value_lines(self) -> None:
    path = write_docx(
        self.temp_dir / "heading.docx",
        paragraphs=[("张三", "Heading 1"), "年龄：28岁", "性别：女"],
    )
    result = parse_docx_knowledge_file(path, source_id="kb-heading")
    self.assertEqual(
        {(item.field, item.value) for item in result.facts},
        {("年龄", "28岁"), ("性别", "女")},
    )

def test_narrative_sentence_remains_chunk_only(self) -> None:
    path = write_docx(
        self.temp_dir / "narrative.docx",
        paragraphs=["张三是一名工程师，今年二十八岁，性格开朗。"],
    )
    result = parse_docx_knowledge_file(path, source_id="kb-narrative")
    self.assertEqual(result.facts, ())
    self.assertIn("二十八岁", "\n".join(item.text for item in result.chunks))

def test_legacy_parse_function_still_returns_list_of_chunks(self) -> None:
    path = write_docx(
        self.temp_dir / "compat.docx",
        paragraphs=["姓名：张三，年龄：28岁"],
    )
    chunks = parse_knowledge_file(path, "kb-compat", "文档")
    self.assertIsInstance(chunks, list)
    self.assertGreater(len(chunks), 0)
```

- [ ] **Step 2: Run tests and verify RED**

```powershell
Push-Location backend
uv run --project . --group dev python -m unittest tests.test_docx_parser tests.test_knowledge_ingestion_pipeline -v
Pop-Location
```

Expected: FAIL because block-aware parsing does not exist.

- [ ] **Step 3: Implement the parse bundle and extraction rules**

Add:

```python
@dataclass(frozen=True, slots=True)
class KnowledgeParseResult:
    chunks: tuple[KnowledgeChunkModel, ...]
    facts: tuple[KnowledgeFactModel, ...]

def parse_docx_knowledge_file(path: Path, source_id: str) -> KnowledgeParseResult:
    blocks = read_docx_blocks(path)
    chunks, block_chunk_ids = chunk_docx_blocks(source_id, blocks)
    facts = extract_docx_facts(source_id, blocks, block_chunk_ids)
    return KnowledgeParseResult(chunks=chunks, facts=facts)

def parse_knowledge_file_result(
    path: Path, source_id: str, source_type: str
) -> KnowledgeParseResult:
    if path.suffix.lower() == ".docx":
        return parse_docx_knowledge_file(path, source_id)
    return KnowledgeParseResult(
        chunks=tuple(chunk_text(source_id, extract_text(path, source_type))),
        facts=(),
    )
```

Read paragraphs with style names and tables in document order. Preserve locators `paragraph` or `table/row/column` in chunk metadata. Keep a table row intact; split a single over-600-character block with the current overlap rules.

Use these centralized aliases:

```python
FACT_FIELD_ALIASES = {
    "年龄": ("年龄", "岁数", "几岁", "多大"),
    "性别": ("性别", "男女"),
    "职务": ("职务", "职位", "岗位", "担任"),
}
FACT_ENTITY_ALIASES = ("姓名", "人员", "员工", "人物", "名称")
```

Extract facts only for:

1. A paragraph containing one entity key/value and one or more recognized field key/values separated by `，,；;` or line breaks.
2. A heading-style paragraph followed by recognized field key/value paragraphs until the next heading.
3. A table whose first row contains exactly one entity alias column; recognized field headers create facts for each non-empty row cell.

Set confidence to `0.99` for table rows, `0.97` for inline records, and `0.95` for heading records. Stop values at the next delimiter and reject blank values. Generate stable fact IDs from SHA-256 of source ID, chunk ID, entity key, field key, and value. Do not extract from narrative prose.

Keep `parse_knowledge_file()` as:

```python
def parse_knowledge_file(
    path: Path, source_id: str, source_type: str
) -> list[KnowledgeChunkModel]:
    return list(parse_knowledge_file_result(path, source_id, source_type).chunks)
```

- [ ] **Step 4: Run parser tests and commit**

```powershell
Push-Location backend
uv run --project . --group dev python -m unittest tests.test_docx_parser tests.test_knowledge_ingestion_pipeline -v
Pop-Location
git add backend/app/docx_parser.py backend/app/text_parser.py backend/tests/test_docx_parser.py backend/tests/test_knowledge_ingestion_pipeline.py
git commit -m "feat: extract high-confidence DOCX facts"
```

### Task 3: Persist, replace, and permission-filter facts

**Files:**
- Modify: `backend/app/models.py`
- Modify: `backend/app/repository.py`
- Modify: `backend/app/sql_repository.py`
- Modify: `backend/app/ingestion.py`
- Modify: `backend/tests/test_sql_repository.py`
- Modify: `backend/tests/test_knowledge_ingestion_pipeline.py`

**Interfaces:**
- Consumes: `KnowledgeParseResult`, `WordFactRepository`, source status, and retrieval permission tags.
- Produces: atomic fact replacement and exact source-scoped fact queries.

- [ ] **Step 1: Write failing repository tests**

```python
def test_complete_indexing_replaces_old_source_facts(self) -> None:
    self.seed_indexed_source("kb-people")
    first = KnowledgeFactModel.create(
        id="fact-old",
        source_id="kb-people",
        chunk_id="chunk-old",
        entity="张三",
        field="年龄",
        value="27岁",
        confidence=0.97,
        locator={"paragraph": 1},
    )
    self.repository.complete_knowledge_source_indexing(
        "kb-people", [chunk("chunk-old")], facts=[first]
    )
    second = replace(first, id="fact-new", chunk_id="chunk-new", value="28岁")
    self.repository.complete_knowledge_source_indexing(
        "kb-people", [chunk("chunk-new")], facts=[second]
    )
    matches = self.repository.find_knowledge_facts(
        WordFactualIntent("张三", "张三", "年龄", "年龄")
    )
    self.assertEqual([item.fact.value for item in matches], ["28岁"])

def test_query_requires_indexed_permitted_source(self) -> None:
    self.seed_fact("kb-private", status="解析中", classification="内部·机密")
    self.seed_fact("kb-public", status="已索引", classification="公开")
    intent = WordFactualIntent("张三", "张三", "年龄", "年龄")
    self.assertEqual(
        self.repository.find_knowledge_facts(
            intent, permission_tags=("内部·机密",)
        ),
        [],
    )
    matches = self.repository.find_knowledge_facts(
        intent, permission_tags=("公开",)
    )
    self.assertEqual([item.fact.source_id for item in matches], ["kb-public"])

def test_docx_queue_persists_facts_before_qdrant_publication(self) -> None:
    queue = KnowledgeIngestionQueue(
        self.repository,
        index_lifecycle=RecordingLifecycle(self.events),
    )
    queue.process("kb-people", self.docx_path, "文档")
    self.assertEqual(self.events, ["postgres-facts", "qdrant-upsert"])
```

- [ ] **Step 2: Run tests and verify RED**

```powershell
Push-Location backend
uv run --project . --group dev python -m unittest tests.test_sql_repository tests.test_knowledge_ingestion_pipeline -v
Pop-Location
```

Expected: FAIL because repositories accept chunks only.

- [ ] **Step 3: Implement replacement/query behavior**

Add `knowledge_facts_by_source: dict[str, list[KnowledgeFactModel]]` to `ChatState` and extend the repository protocol:

```python
def complete_knowledge_source_indexing(
    self,
    source_id: str,
    chunks: list[KnowledgeChunkModel],
    *,
    facts: Sequence[KnowledgeFactModel] = (),
) -> KnowledgeSourceModel:
    raise NotImplementedError

def replace_knowledge_facts(
    self, source_id: str, facts: Sequence[KnowledgeFactModel]
) -> None:
    raise NotImplementedError

def find_knowledge_facts(
    self,
    intent: WordFactualIntent,
    *,
    permission_tags: Sequence[str] = (),
) -> list[WordFactMatch]:
    raise NotImplementedError
```

In-memory replacement occurs under the current lock and clears facts during source failure, reindex, or deletion.

SQL replacement uses the same transaction as chunk replacement: delete old source facts, delete old chunks, insert embedded chunks, validate every fact's source/chunk belongs to the incoming bundle, insert facts, and mark the source indexed. `find_knowledge_facts()` joins `knowledge_facts` to `knowledge_sources`, requires `status = '已索引'`, applies `classification.in_(permission_tags)` when tags are non-empty, exact-matches both normalized keys, and orders by source name then fact ID.

Update `KnowledgeIngestionQueue._process()` to call `parse_knowledge_file_result()` and pass both chunks and facts. The existing retrieval publication callback remains after PostgreSQL completion.

- [ ] **Step 4: Run persistence tests and commit**

```powershell
Push-Location backend
uv run --project . --group dev python -m unittest tests.test_sql_repository tests.test_knowledge_ingestion_pipeline tests.test_knowledge_upload -v
Pop-Location
git add backend/app/models.py backend/app/repository.py backend/app/sql_repository.py backend/app/ingestion.py backend/tests/test_sql_repository.py backend/tests/test_knowledge_ingestion_pipeline.py
git commit -m "feat: persist and scope Word facts"
```

### Task 4: Resolve exact questions and produce bounded answers

**Files:**
- Modify: `backend/app/word_facts.py`
- Create: `backend/app/word_fact_answer.py`
- Modify: `backend/tests/test_word_facts.py`
- Create: `backend/tests/test_word_fact_answer.py`

**Interfaces:**
- Consumes: `WordFactRepository.find_knowledge_facts()` and normal Agent result/message contracts.
- Produces: `resolve_word_factual_intent()`, `validate_word_fact_answer()`, and `WordFactAnswerService.try_answer()`.

- [ ] **Step 1: Write failing intent and answer tests**

```python
def test_age_and_job_aliases_resolve_canonical_fields(self) -> None:
    self.assertEqual(
        resolve_word_factual_intent("张三几岁"),
        WordFactualIntent("张三", "张三", "年龄", "年龄"),
    )
    job = resolve_word_factual_intent("请问张三担任什么职位？")
    self.assertIsInstance(job, WordFactualIntent)
    assert isinstance(job, WordFactualIntent)
    self.assertEqual(job.field, "职务")

def test_open_introduction_is_not_factual(self) -> None:
    self.assertIsNone(resolve_word_factual_intent("介绍张三"))

def test_age_answer_contains_no_gender_or_job(self) -> None:
    service = WordFactAnswerService(FakeFacts([match("张三", "年龄", "28岁")]))
    result = service.try_answer("conv-1", "张三几岁", "quick", [])
    self.assertIsNotNone(result)
    assert result is not None
    text = result.reply.paragraphs[0].text
    self.assertEqual(text, "张三的年龄是28岁。")
    self.assertNotIn("性别", text)
    self.assertNotIn("职务", text)

def test_conflicting_values_return_source_clarification(self) -> None:
    service = WordFactAnswerService(
        FakeFacts(
            [
                match("张三", "年龄", "28岁", source_id="kb-a"),
                match("张三", "年龄", "29岁", source_id="kb-b"),
            ]
        )
    )
    result = service.try_answer("conv-1", "张三几岁", "quick", [])
    self.assertIsNotNone(result)
    assert result is not None
    self.assertIn("存在多个年龄值", result.reply.paragraphs[0].text)
    self.assertNotIn("张三的年龄是28岁", result.reply.paragraphs[0].text)

def test_same_value_from_multiple_sources_still_clarifies_same_name_entity(self) -> None:
    service = WordFactAnswerService(
        FakeFacts(
            [
                match("张三", "年龄", "28岁", source_id="kb-a"),
                match("张三", "年龄", "28岁", source_id="kb-b"),
            ]
        )
    )
    result = service.try_answer("conv-1", "张三几岁", "quick", [])
    self.assertIsNotNone(result)
    assert result is not None
    self.assertIn("请确认来源", result.reply.paragraphs[0].text)

def test_missing_target_field_returns_not_found_not_rag(self) -> None:
    result = WordFactAnswerService(FakeFacts([])).try_answer(
        "conv-1", "张三几岁", "deep", []
    )
    self.assertIsNotNone(result)
    assert result is not None
    self.assertEqual(result.reply.paragraphs[0].text, "未找到张三的年龄。")
```

- [ ] **Step 2: Run tests and verify RED**

```powershell
Push-Location backend
uv run --project . --group dev python -m unittest tests.test_word_facts tests.test_word_fact_answer -v
Pop-Location
```

Expected: FAIL because the resolver and answer service do not exist.

- [ ] **Step 3: Implement exact intent, conflicts, citations, and validation**

Define:

```python
@dataclass(frozen=True, slots=True)
class WordFactClarification:
    message: str
    candidates: tuple[str, ...]

WordFactualResolution = WordFactualIntent | WordFactClarification | None

def resolve_word_factual_intent(question: str) -> WordFactualResolution:
    normalized, positions = normalize_question_with_positions(question)
    field_matches = find_longest_field_aliases(normalized)
    if not field_matches:
        return None
    fields = tuple(dict.fromkeys(item.field for item in field_matches))
    if len(fields) != 1:
        return WordFactClarification(
            "一次只能查询一个事实字段，请选择",
            fields,
        )
    entity = extract_single_entity(question, field_matches)
    if isinstance(entity, WordFactClarification):
        return entity
    field = fields[0]
    return WordFactualIntent(
        entity=entity,
        entity_normalized=normalize_fact_key(entity),
        field=field,
        field_normalized=normalize_fact_key(field),
    )
```

Strip only bounded polite prefixes (`请问`, `能否告诉我`, `帮我查一下`), field wording, particles (`的/是/是什么/吗/呢`), and punctuation. Empty entity or independent entity separators (`和/以及/、`) return clarification.

Implement:

```python
class WordFactAnswerService:
    def __init__(
        self,
        repository: WordFactRepository,
        permission_tags: Sequence[str] = (),
    ) -> None:
        self._repository = repository
        self._permission_tags = tuple(permission_tags)

    def try_answer(
        self,
        conversation_id: str,
        content: str,
        mode: ComposerMode,
        previous_messages: Sequence[ChatMessageModel],
    ) -> AgentRunResult | None:
        del previous_messages
        resolution = resolve_word_factual_intent(content)
        if resolution is None:
            return None
        if isinstance(resolution, WordFactClarification):
            return build_word_fact_run(conversation_id, content, mode, resolution.message)
        matches = self._repository.find_knowledge_facts(
            resolution,
            permission_tags=self._permission_tags,
        )
        return answer_word_fact(conversation_id, content, mode, resolution, matches)
```

Deduplicate identical records within each source. No matches return `未找到{entity}的{field}。`. Exactly one source with one distinct value returns `{entity}的{field}是{value}。`. More than one source returns source clarification to protect against same-name entities, regardless of whether the values match. Multiple values inside one source also return conflict clarification. Citations use excerpt `{field}：{value}` only.

`validate_word_fact_answer()` rejects configured field labels other than `intent.field` and rejects values not present in the selected matches. With no matches it accepts only the exact not-found template; with one source/value it accepts only the exact value template; with a conflict it accepts only the source-clarification template and no value assertion. Run it before building the assistant message. Every completed fact route has one `query_word_fact` step and no LLM/retrieval dependency.

- [ ] **Step 4: Run answer tests and commit**

```powershell
Push-Location backend
uv run --project . --group dev python -m unittest tests.test_word_facts tests.test_word_fact_answer -v
Pop-Location
git add backend/app/word_facts.py backend/app/word_fact_answer.py backend/tests/test_word_facts.py backend/tests/test_word_fact_answer.py
git commit -m "feat: answer Word field facts exactly"
```

### Task 5: Route factual answers before RAG and document reindexing

**Files:**
- Modify: `backend/app/repository.py`
- Modify: `backend/app/sql_repository.py`
- Modify: `backend/app/main.py`
- Modify: `backend/tests/test_agent.py`
- Modify: `backend/tests/test_word_fact_answer.py`
- Modify: `README.md`
- Modify: `deploy/offline/README.md`

**Interfaces:**
- Consumes: optional `StructuredAnswerService`, `WordFactAnswerService`, and existing Agent.
- Produces: greeting → Excel structured → Word factual → document RAG order.

- [ ] **Step 1: Write failing repository route tests**

```python
def test_fact_route_precedes_agent_and_llm(self) -> None:
    repository = self.build_repository(word_fact_service=self.fact_service)
    _, _, messages = repository.send_message("conv-1", "张三几岁", "deep")
    self.assertEqual(messages[-1].paragraphs[0].text, "张三的年龄是28岁。")
    self.assertEqual(self.recording_llm.generation_calls, 0)
    self.assertEqual(self.search_calls, 0)
    self.assertEqual(self.inspect_calls, 0)

def test_open_word_question_continues_to_hybrid_rag(self) -> None:
    repository = self.build_repository(word_fact_service=self.fact_service)
    repository.send_message("conv-1", "介绍张三", "deep")
    self.assertGreater(self.search_calls, 0)
    self.assertEqual(self.recording_llm.generation_calls, 1)

def test_missing_fact_is_terminal_and_never_inspects_document(self) -> None:
    repository = self.build_repository(
        word_fact_service=WordFactAnswerService(FakeFacts([]))
    )
    repository.send_message("conv-1", "张三几岁", "deep")
    self.assertEqual(self.search_calls, 0)
    self.assertEqual(self.inspect_calls, 0)
```

- [ ] **Step 2: Run routing tests and verify RED**

```powershell
Push-Location backend
uv run --project . --group dev python -m unittest tests.test_word_fact_answer tests.test_agent tests.test_sql_repository -v
Pop-Location
```

Expected: FAIL because repositories have no factual service injection.

- [ ] **Step 3: Wire the service and write the reindex contract**

Add `word_fact_service: WordFactAnswerService | None = None` to both repository constructors without changing existing defaults. Use this exact order:

```python
agent_result = self._agent.try_answer_greeting(
    conversation_id=conversation_id,
    content=clean_content,
    mode=mode,
)
if agent_result is None and self._structured_service is not None:
    agent_result = self._structured_service.try_answer(
        conversation_id=conversation_id,
        content=clean_content,
        mode=mode,
        previous_messages=previous_messages,
    )
if agent_result is None and self._word_fact_service is not None:
    agent_result = self._word_fact_service.try_answer(
        conversation_id=conversation_id,
        content=clean_content,
        mode=mode,
        previous_messages=previous_messages,
    )
if agent_result is None:
    agent_result = self._agent.run(
        conversation_id=conversation_id,
        content=clean_content,
        mode=mode,
        previous_messages=previous_messages,
    )
```

Construct the production fact service with the SQL repository and the same `retrieval_permission_tags` used by normal retrieval. Closing the service is a no-op.

Add this exact operator sequence to both runbooks:

```text
1. Apply Alembic revision 20260811_07.
2. Confirm every existing Word source still has a readable file_path.
3. POST /api/knowledge/sources/{source_id}/reindex once for each Word source.
4. Keep the old retrieval publication active until the normal Qdrant publication fence completes.
5. Verify records > 0, knowledge_facts contains rows, and “张三几岁” returns only age.
6. If conflicts appear, disable the factual route and inspect extracted facts; do not route the question to unrelated Word RAG.
```

State explicitly that Word sources require reindexing, while published Excel tables do not require re-uploading.

- [ ] **Step 4: Run the Word gate and commit**

```powershell
Push-Location backend
uv run --project . --group dev python -m unittest tests.test_docx_parser tests.test_word_facts tests.test_word_fact_answer tests.test_agent tests.test_sql_repository tests.test_knowledge_ingestion_pipeline -v
uv run --project . --group dev ruff format --check app tests
uv run --project . --group dev ruff check app tests
Pop-Location
git diff --check
git add backend/app/repository.py backend/app/sql_repository.py backend/app/main.py backend/tests/test_agent.py backend/tests/test_word_fact_answer.py README.md deploy/offline/README.md
git commit -m "feat: route Word factual questions before RAG"
```

Expected: exact factual questions never invoke search, reranking, inspection, or the LLM; open Word questions retain the existing hybrid RAG path.
