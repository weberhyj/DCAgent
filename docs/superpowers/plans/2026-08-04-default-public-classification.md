# Default Public Classification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make newly created or uploaded knowledge sources default to the Chinese classification `公开`, with matching retrieval permission defaults and deployment guidance.

**Architecture:** Keep the existing classification and permission-filtering interfaces unchanged. Align the API defaults and all checked-in environment templates on one literal value, `公开`, while preserving explicitly supplied classifications and leaving existing database rows untouched.

**Tech Stack:** Python 3.11+, FastAPI, Pydantic v2, unittest/pytest, Ruff, Ubuntu Bash deployment templates.

---

## File map

- `backend/app/routes.py`: multipart knowledge-upload default classification.
- `backend/app/schemas.py`: JSON knowledge-source request default classification.
- `backend/tests/test_knowledge_upload.py`: upload default and explicit override behavior.
- `backend/tests/test_api_contract.py`: JSON source-creation default behavior.
- `backend/tests/test_sql_repository.py`: retrieval visibility regression for a public source.
- `.env.example`, `backend/.env.example`, `deploy/offline/.env.example`: matching default retrieval permission tag.
- `tools/tests/test_structured_deployment_contract.py`: environment and documentation contract checks.
- `README.md`, `docs/intranet-deployment-configuration.md`: operator-facing default and upgrade guidance.

### Task 1: Make API-created knowledge sources default to public

**Files:**
- Modify: `backend/tests/test_knowledge_upload.py`
- Modify: `backend/tests/test_api_contract.py`
- Modify: `backend/app/routes.py:499-504`
- Modify: `backend/app/schemas.py:610-619`

- [ ] **Step 1: Write the failing multipart-upload test**

Add this test to `KnowledgeUploadTest` in `backend/tests/test_knowledge_upload.py`:

```python
def test_upload_without_classification_defaults_to_public(self) -> None:
    response = self.client.post(
        "/api/knowledge/uploads",
        files={"file": ("public-policy.txt", b"public policy" * 20, "text/plain")},
    )

    self.assertEqual(response.status_code, 200)
    self.assertEqual(response.json()[0]["classification"], "公开")
```

Keep the existing tests that submit `data={"classification": "内部"}` or `内部·机密`; they prove explicit values are not overwritten.

- [ ] **Step 2: Write the failing JSON source-creation test**

Add this test to `ApiContractTest` in `backend/tests/test_api_contract.py`:

```python
def test_adds_knowledge_source_with_public_default_classification(self) -> None:
    response = self.client.post(
        "/api/knowledge/sources",
        json={"name": "公开制度.txt", "sourceType": "TXT"},
    )

    self.assertEqual(response.status_code, 200)
    self.assertEqual(response.json()[0]["classification"], "公开")
```

- [ ] **Step 3: Run both tests and verify RED**

Run:

```bash
cd backend
uv run pytest tests/test_knowledge_upload.py::KnowledgeUploadTest::test_upload_without_classification_defaults_to_public tests/test_api_contract.py::ApiContractTest::test_adds_knowledge_source_with_public_default_classification -q
```

Expected: both tests fail because the current defaults return `内部·机密`.

- [ ] **Step 4: Change the two API defaults**

In `backend/app/routes.py`, change the upload parameter to:

```python
classification: str = Form(default="公开"),
```

In `backend/app/schemas.py`, change `KnowledgeSourceRequest` to:

```python
classification: str = Field(default="公开", min_length=1, max_length=80)
```

- [ ] **Step 5: Run the focused API tests and verify GREEN**

Run:

```bash
cd backend
uv run pytest tests/test_knowledge_upload.py tests/test_api_contract.py -q
```

Expected: all tests in both files pass.

- [ ] **Step 6: Commit the API default change**

```bash
git add backend/app/routes.py backend/app/schemas.py backend/tests/test_knowledge_upload.py backend/tests/test_api_contract.py
git commit -m "feat: default knowledge sources to public"
```

### Task 2: Align the deployment permission defaults and retrieval regression

**Files:**
- Modify: `tools/tests/test_structured_deployment_contract.py:78-94`
- Modify: `backend/tests/test_sql_repository.py:470-500`
- Modify: `.env.example:22`
- Modify: `backend/.env.example:22`
- Modify: `deploy/offline/.env.example:26`

- [ ] **Step 1: Change the environment contract expectation first**

In `test_env_examples_define_structured_rollout_contract`, change the permission assertion to:

```python
self.assertEqual(values["RETRIEVAL_PERMISSION_TAGS"], "公开")
```

- [ ] **Step 2: Add a public-source retrieval regression test**

Add this test beside the existing permission-classification filter test in `backend/tests/test_sql_repository.py`:

```python
def test_default_public_scope_can_retrieve_public_source(self) -> None:
    self.repository.add_uploaded_knowledge_source(
        source_id="kb-public",
        name="public-policy.txt",
        source_type="TXT",
        classification="公开",
        records=0,
        file_path="public-policy.txt",
        file_size=64,
        mime_type="text/plain",
    )
    self.repository.complete_knowledge_source_indexing(
        "kb-public",
        [
            KnowledgeChunkModel(
                id="chunk-kb-public",
                source_id="kb-public",
                chunk_index=0,
                text="公开制度规定访客需要在前台登记。",
                token_count=18,
            )
        ],
    )
    scoped = SqlChatRepository(self.database, retrieval_permission_tags=("公开",))

    hits = scoped.search_knowledge_chunks("访客前台登记", limit=5)

    self.assertEqual([hit.source.id for hit in hits], ["kb-public"])
```

- [ ] **Step 3: Run the environment test and verify RED**

Run:

```bash
uv run --project backend --group dev python -m unittest tools.tests.test_structured_deployment_contract.StructuredDeploymentContractTests.test_env_examples_define_structured_rollout_contract
```

Expected: failure showing `internal != 公开` for the checked-in environment templates.

- [ ] **Step 4: Run the repository regression before configuration edits**

Run:

```bash
cd backend
uv run pytest tests/test_sql_repository.py::SqlRepositoryTest::test_default_public_scope_can_retrieve_public_source -q
```

Expected: PASS, documenting that matching classification and permission values are retrievable with the existing repository behavior.

- [ ] **Step 5: Align all environment templates**

Set the active assignment in each file to:

```env
RETRIEVAL_PERMISSION_TAGS=公开
```

Apply it to `.env.example`, `backend/.env.example`, and `deploy/offline/.env.example`. Do not edit an operator's untracked production `.env`.

- [ ] **Step 6: Run the focused contracts and verify GREEN**

Run:

```bash
uv run --project backend --group dev python -m unittest tools.tests.test_structured_deployment_contract.StructuredDeploymentContractTests.test_env_examples_define_structured_rollout_contract
cd backend
uv run pytest tests/test_sql_repository.py::SqlRepositoryTest::test_default_public_scope_can_retrieve_public_source -q
```

Expected: both commands pass.

- [ ] **Step 7: Commit the aligned defaults**

```bash
git add .env.example backend/.env.example deploy/offline/.env.example tools/tests/test_structured_deployment_contract.py backend/tests/test_sql_repository.py
git commit -m "fix: align public retrieval defaults"
```

### Task 3: Document the default and upgrade behavior

**Files:**
- Modify: `tools/tests/test_structured_deployment_contract.py`
- Modify: `README.md`
- Modify: `docs/intranet-deployment-configuration.md`

- [ ] **Step 1: Add a failing documentation contract**

Add this method to `StructuredDeploymentContractTests`:

```python
def test_public_classification_default_is_documented(self) -> None:
    for path in (
        REPO_ROOT / "README.md",
        REPO_ROOT / "docs" / "intranet-deployment-configuration.md",
    ):
        text = path.read_text(encoding="utf-8")
        with self.subTest(path=path.relative_to(REPO_ROOT)):
            self.assertIn("RETRIEVAL_PERMISSION_TAGS=公开", text)
            self.assertIn("现有文档不会自动迁移", text)
```

- [ ] **Step 2: Run the documentation contract and verify RED**

Run:

```bash
uv run --project backend --group dev python -m unittest tools.tests.test_structured_deployment_contract.StructuredDeploymentContractTests.test_public_classification_default_is_documented
```

Expected: failure because neither document contains the new deployment note.

- [ ] **Step 3: Add the README deployment note**

After the retrieval-mode explanation in `README.md`, add:

````markdown
新上传知识库文档的默认分类为“公开”，默认部署模板使用：

```env
RETRIEVAL_PERMISSION_TAGS=公开
```

显式提交的其他分类仍会保留。升级不会修改数据库中的历史分类，现有文档不会自动迁移；如需采用新默认值，请删除后重新导入并等待索引完成。
````

- [ ] **Step 4: Add the intranet deployment note**

In the Qdrant retrieval configuration section of `docs/intranet-deployment-configuration.md`, extend the environment example to include:

```env
RETRIEVAL_PERMISSION_TAGS=公开
```

Immediately below it add:

```markdown
后续未显式选择分类的新上传文档默认使用“公开”，该值必须与检索权限标签保持一致。显式分类不会被覆盖。升级不会批量修改数据库，现有文档不会自动迁移；管理员可以删除旧文档后重新导入。
```

- [ ] **Step 5: Run the documentation contract and verify GREEN**

Run:

```bash
uv run --project backend --group dev python -m unittest tools.tests.test_structured_deployment_contract.StructuredDeploymentContractTests.test_public_classification_default_is_documented
```

Expected: PASS.

- [ ] **Step 6: Commit the documentation change**

```bash
git add README.md docs/intranet-deployment-configuration.md tools/tests/test_structured_deployment_contract.py
git commit -m "docs: explain public knowledge defaults"
```

### Task 4: Final verification

**Files:**
- Verify all modified files.

- [ ] **Step 1: Run focused backend tests**

```bash
cd backend
uv run pytest tests/test_knowledge_upload.py tests/test_api_contract.py tests/test_sql_repository.py -q
```

Expected: all selected backend tests pass.

- [ ] **Step 2: Run deployment contract tests**

```bash
uv run --project backend --group dev python -m unittest tools.tests.test_structured_deployment_contract
```

Expected: all structured deployment contract tests pass.

- [ ] **Step 3: Run Ruff checks**

```bash
cd backend
uv run ruff check app tests
uv run ruff format --check app tests
```

Expected: both commands exit successfully with no violations.

- [ ] **Step 4: Verify formatting and scope**

```bash
git diff --check
git status -sb
git diff main...HEAD --stat
```

Expected: no whitespace errors; only the planned code, tests, templates, and documentation are changed.

- [ ] **Step 5: Commit any verification-only correction**

If formatting changes were required, commit only those files:

```bash
git add backend/app backend/tests tools/tests README.md docs .env.example backend/.env.example deploy/offline/.env.example
git commit -m "style: normalize public default changes"
```

If no correction was required, do not create an empty commit.
