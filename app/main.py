import hashlib
import json

from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.config import PROJECT_ROOT, get_settings
from app.db import db_session
from app.repository import Repository
from app.schemas import (
    BadCase,
    BadCaseCreate,
    BadCaseUpdate,
    GeneratedDocument,
    GenerationRequest,
    ReviewRequest,
)
from app.services.embedding import build_embedding_service
from app.services.generator import RuleDocumentGenerator
from app.services.llm import build_llm_client
from app.services.retrieval import RetrievalService


settings = get_settings()
embedding_service = build_embedding_service(settings)
retrieval_service = RetrievalService(embedding_service, settings)
document_generator = RuleDocumentGenerator(build_llm_client(settings))

app = FastAPI(title="Rule Document RAG", version="0.1.0")
app.mount(
    "/static",
    StaticFiles(directory=PROJECT_ROOT / "app" / "static"),
    name="static",
)
templates = Jinja2Templates(directory=PROJECT_ROOT / "app" / "templates")


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "embedding_mode": settings.embedding_mode,
        "llm_mode": settings.llm_mode,
    }


@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request) -> HTMLResponse:
    with db_session() as conn:
        repository = Repository(conn)
        requirements = repository.list_requirements()
        documents = repository.list_generated_documents()
        bad_cases = repository.list_bad_cases()

    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "requirements": requirements,
            "documents": documents,
            "bad_cases": bad_cases,
            "settings": settings,
        },
    )


@app.get("/api/requirements")
def list_requirements():
    with db_session() as conn:
        return Repository(conn).list_requirements()


@app.get("/api/documents", response_model=list[GeneratedDocument])
def list_documents() -> list[GeneratedDocument]:
    with db_session() as conn:
        return Repository(conn).list_generated_documents()


@app.get("/api/documents/{document_id}", response_model=GeneratedDocument)
def get_document(document_id: str) -> GeneratedDocument:
    with db_session() as conn:
        document = Repository(conn).get_generated_document(document_id)
    if document is None:
        raise HTTPException(status_code=404, detail="document not found")
    return document


@app.post("/api/generate", response_model=GeneratedDocument)
def generate_document(request: GenerationRequest) -> GeneratedDocument:
    with db_session() as conn:
        return _generate_with_repository(Repository(conn), request)


@app.post("/api/documents/{document_id}/review", response_model=GeneratedDocument)
def review_document(document_id: str, request: ReviewRequest) -> GeneratedDocument:
    with db_session() as conn:
        repository = Repository(conn)
        document = repository.review_document(
            document_id=document_id,
            status=request.status,
            reviewer_notes=request.reviewer_notes,
            doc=request.doc,
        )
        _maybe_create_bad_case_from_review(
            repository=repository,
            document=document,
            status=request.status,
            reviewer_notes=request.reviewer_notes,
        )
    if document is None:
        raise HTTPException(status_code=404, detail="document not found")
    return document


@app.get("/api/bad-cases", response_model=list[BadCase])
def list_bad_cases() -> list[BadCase]:
    with db_session() as conn:
        return Repository(conn).list_bad_cases()


@app.get("/api/bad-cases/{bad_case_id}", response_model=BadCase)
def get_bad_case(bad_case_id: str) -> BadCase:
    with db_session() as conn:
        bad_case = Repository(conn).get_bad_case(bad_case_id)
    if bad_case is None:
        raise HTTPException(status_code=404, detail="bad case not found")
    return bad_case


@app.post("/api/bad-cases", response_model=BadCase)
def create_bad_case(request: BadCaseCreate) -> BadCase:
    with db_session() as conn:
        repository = Repository(conn)
        bad_case = _upsert_bad_case_from_request(repository, request)
    return bad_case


@app.patch("/api/bad-cases/{bad_case_id}", response_model=BadCase)
def patch_bad_case(bad_case_id: str, request: BadCaseUpdate) -> BadCase:
    with db_session() as conn:
        repository = Repository(conn)
        existing = repository.get_bad_case(bad_case_id)
        if existing is None:
            raise HTTPException(status_code=404, detail="bad case not found")
        updated = existing.model_copy(update=request.model_dump(exclude_unset=True))
        bad_case = _upsert_bad_case(repository, updated)
    return bad_case


@app.delete("/api/bad-cases/{bad_case_id}")
def delete_bad_case(bad_case_id: str) -> dict[str, bool]:
    with db_session() as conn:
        deleted = Repository(conn).delete_bad_case(bad_case_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="bad case not found")
    return {"deleted": True}


@app.post("/ui/generate")
def ui_generate(
    requirement_id: str = Form(...),
    rule_name: str = Form(...),
    rule_code: str = Form(...),
) -> RedirectResponse:
    with db_session() as conn:
        _generate_with_repository(
            Repository(conn),
            GenerationRequest(
                requirement_id=requirement_id,
                rule_name=rule_name,
                rule_code=rule_code,
            ),
        )
    return RedirectResponse("/", status_code=303)


@app.post("/ui/documents/{document_id}/review")
def ui_review(
    document_id: str,
    status: str = Form("reviewed"),
    reviewer_notes: str = Form(""),
) -> RedirectResponse:
    with db_session() as conn:
        repository = Repository(conn)
        document = repository.review_document(
            document_id=document_id,
            status=status,
            reviewer_notes=reviewer_notes,
        )
        _maybe_create_bad_case_from_review(
            repository=repository,
            document=document,
            status=status,
            reviewer_notes=reviewer_notes,
        )
    if document is None:
        raise HTTPException(status_code=404, detail="document not found")
    return RedirectResponse("/", status_code=303)


@app.post("/ui/bad-cases")
def ui_create_bad_case(
    rule_code: str = Form(...),
    rule_name: str = Form(...),
    title: str = Form(...),
    bad_summary: str = Form(...),
    failure_reason: str = Form(...),
    corrected_hint: str = Form(""),
    metadata_json: str = Form("{}"),
) -> RedirectResponse:
    with db_session() as conn:
        repository = Repository(conn)
        request = BadCaseCreate(
            rule_code=rule_code,
            rule_name=rule_name,
            title=title,
            bad_summary=bad_summary,
            failure_reason=failure_reason,
            corrected_hint=corrected_hint,
            metadata=_parse_metadata(metadata_json),
        )
        _upsert_bad_case_from_request(repository, request)
    return RedirectResponse("/", status_code=303)


@app.post("/ui/bad-cases/{bad_case_id}/update")
def ui_update_bad_case(
    bad_case_id: str,
    rule_code: str = Form(...),
    rule_name: str = Form(...),
    title: str = Form(...),
    bad_summary: str = Form(...),
    failure_reason: str = Form(...),
    corrected_hint: str = Form(""),
    metadata_json: str = Form("{}"),
) -> RedirectResponse:
    with db_session() as conn:
        repository = Repository(conn)
        existing = repository.get_bad_case(bad_case_id)
        if existing is None:
            raise HTTPException(status_code=404, detail="bad case not found")
        updated = existing.model_copy(
            update={
                "rule_code": rule_code,
                "rule_name": rule_name,
                "title": title,
                "bad_summary": bad_summary,
                "failure_reason": failure_reason,
                "corrected_hint": corrected_hint,
                "metadata": _parse_metadata(metadata_json),
            }
        )
        _upsert_bad_case(repository, updated)
    return RedirectResponse("/", status_code=303)


@app.post("/ui/bad-cases/{bad_case_id}/delete")
def ui_delete_bad_case(bad_case_id: str) -> RedirectResponse:
    with db_session() as conn:
        deleted = Repository(conn).delete_bad_case(bad_case_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="bad case not found")
    return RedirectResponse("/", status_code=303)


def _generate_with_repository(
    repository: Repository,
    request: GenerationRequest,
) -> GeneratedDocument:
    requirement = repository.get_requirement(request.requirement_id)
    if requirement is None:
        raise HTTPException(status_code=404, detail="requirement not found")

    contexts = retrieval_service.retrieve(
        repository=repository,
        requirement=requirement,
        rule_name=request.rule_name,
        top_k=request.top_k,
    )
    bad_cases = retrieval_service.retrieve_bad_cases(
        repository=repository,
        requirement=requirement,
        rule_name=request.rule_name,
        top_k=request.top_k,
    )
    result = document_generator.generate(
        requirement=requirement,
        contexts=contexts,
        bad_cases=bad_cases,
        rule_name=request.rule_name,
        rule_code=request.rule_code,
    )
    return repository.create_generated_document(
        request_rule_code=request.rule_code,
        request_rule_name=request.rule_name,
        requirement_id=request.requirement_id,
        doc=result.doc,
        prompt=result.prompt,
        raw_response=result.raw_response,
        retrieved_context=contexts,
    )


def _upsert_bad_case(repository: Repository, bad_case: BadCase) -> BadCase:
    repository.upsert_bad_case(bad_case, _bad_case_embedding_text(bad_case))
    persisted = repository.get_bad_case(bad_case.id)
    if persisted is None:
        raise RuntimeError("bad case was not persisted")
    return persisted


def _upsert_bad_case_from_request(
    repository: Repository, request: BadCaseCreate
) -> BadCase:
    bad_case = BadCase(
        id=request.id or _build_bad_case_id(request.rule_code, request.title),
        rule_code=request.rule_code,
        rule_name=request.rule_name,
        title=request.title,
        bad_summary=request.bad_summary,
        failure_reason=request.failure_reason,
        corrected_hint=request.corrected_hint,
        metadata=request.metadata,
    )
    return _upsert_bad_case(repository, bad_case)


def _maybe_create_bad_case_from_review(
    *,
    repository: Repository,
    document: GeneratedDocument | None,
    status: str,
    reviewer_notes: str,
) -> None:
    if document is None:
        return

    normalized_status = status.lower()
    notes = reviewer_notes.strip()
    if normalized_status not in {"reviewed", "rejected"}:
        return
    if normalized_status == "reviewed" and not notes:
        return

    bad_case = BadCase(
        id=f"feedback::{document.id}",
        rule_code=document.rule_code,
        rule_name=document.rule_name,
        title=f"审核反馈：{document.doc.rule_name}",
        bad_summary=_build_bad_summary(document, normalized_status, notes),
        failure_reason=notes or f"审核结论为 {normalized_status}",
        corrected_hint=_build_corrected_hint(document),
        metadata={
            "source": "review_feedback",
            "source_document_id": document.id,
            "source_requirement_id": document.requirement_id,
            "source_status": normalized_status,
        },
    )
    _upsert_bad_case(repository, bad_case)


def _build_bad_summary(document: GeneratedDocument, status: str, notes: str) -> str:
    summary = document.doc.document_summary.strip()
    base = f"审核状态={status}; 规则={document.rule_code}; 摘要={summary}"
    if notes:
        return f"{base}; 反馈={notes}"
    return base


def _build_corrected_hint(document: GeneratedDocument) -> str:
    return (
        "根据审核反馈修订文档，优先检查字段、阈值、异常处理和脱敏约束；"
        f"建议重新对照规则 {document.rule_code} 的需求与上下文生成。"
    )


def _bad_case_embedding_text(bad_case: BadCase) -> list[float]:
    return embedding_service.embed_text(
        "\n".join(
            [
                bad_case.rule_code,
                bad_case.rule_name,
                bad_case.title,
                bad_case.bad_summary,
                bad_case.failure_reason,
                bad_case.corrected_hint,
            ]
        )
    )


def _build_bad_case_id(rule_code: str, title: str) -> str:
    digest = hashlib.blake2b(
        f"{rule_code}\n{title.strip()}".encode("utf-8"),
        digest_size=8,
    ).hexdigest()
    return f"bad::{rule_code}::{digest}"


def _parse_metadata(metadata_json: str) -> dict:
    text = metadata_json.strip() or "{}"
    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise HTTPException(status_code=400, detail="metadata_json must be valid JSON") from exc
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="metadata_json must be a JSON object")
    return payload
