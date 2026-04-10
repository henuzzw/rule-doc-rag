from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from app.config import PROJECT_ROOT, get_settings
from app.db import db_session
from app.repository import Repository
from app.schemas import GeneratedDocument, GenerationRequest, ReviewRequest
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

    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "requirements": requirements,
            "documents": documents,
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
        document = Repository(conn).review_document(
            document_id=document_id,
            status=request.status,
            reviewer_notes=request.reviewer_notes,
            doc=request.doc,
        )
    if document is None:
        raise HTTPException(status_code=404, detail="document not found")
    return document


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
        document = Repository(conn).review_document(
            document_id=document_id,
            status=status,
            reviewer_notes=reviewer_notes,
        )
    if document is None:
        raise HTTPException(status_code=404, detail="document not found")
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
    result = document_generator.generate(
        requirement=requirement,
        contexts=contexts,
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
