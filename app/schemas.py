from typing import Any, Literal

from pydantic import BaseModel, Field


DocumentStatus = Literal["draft", "reviewed", "published", "rejected"]


class Requirement(BaseModel):
    id: str
    title: str
    business_background: str
    source_text: str


class KnowledgeChunk(BaseModel):
    id: str
    source_type: Literal["requirement", "history_rule", "manual"]
    source_id: str
    title: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievedContext(KnowledgeChunk):
    score: float


class RuleExample(BaseModel):
    title: str
    input: dict[str, Any]
    expected_output: dict[str, Any]
    explanation: str


class RuleDocumentContent(BaseModel):
    rule_name: str
    rule_code: str
    document_summary: str
    business_goal: str
    trigger_timing: str
    input_fields: list[str]
    output_fields: list[str]
    decision_logic: list[str]
    pseudo_code: str
    examples: list[RuleExample]
    exception_handling: list[str]
    dependencies: list[str]
    change_risk: str
    test_points: list[str]


class GenerationRequest(BaseModel):
    requirement_id: str
    rule_name: str
    rule_code: str
    top_k: int | None = None


class ReviewRequest(BaseModel):
    status: DocumentStatus = "reviewed"
    reviewer_notes: str = ""
    doc: RuleDocumentContent | None = None


class GeneratedDocument(BaseModel):
    id: str
    rule_code: str
    rule_name: str
    requirement_id: str
    status: DocumentStatus
    doc: RuleDocumentContent
    retrieved_context: list[RetrievedContext]
    reviewer_notes: str
    created_at: str
    updated_at: str

