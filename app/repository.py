from collections.abc import Sequence
from datetime import datetime
from typing import Any
from uuid import uuid4

from psycopg import Connection
from psycopg.types.json import Jsonb

from app.schemas import (
    BadCase,
    GeneratedDocument,
    KnowledgeChunk,
    Requirement,
    RetrievedContext,
    RetrievedBadCase,
    RuleDocumentContent,
)


def to_vector_literal(vector: Sequence[float]) -> str:
    return "[" + ",".join(f"{float(value):.8f}" for value in vector) + "]"


def _iso(value: Any) -> str:
    if isinstance(value, datetime):
        return value.isoformat()
    return str(value)


class Repository:
    def __init__(self, conn: Connection):
        self.conn = conn

    def upsert_requirement(self, requirement: Requirement) -> None:
        self.conn.execute(
            """
            INSERT INTO requirements (id, title, business_background, source_text)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (id) DO UPDATE SET
              title = EXCLUDED.title,
              business_background = EXCLUDED.business_background,
              source_text = EXCLUDED.source_text
            """,
            (
                requirement.id,
                requirement.title,
                requirement.business_background,
                requirement.source_text,
            ),
        )

    def list_requirements(self) -> list[Requirement]:
        rows = self.conn.execute(
            """
            SELECT id, title, business_background, source_text
            FROM requirements
            ORDER BY created_at DESC, id
            """
        ).fetchall()
        return [Requirement.model_validate(row) for row in rows]

    def get_requirement(self, requirement_id: str) -> Requirement | None:
        row = self.conn.execute(
            """
            SELECT id, title, business_background, source_text
            FROM requirements
            WHERE id = %s
            """,
            (requirement_id,),
        ).fetchone()
        return Requirement.model_validate(row) if row else None

    def upsert_knowledge_chunk(
        self, chunk: KnowledgeChunk, embedding: Sequence[float]
    ) -> None:
        embedding_literal = to_vector_literal(embedding)
        self.conn.execute(
            """
            INSERT INTO knowledge_chunks (
              id, source_type, source_id, title, content, metadata, embedding
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s::vector)
            ON CONFLICT (id) DO UPDATE SET
              source_type = EXCLUDED.source_type,
              source_id = EXCLUDED.source_id,
              title = EXCLUDED.title,
              content = EXCLUDED.content,
              metadata = EXCLUDED.metadata,
              embedding = EXCLUDED.embedding
            """,
            (
                chunk.id,
                chunk.source_type,
                chunk.source_id,
                chunk.title,
                chunk.content,
                Jsonb(chunk.metadata),
                embedding_literal,
            ),
        )

    def upsert_bad_case(self, bad_case: BadCase, embedding: Sequence[float]) -> None:
        embedding_literal = to_vector_literal(embedding)
        self.conn.execute(
            """
            INSERT INTO bad_cases (
              id, rule_code, rule_name, title, bad_summary, failure_reason,
              corrected_hint, metadata, embedding
            )
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::vector)
            ON CONFLICT (id) DO UPDATE SET
              rule_code = EXCLUDED.rule_code,
              rule_name = EXCLUDED.rule_name,
              title = EXCLUDED.title,
              bad_summary = EXCLUDED.bad_summary,
              failure_reason = EXCLUDED.failure_reason,
              corrected_hint = EXCLUDED.corrected_hint,
              metadata = EXCLUDED.metadata,
              embedding = EXCLUDED.embedding
            """,
            (
                bad_case.id,
                bad_case.rule_code,
                bad_case.rule_name,
                bad_case.title,
                bad_case.bad_summary,
                bad_case.failure_reason,
                bad_case.corrected_hint,
                Jsonb(bad_case.metadata),
                embedding_literal,
            ),
        )

    def list_bad_cases(self, limit: int = 100) -> list[BadCase]:
        rows = self.conn.execute(
            """
            SELECT id, rule_code, rule_name, title, bad_summary, failure_reason,
                   corrected_hint, metadata
            FROM bad_cases
            ORDER BY created_at DESC, id
            LIMIT %s
            """,
            (limit,),
        ).fetchall()
        return [BadCase.model_validate(row) for row in rows]

    def get_bad_case(self, bad_case_id: str) -> BadCase | None:
        row = self.conn.execute(
            """
            SELECT id, rule_code, rule_name, title, bad_summary, failure_reason,
                   corrected_hint, metadata
            FROM bad_cases
            WHERE id = %s
            """,
            (bad_case_id,),
        ).fetchone()
        return BadCase.model_validate(row) if row else None

    def delete_bad_case(self, bad_case_id: str) -> bool:
        result = self.conn.execute(
            "DELETE FROM bad_cases WHERE id = %s",
            (bad_case_id,),
        )
        return result.rowcount > 0

    def search_knowledge(
        self, query_embedding: Sequence[float], top_k: int
    ) -> list[RetrievedContext]:
        query_literal = to_vector_literal(query_embedding)
        rows = self.conn.execute(
            """
            SELECT
              id,
              source_type,
              source_id,
              title,
              content,
              metadata,
              1 - (embedding <=> %s::vector) AS score
            FROM knowledge_chunks
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (query_literal, query_literal, top_k),
        ).fetchall()
        return [RetrievedContext.model_validate(row) for row in rows]

    def search_bad_cases(
        self, query_embedding: Sequence[float], top_k: int
    ) -> list[RetrievedBadCase]:
        query_literal = to_vector_literal(query_embedding)
        rows = self.conn.execute(
            """
            SELECT
              id,
              rule_code,
              rule_name,
              title,
              bad_summary,
              failure_reason,
              corrected_hint,
              metadata,
              1 - (embedding <=> %s::vector) AS score
            FROM bad_cases
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (query_literal, query_literal, top_k),
        ).fetchall()
        return [self._map_retrieved_bad_case(row) for row in rows]

    def create_generated_document(
        self,
        *,
        request_rule_code: str,
        request_rule_name: str,
        requirement_id: str,
        doc: RuleDocumentContent,
        prompt: str,
        raw_response: str,
        retrieved_context: list[RetrievedContext],
    ) -> GeneratedDocument:
        document_id = str(uuid4())
        self.conn.execute(
            """
            INSERT INTO generated_documents (
              id,
              rule_code,
              rule_name,
              requirement_id,
              status,
              doc_json,
              prompt,
              raw_response,
              retrieved_context
            )
            VALUES (%s, %s, %s, %s, 'draft', %s, %s, %s, %s)
            """,
            (
                document_id,
                request_rule_code,
                request_rule_name,
                requirement_id,
                Jsonb(doc.model_dump(mode="json")),
                prompt,
                raw_response,
                Jsonb([item.model_dump(mode="json") for item in retrieved_context]),
            ),
        )
        created = self.get_generated_document(document_id)
        if created is None:
            raise RuntimeError("generated document was not persisted")
        return created

    def list_generated_documents(self) -> list[GeneratedDocument]:
        rows = self.conn.execute(
            """
            SELECT
              id,
              rule_code,
              rule_name,
              requirement_id,
              status,
              doc_json,
              retrieved_context,
              reviewer_notes,
              created_at,
              updated_at
            FROM generated_documents
            ORDER BY created_at DESC
            LIMIT 50
            """
        ).fetchall()
        return [self._map_generated_document(row) for row in rows]

    def get_generated_document(self, document_id: str) -> GeneratedDocument | None:
        row = self.conn.execute(
            """
            SELECT
              id,
              rule_code,
              rule_name,
              requirement_id,
              status,
              doc_json,
              retrieved_context,
              reviewer_notes,
              created_at,
              updated_at
            FROM generated_documents
            WHERE id = %s
            """,
            (document_id,),
        ).fetchone()
        return self._map_generated_document(row) if row else None

    def review_document(
        self,
        *,
        document_id: str,
        status: str,
        reviewer_notes: str,
        doc: RuleDocumentContent | None = None,
    ) -> GeneratedDocument | None:
        if doc is None:
            self.conn.execute(
                """
                UPDATE generated_documents
                SET status = %s,
                    reviewer_notes = %s,
                    updated_at = now()
                WHERE id = %s
                """,
                (status, reviewer_notes, document_id),
            )
        else:
            self.conn.execute(
                """
                UPDATE generated_documents
                SET status = %s,
                    reviewer_notes = %s,
                    doc_json = %s,
                    updated_at = now()
                WHERE id = %s
                """,
                (
                    status,
                    reviewer_notes,
                    Jsonb(doc.model_dump(mode="json")),
                    document_id,
                ),
            )
        return self.get_generated_document(document_id)

    @staticmethod
    def _map_generated_document(row: dict[str, Any]) -> GeneratedDocument:
        payload = dict(row)
        payload["doc"] = payload.pop("doc_json")
        payload["created_at"] = _iso(payload["created_at"])
        payload["updated_at"] = _iso(payload["updated_at"])
        return GeneratedDocument.model_validate(payload)

    @staticmethod
    def _map_retrieved_bad_case(row: dict[str, Any]) -> RetrievedBadCase:
        return RetrievedBadCase.model_validate(dict(row))

