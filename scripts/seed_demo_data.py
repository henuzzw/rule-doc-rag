from __future__ import annotations

import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.db import db_session  # noqa: E402
from app.repository import Repository  # noqa: E402
from app.schemas import KnowledgeChunk, Requirement  # noqa: E402
from app.services.embedding import build_embedding_service  # noqa: E402


def main() -> None:
    embedder = build_embedding_service()
    requirements = _read_json("requirements.json")
    history_rules = _read_json("history_rules.json")

    with db_session() as conn:
        repository = Repository(conn)

        for item in requirements:
            requirement = Requirement.model_validate(item)
            repository.upsert_requirement(requirement)
            _upsert_chunk(
                repository,
                embedder,
                KnowledgeChunk(
                    id=f"req::{requirement.id}",
                    source_type="requirement",
                    source_id=requirement.id,
                    title=requirement.title,
                    content="\n".join(
                        [
                            f"需求标题：{requirement.title}",
                            f"业务背景：{requirement.business_background}",
                            requirement.source_text,
                        ]
                    ),
                    metadata={"dataset": "demo_requirements"},
                ),
            )

        for item in history_rules:
            chunk = KnowledgeChunk.model_validate(item)
            _upsert_chunk(repository, embedder, chunk)

    print(
        f"seed completed: {len(requirements)} requirements, "
        f"{len(history_rules)} historical rule documents"
    )


def _upsert_chunk(repository, embedder, chunk: KnowledgeChunk) -> None:
    embedding = embedder.embed_text(f"{chunk.title}\n{chunk.content}")
    repository.upsert_knowledge_chunk(chunk, embedding)


def _read_json(filename: str):
    return json.loads((PROJECT_ROOT / "data" / filename).read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()

