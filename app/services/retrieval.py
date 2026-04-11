from app.config import Settings, get_settings
from app.repository import Repository
from app.schemas import Requirement, RetrievedBadCase, RetrievedContext
from app.services.embedding import EmbeddingService


class RetrievalService:
    def __init__(
        self,
        embedding_service: EmbeddingService,
        settings: Settings | None = None,
    ):
        self.embedding_service = embedding_service
        self.settings = settings or get_settings()

    def retrieve(
        self,
        *,
        repository: Repository,
        requirement: Requirement,
        rule_name: str,
        top_k: int | None = None,
    ) -> list[RetrievedContext]:
        query_text = "\n".join(
            [
                f"规则名称: {rule_name}",
                f"需求标题: {requirement.title}",
                f"业务背景: {requirement.business_background}",
                requirement.source_text,
            ]
        )
        query_embedding = self.embedding_service.embed_text(query_text)
        return repository.search_knowledge(
            query_embedding=query_embedding,
            top_k=top_k or self.settings.rag_top_k,
        )

    def retrieve_bad_cases(
        self,
        *,
        repository: Repository,
        requirement: Requirement,
        rule_name: str,
        top_k: int | None = None,
    ) -> list[RetrievedBadCase]:
        query_text = "\n".join(
            [
                f"瑙勫垯鍚嶇О: {rule_name}",
                f"闇€姹傛爣棰? {requirement.title}",
                f"涓氬姟鑳屾櫙: {requirement.business_background}",
                requirement.source_text,
            ]
        )
        query_embedding = self.embedding_service.embed_text(query_text)
        return repository.search_bad_cases(
            query_embedding=query_embedding,
            top_k=top_k or min(self.settings.rag_top_k, 3),
        )

