from __future__ import annotations

import json
import re
from dataclasses import dataclass

from pydantic import ValidationError

from app.schemas import (
    Requirement,
    RetrievedBadCase,
    RetrievedContext,
    RuleDocumentContent,
)
from app.services.llm import LLMClient


@dataclass(frozen=True)
class GenerationResult:
    doc: RuleDocumentContent
    prompt: str
    raw_response: str


class RuleDocumentGenerator:
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    def generate(
        self,
        *,
        requirement: Requirement,
        contexts: list[RetrievedContext],
        bad_cases: list[RetrievedBadCase],
        rule_name: str,
        rule_code: str,
    ) -> GenerationResult:
        prompt = build_prompt(
            requirement=requirement,
            contexts=contexts,
            bad_cases=bad_cases,
            rule_name=rule_name,
            rule_code=rule_code,
        )
        raw_response = self.llm_client.generate(
            prompt,
            generation_hint={
                "requirement": requirement.model_dump(mode="json"),
                "contexts": [item.model_dump(mode="json") for item in contexts],
                "bad_cases": [item.model_dump(mode="json") for item in bad_cases],
                "rule_name": rule_name,
                "rule_code": rule_code,
            },
        )
        return GenerationResult(
            doc=parse_rule_document(raw_response),
            prompt=prompt,
            raw_response=raw_response,
        )


def build_prompt(
    *,
    requirement: Requirement,
    contexts: list[RetrievedContext],
    bad_cases: list[RetrievedBadCase],
    rule_name: str,
    rule_code: str,
) -> str:
    context_block = _format_contexts(contexts)
    bad_case_block = _format_bad_cases(bad_cases)
    schema_json = json.dumps(
        RuleDocumentContent.model_json_schema(),
        ensure_ascii=False,
        indent=2,
    )
    return f"""你要为规则引擎平台生成“单条规则”的说明文档。

目标规则：
- rule_name: {rule_name}
- rule_code: {rule_code}

当前需求文档：
标题：{requirement.title}
业务背景：{requirement.business_background}
需求正文：
{requirement.source_text}

RAG 检索到的上下文：
{context_block}

历史 bad case / 反例约束：
- 这些案例代表历史错误或审核否决模式，只能作为反例参考，生成时必须规避。
{bad_case_block}

输出要求：
1. 只输出 JSON，不要输出 Markdown、解释文本或代码块。
2. 不要编造需求中不存在的外部系统名称；如果是推断，请在对应字段中写“推断：...”。
3. decision_logic 必须是业务人员能读懂的步骤，不要只写技术表达式。
4. examples 至少 2 个，覆盖命中和未命中。
5. exception_handling 必须覆盖字段缺失、外部依赖异常、审计/脱敏。
6. 输出 JSON 必须符合以下 JSON Schema：
{schema_json}
"""


def parse_rule_document(raw_response: str) -> RuleDocumentContent:
    json_text = _extract_json(raw_response)
    try:
        payload = json.loads(json_text)
        return RuleDocumentContent.model_validate(payload)
    except (json.JSONDecodeError, ValidationError) as exc:
        raise ValueError(f"LLM output is not a valid rule document: {exc}") from exc


def _format_contexts(contexts: list[RetrievedContext]) -> str:
    if not contexts:
        return "未检索到上下文。"

    blocks = []
    for index, item in enumerate(contexts, start=1):
        content = item.content[:1400]
        blocks.append(
            f"""[context-{index}]
source_type: {item.source_type}
source_id: {item.source_id}
title: {item.title}
score: {item.score:.4f}
metadata: {json.dumps(item.metadata, ensure_ascii=False)}
content:
{content}"""
        )
    return "\n\n".join(blocks)


def _format_bad_cases(bad_cases: list[RetrievedBadCase]) -> str:
    if not bad_cases:
        return "未检索到 bad case，请继续按当前需求和上下文制作规则。"

    blocks = []
    for index, item in enumerate(bad_cases, start=1):
        blocks.append(
            f"""[badcase-{index}]
rule_code: {item.rule_code}
rule_name: {item.rule_name}
title: {item.title}
score: {item.score:.4f}
metadata: {json.dumps(item.metadata, ensure_ascii=False)}
bad_summary:
{item.bad_summary[:900]}
failure_reason:
{item.failure_reason[:900]}
corrected_hint:
{item.corrected_hint[:900]}"""
        )
    return "\n\n".join(blocks)


def _extract_json(raw_response: str) -> str:
    text = raw_response.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.S)
    if fenced:
        return fenced.group(1)

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return text[start : end + 1]
    return text
