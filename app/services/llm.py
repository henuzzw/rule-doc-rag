from __future__ import annotations

import json
import re
from typing import Any, Protocol

from app.config import Settings, get_settings


class LLMClient(Protocol):
    def generate(
        self,
        prompt: str,
        *,
        generation_hint: dict[str, Any],
    ) -> str:
        """Return a JSON string that follows the rule document schema."""


class MockLLMClient:
    """Offline generator for demo, tests and screenshots."""

    def generate(
        self,
        prompt: str,
        *,
        generation_hint: dict[str, Any],
    ) -> str:
        requirement = generation_hint["requirement"]
        contexts = generation_hint["contexts"]
        rule_name = generation_hint["rule_name"]
        rule_code = generation_hint["rule_code"]

        source_text = requirement["source_text"]
        fields = _extract_fields(source_text)
        if not fields:
            fields = ["application_id", "customer_id", "rule_input"]

        context_titles = [item["title"] for item in contexts[:3]]
        logic = _extract_logic(source_text)

        payload = {
            "rule_name": rule_name,
            "rule_code": rule_code,
            "document_summary": (
                f"根据需求《{requirement['title']}》生成的规则说明。"
                f"系统会在规则触发时读取关键入参，执行条件判断，并输出命中结果。"
            ),
            "business_goal": requirement["business_background"],
            "trigger_timing": "进件/授信/额度调整等决策流程中，规则编排平台调用该规则节点时触发。",
            "input_fields": fields,
            "output_fields": ["hit", "decision", "reason_code", "reason_message"],
            "decision_logic": logic,
            "pseudo_code": _build_pseudo_code(rule_code, fields, logic),
            "examples": [
                {
                    "title": "规则命中示例",
                    "input": {fields[0]: "示例值", "application_status": "SUBMITTED"},
                    "expected_output": {
                        "hit": True,
                        "decision": "REJECT_OR_REVIEW",
                        "reason_code": rule_code,
                    },
                    "explanation": "当申请信息满足规则阈值或黑名单条件时，输出命中并进入拒绝或人工复核链路。",
                },
                {
                    "title": "规则未命中示例",
                    "input": {fields[0]: "正常值", "application_status": "SUBMITTED"},
                    "expected_output": {
                        "hit": False,
                        "decision": "PASS",
                        "reason_code": "",
                    },
                    "explanation": "当输入字段完整且未满足风险条件时，规则返回通过。",
                },
            ],
            "exception_handling": [
                "核心入参缺失时返回 NOT_AVAILABLE，并记录字段级缺失原因。",
                "外部服务超时或不可用时按配置降级：可返回人工复核，也可跳过并记录告警。",
                "禁止在规则日志中输出身份证号、手机号等完整敏感信息。",
            ],
            "dependencies": context_titles or ["需求文档", "历史规则文档"],
            "change_risk": "调整阈值、名单来源或外部服务降级策略时，可能影响拒绝率、复核率和通过率，需要灰度验证。",
            "test_points": [
                "覆盖命中、未命中、边界阈值、空值、异常返回等分支。",
                "使用历史生产样本回放，对比规则上线前后的命中率差异。",
                "校验 reason_code、决策结果、审计日志和脱敏输出是否符合平台规范。",
            ],
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)


class DeepSeekLLMClient:
    def __init__(self, settings: Settings):
        if not settings.deepseek_api_key:
            raise ValueError("DEEPSEEK_API_KEY is required when LLM_MODE=deepseek")

        self.settings = settings
        self.client = _build_openai_client(
            settings=settings,
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
        )

    def generate(
        self,
        prompt: str,
        *,
        generation_hint: dict[str, Any],
    ) -> str:
        completion = self.client.chat.completions.create(
            model=self.settings.deepseek_model,
            temperature=0.2,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": "你是资深规则引擎专家和技术文档作者，只输出严格 JSON。",
                },
                {"role": "user", "content": prompt},
            ],
        )
        return completion.choices[0].message.content or "{}"


class QianfanCodingLLMClient:
    """Baidu Qianfan coding endpoint via OpenAI-compatible chat completions."""

    def __init__(self, settings: Settings):
        if not settings.qianfan_api_key:
            raise ValueError("QIANFAN_API_KEY is required when LLM_MODE=qianfan")

        self.settings = settings
        self.client = _build_openai_client(
            settings=settings,
            api_key=settings.qianfan_api_key,
            base_url=settings.qianfan_base_url,
        )

    def generate(
        self,
        prompt: str,
        *,
        generation_hint: dict[str, Any],
    ) -> str:
        completion = self.client.chat.completions.create(
            model=self.settings.qianfan_model,
            temperature=0.2,
            response_format={"type": "json_object"},
            max_tokens=self.settings.qianfan_max_tokens,
            messages=[
                {
                    "role": "system",
                    "content": "你是资深规则引擎专家和技术文档作者，只输出严格 JSON。",
                },
                {"role": "user", "content": prompt},
            ],
        )
        return completion.choices[0].message.content or "{}"


def build_llm_client(settings: Settings | None = None) -> LLMClient:
    settings = settings or get_settings()
    if settings.llm_mode == "deepseek":
        return DeepSeekLLMClient(settings)
    if settings.llm_mode == "qianfan":
        return QianfanCodingLLMClient(settings)
    return MockLLMClient()


def _build_openai_client(*, settings: Settings, api_key: str, base_url: str):
    import httpx
    from openai import OpenAI

    client_kwargs: dict[str, Any] = {
        "api_key": api_key,
        "base_url": base_url,
    }
    proxy_url = settings.https_proxy or settings.http_proxy
    if proxy_url:
        client_kwargs["http_client"] = httpx.Client(proxy=proxy_url)
    return OpenAI(**client_kwargs)


def _extract_fields(text: str) -> list[str]:
    dotted = re.findall(r"\b[a-z][a-z0-9_]*(?:\.[a-z0-9_]+)+\b", text)
    snake = re.findall(r"`([a-z][a-z0-9_]+)`", text)
    seen: set[str] = set()
    fields: list[str] = []
    for field in dotted + snake:
        if field not in seen:
            seen.add(field)
            fields.append(field)
    return fields[:12]


def _extract_logic(text: str) -> list[str]:
    candidates = []
    for raw_line in re.split(r"[\n。；;]", text):
        line = raw_line.strip(" -\t")
        if len(line) < 6:
            continue
        if any(keyword in line for keyword in ("当", "若", "如果", "命中", "阈值", "拒绝", "复核", "通过")):
            candidates.append(line)

    if candidates:
        return candidates[:8]
    return [
        "读取规则入参并完成字段完整性校验。",
        "按需求文档定义的条件表达式执行判断。",
        "命中风险条件时输出 hit=true，并返回对应 reason_code。",
        "未命中时输出 hit=false，允许后续规则继续执行。",
    ]


def _build_pseudo_code(rule_code: str, fields: list[str], logic: list[str]) -> str:
    first_field = fields[0]
    first_logic = logic[0] if logic else "满足需求中定义的风险条件"
    return "\n".join(
        [
            f"function {rule_code}(input):",
            f"    assert_present(input.{first_field})",
            f"    if {first_logic}:",
            f"        return hit('{rule_code}', decision='REJECT_OR_REVIEW')",
            "    return pass()",
        ]
    )

