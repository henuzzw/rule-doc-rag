import json
import math

from app.schemas import Requirement, RetrievedBadCase, RetrievedContext
from app.services.embedding import HashEmbeddingService
from app.services.generator import (
    RuleDocumentGenerator,
    build_prompt,
    parse_rule_document,
)
from app.services.llm import MockLLMClient


def test_hash_embedding_has_expected_dimension_and_is_normalized():
    service = HashEmbeddingService(dimension=64)

    vector = service.embed_text("手机号三要素 customer.phone customer.id_no")

    assert len(vector) == 64
    assert math.isclose(
        math.sqrt(sum(value * value for value in vector)),
        1.0,
        rel_tol=1e-4,
    )


def test_parse_rule_document_accepts_json_fence():
    raw = """```json
{
  "rule_name": "年龄准入",
  "rule_code": "RULE_AGE",
  "document_summary": "summary",
  "business_goal": "goal",
  "trigger_timing": "trigger",
  "input_fields": ["customer.age"],
  "output_fields": ["hit"],
  "decision_logic": ["年龄小于阈值时拒绝"],
  "pseudo_code": "if age < 22: reject()",
  "examples": [
    {
      "title": "命中",
      "input": {"customer.age": 18},
      "expected_output": {"hit": true},
      "explanation": "年龄过低"
    }
  ],
  "exception_handling": ["年龄为空时复核"],
  "dependencies": ["需求文档"],
  "change_risk": "影响准入率",
  "test_points": ["边界值测试"]
}
```"""

    document = parse_rule_document(raw)

    assert document.rule_code == "RULE_AGE"
    assert document.examples[0].expected_output["hit"] is True


def test_build_prompt_includes_bad_cases():
    requirement = Requirement(
        id="REQ-1",
        title="示例需求",
        business_background="用于验证 bad case 是否进入 prompt",
        source_text="规则目标：示例。",
    )
    context = RetrievedContext(
        id="ctx-1",
        source_type="history_rule",
        source_id="H-1",
        title="历史规则",
        content="历史内容",
        metadata={},
        score=0.91,
    )
    bad_case = RetrievedBadCase(
        id="bad-1",
        rule_code="RULE_X",
        rule_name="示例规则",
        title="把超时当通过",
        bad_summary="超时后直接放行。",
        failure_reason="放大风险。",
        corrected_hint="应转人工复核。",
        metadata={"severity": "high"},
        score=0.88,
    )

    prompt = build_prompt(
        requirement=requirement,
        contexts=[context],
        bad_cases=[bad_case],
        rule_name="示例规则",
        rule_code="RULE_X",
    )

    assert "反例约束" in prompt
    assert "把超时当通过" in prompt
    assert "应转人工复核" in prompt


def test_mock_generator_uses_requirement_and_context():
    generator = RuleDocumentGenerator(MockLLMClient())
    requirement = Requirement(
        id="REQ-1",
        title="手机号实名校验",
        business_background="降低冒名申请风险",
        source_text=(
            "入参字段：customer.phone、customer.id_no、operator.verify_result。\n"
            "当 operator.verify_result = NOT_MATCH，规则命中并拒绝。"
        ),
    )
    context = RetrievedContext(
        id="ctx-1",
        source_type="history_rule",
        source_id="H-1",
        title="历史规则文档：身份证姓名一致性校验",
        content="历史上下文",
        metadata={},
        score=0.91,
    )
    bad_case = RetrievedBadCase(
        id="bad-1",
        rule_code="RULE_PHONE_REALNAME",
        rule_name="手机号三要素实名一致性校验",
        title="超时直接通过",
        bad_summary="运营商超时后放行。",
        failure_reason="违背兜底策略。",
        corrected_hint="应进入人工复核。",
        metadata={},
        score=0.82,
    )

    result = generator.generate(
        requirement=requirement,
        contexts=[context],
        bad_cases=[bad_case],
        rule_name="手机号三要素实名一致性校验",
        rule_code="RULE_PHONE_REALNAME",
    )

    as_json = json.loads(result.raw_response)
    assert as_json["rule_code"] == "RULE_PHONE_REALNAME"
    assert "operator.verify_result" in result.doc.input_fields
    assert "历史规则文档：身份证姓名一致性校验" in result.doc.dependencies
