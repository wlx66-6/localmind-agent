from __future__ import annotations

from app.generation.prompt import REFUSAL_TEXT, is_refusal


def test_exact_marker_is_a_refusal() -> None:
    assert is_refusal(REFUSAL_TEXT)


def test_paraphrase_after_the_marker_is_a_refusal() -> None:
    # The cloud model continues the sentence after the marker.
    answer = "资料中未找到足够依据。现有资料只说明了备份范围，没有提到频率。"
    assert is_refusal(answer)


def test_local_model_dropping_the_tail_is_still_a_refusal() -> None:
    # Observed on qwen3.5:4b during the stage 5 evaluation: the refusal is
    # real, only the wording of the marker changed.
    answer = "资料中未找到关于“如何同步索引”的具体操作步骤或工具推荐。"
    assert is_refusal(answer)


def test_leading_whitespace_is_ignored() -> None:
    assert is_refusal("\n\n  资料中未找到足够依据。")


def test_normal_answer_is_not_a_refusal() -> None:
    assert not is_refusal("RRF 公式中的 k 通常取 60 [1]。")


def test_marker_in_the_middle_is_not_a_refusal() -> None:
    # Only the opening characters matter, so a quoted phrase does not flip a
    # real answer into a refusal.
    answer = "根据资料 [1]，提示词要求模型在无依据时回答“资料中未找到足够依据”。"
    assert not is_refusal(answer)
