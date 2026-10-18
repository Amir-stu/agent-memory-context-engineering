from datetime import date

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.context_builder import (
    EMPTY_MEMORY_NOTE,
    build_context,
    build_system_prompt,
    format_memory_block,
    latest_user_text,
)
from src.memory_manager import MemoryRecord

TODAY = date(2026, 10, 18)


def record(memory_id: str, text: str, score: float | None = None) -> MemoryRecord:
    return MemoryRecord(id=memory_id, text=text, score=score)


def test_format_memory_block_when_empty_returns_empty_note():
    assert format_memory_block((), max_chars=500) == EMPTY_MEMORY_NOTE


def test_format_memory_block_when_scored_orders_highest_score_first():
    block = format_memory_block((record("a", "Likes tea", 0.2), record("b", "Name is Zed", 0.9)), max_chars=500)
    assert block.splitlines() == ["- Name is Zed", "- Likes tea"]


def test_format_memory_block_when_unscored_keeps_input_order():
    block = format_memory_block((record("a", "First"), record("b", "Second")), max_chars=500)
    assert block.splitlines() == ["- First", "- Second"]


def test_format_memory_block_when_duplicate_text_keeps_one_line():
    block = format_memory_block((record("a", "Uses Python", 0.5), record("b", "uses python ", 0.4)), max_chars=500)
    assert block.splitlines() == ["- Uses Python"]


def test_format_memory_block_when_over_budget_drops_whole_lowest_ranked_items():
    memories = (record("a", "x" * 30, 0.9), record("b", "y" * 30, 0.5), record("c", "z" * 30, 0.1))
    block = format_memory_block(memories, max_chars=70)
    assert block.splitlines() == ["- " + "x" * 30, "- " + "y" * 30]


def test_format_memory_block_when_first_item_exceeds_budget_returns_empty_note():
    assert format_memory_block((record("a", "x" * 100),), max_chars=10) == EMPTY_MEMORY_NOTE


def test_format_memory_block_when_text_blank_skips_it():
    block = format_memory_block((record("a", "   "), record("b", "Real fact")), max_chars=500)
    assert block == "- Real fact"


def test_build_system_prompt_when_memories_present_wraps_them_in_tagged_section():
    prompt = build_system_prompt((record("a", "Name is Zed"),), today=TODAY, max_memory_chars=500)
    assert "<user_memory>\n- Name is Zed\n</user_memory>" in prompt
    assert "2026-10-18" in prompt


def test_build_system_prompt_when_no_memories_says_so():
    prompt = build_system_prompt((), today=TODAY, max_memory_chars=500)
    assert f"<user_memory>\n{EMPTY_MEMORY_NOTE}\n</user_memory>" in prompt


def test_build_context_when_history_long_keeps_only_window_after_system_prompt():
    history = [HumanMessage(content=f"q{index}") for index in range(10)]
    context = build_context("SYS", history, window=3)
    assert isinstance(context[0], SystemMessage)
    assert context[0].content == "SYS"
    assert [message.content for message in context[1:]] == ["q7", "q8", "q9"]


def test_build_context_when_window_cuts_mid_exchange_starts_on_user_turn():
    history = [HumanMessage(content="q1"), AIMessage(content="a1"), HumanMessage(content="q2")]
    context = build_context("SYS", history, window=2)
    assert [message.content for message in context[1:]] == ["q2"]


def test_build_context_when_history_is_old_system_messages_drops_them():
    history = [SystemMessage(content="stale"), HumanMessage(content="q")]
    assert [message.content for message in build_context("SYS", history, window=5)] == ["SYS", "q"]


def test_latest_user_text_returns_last_human_message():
    history = [HumanMessage(content="first"), AIMessage(content="a"), HumanMessage(content="last")]
    assert latest_user_text(history) == "last"


def test_latest_user_text_when_no_human_message_returns_empty_string():
    assert latest_user_text([AIMessage(content="a")]) == ""
