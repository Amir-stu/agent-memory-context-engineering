from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.agent_graph import scoped_thread_id

USER = "user-zed"


def test_reply_when_memory_exists_injects_it_into_system_prompt(agent, memory_manager, chat_model):
    memory_manager.add_fact(USER, "Prefers DuckDB for analytics work")

    turn = agent.reply(USER, "t1", "Which database for analytics?")

    system_message = chat_model.calls[-1][0]
    assert isinstance(system_message, SystemMessage)
    assert "- Prefers DuckDB for analytics work" in system_message.content
    assert [record.text for record in turn.memories] == ["Prefers DuckDB for analytics work"]


def test_reply_when_called_writes_turn_to_long_term_memory(agent, memory_manager):
    agent.reply(USER, "t1", "My name is Zed Testperson")
    assert memory_manager.wait_for_pending() == ()
    assert [record.text for record in memory_manager.list_memories(USER)] == ["My name is Zed Testperson"]


def test_reply_when_same_thread_keeps_short_term_history(agent, chat_model):
    agent.reply(USER, "t1", "first question")
    agent.reply(USER, "t1", "second question")

    sent = chat_model.calls[-1]
    assert [type(message) for message in sent] == [SystemMessage, HumanMessage, AIMessage, HumanMessage]
    assert sent[1].content == "first question"


def test_reply_when_new_thread_starts_without_short_term_history(agent, chat_model):
    agent.reply(USER, "t1", "first question")
    agent.reply(USER, "t2", "fresh start")
    assert [message.content for message in chat_model.calls[-1][1:]] == ["fresh start"]


def test_reply_when_history_exceeds_window_trims_oldest_turns(agent, chat_model, settings):
    for index in range(settings.history_window):
        agent.reply(USER, "t1", f"question {index}")
    sent = chat_model.calls[-1]
    assert len(sent) - 1 <= settings.history_window
    assert isinstance(sent[1], HumanMessage)


def test_reply_when_other_user_has_memories_does_not_leak_them(agent, memory_manager, chat_model):
    memory_manager.add_fact("user-quux", "Secret project codename Bluefin")
    agent.reply(USER, "t1", "Tell me about project codename Bluefin")
    assert "Bluefin" not in chat_model.calls[-1][0].content


def test_history_when_two_users_share_thread_name_keeps_them_apart(agent):
    agent.reply(USER, "main", "from zed")
    agent.reply("user-quux", "main", "from quux")
    assert next(message.content for message in agent.history(USER, "main")) == "from zed"
    assert next(message.content for message in agent.history("user-quux", "main")) == "from quux"


def test_clear_thread_when_called_drops_history_but_keeps_long_term_memory(agent, memory_manager):
    agent.reply(USER, "t1", "My goal is an ML platform role")
    memory_manager.wait_for_pending()

    agent.clear_thread(USER, "t1")

    assert agent.history(USER, "t1") == []
    assert len(memory_manager.list_memories(USER)) == 1


def test_list_threads_returns_only_this_users_threads(agent):
    agent.reply(USER, "alpha", "hi")
    agent.reply(USER, "beta", "hi")
    agent.reply("user-quux", "gamma", "hi")
    assert agent.list_threads(USER) == ("alpha", "beta")


def test_scoped_thread_id_prefixes_user_id():
    assert scoped_thread_id("user-zed", "t1") == "user-zed::t1"


def test_sqlite_checkpointer_when_agent_restarts_restores_thread_history(chat_model, fake_backend, settings):
    from src.agent_graph import CompanionAgent, open_checkpointer
    from src.memory_manager import MemoryManager

    first = CompanionAgent(chat_model, MemoryManager(fake_backend), open_checkpointer(settings), settings)
    first.reply(USER, "t1", "remember this thread")
    first.close()

    restarted = CompanionAgent(chat_model, MemoryManager(fake_backend), open_checkpointer(settings), settings)
    assert restarted.history(USER, "t1")[0].content == "remember this thread"
    restarted.close()
