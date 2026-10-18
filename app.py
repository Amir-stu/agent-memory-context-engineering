"""Streamlit chat with a live Memory Inspector.

streamlit run app.py
"""

from datetime import datetime

import streamlit as st
from dotenv import load_dotenv

from src.agent_graph import CompanionAgent, create_agent
from src.config import load_settings
from src.memory_manager import MemoryRecord, validate_identifier

DEFAULT_USER_ID = "demo-user"
DEFAULT_THREAD_ID = "main"


@st.cache_resource
def get_agent() -> CompanionAgent:
    load_dotenv()
    return create_agent(load_settings())


def new_thread_id() -> str:
    return f"thread-{datetime.now():%Y%m%d-%H%M%S}"


def select_user() -> str | None:
    raw_user_id = st.sidebar.text_input("User ID", value=DEFAULT_USER_ID, help="Each user has separate memory.")
    try:
        return validate_identifier(raw_user_id, "user_id")
    except ValueError as error:
        st.sidebar.error(str(error))
        return None


def select_thread(agent: CompanionAgent, user_id: str) -> str:
    threads = agent.list_threads(user_id)
    current = st.session_state.get("thread_id", DEFAULT_THREAD_ID)
    options = tuple(dict.fromkeys((current, *threads)))
    thread_id = st.sidebar.selectbox("Conversation thread", options, index=options.index(current))
    st.session_state["thread_id"] = thread_id

    new_col, clear_col = st.sidebar.columns(2)
    if new_col.button("New thread", use_container_width=True):
        st.session_state["thread_id"] = new_thread_id()
        st.rerun()
    if clear_col.button("Clear thread", use_container_width=True, help="Deletes this thread's messages only."):
        agent.clear_thread(user_id, thread_id)
        st.rerun()
    return thread_id


def render_history(agent: CompanionAgent, user_id: str, thread_id: str) -> None:
    for message in agent.history(user_id, thread_id):
        role = "user" if message.type == "human" else "assistant"
        st.chat_message(role).markdown(message.text)


def render_injected(memories: tuple[MemoryRecord, ...]) -> None:
    with st.expander(f"Context engineering: {len(memories)} memories injected into this turn"):
        if not memories:
            st.caption("No stored memory matched this message.")
        for memory in memories:
            score = f"{memory.score:.2f}" if memory.score is not None else "n/a"
            st.markdown(f"- `{score}` {memory.text}")


def handle_input(agent: CompanionAgent, user_id: str, thread_id: str) -> None:
    prompt = st.chat_input("Message your companion")
    if not prompt:
        return
    st.chat_message("user").markdown(prompt)
    try:
        turn = agent.reply(user_id, thread_id, prompt)
    except Exception as error:  # surface provider/auth errors in the UI instead of a blank page
        st.error(f"The model call failed: {error}")
        return
    with st.chat_message("assistant"):
        st.markdown(turn.answer)
        render_injected(turn.memories)


def render_memory_inspector(agent: CompanionAgent, user_id: str) -> None:
    sidebar = st.sidebar
    sidebar.divider()
    sidebar.subheader("Current remembered facts about you")
    with sidebar.status("Saving new memories…", expanded=False) as status:
        errors = agent.memory.wait_for_pending()
        status.update(label="Memory up to date", state="complete")
    for error in errors:
        sidebar.warning(f"A memory write failed: {error}")

    memories = agent.memory.list_memories(user_id)
    if not memories:
        sidebar.caption("Nothing stored yet. Tell the companion about yourself.")
    for memory in memories:
        edited = sidebar.text_area("fact", memory.text, key=f"text-{memory.id}", label_visibility="collapsed")
        save_col, delete_col = sidebar.columns(2)
        if save_col.button("Save", key=f"save-{memory.id}", disabled=edited.strip() == memory.text):
            agent.memory.edit(memory.id, edited)
            st.rerun()
        if delete_col.button("Forget", key=f"forget-{memory.id}"):
            agent.memory.forget(memory.id)
            st.rerun()

    with sidebar.form("add-fact", clear_on_submit=True):
        fact = st.text_input("Add a fact manually")
        if st.form_submit_button("Remember") and fact.strip():
            agent.memory.add_fact(user_id, fact)
            st.rerun()


def main() -> None:
    st.set_page_config(page_title="Persistent AI Companion", layout="wide")
    st.title("Persistent AI Companion")
    st.caption("Short-term memory lives in the thread. Long-term memory follows the user across threads.")

    agent = get_agent()
    user_id = select_user()
    if user_id is None:
        return
    thread_id = select_thread(agent, user_id)
    render_history(agent, user_id, thread_id)
    handle_input(agent, user_id, thread_id)
    render_memory_inspector(agent, user_id)


main()
