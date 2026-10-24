import uuid
import streamlit as st
from langgraph_backend import chatbot, model, get_health_response, retrieve_all_threads
from langchain_core.messages import HumanMessage, AIMessage

st.set_page_config(page_title="Clinical Intelligence Agent", page_icon="⚕️")
st.markdown(
    "<h1 style='text-align: center; color: #2E86C1;'>🩺 AI Health Companion</h1>",
    unsafe_allow_html=True,
)
st.caption(
    "<p style='text-align: center; '>Your personalized wellness and nutrition guide</p>",
    unsafe_allow_html=True,
)


# =========================== Utilities ===========================
def generate_thread_id():
    return uuid.uuid4()


def reset_chat():
    thread_id = generate_thread_id()
    st.session_state["thread_id"] = thread_id  # current thread
    add_thread(thread_id)
    st.session_state["chat_history"] = []


def add_thread(thread_id):
    if thread_id not in st.session_state["chat_threads"]:
        st.session_state["chat_threads"].append(thread_id)


def load_conversation(thread_id):
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    return state.values.get("messages", [])


# ======================= Session Initialization ===================
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thread_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_threads()

if "chat_titles" not in st.session_state:
    st.session_state["chat_titles"] = {}

add_thread(st.session_state["thread_id"])


def structured_messages(messages):
    temp_messages = []

    for msg in messages:
        if isinstance(msg, HumanMessage):
            role = "user"
        else:
            role = "assistant"
        temp_messages.append({"role": role, "content": msg.content})

    return temp_messages


def generate_heading(thread_id):
    if thread_id in st.session_state["chat_titles"]:
        return st.session_state["chat_titles"][thread_id]

    messages = load_conversation(thread_id)

    # Convert conversation to readable plain text
    history_text = "\n".join(
        f"{getattr(m, 'type', 'message')}: {m.content}" for m in messages
    )

    prompt = (
        "Generate a very short, concise title (max 4-5 words) summarizing this chat. "
        "Do not use quotes or introductory words:\n\n"
        f"{history_text}"
    )

    response = model.invoke([HumanMessage(content=prompt)])
    title = response.content.strip()
    st.session_state["chat_titles"][thread_id] = title
    return title


# ============================ Sidebar ============================
st.sidebar.title("LangGraph MCP Chatbot")

if st.sidebar.button("New Chat"):
    reset_chat()
    st.rerun()

st.sidebar.header("My Conversations")

for thread_id in st.session_state["chat_threads"][::-1]:
    messages = load_conversation(thread_id)

    if messages and st.sidebar.button(
        generate_heading(thread_id), key=f"btn_{thread_id}"
    ):
        st.session_state["thread_id"] = thread_id
        messages = load_conversation(thread_id)
        st.session_state["chat_history"] = messages
        st.rerun()

# ============================ Main UI ============================

# Render history
for msg in st.session_state["chat_history"]:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    avatar_icon = "👤" if role == "user" else "🏥"
    with st.chat_message(role, avatar=avatar_icon):
        st.markdown(msg.content)

chat_box = st.chat_input("How can I assist you")

if chat_box:
    st.session_state.chat_history.append(HumanMessage(content=chat_box))
    with st.chat_message("user", avatar="👤"):
        st.markdown(chat_box)

    with st.chat_message("assistant", avatar="🏥"):
        with st.spinner("Thinking..."):
            stream_generator = get_health_response(
                user_input=chat_box, thread_id=st.session_state.thread_id
            )
            final_result = st.write_stream(stream_generator)

    st.session_state["chat_history"].append(AIMessage(content=final_result))
    st.rerun()
