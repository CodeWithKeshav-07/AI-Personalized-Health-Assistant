import streamlit as st
from main import get_health_response
from langchain.messages import HumanMessage, AIMessage, SystemMessage


st.markdown("<h1 style='text-align: center; color: #2E86C1;'>🩺 AI Health Companion</h1>", unsafe_allow_html=True)
st.caption("<p style='text-align: center; '>Your personalized wellness and nutrition guide</p>", unsafe_allow_html=True)


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for msg in st.session_state.chat_history:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    avatar_icon = "👤" if role == "user" else "🏥" 
    with st.chat_message(role, avatar=avatar_icon):
        st.markdown(msg.content)
        
        
chat_box = st.chat_input(
    "How can I assist you",
)
if chat_box:
    st.session_state.chat_history.append(HumanMessage(content=chat_box))
    with st.chat_message("user",avatar="👤"):
        st.markdown(chat_box)
        
    with st.chat_message("assistant",avatar="🏥"):
        with st.spinner("Thinking..."):
          stream_generator = get_health_response(
                user_input=chat_box,
                chat_history=st.session_state.chat_history[:-1]
            )
          final_result = st.write_stream(stream_generator)
    
    st.session_state.chat_history.append(AIMessage(content=final_result))
    


