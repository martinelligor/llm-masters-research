

import os
import json
import time
import requests
import datetime
import streamlit as st

from redis import Redis
from llm_api.utils.prompt import SYSTEM_PROMPT_1, SYSTEM_PROMPT_2, SYSTEM_PROMPT_3

MODELS = {
    'OpenAI GPT-4o': 'gpt-4o',
    'Fine-Tuned GPT-4o-mini': 'ft:gpt-4o-mini-2024-07-18:personal::BK1CWj1i',
    'OpenAI GPT-4o-mini': 'gpt-4o-mini',
    'Claude 3 Haiku': 'claude-3-haiku-20240307',
    'Claude 3 Opus': 'claude-3-opus-20240229',
    'Claude 3 Sonnet': 'claude-3-sonnet-20240229',
    'Claude 3.5 Sonnet': 'claude-3-5-sonnet-20240620',
    'Llama 3.2 3B': 'huggingface/meta-llama/Llama-3.2-3B-Instruct',
    'Llama 3.1 8B': 'huggingface/meta-llama/Llama-3.1-8B-Instruct',
    'Mistral 7B': 'huggingface/mistralai/Mistral-7B-Instruct-v0.3',
    'Mixtral 8x7b': 'huggingface/mistralai/Mixtral-8x7B-Instruct-v0.1',
}


# Sample users with roles
USERS = {
    "admin": {"password": "admin123", "role": "admin"},
    "hr_department": {"password": "hr_dept123", "role": "hr"},
    "user": {"password": "user123", "role": "user"},
}


def login():
    st.title("Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    login_button = st.button("Login")

    if login_button:
        user = USERS.get(username)
        if user and user["password"] == password:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.session_state.role = user["role"]
            st.success(f"Welcome {username}!")

            time.sleep(1)
            st.rerun()
        else:
            st.error("Invalid credentials")


def initialize_chat():
    st.sidebar.title("🔬 LLM Research Chat 🔬")
    llm_name = select_model()
    system_prompt = select_system_prompt()

    if not llm_name:
        st.stop()

    chat_id = f"model_{MODELS.get(llm_name)}"

    if chat_id not in st.session_state.keys():
        st.session_state[chat_id] = []

    prompt = st.chat_input(f"Ask '{llm_name}' a question ...")

    return prompt, chat_id, MODELS.get(llm_name), system_prompt


def select_model():
    model_names = [model for model in MODELS.keys()]

    llm_name = st.sidebar.selectbox(
        label=f"Choose Agent (available {len(MODELS)})", options=model_names, index=0)

    if llm_name:
        return llm_name


def select_system_prompt():
    system_prompt = st.sidebar.selectbox(
        "Choose System Prompt",
        ["SYSTEM_PROMPT_1", "SYSTEM_PROMPT_2", "SYSTEM_PROMPT_3"]
    )

    if system_prompt == "SYSTEM_PROMPT_1":
        return SYSTEM_PROMPT_1
    elif system_prompt == "SYSTEM_PROMPT_2":
        return SYSTEM_PROMPT_2

    return SYSTEM_PROMPT_3


def clear_chat(chat_id):
    redis_client = Redis(
        host=os.getenv("REDIS_HOST"),
        port=os.getenv("REDIS_PORT"),
        password=os.getenv("REDIS_PASSWORD")
    )
    redis_client.delete(chat_id)

    st.session_state[chat_id] = []


def print_chat_history_timeline(chat_history_key):
    for message in st.session_state[chat_history_key]:
        role = message["role"]

        if role == "user":
            with st.chat_message("user", avatar="👨🏼‍🔬"):
                question = message["content"]
                st.markdown(f"{question}", unsafe_allow_html=True)

        elif role == "assistant":
            with st.chat_message("assistant", avatar="🤖"):
                st.markdown(message["content"], unsafe_allow_html=True)


def save_conversation(chat_id):
    OUTPUT_DIR = "llm_conversations"
    OUTPUT_DIR = os.path.join(os.getcwd(), OUTPUT_DIR)

    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    filename = f"{OUTPUT_DIR}/{timestamp}"

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    if st.session_state[chat_id]:
        if st.sidebar.button("Save conversation"):
            with open(f"{filename}.json", "w") as f:
                json.dump(st.session_state[chat_id], f, indent=4)

            st.success(f"Conversation saved to {filename}.json")


def chat(model, user_question, chat_id, system_prompt):
    print_chat_history_timeline(chat_id)

    # run the model
    if user_question:
        prompt = user_question
        st.session_state[chat_id].append({"content": f"{prompt}", "role": "user"})
        with st.chat_message("question", avatar="🧑‍🚀"):
            st.write(user_question)

        messages = [
            dict(content=message["content"],
                 role=message["role"]) for message in st.session_state[chat_id]
        ]

        def llm_stream():
            response = requests.post(
                url="http://localhost:8099/get_answer",
                json={
                    "question": prompt,
                    "model": model,
                    "role": st.session_state.role,
                    "system_prompt": system_prompt,
                    "kb_id": "llm_research",
                    "get_references": "true",
                    "thread_id": chat_id
                }
            )

            return response.json()

        # streaming response
        with st.chat_message("response", avatar="🤖"):
            chat_box = st.empty()
            response_message = llm_stream()
            chat_box.write(f'{response_message["answer"]}\n\n**References:**\n')
            st.json(response_message["references"], expanded=False)

        st.session_state[chat_id].append(
            {"content": f"{response_message['answer']}", "role": "assistant"})

        print(st.session_state[chat_id])
        return response_message['answer']


if __name__ == "__main__":
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False

    if not st.session_state.logged_in:
        login()

    if st.session_state.logged_in:
        prompt, chat_id, model, system_prompt = initialize_chat()

        knowledge_base = {}
        if prompt and len(prompt) > 0:
            chat(model, prompt, chat_id, system_prompt)

        if st.session_state[chat_id]:
            clear_conversation = st.sidebar.button("Clear chat")

            if clear_conversation:
                clear_chat(chat_id)
                st.rerun()

        save_conversation(chat_id)

        if st.session_state[chat_id]:
            _ = [st.sidebar.write("") for i in range(43)]
        else:
            _ = [st.sidebar.write("") for i in range(50)]

        logout = st.sidebar.button("Logout")
        if logout:
            clear_chat(chat_id)
            st.session_state[chat_id] = []
            st.session_state.logged_in = False
            st.rerun()
