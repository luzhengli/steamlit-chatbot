# from openai import OpenAI
from http import client
import streamlit as st
from langchain_community.llms import Tongyi
from langchain_core.runnables import ConfigurableField

st.title("Chat Robot")


# dashscope_api_key = st.secrets["DASHSCOPE_API_KEY"]
dashscope_api_key = st.sidebar.text_input('DashScope API Key', type='password')
if not dashscope_api_key.startswith('sk-'):
    st.warning('Please enter your DashScope API key!', icon='⚠')
    st.stop()

@st.cache_resource()
def get_model(api_key: str):
    return Tongyi(model='qwen-turbo', dashscope_api_key=api_key).configurable_fields(
    top_p=ConfigurableField( # 支持运行时传入模型参数
        id="llm_top_p",
        name="LLM top_p",
        description="The top_p of the LLM",
    )
)
client = get_model(dashscope_api_key)

# if "qwen_model" not in st.session_state:
#     st.session_state["qwen_model"] = "qwen-turbo"

if "messages" not in st.session_state:
    st.session_state.messages = []

# 每次运行一次都会显示所有历史对话
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 每次输入对话后，都会将本轮对话添加到历史会话中
if prompt := st.chat_input("What is up?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        stream = client.stream(st.session_state.messages)
        response = st.write_stream(stream)
    st.session_state.messages.append({"role": "assistant", "content": response})
