import os
import streamlit as st
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.llms import Tongyi
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.callbacks.base import BaseCallbackHandler
from langchain.prompts import PromptTemplate
from langchain.schema import Document
import time
from dotenv import load_dotenv
import json
from datetime import datetime

# 页面配置
st.set_page_config(page_title="本地知识库RAG", layout="wide")

# 初始化会话状态
if "conversations" not in st.session_state:
    st.session_state.conversations = []
if "documents" not in st.session_state:
    st.session_state.documents = []
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None
if "history_files" not in st.session_state:
    st.session_state.history_files = []

# 验证API key
@st.cache_data
def is_valid_api_key(api_key):
    return api_key.startswith("sk-")

# 加载文档
@st.cache_resource
def load_document(file):
    name, extension = os.path.splitext(file.name)
    if extension == ".pdf":
        loader = PyPDFLoader(file.name)
    elif extension == ".docx":
        loader = Docx2txtLoader(file.name)
    else:
        loader = TextLoader(file.name)
    return loader.load()

# 处理文档
@st.cache_resource
def process_documents(_documents):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(_documents)
    embeddings = DashScopeEmbeddings(model="text-embedding-v1")
    vectorstore = Chroma.from_documents(texts, embeddings)
    return vectorstore

# 流式输出处理器
class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)
        
# JSON 编码器
class ConversationEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, Document):
            return {
                "_type": "Document",
                "page_content": obj.page_content,
                "metadata": obj.metadata
            }
        return super().default(obj)

# JSON 解码器
def conversation_object_hook(obj):
    if '_type' in obj and obj['_type'] == 'Document':
        return Document(page_content=obj['page_content'], metadata=obj['metadata'])
    return obj

# 保存对话历史
def save_conversation(conversations):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"conversation_history_{timestamp}.json"
    return filename

# 加载对话历史
def load_conversation(filename):
    with open(filename, "r") as f:
        return json.load(f, object_hook=conversation_object_hook)

# 更新历史文件列表
def update_history_files():
    st.session_state.history_files = [f for f in os.listdir() if f.startswith("conversation_history_") and f.endswith(".json")]
    st.session_state.history_files.sort(reverse=True)  # 最新的对话显示在最上面

# 主应用函数
def main():
    st.title("本地知识库RAG")

    # 左侧栏
    with st.sidebar:
        st.subheader("API Key")
        api_key = st.text_input("输入DashScope API Key", type="password")
        if not is_valid_api_key(api_key):
            st.error("请输入有效的API Key")
            return

        os.environ["DASHSCOPE_API_KEY"] = api_key

        st.subheader("文档")
        uploaded_files = st.file_uploader("上传文档", accept_multiple_files=True)
        if uploaded_files:
            for file in uploaded_files:
                if file.name not in [doc.metadata["source"] for doc in st.session_state.documents]:
                    doc = load_document(file)
                    st.session_state.documents.extend(doc)
            
            # if st.button("处理文档"):
                with st.spinner("正在处理文档..."):
                    st.session_state.vectorstore = process_documents(st.session_state.documents)
                st.success("文档处理完成！")
                # print('123')
                

        # 历史对话选择
        st.subheader("历史对话")
        update_history_files()

        col1, col2 = st.columns([3, 1])
        with col1:
            selected_history = st.selectbox("选择历史对话", st.session_state.history_files)
        with col2:
            if st.button("新对话"):
                st.session_state.conversations = []
                st.rerun()

        if selected_history:
            if st.button("加载选中的对话"):
                try:
                    st.session_state.conversations = load_conversation(selected_history)
                    st.success(f"成功加载对话: {selected_history}")
                    st.rerun()
                except Exception as e:
                    st.error(f"加载对话时出错: {str(e)}")

    # 右侧对话区
    if st.session_state.vectorstore:
        # 显示对话历史
        for message in st.session_state.conversations:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if "source_documents" in message:
                    with st.expander("参考资料"):
                        for doc in message["source_documents"]:
                            st.markdown(f"- **{doc.metadata['source']}**: {doc.page_content[:100]}...")

        # 用户输入
        user_input = st.chat_input("请输入您的问题")
        if user_input:
            st.session_state.conversations.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            llm = Tongyi(model_name="qwen-plus", streaming=True)
            
            prompt_template = """使用以下上下文来回答问题。如果你不知道答案，就说你不知道，不要试图编造答案。

            上下文: {context}

            问题: {question}

            回答:"""
            PROMPT = PromptTemplate(
                template=prompt_template, input_variables=["context", "question"]
            )
            
            chain_type_kwargs = {"prompt": PROMPT}
            qa_chain = RetrievalQA.from_chain_type(
                llm,
                retriever=st.session_state.vectorstore.as_retriever(),
                chain_type="stuff",
                return_source_documents=True,
                chain_type_kwargs=chain_type_kwargs
            )
            
            with st.chat_message("assistant"):
                response_container = st.empty()
                stream_handler = StreamHandler(response_container)
                
                with st.spinner("AI正在思考..."):
                    result = qa_chain({"query": user_input}, callbacks=[stream_handler])
                    
                    if result["source_documents"]:
                        with st.expander("参考资料"):
                            for doc in result["source_documents"]:
                                st.markdown(f"- **{doc.metadata['source']}**: {doc.page_content[:100]}...")
            
            st.session_state.conversations.append({"role": "assistant", "content": result["result"], "source_documents": result["source_documents"]})
            
        # 保存对话按钮
        if st.session_state.conversations:
            save_container = st.empty()
            if save_container.button("保存当前对话"):
                save_form = st.form(key="save_form")
                with save_form:
                    conversation_name = st.text_input("输入对话名称（可选）", value="")
                    submit_button = st.form_submit_button("确认保存")

                if submit_button:
                    if conversation_name:
                        filename = f"conversation_history_{conversation_name}.json"
                    else:
                        filename = save_conversation(st.session_state.conversations)
                    
                    with open(filename, "w") as f:
                        json.dump(st.session_state.conversations, f, cls=ConversationEncoder)
                    st.success(f"对话已保存到 {filename}")
                    update_history_files()
                    save_form.empty()
                    save_container.empty()
                    st.rerun()        
       

if __name__ == "__main__":
    main()
