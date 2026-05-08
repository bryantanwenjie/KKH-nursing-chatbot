import os
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import streamlit as st

# The Final Fix: Importing from the classic package
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool

# ... the rest of your app.py code ...

# ... the rest of your UI and Agent logic ...

st.success("✅ LangChain loaded successfully! The app is running.")

# ... the rest of your app.py code ...

# --- PAGE CONFIG ---
st.set_page_config(page_title="KKH Nursing Assistant", page_icon="🏥")
st.title("🏥 KKH Clinical Nursing Assistant")

# --- AUTHENTICATION ---
if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

# --- TOOLS & RAG ---
@st.cache_resource
def initialize_retriever():
    loader = PyPDFLoader("Section 01 - Medical Emergencies.pdf")
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(docs)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)
    return vectorstore.as_retriever()

retriever = initialize_retriever()

@tool
def search_nursing_protocols(query: str) -> str:
    """Search the KKH Medical Emergencies PDF for clinical guidelines."""
    docs = retriever.invoke(query)
    return "\n\n".join([doc.page_content for doc in docs])

@tool
def calculate_fluid_requirement(weight_kg: float) -> str:
    """Calculates daily fluid requirements (Holliday-Segar Formula)."""
    if weight_kg <= 10: res = weight_kg * 100
    elif weight_kg <= 20: res = 1000 + (weight_kg - 10) * 50
    else: res = 1500 + (weight_kg - 20) * 20
    return f"The calculated fluid requirement is {res} mL/day."

tools = [calculate_fluid_requirement, search_nursing_protocols]

# --- AGENT SETUP ---
llm = ChatGoogleGenerativeAI(model="gemini-3-flash", temperature=0)
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a KKH Clinical Nursing Assistant. Be precise and professional."),
    ("placeholder", "{chat_history}"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# --- CHAT INTERFACE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_input := st.chat_input("How can I assist with clinical protocols today?"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        response = agent_executor.invoke({
            "input": user_input,
            "chat_history": st.session_state.messages[:-1] 
        })
        full_response = response["output"]
        st.markdown(full_response)
    
    st.session_state.messages.append({"role": "assistant", "content": full_response})