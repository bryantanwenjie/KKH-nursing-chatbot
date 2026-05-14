import base64
import os
import re
import time
import streamlit as st

# --- ORIGINAL IMPORTS ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent

# --- JOESON: Logic Imports ---
from streamlit_mic_recorder import speech_to_text
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS

# --- PAGE CONFIG ---
st.set_page_config(page_title="NursBot | Clinical AI", page_icon="🏥", layout="wide")

# --- AUTHENTICATION ---
if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

# --- JOESON: Azure Environment Setup ---
if "AZURE_OPENAI_API_KEY" in st.secrets:
    os.environ["AZURE_OPENAI_API_KEY"] = st.secrets["AZURE_OPENAI_API_KEY"]
    os.environ["AZURE_OPENAI_ENDPOINT"] = st.secrets["AZURE_OPENAI_ENDPOINT"]
    os.environ["AZURE_OPENAI_API_VERSION"] = st.secrets["AZURE_OPENAI_API_VERSION"]
    AZURE_CHAT_DEPLOY = st.secrets.get("AZURE_OPENAI_CHAT_DEPLOYMENT")
    AZURE_EMBED_DEPLOY = st.secrets.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")

# ==========================================
# ======= ORIGINAL: BACKEND & LOGIC ========
# ==========================================
@st.cache_resource(show_spinner=False)
def initialize_retriever():
    persist_directory = "./chroma_db"
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    try:
        if os.path.exists(persist_directory) and os.listdir(persist_directory):
            vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
            return vectorstore.as_retriever()
        
        pdf_path = "Section 01 - Medical Emergencies.pdf"
        if not os.path.exists(pdf_path): return None
        
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_documents(docs)
        vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings, persist_directory=persist_directory)
        return vectorstore.as_retriever()
    except Exception as e:
        st.error(f"🚨 Vector DB Error: {str(e)}")
        return None

retriever = initialize_retriever()

@tool
def search_nursing_protocols(query: str) -> str:
    """Original: Search clinical guidelines tool"""
    if not retriever: return "Error: Database not initialized."
    docs = retriever.invoke(query)
    results = [f"[Source: Page {doc.metadata.get('page', 'Unknown')}]\n{doc.page_content}" for doc in docs]
    return "\n\n---\n\n".join(results)

@tool
def calculate_fluid_requirement(weight_kg: float) -> str:
    """Original: Holliday-Segar Formula tool"""
    if weight_kg <= 10: res = weight_kg * 100
    elif weight_kg <= 20: res = 1000 + (weight_kg - 10) * 50
    else: res = 1500 + (weight_kg - 20) * 20
    return f"The calculated fluid requirement is {res} mL/day."

tools = [calculate_fluid_requirement, search_nursing_protocols]
llm_gemini = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=0)
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a professional KKH Clinical Nursing Assistant. Always use tools for protocols."),
    ("placeholder", "{chat_history}"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])
agent = create_tool_calling_agent(llm_gemini, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)

# ==========================================
# ======= JOESON: BACKEND & LOGIC ==========
# ==========================================
if "AZURE_OPENAI_API_KEY" in st.secrets:
    azure_llm = AzureChatOpenAI(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        azure_deployment=AZURE_CHAT_DEPLOY,
    )
    azure_embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"],
        api_key=os.environ["AZURE_OPENAI_API_KEY"],
        api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        azure_deployment=AZURE_EMBED_DEPLOY,
    )

@st.cache_resource
def get_joeson_vectorstore():
    """JOESON: FAISS Vector Store using Azure"""
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PDF_PATHS = [os.path.join(BASE_DIR, "Section 01 - Medical Emergencies.pdf"), os.path.join(BASE_DIR, "formula.pdf")]
    docs = []
    for p in PDF_PATHS:
        if os.path.exists(p): docs.extend(PyPDFLoader(p).load())
    chunks = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150).split_documents(docs)
    return FAISS.from_documents(chunks, azure_embeddings)

def joeson_answer_logic(question):
    """JOESON: Azure model logic with specialized calculations"""
    if "last_topic" not in st.session_state: st.session_state.last_topic = ""
    q_low = question.lower()
    if "fluid" in q_low: st.session_state.last_topic = "fluid"
    if any(x in q_low for x in ["bp", "systolic"]): st.session_state.last_topic = "bp"
    
    numbers = re.findall(r"\d+\.?\d*", question)
    if st.session_state.last_topic == "fluid" and numbers:
        weight = float(numbers[0])
        daily = min(weight * 100 if weight <= 10 else 1000 + (weight-10)*50 if weight <= 20 else 1500 + (weight-20)*20, 2500)
        st.session_state.last_topic = ""
        return f"**Azure Result:** Fluid requirement is {daily} ml/day."

    vs = get_joeson_vectorstore()
    context = "\n\n".join([d.page_content for d in vs.as_retriever().invoke(question)])
    return azure_llm.invoke(f"Context: {context}\n\nQuestion: {question}").content

# ==========================================
# =========== ORIGINAL: UI SETUP ===========
# ==========================================
if "app_started" not in st.session_state: st.session_state.app_started = False
if "dark_mode" not in st.session_state: st.session_state.dark_mode = True 
if "studio_expanded" not in st.session_state: st.session_state.studio_expanded = False 
if "chat_sessions" not in st.session_state: st.session_state.chat_sessions = {"New Chat": []}
if "current_chat" not in st.session_state: st.session_state.current_chat = "New Chat"
if "app_mode" not in st.session_state: st.session_state.app_mode = "Gemini Agent"
if "studio_prompt_trigger" not in st.session_state: st.session_state.studio_prompt_trigger = None

bg_color = "#131314" if st.session_state.dark_mode else "#FFFFFF"
text_main = "#E3E3E3" if st.session_state.dark_mode else "#1F2937"
card_bg = "#1E1F20" if st.session_state.dark_mode else "#F9FAFB"

st.markdown(f"""
<style>
    [data-testid="stAppViewContainer"] {{ background: {bg_color}; }}
    .hero-title {{ font-size: 3.8rem; font-weight: 800; color: {text_main}; }}
    .hero-title span {{ color: #0d9488; }}
    .badge {{ background-color: #E8F0FE; color: #1A73E8; padding: 6px 16px; border-radius: 20px; font-weight: 700; }}
    .breadcrumb {{ color: #C4C7C5; font-size: 12px; border-bottom: 1px solid #444746; padding: 10px 0; margin-bottom: 20px; }}
    .studio-btn-wrapper div[data-testid="stButton"] button {{
        width: 100%; text-align: left; background-color: {bg_color}; border: 1px solid #444746; border-radius: 12px; padding: 15px; color: {text_main};
    }}
</style>
""", unsafe_allow_html=True)

# --- VIEW 1: LANDING PAGE ---
if not st.session_state.app_started:
    col1, col2 = st.columns([1.2, 1], gap="large")
    with col1:
        st.markdown('<div class="badge">✨ AI-POWERED CLINICAL ASSISTANT</div>', unsafe_allow_html=True)
        st.markdown('<div class="hero-title">Smarter Nursing<br>with <span>AI</span> Support</div>', unsafe_allow_html=True)
        if st.button("Try Chatbot ➔", type="primary", key="try_btn"):
            st.session_state.app_started = True
            st.rerun()
    with col2: st.info("Visual Placeholder: 'nurse.png'")

# --- VIEW 2: CHAT INTERFACE ---
else:
    with st.sidebar:
        st.markdown(f"<h3 style='color:{text_main};'>🩺 NursBot</h3>", unsafe_allow_html=True)
        # JOESON: Integrating the engine toggle into your sidebar
        st.session_state.app_mode = st.radio("AI Engine:", ["Gemini Agent", "Azure (Joeson)"])
        
        if st.button("➕ New chat", type="primary", use_container_width=True):
            name = f"Chat {len(st.session_state.chat_sessions)+1}"
            st.session_state.chat_sessions[name] = []
            st.session_state.current_chat = name
            st.rerun()
        
        st.divider()
        for chat_id in reversed(list(st.session_state.chat_sessions.keys())):
            if st.button(f"💬 {chat_id[:15]}...", key=f"h_{chat_id}", use_container_width=True):
                st.session_state.current_chat = chat_id
                st.rerun()

    chat_col, studio_col = st.columns([3, 1] if st.session_state.studio_expanded else [1, 0.01])
    
    with chat_col:
        st.markdown("<div class='breadcrumb'>📁 KKH Workspace / Section 01</div>", unsafe_allow_html=True)
        if st.button("⚡ Studio", use_container_width=True):
            st.session_state.studio_expanded = not st.session_state.studio_expanded
            st.rerun()

        chat_container = st.container(height=450, border=False)
        with chat_container:
            for m in st.session_state.chat_sessions[st.session_state.current_chat]:
                with st.chat_message(m["role"]): st.markdown(m["content"])

        with st.popover("➕ Tools & Attachments"):
            app_mode = st.selectbox("Mode:", ["Clinical Text", "Vision", "Voice (Joeson)"])
            # JOESON: Speech-to-Text integration
            if app_mode == "Voice (Joeson)":
                voice_text = speech_to_text(language="en", use_container_width=True, just_once=True, key="JOE_STT")
                if voice_text:
                    st.session_state.chat_sessions[st.session_state.current_chat].append({"role": "user", "content": voice_text})
                    st.rerun()

        user_input = st.chat_input("Message NursBot...")
        if st.session_state.studio_prompt_trigger:
            user_input = st.session_state.studio_prompt_trigger
            st.session_state.studio_prompt_trigger = None

        if user_input:
            st.session_state.chat_sessions[st.session_state.current_chat].append({"role": "user", "content": user_input})
            st.rerun()

    if len(st.session_state.chat_sessions[st.session_state.current_chat]) > 0 and st.session_state.chat_sessions[st.session_state.current_chat][-1]["role"] == "user":
        with chat_col:
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    latest = st.session_state.chat_sessions[st.session_state.current_chat][-1]["content"]
                    # JOESON: Logic routing
                    if st.session_state.app_mode == "Azure (Joeson)":
                        ans = joeson_answer_logic(latest)
                    else:
                        res = agent_executor.invoke({"input": latest, "chat_history": []})
                        ans = res["output"]
                    st.markdown(ans)
                    st.session_state.chat_sessions[st.session_state.current_chat].append({"role": "assistant", "content": ans})
                    st.rerun()

    if st.session_state.studio_expanded:
        with studio_col:
            st.markdown(f"<h3 style='color:{text_main};'>Studio</h3>", unsafe_allow_html=True)
            if st.button("📝 Generate Quiz"):
                st.session_state.studio_prompt_trigger = "Generate a 3-question MCQ quiz."
                st.rerun()