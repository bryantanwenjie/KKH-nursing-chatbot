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
from langchain_core.messages import HumanMessage, AIMessage
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent

# --- JOESON: Speech-to-Text & Azure Imports ---
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
    # Added deployments for chat and embeddings
    AZURE_CHAT_DEPLOY = st.secrets.get("AZURE_OPENAI_CHAT_DEPLOYMENT")
    AZURE_EMBED_DEPLOY = st.secrets.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")

# ==========================================
# ======= ORIGINAL: BACKEND & LOGIC ========
# ==========================================

@st.cache_resource(show_spinner=False)
def initialize_retriever():
    """Original: Chroma Vector DB initialization"""
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
    warning = ""
    if weight_kg < 2.0 or weight_kg > 80.0:
        warning = "\n\n⚠️ **CLINICAL WARNING:** Weight outside standard pediatric ranges."
    if weight_kg <= 10: res = weight_kg * 100
    elif weight_kg <= 20: res = 1000 + (weight_kg - 10) * 50
    else: res = 1500 + (weight_kg - 20) * 20
    return f"The calculated fluid requirement is {res} mL/day.{warning}"

tools = [calculate_fluid_requirement, search_nursing_protocols]
llm_gemini = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a professional KKH Clinical Nursing Assistant. Use search_nursing_protocols for all KKH-specific info."),
    ("placeholder", "{chat_history}"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_tool_calling_agent(llm_gemini, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)

# ==========================================
# ======= JOESON: BACKEND & LOGIC ==========
# ==========================================

# JOESON: Azure Model & Embeddings Setup
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
    """JOESON: FAISS Vector Store using Azure Embeddings"""
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PDF_PATHS = [
        os.path.join(BASE_DIR, "Section 01 - Medical Emergencies.pdf"),
        os.path.join(BASE_DIR, "formula.pdf")
    ]
    all_documents = []
    for path in PDF_PATHS:
        if os.path.exists(path):
            all_documents.extend(PyPDFLoader(path).load())
    
    chunks = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150).split_documents(all_documents)
    return FAISS.from_documents(chunks, azure_embeddings)

def joeson_answer_logic(question):
    """JOESON: Stateful logic for BP/Fluids and Azure RAG"""
    if "last_topic" not in st.session_state: st.session_state.last_topic = ""
    
    q_low = question.lower()
    if "fluid" in q_low: st.session_state.last_topic = "fluid"
    if any(x in q_low for x in ["bp", "blood pressure", "systolic"]): st.session_state.last_topic = "bp"

    numbers = re.findall(r"\d+\.?\d*", question)
    
    # JOESON: Calculation routing
    if st.session_state.last_topic == "fluid" and numbers:
        weight = float(numbers[0])
        daily = min(weight * 100 if weight <= 10 else 1000 + (weight-10)*50 if weight <= 20 else 1500 + (weight-20)*20, 2500)
        st.session_state.last_topic = ""
        return f"**Azure Fluid Calculation:** {daily} ml/day based on weight {weight}kg."

    # JOESON: RAG Response
    vectorstore = get_joeson_vectorstore()
    docs = vectorstore.as_retriever(search_kwargs={"k": 3}).invoke(question)
    context = "\n\n".join([d.page_content for d in docs])
    
    prompt = f"You are a nursing assistant. Answer ONLY using the PDF context.\n\nContext: {context}\n\nQuestion: {question}"
    return azure_llm.invoke(prompt).content

# ==========================================
# =========== FRONTEND UI ==================
# ==========================================

# --- STATE MANAGEMENT ---
if "app_started" not in st.session_state: st.session_state.app_started = False
if "dark_mode" not in st.session_state: st.session_state.dark_mode = True 
if "chat_sessions" not in st.session_state: st.session_state.chat_sessions = {"New Chat": []}
if "current_chat" not in st.session_state: st.session_state.current_chat = "New Chat"
if "app_mode" not in st.session_state: st.session_state.app_mode = "Gemini Agent"
if "studio_expanded" not in st.session_state: st.session_state.studio_expanded = False
if "studio_prompt_trigger" not in st.session_state: st.session_state.studio_prompt_trigger = None

def get_langchain_history(messages):
    return [HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"]) for m in messages]

# --- THEME COLORS & CSS ---
bg_color = "#131314" if st.session_state.dark_mode else "#FFFFFF"
text_main = "#E3E3E3" if st.session_state.dark_mode else "#1F2937"
divider_color = "#444746" if st.session_state.dark_mode else "#E5E7EB"

st.markdown(f"""<style>
    [data-testid="stAppViewContainer"] {{ background: {bg_color}; }}
    .breadcrumb {{ color: #C4C7C5; font-size: 12px; border-bottom: 1px solid {divider_color}; margin-bottom: 20px; padding: 10px 0; }}
    .disclaimer-box {{ font-size: 11px; color: #C4C7C5; background: rgba(255,255,255,0.05); padding: 10px; border-radius: 8px; border: 1px solid {divider_color}; }}
</style>""", unsafe_allow_html=True)

# --- VIEW 1: LANDING ---
if not st.session_state.app_started:
    st.markdown(f"<h1 style='color:{text_main}; text-align:center;'>🩺 NursBot</h1>", unsafe_allow_html=True)
    if st.button("Get Started", type="primary", use_container_width=True):
        st.session_state.app_started = True
        st.rerun()

# --- VIEW 2: CHAT INTERFACE ---
else:
    with st.sidebar:
        st.markdown(f"<h3 style='color:{text_main};'>Settings</h3>", unsafe_allow_html=True)
        
        # JOESON: Mode switcher
        st.session_state.app_mode = st.radio(
            "Select AI Engine:",
            ["Gemini Agent", "Azure Model (Joeson)"],
            help="Choose between the original Gemini Agent or Joeson's Azure RAG model."
        )
        st.divider()
        if st.button("➕ New Chat", use_container_width=True):
            name = f"Chat {len(st.session_state.chat_sessions)+1}"
            st.session_state.chat_sessions[name] = []
            st.session_state.current_chat = name
            st.rerun()
        st.markdown('<div class="disclaimer-box">⚠️ Use for educational purposes only. Verify with KKH protocols.</div>', unsafe_allow_html=True)

    chat_col, studio_col = st.columns([3, 1] if st.session_state.studio_expanded else [1, 0.001])
    
    with chat_col:
        st.markdown("<div class='breadcrumb'>📁 KKH Workspace / Section 01 - Medical Emergencies</div>", unsafe_allow_html=True)
        
        chat_container = st.container(height=450, border=False)
        with chat_container:
            for m in st.session_state.chat_sessions[st.session_state.current_chat]:
                with st.chat_message(m["role"]): st.markdown(m["content"])

        # JOESON: Tools Popover with STT
        with st.popover("➕ Tools"):
            app_cap = st.selectbox("Capability:", ["Text/Docs", "Voice (Joeson)", "Vision"])
            if app_cap == "Voice (Joeson)":
                st.write("Click to speak:")
                voice_data = speech_to_text(language="en", use_container_width=True, just_once=True, key="JOESON_MIC")
                if voice_data:
                    st.session_state.chat_sessions[st.session_state.current_chat].append({"role": "user", "content": voice_data})
                    st.rerun()

        user_input = st.chat_input("Ask NursBot...")
        if st.session_state.studio_prompt_trigger:
            user_input = st.session_state.studio_prompt_trigger
            st.session_state.studio_prompt_trigger = None

        if user_input:
            st.session_state.chat_sessions[st.session_state.current_chat].append({"role": "user", "content": user_input})
            st.rerun()

    # --- EXECUTION ENGINE ---
    current_msgs = st.session_state.chat_sessions[st.session_state.current_chat]
    if len(current_msgs) > 0 and current_msgs[-1]["role"] == "user":
        with chat_col:
            with st.chat_message("assistant"):
                with st.spinner("Processing clinical data..."):
                    try:
                        latest_q = current_msgs[-1]["content"]
                        # JOESON: Choice routing
                        if st.session_state.app_mode == "Azure Model (Joeson)":
                            ans = joeson_answer_logic(latest_q)
                        else:
                            history = get_langchain_history(current_msgs[:-1])
                            res = agent_executor.invoke({"input": latest_q, "chat_history": history})
                            ans = res["output"]
                        
                        st.markdown(ans)
                        st.session_state.chat_sessions[st.session_state.current_chat].append({"role": "assistant", "content": ans})
                        st.rerun()
                    except Exception as e:
                        st.error(f"Error: {e}")

    # --- STUDIO PANE ---
    if st.button("⚡ Studio", use_container_width=True):
        st.session_state.studio_expanded = not st.session_state.studio_expanded
        st.rerun()