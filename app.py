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

# --- JOESON: Logic & STT Imports ---
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
    except: return None

retriever = initialize_retriever()

@tool
def search_nursing_protocols(query: str) -> str:
    """Original Tool: PDF Retrieval"""
    if not retriever: return "Error: Database not initialized."
    docs = retriever.invoke(query)
    return "\n\n---\n\n".join([f"[Page {d.metadata.get('page', '?')}]\n{d.page_content}" for d in docs])

@tool
def calculate_fluid_requirement(weight_kg: float) -> str:
    """Original Tool: Calculations"""
    if weight_kg <= 10: res = weight_kg * 100
    elif weight_kg <= 20: res = 1000 + (weight_kg - 10) * 50
    else: res = 1500 + (weight_kg - 20) * 20
    return f"The calculated fluid requirement is {res} mL/day."

tools = [calculate_fluid_requirement, search_nursing_protocols]
llm_gemini = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=0)
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a professional KKH Clinical Nursing Assistant."),
    ("placeholder", "{chat_history}"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])
agent_executor = AgentExecutor(agent=create_tool_calling_agent(llm_gemini, tools, prompt), tools=tools, verbose=False)

# ==========================================
# ======= JOESON: BACKEND & LOGIC ==========
# ==========================================
if "AZURE_OPENAI_API_KEY" in st.secrets:
    azure_llm = AzureChatOpenAI(azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"], api_key=os.environ["AZURE_OPENAI_API_KEY"], api_version=os.environ["AZURE_OPENAI_API_VERSION"], azure_deployment=AZURE_CHAT_DEPLOY)
    azure_embeddings = AzureOpenAIEmbeddings(azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT"], api_key=os.environ["AZURE_OPENAI_API_KEY"], api_version=os.environ["AZURE_OPENAI_API_VERSION"], azure_deployment=AZURE_EMBED_DEPLOY)

@st.cache_resource
def get_joeson_vectorstore():
    """JOESON: FAISS setup"""
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    PDF_PATHS = [os.path.join(BASE_DIR, "Section 01 - Medical Emergencies.pdf"), os.path.join(BASE_DIR, "formula.pdf")]
    docs = []
    for p in PDF_PATHS:
        if os.path.exists(p): docs.extend(PyPDFLoader(p).load())
    chunks = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150).split_documents(docs)
    return FAISS.from_documents(chunks, azure_embeddings)

def calculate_fluids(weight):
    """Joeson's Fluid Calculation Logic"""
    weight = float(weight)
    if weight <= 10:
        daily = weight * 100
        hourly = weight * 4
    elif weight <= 20:
        daily = 1000 + ((weight - 10) * 50)
        hourly = 40 + ((weight - 10) * 2)
    else:
        daily = 1500 + ((weight - 20) * 20)
        hourly = 60 + ((weight - 20) * 1)
    
    daily = min(daily, 2500)
    
    return f"**Fluid Maintenance Calculation:**\n\nWeight: {weight} kg\n\nDaily fluid requirement: {daily:.0f} ml/day\nHourly fluid requirement: {hourly:.0f} ml/hr"

def calculate_systolic_bp(age):
    """Joeson's BP Calculation Logic"""
    age = float(age)
    if age < 1/12:
        return "**Expected systolic BP:** > 60 mmHg"
    elif age < 1:
        return "**Expected systolic BP:** > 70 mmHg"
    elif age <= 10:
        sbp = 70 + (age * 2)
        return f"**Expected systolic BP:** > {sbp:.0f} mmHg"
    else:
        return "This PDF formula only covers children up to 10 years old."

def joeson_answer_logic(question):
    """JOESON: Calculation & RAG Logic with Memory"""
    question_lower = question.lower()

    # Save topic memory
    if "fluid" in question_lower or "maintenance" in question_lower:
        st.session_state.last_topic = "fluid"
    if "bp" in question_lower or "blood pressure" in question_lower or "systolic" in question_lower:
        st.session_state.last_topic = "bp"

    numbers = re.findall(r"\d+\.?\d*", question)

    # Ask for missing information
    if st.session_state.last_topic == "fluid" and not numbers:
        return "Please provide patient weight in kg."
    if st.session_state.last_topic == "bp" and not numbers:
        return "Please provide patient age."

    # Follow-up fluid calculation
    if st.session_state.last_topic == "fluid" and numbers:
        weight = numbers
        st.session_state.last_topic = "" # Reset topic
        return calculate_fluids(weight)

    # Follow-up BP calculation
    if st.session_state.last_topic == "bp" and numbers:
        age = numbers
        st.session_state.last_topic = "" # Reset topic
        return calculate_systolic_bp(age)

    # Original RAG retrieval
    vs = get_joeson_vectorstore()
    retriever = vs.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in docs])

    prompt = f"""You are a nursing education assistant.
Answer ONLY using the PDF context below.
If the answer is not in the PDF, say: I cannot find this information in the PDF.

PDF Context:
{context}

Question:
{question}

Answer:"""

    return azure_llm.invoke(prompt).content


# ==========================================
# =========== ORIGINAL: UI & CSS ===========
# ==========================================
if "app_started" not in st.session_state: st.session_state.app_started = False
if "dark_mode" not in st.session_state: st.session_state.dark_mode = True 
if "studio_expanded" not in st.session_state: st.session_state.studio_expanded = False 
if "chat_sessions" not in st.session_state: st.session_state.chat_sessions = {"New Chat": []}
if "current_chat" not in st.session_state: st.session_state.current_chat = "New Chat"
if "last_topic" not in st.session_state: st.session_state.last_topic = "" # Added for Joeson's logic

bg_color = "#131314" if st.session_state.dark_mode else "#FFFFFF"
text_main = "#E3E3E3" if st.session_state.dark_mode else "#1F2937"

st.markdown(f"""
<style>
    [data-testid="stAppViewContainer"] {{ background: {bg_color}; }}
    .hero-title {{ font-size: 3.8rem; font-weight: 800; color: {text_main}; }}
    .hero-title span {{ color: #0d9488; }}
    .badge {{ background-color: #E8F0FE; color: #1A73E8; padding: 6px 16px; border-radius: 20px; font-weight: 700; }}
    .breadcrumb {{ color: #C4C7C5; font-size: 12px; border-bottom: 1px solid #444746; padding: 10px 0; margin-bottom: 20px; }}
    /* Joeson's Right Panel Styling */
    .joeson-card {{ background: rgba(255,255,255,0.05); padding: 15px; border-radius: 12px; border: 1px solid #0d9488; margin-bottom: 10px; }}
</style>
""", unsafe_allow_html=True)

# --- LANDING PAGE ---
if not st.session_state.app_started:
    col1, col2 = st.columns([1.2, 1], gap="large")
    with col1:
        st.markdown('<div class="badge">✨ AI-POWERED CLINICAL ASSISTANT</div>', unsafe_allow_html=True)
        st.markdown('<div class="hero-title">Smarter Nursing<br>with <span>AI</span> Support</div>', unsafe_allow_html=True)
        if st.button("Try Chatbot ➔", type="primary"):
            st.session_state.app_started = True
            st.rerun()
    with col2: st.info("nurse.png")

# --- MAIN APP ---
else:
    with st.sidebar:
        st.markdown(f"<h3 style='color:{text_main};'>🩺 NursBot</h3>", unsafe_allow_html=True)
        if st.button("➕ New chat", type="primary", use_container_width=True):
            name = f"Chat {len(st.session_state.chat_sessions)+1}"
            st.session_state.chat_sessions[name] = []
            st.session_state.current_chat = name
            st.rerun()
        st.divider()
        for chat_id in reversed(list(st.session_state.chat_sessions.keys())):
            if st.button(f"💬 {chat_id}", key=f"h_{chat_id}", use_container_width=True):
                st.session_state.current_chat = chat_id
                st.rerun()

    chat_col, studio_col = st.columns([3, 1.2] if st.session_state.studio_expanded else [1, 0.01])
    
    with chat_col:
        st.markdown("<div class='breadcrumb'>📁 KKH Workspace / Original Gemini Agent</div>", unsafe_allow_html=True)
        if st.button("⚡ Open Joeson's Studio" if not st.session_state.studio_expanded else "✖ Close Studio", use_container_width=True):
            st.session_state.studio_expanded = not st.session_state.studio_expanded
            st.rerun()

        chat_container = st.container(height=450, border=False)
        with chat_container:
            for m in st.session_state.chat_sessions[st.session_state.current_chat]:
                with st.chat_message(m["role"]): st.markdown(m["content"])

        user_input = st.chat_input("Message Gemini Agent...")
        if user_input:
            st.session_state.chat_sessions[st.session_state.current_chat].append({"role": "user", "content": user_input})
            with chat_container:
                with st.chat_message("assistant"):
                    res = agent_executor.invoke({"input": user_input, "chat_history": []})
                    st.markdown(res["output"])
                    st.session_state.chat_sessions[st.session_state.current_chat].append({"role": "assistant", "content": res["output"]})
            st.rerun()

    # ==========================================
    # ======= JOESON: RIGHT PANEL (STUDIO) =====
    # ==========================================
    if st.session_state.studio_expanded:
        with studio_col:
            st.markdown(f"<h3 style='color:#0d9488;'>Joeson's Studio</h3>", unsafe_allow_html=True)
            
            with st.container(border=True):
                st.markdown("**Joeson's Voice Input**")
                # JOESON: Speech to Text
                voice_text = speech_to_text(language="en", use_container_width=True, just_once=True, key="JOE_MIC_UI")
                if voice_text:
                    st.info(f"Joeson Heard: {voice_text}")
                    with st.spinner("Joeson's Model is thinking..."):
                        ans = joeson_answer_logic(voice_text)
                        st.success(ans)

            st.divider()
            
            st.markdown("**Joeson's Azure Model**")
            joe_q = st.text_area("Ask Joeson's Model directly:", placeholder="e.g. Calculate fluid for 15kg", key="joe_input")
            if st.button("Run Joeson's Logic"):
                if joe_q:
                    with st.spinner("Azure RAG Processing..."):
                        ans = joeson_answer_logic(joe_q)
                        st.markdown(f"<div class='joeson-card'>{ans}</div>", unsafe_allow_html=True)
                else:
                    st.warning("Please enter a query for Joeson.")

            st.divider()
            st.markdown("Original Studio Actions")
            if st.button("📝 Generate Quiz (Gemini)"):
                st.info("Quiz feature triggered")