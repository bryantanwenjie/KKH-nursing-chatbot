import base64
import os
import re
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import streamlit as st

# --- THE MEMORY NUKE ---
if "nuke_complete" not in st.session_state:
    st.session_state.clear()
    st.session_state.nuke_complete = True
# -----------------------

from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage

# --- PAGE CONFIG ---
# We moved this to the top so it applies to the whole app immediately
st.set_page_config(page_title="NursBot | Clinical AI", page_icon="🏥", layout="wide")

# --- AUTHENTICATION ---
if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

# ==========================================
# ======= BACKEND (DO NOT TOUCH) ===========
# ==========================================
@st.cache_resource(show_spinner=False)
def initialize_retriever():
    try:
        with st.spinner("Building unlimited Vector Database with Hugging Face..."):
            loader = PyPDFLoader("Section 01 - Medical Emergencies.pdf")
            docs = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            chunks = text_splitter.split_documents(docs)

            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)
            
            return vectorstore.as_retriever()
    
    except Exception as e:
        st.error(f"🚨 Vector DB Error: {str(e)}")
        st.stop()

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

llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a strictly professional KKH Clinical Nursing Assistant. 
    
    Your ONLY purpose is to answer questions related to clinical protocols, nursing guidelines, and medical topics based on the provided KKH documents. 
    
    CRITICAL RULES:
    1. If a user asks a question unrelated to healthcare, nursing, or KKH (e.g., recipes, general technology, movies, casual chat), you MUST politely refuse to answer. 
    2. Do NOT use your general world knowledge to answer off-topic questions. 
    3. If refusing, gently remind the user that you are a clinical assistant and ask how you can help with medical protocols today."""),
    ("placeholder", "{chat_history}"),
    ("placeholder", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)


# ==========================================
# =========== FRONTEND UI ==================
# ==========================================

# Custom CSS to match the Base44 Design
st.markdown("""
<style>
    .badge {
        background-color: #E8F0FE;
        color: #1A73E8;
        padding: 6px 16px;
        border-radius: 20px;
        font-size: 13px;
        font-weight: 700;
        display: inline-block;
        margin-bottom: 1rem;
        border: 1px solid #D2E3FC;
    }
    .hero-title {
        font-size: 3.5rem;
        font-weight: 800;
        line-height: 1.1;
        margin-bottom: 1.5rem;
        color: #1f2937;
    }
    .hero-title span {
        color: #0d9488; /* Teal color for AI */
    }
    .hero-subtitle {
        font-size: 1.2rem;
        color: #4b5563;
        margin-bottom: 2rem;
        line-height: 1.6;
    }
    .stat-number {
        font-size: 1.8rem;
        font-weight: 800;
        color: #111827;
        margin-bottom: -5px;
    }
    .stat-label {
        font-size: 0.9rem;
        color: #6b7280;
    }
</style>
""", unsafe_allow_html=True)

# App State Management
if "app_started" not in st.session_state:
    st.session_state.app_started = False

# --- VIEW 1: LANDING PAGE ---
if not st.session_state.app_started:
    # Adding some top spacing
    st.write("<br><br>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1.2, 1], gap="large")
    
    with col1:
        st.markdown('<div class="badge">✨ AI-POWERED CLINICAL ASSISTANT</div>', unsafe_allow_html=True)
        st.markdown('<div class="hero-title">Smarter Nursing<br>with <span>AI</span> Support</div>', unsafe_allow_html=True)
        st.markdown('<div class="hero-subtitle">A smart chatbot designed for nurses — access clinical protocols, perform medical calculations, and learn on the go. All in one place, available 24/7.</div>', unsafe_allow_html=True)
        
        btn_col1, btn_col2 = st.columns(2)
        with btn_col1:
            if st.button("Try Chatbot ➔", type="primary", use_container_width=True):
                st.session_state.app_started = True
                st.rerun()
        with btn_col2:
            st.button("Learn More")
            
        st.write("<br><br>", unsafe_allow_html=True)
        
        # Stats section
        stat1, stat2, stat3 = st.columns(3)
        with stat1:
            st.markdown('<div class="stat-number">24/7</div><div class="stat-label">Available</div>', unsafe_allow_html=True)
        with stat2:
            st.markdown('<div class="stat-number">500+</div><div class="stat-label">Protocols</div>', unsafe_allow_html=True)
        with stat3:
            st.markdown('<div class="stat-number">98%</div><div class="stat-label">Accuracy</div>', unsafe_allow_html=True)

    with col2:
        # We use a placeholder medical illustration similar to your design
        st.image("https://img.freepik.com/free-vector/doctor-nurse-concept-illustration_114360-15555.jpg", use_container_width=True)

# --- VIEW 2: CHAT INTERFACE ---
else:
    # Sidebar only shows when chat is active
    with st.sidebar:
        if st.button("⬅ Back to Home"):
            st.session_state.app_started = False
            st.rerun()
            
        st.divider()
        st.header("📷 Clinical Vision")
        uploaded_image = st.file_uploader("Upload monitor, chart, or clinical image", type=["png", "jpg", "jpeg"])
        
        if uploaded_image:
            st.image(uploaded_image, caption="Image ready for analysis", use_container_width=True)

    st.title("🏥 NursBot Clinical Assistant")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if user_input := st.chat_input("How can I assist with clinical protocols today?"):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            try:
                # 1. Properly package the input
                if uploaded_image is not None:
                    img_bytes = uploaded_image.getvalue()
                    encoded_img = base64.b64encode(img_bytes).decode("utf-8")
                    image_data = f"data:image/jpeg;base64,{encoded_img}"
                    
                    agent_input = [
                        HumanMessage(content=[
                            {"type": "text", "text": user_input},
                            {"type": "image_url", "image_url": {"url": image_data}}
                        ])
                    ]
                else:
                    agent_input = [HumanMessage(content=user_input)]

                # 2. Run the agent
                response = agent_executor.invoke({
                    "input": agent_input,
                    "chat_history": st.session_state.messages[:-1] 
                })
                
                # 3. Clean the response
                raw_output = str(response.get("output", ""))
                match = re.search(r"'text':\s*['\"](.*?)['\"],\s*'index':", raw_output, re.DOTALL)
                
                if match:
                    full_response = match.group(1).replace('\\n', '\n').replace('\\t', '\t').replace("\\'", "'")
                else:
                    full_response = raw_output

                st.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                
            except Exception as e:
                st.error(f"🚨 Error: {str(e)}")