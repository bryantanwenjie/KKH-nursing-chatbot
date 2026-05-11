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
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage

# --- PAGE CONFIG ---
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

# --- STATE MANAGEMENT ---
if "app_started" not in st.session_state:
    st.session_state.app_started = False
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False
if "studio_expanded" not in st.session_state:
    st.session_state.studio_expanded = True
if "messages" not in st.session_state:
    st.session_state.messages = []

def toggle_theme():
    st.session_state.dark_mode = not st.session_state.dark_mode

# --- DYNAMIC THEME COLORS ---
if st.session_state.dark_mode:
    bg_color = "radial-gradient(circle at top left, #1F2937, #111827)"
    text_main = "#F9FAFB"
    text_sub = "#9CA3AF"
    nav_color = "#D1D5DB"
    divider_color = "#374151"
    sidebar_bg = "transparent"
else:
    bg_color = "radial-gradient(circle at top left, #FFFFFF, #F0F4F8)"
    text_main = "#1F2937"
    text_sub = "#4B5563"
    nav_color = "#4B5563"
    divider_color = "#E5E7EB"
    sidebar_bg = "transparent"

# --- CUSTOM CSS ---
st.markdown(f"""
<style>
    /* Apply Background Color */
    [data-testid="stAppViewContainer"] {{
        background: {bg_color};
    }}
    [data-testid="stHeader"] {{
        background: transparent;
    }}
    /* Blue Primary Buttons */
    div[data-testid="stButton"] button[kind="primary"] {{
        background-color: #1A73E8 !important;
        border-color: #1A73E8 !important;
        color: white !important;
        border-radius: 8px !important;
    }}
    div[data-testid="stButton"] button[kind="primary"]:hover {{
        background-color: #1557B0 !important;
        border-color: #1557B0 !important;
    }}
    /* Typography & Layout for Landing Page */
    .nav-links {{ display: flex; justify-content: center; gap: 30px; font-size: 14px; font-weight: 600; color: {nav_color}; margin-top: 10px; }}
    .badge {{ background-color: #E8F0FE; color: #1A73E8; padding: 6px 16px; border-radius: 20px; font-size: 13px; font-weight: 700; display: inline-block; margin-bottom: 1rem; border: 1px solid #D2E3FC; }}
    .hero-title {{ font-size: 3.8rem; font-weight: 800; line-height: 1.1; margin-bottom: 1.5rem; color: {text_main}; }}
    .hero-title span {{ color: #0d9488; }}
    .hero-subtitle {{ font-size: 1.2rem; color: {text_sub}; margin-bottom: 2rem; line-height: 1.6; }}
    .stats-container {{ display: flex; align-items: center; gap: 2rem; margin-top: 1rem; }}
    .stat-box {{ display: flex; flex-direction: column; }}
    .stat-divider {{ height: 45px; width: 2px; background-color: {divider_color}; }}
    .stat-number {{ font-size: 1.8rem; font-weight: 800; color: {text_main}; margin-bottom: -5px; }}
    .stat-label {{ font-size: 0.9rem; color: {text_sub}; }}
    
    /* Studio Cards */
    .studio-card {{
        background-color: {bg_color};
        border: 1px solid {divider_color};
        border-radius: 12px;
        padding: 15px;
        margin-bottom: 12px;
        cursor: pointer;
        transition: 0.2s;
        color: {text_main};
        font-weight: 500;
        display: flex;
        align-items: center;
        gap: 10px;
    }}
    .studio-card:hover {{ border-color: #1A73E8; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }}
    
    /* Clean up history buttons */
    .history-btn {{
        width: 100%; text-align: left; background: none; border: none; color: {text_sub};
        padding: 8px; border-radius: 6px; cursor: pointer; font-size: 14px;
    }}
    .history-btn:hover {{ background: rgba(0,0,0,0.05); }}
</style>
""", unsafe_allow_html=True)


# --- VIEW 1: LANDING PAGE ---
if not st.session_state.app_started:
    nav1, nav2, nav3, nav4 = st.columns([1.5, 4, 0.5, 1], gap="small")
    with nav1: st.markdown(f"<h3 style='margin-top: -5px; color: {text_main};'>🩺 NursBot</h3>", unsafe_allow_html=True)
    with nav2: st.markdown('<div class="nav-links"><span>Features</span><span>How It Works</span><span>Benefits</span><span>Demo</span><span>About</span></div>', unsafe_allow_html=True)
    with nav3:
        icon = "☀️" if st.session_state.dark_mode else "🌙"
        st.button(icon, on_click=toggle_theme, key="theme_btn")
    with nav4:
        if st.button("Get Started", type="primary", use_container_width=True):
            st.session_state.app_started = True
            st.rerun()
            
    st.divider() 
    st.write("<br><br>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1.2, 1], gap="large")
    with col1:
        st.markdown('<div class="badge">✨ AI-POWERED CLINICAL ASSISTANT</div>', unsafe_allow_html=True)
        st.markdown('<div class="hero-title">Smarter Nursing<br>with <span>AI</span> Support</div>', unsafe_allow_html=True)
        st.markdown('<div class="hero-subtitle">A smart chatbot designed for nurses — access clinical protocols, perform medical calculations, and learn on the go. All in one place, available 24/7.</div>', unsafe_allow_html=True)
        btn_col1, btn_col2, _ = st.columns([1, 1, 1.5]) 
        with btn_col1:
            if st.button("Try Chatbot ➔", type="primary", use_container_width=True, key="try_btn"):
                st.session_state.app_started = True
                st.rerun()
        with btn_col2: st.button("Learn More", use_container_width=True)
        st.write("<br><br>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="stats-container">
            <div class="stat-box"><div class="stat-number">24/7</div><div class="stat-label">Available</div></div><div class="stat-divider"></div>
            <div class="stat-box"><div class="stat-number">500+</div><div class="stat-label">Protocols</div></div><div class="stat-divider"></div>
            <div class="stat-box"><div class="stat-number">98%</div><div class="stat-label">Accuracy</div></div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.image("nurse.png", use_container_width=True)


# --- VIEW 2: NOTEBOOKLM / CHATGPT STYLE APP ---
else:
    # 1. LEFT PANE: ChatGPT-Style Sidebar
    with st.sidebar:
        if st.button("⬅ Back to Home"):
            st.session_state.app_started = False
            st.rerun()
            
        st.write("<br>", unsafe_allow_html=True)
        
        # New Chat Button
        if st.button("➕ New Chat", type="primary", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
            
        st.divider()
        
        # Mock Chat History
        st.markdown(f"<p style='color:{text_sub}; font-size:12px; font-weight:bold;'>TODAY</p>", unsafe_allow_html=True)
        st.button("💬 Fluid Requirement Calc", use_container_width=True, key="hist1")
        st.button("💬 Medical Emergencies...", use_container_width=True, key="hist2")
        
        st.write("<br>", unsafe_allow_html=True)
        st.markdown(f"<p style='color:{text_sub}; font-size:12px; font-weight:bold;'>YESTERDAY</p>", unsafe_allow_html=True)
        st.button("💬 Image Analysis - ECG", use_container_width=True, key="hist3")
        st.button("💬 Quiz: Ward Protocols", use_container_width=True, key="hist4")

    # 2. MAIN LAYOUT TOGGLE
    if st.session_state.studio_expanded:
        chat_col, studio_col = st.columns([2.5, 1], gap="large")
    else:
        chat_col = st.container()

    # 3. CENTER PANE: Chat & Input
    with chat_col:
        # Header Row: Title + Toggle Studio Button
        head_title, head_btn = st.columns([4, 1.2])
        with head_title:
            st.markdown(f"<h2 style='color: {text_main};'>🏥 NursBot</h2>", unsafe_allow_html=True)
        with head_btn:
            st.write("<br>", unsafe_allow_html=True)
            toggle_label = "➡️ Hide Studio" if st.session_state.studio_expanded else "⬅️ Show Studio"
            if st.button(toggle_label, use_container_width=True):
                st.session_state.studio_expanded = not st.session_state.studio_expanded
                st.rerun()

        # Render Chat Messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # ChatGPT-Style Attachment/Mode Selector (Lives right above the text input)
        st.write("<br>", unsafe_allow_html=True)
        with st.expander("📎 Attachments & AI Mode", expanded=False):
            app_mode = st.selectbox(
                "Select AI Capability:",
                ["Clinical Vision (Image)", "Video Analysis", "Speech-to-Text", "Clinical Quiz"]
            )
            
            uploaded_file = None
            if app_mode == "Clinical Vision (Image)":
                uploaded_file = st.file_uploader("Upload monitor, chart, or clinical image", type=["png", "jpg", "jpeg"])
            elif app_mode == "Video Analysis":
                uploaded_file = st.file_uploader("Upload clinical procedure video", type=["mp4", "mov"])
            elif app_mode == "Speech-to-Text":
                uploaded_file = st.file_uploader("Upload physician audio notes", type=["wav", "mp3"])
            elif app_mode == "Clinical Quiz":
                st.slider("Number of Questions", 1, 10, 5)
                st.selectbox("Difficulty", ["Beginner Nursing", "Advanced Clinical", "Specialist"])

        # The Chat Input
        if user_input := st.chat_input(f"Message NursBot ({app_mode})..."):
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.markdown(user_input)

            with st.chat_message("assistant"):
                try:
                    # --- TEAM ROUTING LOGIC ---
                    
                    # MEMBER 1: YOUR VISION CODE
                    if app_mode == "Clinical Vision (Image)":
                        if uploaded_file is not None:
                            img_bytes = uploaded_file.getvalue()
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

                        response = agent_executor.invoke({
                            "input": agent_input,
                            "chat_history": st.session_state.messages[:-1] 
                        })
                        
                        raw_output = str(response.get("output", ""))
                        match = re.search(r"'text':\s*['\"](.*?)['\"],\s*'index':", raw_output, re.DOTALL)
                        full_response = match.group(1).replace('\\n', '\n').replace('\\t', '\t').replace("\\'", "'") if match else raw_output
                        
                        st.markdown(full_response)
                        st.session_state.messages.append({"role": "assistant", "content": full_response})

                    # MEMBER 2: VIDEO CODE
                    elif app_mode == "Video Analysis":
                        placeholder_response = "*(Teammate's Video API logic will process this prompt)*"
                        st.markdown(placeholder_response)
                        st.session_state.messages.append({"role": "assistant", "content": placeholder_response})
                        
                    # MEMBER 3: SPEECH CODE
                    elif app_mode == "Speech-to-Text":
                        placeholder_response = "*(Teammate's Whisper/Speech API logic will process this prompt)*"
                        st.markdown(placeholder_response)
                        st.session_state.messages.append({"role": "assistant", "content": placeholder_response})

                    # MEMBER 4: QUIZ CODE
                    elif app_mode == "Clinical Quiz":
                        placeholder_response = "*(Teammate's Quiz Generation logic will process this prompt)*"
                        st.markdown(placeholder_response)
                        st.session_state.messages.append({"role": "assistant", "content": placeholder_response})

                except Exception as e:
                    st.error(f"🚨 Error: {str(e)}")

    # 4. RIGHT PANE: Studio (Only renders if expanded)
    if st.session_state.studio_expanded:
        with studio_col:
            st.markdown(f"<h3 style='color: {text_main};'>Studio</h3>", unsafe_allow_html=True)
            st.write("<br>", unsafe_allow_html=True)
            
            st.markdown('<div class="studio-card">🎙️ <b>Audio Overview</b><br><span style="font-size:12px; color:gray;">Generate podcast</span></div>', unsafe_allow_html=True)
            st.markdown('<div class="studio-card">📝 <b>Generate Quiz</b><br><span style="font-size:12px; color:gray;">Test your knowledge</span></div>', unsafe_allow_html=True)
            st.markdown('<div class="studio-card">📊 <b>Data Table</b><br><span style="font-size:12px; color:gray;">Extract clinical stats</span></div>', unsafe_allow_html=True)
            st.markdown('<div class="studio-card">🎥 <b>Video Summary</b><br><span style="font-size:12px; color:gray;">Analyze procedure</span></div>', unsafe_allow_html=True)