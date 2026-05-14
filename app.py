import base64
import os
import re
import time
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import streamlit as st

from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage

# 👉 JOESON'S CODE: Imports for Azure and Speech-to-Text
from langchain_openai import AzureChatOpenAI
from streamlit_mic_recorder import speech_to_text

# --- PAGE CONFIG ---
st.set_page_config(page_title="NursBot | Clinical AI", page_icon="🏥", layout="wide")

# --- AUTHENTICATION ---
if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

# 👉 JOESON'S CODE: Azure Authentication setup
if "AZURE_OPENAI_API_KEY" in st.secrets:
    os.environ["AZURE_OPENAI_API_KEY"] = st.secrets["AZURE_OPENAI_API_KEY"]
    os.environ["AZURE_OPENAI_ENDPOINT"] = st.secrets["AZURE_OPENAI_ENDPOINT"]
    os.environ["AZURE_OPENAI_API_VERSION"] = st.secrets["AZURE_OPENAI_API_VERSION"]
    os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"] = st.secrets["AZURE_OPENAI_CHAT_DEPLOYMENT"]

# ==========================================
# ======= BACKEND & CLINICAL LOGIC =========
# ==========================================
@st.cache_resource(show_spinner=False)
def initialize_retriever():
    """Initializes the Vector DB with persistent local storage to prevent rebuilding."""
    persist_directory = "./chroma_db"
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    try:
        # Check if database already exists on disk
        if os.path.exists(persist_directory) and os.listdir(persist_directory):
            vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
            return vectorstore.as_retriever(search_kwargs={"k": 3})
            
        # Otherwise, build it for the first time
        with st.spinner("Building Vector Database for the first time... (This will be cached)"):
            pdf_path = "Section 01 - Medical Emergencies.pdf"
            if not os.path.exists(pdf_path):
                st.warning(f"File '{pdf_path}' not found. Vector DB will be empty.")
                return None
                
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            chunks = text_splitter.split_documents(docs)

            vectorstore = Chroma.from_documents(
                documents=chunks, 
                embedding=embeddings, 
                persist_directory=persist_directory
            )
            return vectorstore.as_retriever(search_kwargs={"k": 3})
            
    except Exception as e:
        st.error(f"🚨 Vector DB Error: {str(e)}")
        st.stop()

retriever = initialize_retriever()

@tool
def search_nursing_protocols(query: str) -> str:
    """Search the KKH Medical Emergencies PDF for clinical guidelines."""
    if not retriever:
        return "Error: Database not initialized. Please ensure the PDF is loaded."
        
    docs = retriever.invoke(query)
    results = []
    for doc in docs:
        page = doc.metadata.get('page', 'Unknown Page')
        results.append(f"[Source: Page {page}]\n{doc.page_content}")
    return "\n\n---\n\n".join(results)

@tool
def calculate_fluid_requirement(weight_kg: float) -> str:
    """Calculates daily and hourly fluid requirements (Holliday-Segar Formula)."""
    warning = ""
    if weight_kg < 2.0 or weight_kg > 80.0:
        warning = "\n\n⚠️ **CLINICAL WARNING:** Weight is outside standard pediatric ranges."
        
    # 👉 JOESON'S CODE: Hourly calculations added to your existing base logic
    if weight_kg <= 10: 
        daily = weight_kg * 100
        hourly = weight_kg * 4
    elif weight_kg <= 20: 
        daily = 1000 + ((weight_kg - 10) * 50)
        hourly = 40 + ((weight_kg - 10) * 2)
    else: 
        daily = 1500 + ((weight_kg - 20) * 20)
        hourly = 60 + ((weight_kg - 20) * 1)
    
    # 👉 JOESON'S CODE: Capping the daily fluid at 2500
    daily = min(daily, 2500)
    
    return f"Daily requirement: {daily:.0f} mL/day.\nHourly requirement: {hourly:.0f} mL/hr.{warning}"

# 👉 JOESON'S CODE: His custom BP logic converted into a LangChain tool
@tool
def calculate_systolic_bp(age_years: float) -> str:
    """Calculates the expected systolic blood pressure for children up to 10 years old."""
    if age_years < 1/12:
        return "Expected systolic BP: > 60 mmHg"
    elif age_years < 1:
        return "Expected systolic BP: > 70 mmHg"
    elif age_years <= 10:
        sbp = 70 + (age_years * 2)
        return f"Expected systolic BP: > {sbp:.0f} mmHg"
    else:
        return "Warning: This formula only covers children up to 10 years old."

# 👉 JOESON'S CODE: Added his BP tool to your tools list
tools = [calculate_fluid_requirement, search_nursing_protocols, calculate_systolic_bp]

# Initialize Your Google Model
llm_gemini = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=0)

# 👉 JOESON'S CODE: Initialize his Azure Model
try:
    llm_azure = AzureChatOpenAI(
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT", ""),
        api_key=os.getenv("AZURE_OPENAI_API_KEY", ""),
        api_version=os.getenv("AZURE_OPENAI_API_VERSION", ""),
        azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT", ""),
        temperature=0
    )
except:
    llm_azure = None

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a strictly professional KKH Clinical Nursing Assistant. 
    
    CRITICAL RULES:
    1. If a user asks a question unrelated to healthcare, nursing, or KKH, politely refuse.
    2. Do NOT use general knowledge for KKH protocols; ALWAYS use the `search_nursing_protocols` tool.
    3. When quoting protocols, mention the Source Page Number provided by the tool.
    4. If calculating fluids or BP, clearly display the math and any clinical warnings.
    5. Be concise, structured, and use bullet points for readability.
    """),
    ("placeholder", "{chat_history}"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

# Helper function to format history for LangChain
def get_langchain_history(messages):
    history = []
    for m in messages:
        if m["role"] == "user":
            history.append(HumanMessage(content=m["content"]))
        else:
            history.append(AIMessage(content=m["content"]))
    return history


# ==========================================
# =========== FRONTEND UI ==================
# ==========================================

# --- STATE MANAGEMENT ---
if "app_started" not in st.session_state:
    st.session_state.app_started = False
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True 
if "studio_expanded" not in st.session_state:
    st.session_state.studio_expanded = False 
if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = {"New Chat": []}
if "current_chat" not in st.session_state:
    st.session_state.current_chat = "New Chat"
if "chat_counter" not in st.session_state:
    st.session_state.chat_counter = 1
if "studio_prompt_trigger" not in st.session_state:
    st.session_state.studio_prompt_trigger = None

def toggle_theme():
    st.session_state.dark_mode = not st.session_state.dark_mode

def get_chat_title(chat_id, messages):
    if not messages:
        return chat_id
    for m in messages:
        if m["role"] == "user":
            return m["content"][:20] + "..."
    return chat_id

# --- DYNAMIC THEME COLORS ---
if st.session_state.dark_mode:
    bg_color = "#131314" 
    text_main = "#E3E3E3"
    text_sub = "#C4C7C5"
    nav_color = "#E3E3E3"
    divider_color = "#444746"
    card_bg = "#1E1F20"
    card_hover = "rgba(255,255,255,0.08)"
else:
    bg_color = "#FFFFFF"
    text_main = "#1F2937"
    text_sub = "#4B5563"
    nav_color = "#4B5563"
    divider_color = "#E5E7EB"
    card_bg = "#F9FAFB"
    card_hover = "rgba(26, 115, 232, 0.05)"

# --- CUSTOM CSS ---
st.markdown(f"""
<style>
    [data-testid="stAppViewContainer"] {{ background: {bg_color}; }}
    [data-testid="stHeader"] {{ background: transparent; }}
    
    div[data-testid="stButton"] button[kind="primary"] {{
        background-color: #1A73E8 !important; border-color: #1A73E8 !important; color: white !important; border-radius: 8px !important; font-weight: 600;
    }}
    div[data-testid="stButton"] button[kind="primary"]:hover {{ background-color: #1557B0 !important; }}
    
    .nav-links {{ display: flex; justify-content: center; gap: 30px; font-size: 14px; font-weight: 600; color: {nav_color}; margin-top: 10px; }}
    .badge {{ background-color: #E8F0FE; color: #1A73E8; padding: 6px 16px; border-radius: 20px; font-size: 13px; font-weight: 700; display: inline-block; margin-bottom: 1rem; border: 1px solid #D2E3FC; }}
    .hero-title {{ font-size: 3.8rem; font-weight: 800; line-height: 1.1; margin-bottom: 1.5rem; color: {text_main}; }}
    .hero-title span {{ color: #0d9488; }}
    .hero-subtitle {{ font-size: 1.2rem; color: {text_sub}; margin-bottom: 2rem; line-height: 1.6; }}
    
    .breadcrumb {{ color: {text_sub}; font-size: 12px; font-weight: 600; padding: 10px 0; border-bottom: 1px solid {divider_color}; margin-bottom: 20px; }}
    
    .disclaimer-box {{ font-size: 11px; color: {text_sub}; background: {card_bg}; padding: 10px; border-radius: 8px; margin-top: 30px; text-align: center; border: 1px solid {divider_color}; }}
    
    .studio-btn-wrapper div[data-testid="stButton"] button {{
        width: 100%; text-align: left; background-color: {bg_color}; border: 1px solid {divider_color}; border-radius: 12px; padding: 15px; color: {text_main}; transition: all 0.2s ease;
    }}
    .studio-btn-wrapper div[data-testid="stButton"] button:hover {{
        border-color: #1A73E8; background: {card_hover}; transform: translateY(-2px);
    }}
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
        st.markdown('<div class="hero-subtitle">A smart chatbot designed for nurses — access clinical protocols, perform medical calculations, and learn on the go.</div>', unsafe_allow_html=True)
        btn_col1, btn_col2, _ = st.columns([1, 1, 1.5]) 
        with btn_col1:
            if st.button("Try Chatbot ➔", type="primary", use_container_width=True, key="try_btn"):
                st.session_state.app_started = True
                st.rerun()
        with btn_col2: st.button("Learn More", use_container_width=True)

    with col2:
        try:
            st.image("nurse.png", use_container_width=True)
        except:
            st.info("Visual Placeholder: 'nurse.png' missing")


# --- VIEW 2: APP CHAT INTERFACE ---
else:
    current_messages = st.session_state.chat_sessions[st.session_state.current_chat]

    # 1. LEFT PANE: Sidebar History & Safety
    with st.sidebar:
        st.markdown(f"<h3 style='color:{text_main}; margin-top:-20px;'>🩺 NursBot</h3>", unsafe_allow_html=True)
        
        if st.button("➕ New chat", type="primary", use_container_width=True):
            st.session_state.chat_counter += 1
            new_chat_name = f"Chat {st.session_state.chat_counter}"
            st.session_state.chat_sessions[new_chat_name] = []
            st.session_state.current_chat = new_chat_name
            st.rerun()
            
        st.write("<br>", unsafe_allow_html=True)
        st.markdown(f"<p style='color:{text_sub}; font-size:12px; font-weight:700; letter-spacing: 1px;'>RECENT</p>", unsafe_allow_html=True)
        
        for chat_id in reversed(list(st.session_state.chat_sessions.keys())):
            title = get_chat_title(chat_id, st.session_state.chat_sessions[chat_id])
            is_active = "🔹 " if chat_id == st.session_state.current_chat else "💬 "
            if st.button(f"{is_active}{title}", key=f"hist_{chat_id}", use_container_width=True):
                st.session_state.current_chat = chat_id
                st.rerun()
        
        st.divider()
        if st.button("⬅ Back to Home", use_container_width=True):
            st.session_state.app_started = False
            st.rerun()

        st.markdown(
            f'<div class="disclaimer-box">⚠️ <b>Clinical Disclaimer</b><br>NursBot is an AI assistant. Do not use as a substitute for clinical judgment. Always verify with official KKH Protocol Manuals.</div>', 
            unsafe_allow_html=True
        )

    # 2. MAIN LAYOUT
    if st.session_state.studio_expanded:
        chat_col, studio_col = st.columns([3, 1], gap="large")
    else:
        chat_col = st.container()

    # 3. CENTER PANE: Chat
    with chat_col:
        c1, c2 = st.columns([4, 1])
        with c1:
            st.markdown(f"<div class='breadcrumb'>📁 KKH Workspace / Section 01 - Medical Emergencies</div>", unsafe_allow_html=True)
        with c2:
            toggle_label = "✖ Close Studio" if st.session_state.studio_expanded else "⚡ Open Studio"
            if st.button(toggle_label, use_container_width=True):
                st.session_state.studio_expanded = not st.session_state.studio_expanded
                st.rerun()

        chat_container = st.container(height=450, border=False) 
        
        with chat_container:
            for message in current_messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            
            if len(current_messages) == 0:
                st.markdown(f"<h2 style='color:{text_main}; text-align:center; margin-top:100px;'>How can I help you today?</h2>", unsafe_allow_html=True)

        # Tools Popover
        uploaded_file = None
        spoken_text = None
        app_mode = "Clinical Text & Docs"
        
        with st.popover("➕ Tools & Attachments", help="Upload images, video, or speech"):
            app_mode = st.selectbox("Select AI Capability:", ["Clinical Text & Docs", "Vision (Image)", "Speech-to-Text"])
            
            # 👉 JOESON'S CODE: UI integration to let you pick his Azure LLM
            model_options = ["Gemini (NursBot Default)"]
            if llm_azure: 
                model_options.append("Azure OpenAI (Joeson's Model)")
            selected_model = st.selectbox("Select AI Brain:", model_options)
            
            if app_mode == "Vision (Image)":
                uploaded_file = st.file_uploader("Upload image", type=["png", "jpg", "jpeg"])
                
            # 👉 JOESON'S CODE: Speech-to-text triggering logic
            elif app_mode == "Speech-to-Text":
                st.markdown("<p style='font-size: 14px; font-weight: 600;'>🎤 Click to Speak:</p>", unsafe_allow_html=True)
                spoken_text = speech_to_text(language="en", use_container_width=True, just_once=True, key="STT")

        # Input Handling
        user_input = st.chat_input(f"Message NursBot ({app_mode})...")
        
        if st.session_state.studio_prompt_trigger:
            user_input = st.session_state.studio_prompt_trigger
            st.session_state.studio_prompt_trigger = None 
            
        # 👉 JOESON'S CODE: Capturing microphone output as user_input
        if app_mode == "Speech-to-Text" and spoken_text:
            user_input = spoken_text

        if user_input:
            st.session_state.chat_sessions[st.session_state.current_chat].append({"role": "user", "content": user_input})
            st.rerun() 

    # Run AI Logic
    if len(current_messages) > 0 and current_messages[-1]["role"] == "user":
        latest_user_input = current_messages[-1]["content"]
        with chat_container:
            with st.chat_message("assistant"):
                try:
                    chat_history_lc = get_langchain_history(current_messages[:-1])
                    agent_input = latest_user_input
                    
                    if app_mode == "Vision (Image)" and uploaded_file is not None:
                        img_bytes = uploaded_file.getvalue()
                        encoded_img = base64.b64encode(img_bytes).decode("utf-8")
                        image_data = f"data:image/jpeg;base64,{encoded_img}"
                        agent_input = [HumanMessage(content=[{"type": "text", "text": latest_user_input}, {"type": "image_url", "image_url": {"url": image_data}}])]

                    with st.spinner(f"Analyzing using {selected_model.split(' ')[0]}..."):
                        
                        # 👉 JOESON'S CODE: Dynamically build agent using the selected LLM
                        active_llm = llm_azure if "Azure" in selected_model and llm_azure else llm_gemini
                        
                        agent = create_tool_calling_agent(active_llm, tools, prompt)
                        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)

                        response = agent_executor.invoke({
                            "input": agent_input, 
                            "chat_history": chat_history_lc
                        })
                    
                    raw_output = str(response.get("output", ""))
                    match = re.search(r"'text':\s*['\"](.*?)['\"],\s*'index':", raw_output, re.DOTALL)
                    full_response = match.group(1).replace('\\n', '\n').replace('\\t', '\t').replace("\\'", "'") if match else raw_output
                    
                    def stream_text(text):
                        for word in text.split(" "):
                            yield word + " "
                            time.sleep(0.02)
                    
                    st.write_stream(stream_text(full_response))
                    
                    st.session_state.chat_sessions[st.session_state.current_chat].append({"role": "assistant", "content": full_response})
                    time.sleep(0.1)
                    st.rerun()

                except Exception as e:
                    st.error(f"🚨 Error: {str(e)}")


    # 4. RIGHT PANE: Interactive Studio
    if st.session_state.studio_expanded:
        with studio_col:
            st.markdown(f"<h3 style='color: {text_main}; margin-top:0px;'>Studio</h3>", unsafe_allow_html=True)
            st.markdown(f"<p style='color:{text_sub}; font-size:13px;'>Quick actions based on context:</p>", unsafe_allow_html=True)
            
            st.markdown('<div class="studio-btn-wrapper">', unsafe_allow_html=True)
            
            if st.button("📝 Generate Quiz"):
                st.session_state.studio_prompt_trigger = "Based on our current conversation and the clinical document, generate a 3-question multiple-choice clinical quiz."
                st.rerun()
                
            if st.button("📊 Extract Data Table"):
                st.session_state.studio_prompt_trigger = "Extract all numerical dosages, guidelines, or stats from the current topic and present them in a clear Markdown table."
                st.rerun()

            if st.button("🎙️ Audio Summary"):
                st.session_state.studio_prompt_trigger = "Provide a podcast-style, conversational script summarizing the main points of our current clinical topic."
                st.rerun()
                
            st.markdown('</div>', unsafe_allow_html=True)