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
st.set_page_config(
    page_title="NursBot | Clinical AI",
    page_icon="🏥",
    layout="wide"
)

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

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100
            )

            chunks = text_splitter.split_documents(docs)

            embeddings = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2"
            )

            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings
            )

            return vectorstore.as_retriever()

    except Exception as e:
        st.error(f"🚨 Vector DB Error: {str(e)}")
        st.stop()


retriever = initialize_retriever()


@tool
def search_nursing_protocols(query: str) -> str:
    """Search the KKH Medical Emergencies PDF for clinical guidelines."""

    docs = retriever.invoke(query)

    result = ""

    for doc in docs:
        result += doc.page_content + "\n\n"

    return result


@tool
def calculate_fluid_requirement(weight_kg: float) -> str:
    """Calculates daily fluid requirements (Holliday-Segar Formula)."""

    if weight_kg <= 10:
        res = weight_kg * 100

    elif weight_kg <= 20:
        res = 1000 + (weight_kg - 10) * 50

    else:
        res = 1500 + (weight_kg - 20) * 20

    return "The calculated fluid requirement is " + str(res) + " mL/day."


tools = [
    calculate_fluid_requirement,
    search_nursing_protocols
]

llm = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview",
    temperature=0
)

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are a strictly professional KKH Clinical Nursing Assistant.

Your ONLY purpose is to answer questions related to clinical protocols, nursing guidelines, and medical topics based on the provided KKH documents.

CRITICAL RULES:
1. If a user asks a question unrelated to healthcare, nursing, or KKH (e.g., recipes, general technology, movies, casual chat), you MUST politely refuse to answer.

2. Do NOT use your general world knowledge to answer off-topic questions.

3. If refusing, gently remind the user that you are a clinical assistant and ask how you can help with medical protocols today.
"""
    ),
    ("placeholder", "{chat_history}"),
    ("placeholder", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_tool_calling_agent(
    llm,
    tools,
    prompt
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=False
)

# ==========================================
# =========== SESSION STATES ===============
# ==========================================

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


def toggle_theme():
    st.session_state.dark_mode = not st.session_state.dark_mode


def get_chat_title(chat_id, messages):

    if not messages:
        return chat_id

    for m in messages:
        if m["role"] == "user":
            return m["content"][:20] + "..."

    return chat_id


# ==========================================
# =========== THEME COLORS =================
# ==========================================

if st.session_state.dark_mode:

    bg_color = "#131314"
    text_main = "#E3E3E3"
    text_sub = "#C4C7C5"
    nav_color = "#E3E3E3"
    divider_color = "#444746"

else:

    bg_color = "#FFFFFF"
    text_main = "#1F2937"
    text_sub = "#4B5563"
    nav_color = "#4B5563"
    divider_color = "#E5E7EB"

# ==========================================
# ======== RIGHT SIDEBAR WIDTH =============
# ==========================================

if st.session_state.studio_expanded:

    studio_width = "340px"
    chat_margin = "360px"

else:

    studio_width = "0px"
    chat_margin = "0px"

# ==========================================
# =============== CSS ======================
# ==========================================

st.markdown(f"""
<style>

/* APP BACKGROUND */

[data-testid="stAppViewContainer"] {{
    background: {bg_color};
}}

[data-testid="stHeader"] {{
    background: transparent;
}}

/* BUTTONS */

div[data-testid="stButton"] button[kind="primary"] {{
    background-color: #1A73E8 !important;
    border-color: #1A73E8 !important;
    color: white !important;
    border-radius: 10px !important;
}}

div[data-testid="stButton"] button[kind="primary"]:hover {{
    background-color: #1557B0 !important;
}}

/* SIDEBAR BUTTONS */

.stButton > button {{
    text-align: left !important;
    border-radius: 10px !important;
}}

/* TYPOGRAPHY */

.nav-links {{
    display: flex;
    justify-content: center;
    gap: 30px;
    font-size: 14px;
    font-weight: 600;
    color: {nav_color};
    margin-top: 10px;
}}

.badge {{
    background-color: #E8F0FE;
    color: #1A73E8;
    padding: 6px 16px;
    border-radius: 20px;
    font-size: 13px;
    font-weight: 700;
    display: inline-block;
    margin-bottom: 1rem;
    border: 1px solid #D2E3FC;
}}

.hero-title {{
    font-size: 3.8rem;
    font-weight: 800;
    line-height: 1.1;
    margin-bottom: 1.5rem;
    color: {text_main};
}}

.hero-title span {{
    color: #0d9488;
}}

.hero-subtitle {{
    font-size: 1.2rem;
    color: {text_sub};
    margin-bottom: 2rem;
    line-height: 1.6;
}}

/* RIGHT STUDIO SIDEBAR */

.studio-sidebar {{
    position: fixed;
    top: 0;
    right: 0;
    width: {studio_width};
    height: 100vh;
    overflow-x: hidden;
    transition: 0.3s ease;
    padding-top: 80px;
    z-index: 999;

    border-left: 1px solid {divider_color};

    backdrop-filter: blur(14px);

    background: rgba(19,19,20,0.85);
}}

/* CHAT SHIFT */

.main-chat-area {{
    transition: margin-right 0.3s ease;
    margin-right: {chat_margin};
}}

/* STUDIO CONTENT */

.studio-content {{
    padding: 20px;
}}

/* STUDIO CARD */

.studio-card {{
    background-color: {bg_color};
    border: 1px solid {divider_color};
    border-radius: 14px;
    padding: 16px;
    margin-bottom: 14px;
    cursor: pointer;
    transition: 0.2s;
    color: {text_main};
    display: flex;
    align-items: center;
    gap: 12px;
}}

.studio-card:hover {{
    border-color: #1A73E8;
    transform: translateY(-2px);
}}

/* CHAT INPUT */

[data-testid="stChatInput"] {{
    margin-bottom: 10px;
}}

</style>
""", unsafe_allow_html=True)

# ==========================================
# ============ LANDING PAGE ================
# ==========================================

if not st.session_state.app_started:

    nav1, nav2, nav3, nav4 = st.columns(
        [1.5, 4, 0.5, 1],
        gap="small"
    )

    with nav1:
        st.markdown(
            f"<h3 style='margin-top: -5px; color: {text_main};'>🩺 NursBot</h3>",
            unsafe_allow_html=True
        )

    with nav2:
        st.markdown(
            '''
<div class="nav-links">
<span>Features</span>
<span>How It Works</span>
<span>Benefits</span>
<span>Demo</span>
<span>About</span>
</div>
''',
            unsafe_allow_html=True
        )

    with nav3:

        icon = "☀️"

        if not st.session_state.dark_mode:
            icon = "🌙"

        st.button(
            icon,
            on_click=toggle_theme,
            key="theme_btn"
        )

    with nav4:

        if st.button(
            "Get Started",
            type="primary",
            use_container_width=True
        ):

            st.session_state.app_started = True
            st.rerun()

    st.divider()

    st.write("<br><br>", unsafe_allow_html=True)

    col1, col2 = st.columns([1.2, 1], gap="large")

    with col1:

        st.markdown(
            '<div class="badge">✨ AI-POWERED CLINICAL ASSISTANT</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="hero-title">Smarter Nursing<br>with <span>AI</span> Support</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="hero-subtitle">A smart chatbot designed for nurses — access clinical protocols, perform medical calculations, and learn on the go.</div>',
            unsafe_allow_html=True
        )

        btn_col1, btn_col2, _ = st.columns([1, 1, 1.5])

        with btn_col1:

            if st.button(
                "Try Chatbot ➔",
                type="primary",
                use_container_width=True,
                key="try_btn"
            ):

                st.session_state.app_started = True
                st.rerun()

        with btn_col2:
            st.button(
                "Learn More",
                use_container_width=True
            )

    with col2:
        st.image(
            "nurse.png",
            use_container_width=True
        )

# ==========================================
# =============== CHAT APP =================
# ==========================================

else:

    current_messages = st.session_state.chat_sessions[
        st.session_state.current_chat
    ]

    # ======================================
    # LEFT SIDEBAR
    # ======================================

    with st.sidebar:

        st.markdown(
            f"<h3 style='color:{text_main};'>🩺 NursBot</h3>",
            unsafe_allow_html=True
        )

        if st.button(
            "➕ New chat",
            use_container_width=True
        ):

            st.session_state.chat_counter += 1

            new_chat_name = (
                "Chat " +
                str(st.session_state.chat_counter)
            )

            st.session_state.chat_sessions[new_chat_name] = []

            st.session_state.current_chat = new_chat_name

            st.rerun()

        st.write("<br>", unsafe_allow_html=True)

        st.markdown(
            f"<p style='color:{text_sub}; font-size:14px; font-weight:600;'>Recent</p>",
            unsafe_allow_html=True
        )

        for chat_id in reversed(
            list(st.session_state.chat_sessions.keys())
        ):

            title = get_chat_title(
                chat_id,
                st.session_state.chat_sessions[chat_id]
            )

            is_active = "💬 "

            if chat_id == st.session_state.current_chat:
                is_active = "🔹 "

            if st.button(
                is_active + title,
                key="hist_" + chat_id,
                use_container_width=True
            ):

                st.session_state.current_chat = chat_id
                st.rerun()

        st.divider()

        if st.button(
            "⬅ Back to Home",
            use_container_width=True
        ):

            st.session_state.app_started = False
            st.rerun()

    # ======================================
    # CHAT AREA
    # ======================================

    st.markdown(
        '<div class="main-chat-area">',
        unsafe_allow_html=True
    )

    _, head_btn = st.columns([5, 1])

    with head_btn:

        toggle_label = "⚡ Open Studio"

        if st.session_state.studio_expanded:
            toggle_label = "✖ Close Studio"

        if st.button(
            toggle_label,
            use_container_width=True
        ):

            st.session_state.studio_expanded = (
                not st.session_state.studio_expanded
            )

            st.rerun()

    # ======================================
    # CHAT CONTAINER
    # ======================================

    chat_container = st.container(
        height=550,
        border=False
    )

    with chat_container:

        for message in current_messages:

            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if len(current_messages) == 0:

            st.markdown(
                f"""
<h1 style='color:{text_main};
text-align:center;
margin-top:100px;'>
How can I help you today?
</h1>
""",
                unsafe_allow_html=True
            )

    # ======================================
    # TOOLS POPOVER
    # ======================================

    uploaded_file = None

    app_mode = "Clinical Vision (Image)"

    with st.popover(
        "➕ Tools & Attachments",
        help="Upload images, video, or speech"
    ):

        app_mode = st.selectbox(
            "Select AI Capability:",
            [
                "Clinical Vision (Image)",
                "Video Analysis",
                "Speech-to-Text",
                "Clinical Quiz"
            ]
        )

        if app_mode == "Clinical Vision (Image)":

            uploaded_file = st.file_uploader(
                "Upload image",
                type=["png", "jpg", "jpeg"]
            )

        elif app_mode == "Video Analysis":

            uploaded_file = st.file_uploader(
                "Upload video",
                type=["mp4", "mov"]
            )

        elif app_mode == "Speech-to-Text":

            uploaded_file = st.file_uploader(
                "Upload audio",
                type=["wav", "mp3"]
            )

        elif app_mode == "Clinical Quiz":

            st.slider("Questions", 1, 10, 5)

            st.selectbox(
                "Difficulty",
                [
                    "Beginner",
                    "Advanced",
                    "Specialist"
                ]
            )

    # ======================================
    # CHAT INPUT
    # ======================================

    if user_input := st.chat_input(
        "Message NursBot (" + app_mode + ")..."
    ):

        st.session_state.chat_sessions[
            st.session_state.current_chat
        ].append(
            {
                "role": "user",
                "content": user_input
            }
        )

        st.rerun()

    # ======================================
    # AI RESPONSE
    # ======================================

    if (
        len(current_messages) > 0
        and current_messages[-1]["role"] == "user"
    ):

        latest_user_input = current_messages[-1]["content"]

        with chat_container:

            with st.chat_message("assistant"):

                try:

                    if app_mode == "Clinical Vision (Image)":

                        if uploaded_file is not None:

                            img_bytes = uploaded_file.getvalue()

                            encoded_img = base64.b64encode(
                                img_bytes
                            ).decode("utf-8")

                            image_data = (
                                "data:image/jpeg;base64,"
                                + encoded_img
                            )

                            agent_input = [
                                HumanMessage(
                                    content=[
                                        {
                                            "type": "text",
                                            "text": latest_user_input
                                        },
                                        {
                                            "type": "image_url",
                                            "image_url": {
                                                "url": image_data
                                            }
                                        }
                                    ]
                                )
                            ]

                        else:

                            agent_input = [
                                HumanMessage(
                                    content=latest_user_input
                                )
                            ]

                        with st.spinner(
                            "Analyzing clinical data..."
                        ):

                            response = agent_executor.invoke(
                                {
                                    "input": agent_input,
                                    "chat_history":
                                    current_messages[:-1]
                                }
                            )

                        raw_output = str(
                            response.get("output", "")
                        )

                        match = re.search(
                            r"'text':\s*['\"](.*?)['\"],\s*'index':",
                            raw_output,
                            re.DOTALL
                        )

                        if match:

                            full_response = (
                                match.group(1)
                                .replace('\\n', '\n')
                                .replace('\\t', '\t')
                                .replace("\\'", "'")
                            )

                        else:
                            full_response = raw_output

                        st.markdown(full_response)

                        st.session_state.chat_sessions[
                            st.session_state.current_chat
                        ].append(
                            {
                                "role": "assistant",
                                "content": full_response
                            }
                        )

                        st.rerun()

                    elif app_mode == "Video Analysis":

                        response_text = (
                            "*(Teammate's Video API logic "
                            "will process this prompt)*"
                        )

                        st.markdown(response_text)

                        st.session_state.chat_sessions[
                            st.session_state.current_chat
                        ].append(
                            {
                                "role": "assistant",
                                "content": response_text
                            }
                        )

                        st.rerun()

                    elif app_mode == "Speech-to-Text":

                        response_text = (
                            "*(Teammate's Speech API logic "
                            "will process this prompt)*"
                        )

                        st.markdown(response_text)

                        st.session_state.chat_sessions[
                            st.session_state.current_chat
                        ].append(
                            {
                                "role": "assistant",
                                "content": response_text
                            }
                        )

                        st.rerun()

                    elif app_mode == "Clinical Quiz":

                        response_text = (
                            "*(Teammate's Quiz logic "
                            "will process this prompt)*"
                        )

                        st.markdown(response_text)

                        st.session_state.chat_sessions[
                            st.session_state.current_chat
                        ].append(
                            {
                                "role": "assistant",
                                "content": response_text
                            }
                        )

                        st.rerun()

                except Exception as e:

                    st.error("🚨 Error: " + str(e))

    st.markdown(
        "</div>",
        unsafe_allow_html=True
    )

    # ======================================
    # RIGHT STUDIO SIDEBAR
    # ======================================

    st.markdown(f"""
<div class="studio-sidebar">

    <div class="studio-content">

        <h3 style='color:{text_main}; margin-top:0;'>
            Studio
        </h3>

        <div class="studio-card">
            🎙️
            <div>
                <b>Audio Overview</b><br>
                <span style="font-size:12px; color:gray;">
                    Generate podcast
                </span>
            </div>
        </div>

        <div class="studio-card">
            📝
            <div>
                <b>Generate Quiz</b><br>
                <span style="font-size:12px; color:gray;">
                    Test your knowledge
                </span>
            </div>
        </div>

        <div class="studio-card">
            📊
            <div>
                <b>Data Table</b><br>
                <span style="font-size:12px; color:gray;">
                    Extract clinical stats
                </span>
            </div>
        </div>

        <div class="studio-card">
            🎥
            <div>
                <b>Video Summary</b><br>
                <span style="font-size:12px; color:gray;">
                    Analyze procedure
                </span>
            </div>
        </div>

    </div>

</div>
""", unsafe_allow_html=True)