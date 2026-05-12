import base64
import os
import re

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import streamlit as st

# ==========================================
# PAGE CONFIG
# ==========================================
st.set_page_config(
    page_title="NursBot | Clinical AI",
    page_icon="🏥",
    layout="wide"
)

# ==========================================
# MEMORY RESET
# ==========================================
if "nuke_complete" not in st.session_state:
    st.session_state.clear()
    st.session_state.nuke_complete = True

# ==========================================
# IMPORTS
# ==========================================
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage

# ==========================================
# AUTH
# ==========================================
if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

# ==========================================
# VECTOR DATABASE
# ==========================================
@st.cache_resource(show_spinner=False)
def initialize_retriever():
    try:
        with st.spinner("Building vector database..."):
            loader = PyPDFLoader("Section 01 - Medical Emergencies.pdf")
            docs = loader.load()

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=100
            )

            chunks = splitter.split_documents(docs)

            embeddings = HuggingFaceEmbeddings(
                model_name="all-MiniLM-L6-v2"
            )

            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings
            )

            return vectorstore.as_retriever()

    except Exception as e:
        st.error(f"Vector DB Error: {str(e)}")
        st.stop()


retriever = initialize_retriever()

# ==========================================
# TOOLS
# ==========================================
@tool
def search_nursing_protocols(query: str) -> str:
    """Search nursing protocols."""
    docs = retriever.invoke(query)
    return "\n\n".join([doc.page_content for doc in docs])


@tool
def calculate_fluid_requirement(weight_kg: float) -> str:
    """Calculate fluid requirement."""

    if weight_kg <= 10:
        res = weight_kg * 100
    elif weight_kg <= 20:
        res = 1000 + (weight_kg - 10) * 50
    else:
        res = 1500 + (weight_kg - 20) * 20

    return f"Fluid requirement: {res} mL/day"


tools = [
    calculate_fluid_requirement,
    search_nursing_protocols
]

# ==========================================
# LLM
# ==========================================
llm = ChatGoogleGenerativeAI(
    model="gemini-3-flash-preview",
    temperature=0
)

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
You are a professional KKH Clinical Nursing Assistant.

ONLY answer healthcare and nursing questions.

Reject unrelated questions politely.
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
# SESSION STATE
# ==========================================
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = True

if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = {
        "New Chat": []
    }

if "current_chat" not in st.session_state:
    st.session_state.current_chat = "New Chat"

if "chat_counter" not in st.session_state:
    st.session_state.chat_counter = 1

# ==========================================
# THEME
# ==========================================
if st.session_state.dark_mode:
    bg_color = "#131314"
    panel_color = "#1E1F20"
    border_color = "#2A2B2D"
    text_main = "#E3E3E3"
    text_sub = "#9AA0A6"
else:
    bg_color = "#FFFFFF"
    panel_color = "#F8F9FA"
    border_color = "#E5E7EB"
    text_main = "#1F2937"
    text_sub = "#4B5563"

# ==========================================
# CSS
# ==========================================
st.markdown(f"""
<style>

/* APP */
.stApp {{
    background: {bg_color};
}}

/* REMOVE DEFAULT PADDING */
.block-container {{
    max-width: 100%;
    padding-top: 1rem;
    padding-bottom: 0rem;
}}

/* HIDE SIDEBAR */
[data-testid="stSidebar"] {{
    display: none;
}}

/* PANELS */
.panel {{
    background: {panel_color};
    border: 1px solid {border_color};
    border-radius: 20px;
    padding: 18px;
    height: 88vh;
    overflow-y: auto;
}}

/* CHAT PANEL */
.chat-panel {{
    background: {panel_color};
    border: 1px solid {border_color};
    border-radius: 20px;
    padding: 20px;
    height: 88vh;
    overflow-y: auto;
}}

/* HEADINGS */
.main-title {{
    color: {text_main};
    font-size: 22px;
    font-weight: 700;
}}

.section-title {{
    color: {text_main};
    font-size: 18px;
    font-weight: 600;
    margin-bottom: 15px;
}}

/* CHAT BUTTONS */
.stButton > button {{
    border-radius: 12px !important;
    border: 1px solid {border_color} !important;
    background: transparent !important;
    color: {text_main} !important;
    text-align: left !important;
}}

.stButton > button:hover {{
    border-color: #4285F4 !important;
}}

/* STUDIO CARDS */
.studio-card {{
    border-radius: 18px;
    padding: 18px;
    margin-bottom: 15px;
    font-weight: 600;
    color: white;
    transition: 0.2s;
    cursor: pointer;
}}

.studio-card:hover {{
    transform: translateY(-2px);
}}

.audio {{
    background: linear-gradient(135deg,#5B8DEF,#7B61FF);
}}

.quiz {{
    background: linear-gradient(135deg,#00C896,#00A86B);
}}

.video {{
    background: linear-gradient(135deg,#FF6B6B,#FF8E53);
}}

.data {{
    background: linear-gradient(135deg,#36CFC9,#3A86FF);
}}

/* CHAT INPUT */
[data-testid="stChatInput"] {{
    position: sticky;
    bottom: 0;
    background: {panel_color};
    padding-top: 10px;
}}

/* TEXT */
p, div {{
    color: {text_main};
}}

</style>
""", unsafe_allow_html=True)

# ==========================================
# HELPERS
# ==========================================
def get_chat_title(chat_id, messages):
    if not messages:
        return chat_id

    for m in messages:
        if m["role"] == "user":
            return m["content"][:25] + "..."

    return chat_id

# ==========================================
# CURRENT CHAT
# ==========================================
current_messages = st.session_state.chat_sessions[
    st.session_state.current_chat
]

# ==========================================
# LAYOUT
# ==========================================
left_col, center_col, right_col = st.columns(
    [1.1, 2.4, 1.1],
    gap="medium"
)

# ==========================================
# LEFT PANEL
# ==========================================
with left_col:

    st.markdown(
        '<div class="panel">',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="main-title">🩺 NursBot</div>',
        unsafe_allow_html=True
    )

    st.write("")

    if st.button("➕ New Chat", use_container_width=True):

        st.session_state.chat_counter += 1

        new_chat = f"Chat {st.session_state.chat_counter}"

        st.session_state.chat_sessions[new_chat] = []

        st.session_state.current_chat = new_chat

        st.rerun()

    st.write("")

    st.markdown(
        '<div class="section-title">Recent Chats</div>',
        unsafe_allow_html=True
    )

    for chat_id in reversed(
        list(st.session_state.chat_sessions.keys())
    ):

        title = get_chat_title(
            chat_id,
            st.session_state.chat_sessions[chat_id]
        )

        prefix = "🔹" if chat_id == st.session_state.current_chat else "💬"

        if st.button(
            f"{prefix} {title}",
            key=chat_id,
            use_container_width=True
        ):
            st.session_state.current_chat = chat_id
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

# ==========================================
# CENTER CHAT PANEL
# ==========================================
with center_col:

    st.markdown(
        '<div class="chat-panel">',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-title">Chat</div>',
        unsafe_allow_html=True
    )

    # RENDER MESSAGES
    for message in current_messages:

        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # EMPTY STATE
    if len(current_messages) == 0:

        st.markdown("""
        <div style='
            text-align:center;
            margin-top:120px;
            color:#9AA0A6;
        '>
            <h1>How can I help you today?</h1>
            <p>Ask about protocols, emergencies, medications, or calculations.</p>
        </div>
        """, unsafe_allow_html=True)

    # FILE UPLOAD
    uploaded_file = st.file_uploader(
        "Upload Clinical Image",
        type=["png", "jpg", "jpeg"]
    )

    # CHAT INPUT
    user_input = st.chat_input(
        "Ask NursBot..."
    )

    st.markdown("</div>", unsafe_allow_html=True)

# ==========================================
# AI RESPONSE
# ==========================================
if user_input:

    st.session_state.chat_sessions[
        st.session_state.current_chat
    ].append({
        "role": "user",
        "content": user_input
    })

    current_messages = st.session_state.chat_sessions[
        st.session_state.current_chat
    ]

    with center_col:

        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):

            try:

                if uploaded_file is not None:

                    img_bytes = uploaded_file.getvalue()

                    encoded_img = base64.b64encode(
                        img_bytes
                    ).decode("utf-8")

                    image_data = f"data:image/jpeg;base64,{encoded_img}"

                    agent_input = [
                        HumanMessage(
                            content=[
                                {
                                    "type": "text",
                                    "text": user_input
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
                        HumanMessage(content=user_input)
                    ]

                with st.spinner("Analyzing..."):

                    response = agent_executor.invoke({
                        "input": agent_input,
                        "chat_history": current_messages[:-1]
                    })

                raw_output = str(
                    response.get("output", "")
                )

                match = re.search(
                    r"'text':\s*['\"](.*?)['\"],\s*'index':",
                    raw_output,
                    re.DOTALL
                )

                full_response = (
                    match.group(1)
                    .replace('\\n', '\n')
                    .replace('\\t', '\t')
                    .replace("\\'", "'")
                    if match
                    else raw_output
                )

                st.markdown(full_response)

                st.session_state.chat_sessions[
                    st.session_state.current_chat
                ].append({
                    "role": "assistant",
                    "content": full_response
                })

            except Exception as e:

                st.error(f"Error: {str(e)}")

# ==========================================
# RIGHT PANEL
# ==========================================
with right_col:

    st.markdown(
        '<div class="panel">',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-title">Studio</div>',
        unsafe_allow_html=True
    )

    st.markdown("""
    <div class="studio-card audio">
        🎙️ Audio Overview
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="studio-card quiz">
        📝 Generate Quiz
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="studio-card video">
        🎥 Video Summary
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="studio-card data">
        📊 Data Table
    </div>
    """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)