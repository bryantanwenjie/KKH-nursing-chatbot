import base64
import os
import re
import time
import json
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import streamlit as st
import hashlib
import pyodbc

from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage

# 👉 JOESON'S CODE: Imports for Azure and Speech-to-Text
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
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
    if "AZURE_OPENAI_EMBEDDING_DEPLOYMENT" in st.secrets:
        os.environ["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"] = st.secrets["AZURE_OPENAI_EMBEDDING_DEPLOYMENT"]

# --- AZURE SQL DATABASE CONNECTION + LOGIN SYSTEM ---
def get_db_connection():
    server = st.secrets["AZURE_SQL_SERVER"].strip()
    database = st.secrets["AZURE_SQL_DATABASE"].strip()
    username = st.secrets["AZURE_SQL_USERNAME"].strip()
    password = st.secrets["AZURE_SQL_PASSWORD"]

    if server.startswith("tcp:"):
        server = server.replace("tcp:", "")
    if ",1433" in server:
        server = server.replace(",1433", "")

    conn_str = (
        "DRIVER={ODBC Driver 17 for SQL Server};"
        f"SERVER=tcp:{server},1433;"
        f"DATABASE={database};"
        f"UID={username};"
        f"PWD={password};"
        "Encrypt=yes;"
        "TrustServerCertificate=no;"
        "Connection Timeout=30;"
    )

    try:
        conn = pyodbc.connect(conn_str)
        return conn
    except pyodbc.Error as e:
        st.error("Azure SQL connection failed.")
        st.code(str(e))
        st.info("""
Please check:
1. Azure SQL firewall has your current IP address added.
2. Server name is like: yourserver.database.windows.net
3. Port 1433 is not blocked by school/WiFi/firewall.
4. Username and password are correct.
5. Database name is correct.
""")
        return None

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(full_name, email, password):
    try:
        conn = get_db_connection()
        if conn is None: 
            return False, "Cannot connect to Azure SQL Database."
        cursor = conn.cursor()
        password_hash = hash_password(password)

        query = """
        INSERT INTO users (full_name, email, password_hash)
        VALUES (?, ?, ?)
        """
        cursor.execute(query, (full_name, email, password_hash))
        conn.commit()
        cursor.close()
        conn.close()

        return True, "Registration successful! Please login."

    except pyodbc.IntegrityError:
        return False, "Email already exists. Please login instead."
    except pyodbc.Error as e:
        return False, f"Database error: {str(e)}"
    except Exception as e:
        return False, f"Unexpected error: {str(e)}"

def login_user(email, password):
    try:
        conn = get_db_connection()
        if conn is None: 
            return False, None

        cursor = conn.cursor()
        password_hash = hash_password(password)

        query = """
        SELECT user_id, full_name, email
        FROM users
        WHERE email = ? AND password_hash = ?
        """
        cursor.execute(query, (email, password_hash))
        row = cursor.fetchone()
        cursor.close()
        conn.close()

        if row:
            return True, {
                "user_id": row[0],
                "full_name": row[1],
                "email": row[2]
            }
        return False, None

    except pyodbc.Error as e:
        st.error(f"Database error: {str(e)}")
        return False, None
    except Exception as e:
        st.error(f"Unexpected error: {str(e)}")
        return False, None

# 👉 ZHEN RONG'S CODE: Video Request & SQL Query Processing Functions
def is_video_request(question):
    question = question.lower()
    video_keywords = [
        "video",
        "youtube",
        "tutorial",
        "watch",
        "show me",
        "demonstration",
        "demo",
        "link"
    ]
    return any(word in question for word in video_keywords)

def search_video_tutorial(user_question):
    conn = get_db_connection()
    if conn is None:
        return []
    cursor = conn.cursor()
    q = user_question.lower()

    if (
        "infant cpr" in q
        or "baby cpr" in q
        or "cpr" in q
        or "cardiopulmonary resuscitation" in q
        or "resuscitation" in q
    ):
        search_text = "%cpr%"
    elif (
        "blood pressure" in q
        or "bp" in q
        or "systolic" in q
        or "diastolic" in q
        or "measure blood pressure" in q
        or "check blood pressure" in q
    ):
        search_text = "%blood_pressure%"
    elif (
        "heart rate" in q
        or "pulse" in q
        or "check pulse" in q
        or "check heart rate" in q
    ):
        search_text = "%heart_rate%"
    else:
        search_text = "%" + q + "%"

    query = """
    SELECT TOP 3 title, topic, description, youtube_url
    FROM video_tutorials
    WHERE category LIKE ?
       OR title LIKE ?
       OR topic LIKE ?
       OR description LIKE ?
       OR keywords LIKE ?
    """

    cursor.execute(
        query,
        search_text,
        search_text,
        search_text,
        search_text,
        search_text
    )

    rows = cursor.fetchall()
    cursor.close()
    conn.close()

    videos = []
    for row in rows:
        # Failsafe: if the database returns an unexpected number of columns, skip it
        if len(row) != 4:
            continue
            
        # Cleanly UNPACK the 4 columns directly into 4 separate variables
        raw_title, raw_topic, raw_desc, raw_url = row

        videos.append({
            "title": str(raw_title).strip() if raw_title else "No Title",
            "topic": str(raw_topic).strip() if raw_topic else "No Topic",
            "description": str(raw_desc).strip() if raw_desc else "No Description",
            "youtube_url": str(raw_url).strip() if raw_url else ""
        })
        
    return videos

# ==========================================
# ======= BACKEND & CLINICAL LOGIC =========
# ==========================================
@st.cache_resource(show_spinner=False)
def initialize_retriever():
    persist_directory = "./chroma_db"
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    try:
        if os.path.exists(persist_directory) and os.listdir(persist_directory):
            vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
            return vectorstore.as_retriever(search_kwargs={"k": 3})
            
        with st.spinner("Building Vector Database for the first time... (This will be cached)"):
            pdf_path = "Section 01 - Medical Emergencies.pdf"
            if not os.path.exists(pdf_path):
                st.warning(f"File '{pdf_path}' not found. Vector DB will be empty.")
                return None
                
            loader = PyPDFLoader(pdf_path)
            docs = loader.load()
            
            # 👉 TOKEN OPTIMIZATION: Cut chunk size in half to save tokens
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
            chunks = text_splitter.split_documents(docs)

            vectorstore = Chroma.from_documents(
                documents=chunks, 
                embedding=embeddings, 
                persist_directory=persist_directory
            )
            # 👉 TOKEN OPTIMIZATION: Retrieve top 2 matches instead of top 3
            return vectorstore.as_retriever(search_kwargs={"k": 2})
            
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
        
    if weight_kg <= 10: 
        daily = weight_kg * 100
        hourly = weight_kg * 4
    elif weight_kg <= 20: 
        daily = 1000 + ((weight_kg - 10) * 50)
        hourly = 40 + ((weight_kg - 10) * 2)
    else: 
        daily = 1500 + ((weight_kg - 20) * 20)
        hourly = 60 + ((weight_kg - 20) * 1)
    
    daily = min(daily, 2500)
    
    return f"Daily requirement: {daily:.0f} mL/day.\nHourly requirement: {hourly:.0f} mL/hr.{warning}"

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

@tool
def calculate_parkland_burn_fluid(weight_kg: float, burn_percentage: float) -> str:
    """Calculates Parkland formula fluid replacement for 2nd/3rd degree burns."""
    vol_min = 3 * weight_kg * burn_percentage
    vol_max = 4 * weight_kg * burn_percentage
    
    half_min = vol_min / 2
    half_max = vol_max / 2
    
    return (
        f"Total 24-hour volume: {vol_min:.0f} - {vol_max:.0f} mL.\n"
        f"1st 8 hours: Give {half_min:.0f} - {half_max:.0f} mL.\n"
        f"Next 16 hours: Give remaining {half_min:.0f} - {half_max:.0f} mL."
    )

@tool
def expected_urine_output(weight_kg: float, is_neonate: bool) -> str:
    """Calculates the expected minimum urine output per hour."""
    if is_neonate:
        expected = 0.5 * weight_kg
        return f"For a neonate ({weight_kg} kg), normal expected urine output is > {expected:.1f} mL/hr."
    else:
        expected = 1.0 * weight_kg
        return f"For an infant or older child ({weight_kg} kg), normal expected urine output is > {expected:.1f} mL/hr."

@tool
def check_vitals_ranges(age_years: float) -> str:
    """Returns the normal Heart Rate and Respiratory Rate ranges based on pediatric age."""
    if age_years < 0.25:
        return "Heart Rate: 90 - 180 bpm | Respiratory Rate: 30 - 60 breaths/min"
    elif age_years < 0.5:
        return "Heart Rate: 80 - 160 bpm | Respiratory Rate: 30 - 60 breaths/min"
    elif age_years < 1.0:
        return "Heart Rate: 80 - 140 bpm | Respiratory Rate: 25 - 45 breaths/min"
    elif age_years < 6.0:
        return "Heart Rate: 75 - 130 bpm | Respiratory Rate: 20 - 30 breaths/min"
    elif age_years < 10.0:
        return "Heart Rate: 70 - 110 bpm | Respiratory Rate: 16 - 24 breaths/min"
    elif age_years < 15.0:
        return "Heart Rate: 60 - 90 bpm | Respiratory Rate: 14 - 20 breaths/min"
    else:
        return "Heart Rate: 60 - 90 bpm | Respiratory Rate: 12 - 16 breaths/min"

tools = [
    calculate_fluid_requirement, 
    search_nursing_protocols, 
    calculate_systolic_bp,
    calculate_parkland_burn_fluid,
    expected_urine_output,
    check_vitals_ranges
]

# ==========================================
# ======= LLM INITIALIZATION SECTION =======
# ==========================================

# 1. Bryan's Gemini Model
# Uses the standard GOOGLE_API_KEY from secrets
bryan_gemini_key = st.secrets.get("GOOGLE_API_KEY", "")
llm_gemini_bryan = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",  # Put Bryan's specific model name here
    api_key=bryan_gemini_key,
    temperature=0
)

# 2. Zhen Rong's Gemini Model
# Uses the custom ZHEN_RONG_GOOGLE_API_KEY from secrets
zhen_rong_gemini_key = st.secrets.get("ZHEN_RONG_GOOGLE_API_KEY", "")
llm_gemini_zhenrong = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",  # Put Zhen Rong's specific model name here
    api_key=zhen_rong_gemini_key,
    temperature=0
)

# 3. Joeson's Azure Model (Keep as it is)
try:
    llm_azure = AzureChatOpenAI(
        azure_endpoint=st.secrets["JOESON_AZURE_OPENAI_ENDPOINT"],
        api_key=st.secrets["JOESON_AZURE_OPENAI_API_KEY"],
        api_version=st.secrets["JOESON_AZURE_OPENAI_API_VERSION"],
        azure_deployment=st.secrets["JOESON_AZURE_OPENAI_CHAT_DEPLOYMENT"],
        temperature=0
    )
except:
    llm_azure = None

# --- UPDATE SYSTEM PROMPT BLOCK ---
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a strictly professional KKH Clinical Nursing Assistant. 
    
    CRITICAL RULES:
    1. If a user asks a question unrelated to healthcare, nursing, or KKH, politely refuse.
    2. Do NOT use general knowledge for KKH protocols; ALWAYS use the `search_nursing_protocols` tool.
    3. When quoting protocols, mention the Source Page Number provided by the tool.
    4. If calculating fluids or BP, clearly display the math and any clinical warnings.
    5. Be concise, structured, and use bullet points for readability.
    6. IF the user asks for a video or tutorial, positively acknowledge their request, provide a brief clinical introduction, and state that you have retrieved the video from the database below. DO NOT say you cannot provide videos.
    7. VISION & HANDWRITING RULE: When analyzing uploaded images or handwritten notes, extract the variables EXACTLY as written. If text is rotated or sideways, mentally orient it to read it correctly. If the handwriting is ambiguous or illegible, DO NOT guess or hallucinate the numbers. Halt tool execution immediately and trigger the safety formatting defined in Rule 8.
    8. ZERO HALLUCINATION & SAFETY FORMATTING: You are strictly forbidden from inventing or guessing patient variables (like age, weight, or vitals). If you lack the required data to run a tool, or if an uploaded image is unreadable, invalid, or lacks clinical text, you MUST halt and respond using EXACTLY this Markdown template structure:
    
    ⚠️ **Clinical Extraction Error**
    [Insert a concise 1-sentence explanation of exactly what is missing or why the image cannot be processed.]
    
    **To proceed safely in accordance with KKH protocols, please provide:**
    * Patient's Age / DOB
    * Current Weight (kg)
    * Current Vital Signs (HR, RR, BP, Temp)
    * *Or upload a clearer image of the clinical notes.*
    
    9. ANTI-DIAGNOSIS RULE: You are a nursing assistant, not a doctor. You must NEVER diagnose a patient's condition, suggest a new medical illness, or prescribe medications. If a user asks for a diagnosis based on text, symptoms, or images, explicitly refuse by stating that KKH protocols require a doctor to make official medical diagnoses.
    """),
    ("placeholder", "{chat_history}"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

def get_langchain_history(messages):
    history = []
    for m in messages:
        if m["role"] == "user":
            history.append(HumanMessage(content=m["content"]))
        else:
            history.append(AIMessage(content=m["content"]))
    return history

# ==========================================
# =========== QUIZ GENERATOR LOGIC ==========
# ==========================================

QUIZ_PDF_PATHS = [
    "Section 01 - Medical Emergencies.pdf",
    "Section_01_Medical Emergencies.pdf",
    "Section_01_Medical_Emergencies.pdf",
]

def get_quiz_pdf_path():
    for path in QUIZ_PDF_PATHS:
        if os.path.exists(path):
            return path
    return QUIZ_PDF_PATHS

def check_quiz_env():
    required_keys = [
        "CHEEYOU_AZURE_OPENAI_ENDPOINT",
        "CHEEYOU_AZURE_OPENAI_API_KEY",
        "CHEEYOU_AZURE_OPENAI_API_VERSION",
        "CHEEYOU_AZURE_OPENAI_EMBEDDING_DEPLOYMENT",
        "CHEEYOU_AZURE_OPENAI_CHAT_DEPLOYMENT",
    ]
    return [key for key in required_keys if key not in st.secrets]

def clean_json_response(text):
    text = text.strip()
    text = re.sub(r"^```json", "", text)
    text = re.sub(r"^```", "", text)
    text = re.sub(r"```$", "", text)
    return text.strip()

@st.cache_resource(show_spinner=False)
def create_quiz_vectorstore():
    pdf_path = get_quiz_pdf_path()
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)

    embeddings = AzureOpenAIEmbeddings(
        azure_endpoint=st.secrets["CHEEYOU_AZURE_OPENAI_ENDPOINT"],
        api_key=st.secrets["CHEEYOU_AZURE_OPENAI_API_KEY"],
        api_version=st.secrets["CHEEYOU_AZURE_OPENAI_API_VERSION"],
        azure_deployment=st.secrets["CHEEYOU_AZURE_OPENAI_EMBEDDING_DEPLOYMENT"],
    )
    return FAISS.from_documents(chunks, embeddings)

@st.cache_resource(show_spinner=False)
def create_quiz_llm():
    return AzureChatOpenAI(
        azure_endpoint=st.secrets["CHEEYOU_AZURE_OPENAI_ENDPOINT"],
        api_key=st.secrets["CHEEYOU_AZURE_OPENAI_API_KEY"],
        api_version=st.secrets["CHEEYOU_AZURE_OPENAI_API_VERSION"],
        azure_deployment=st.secrets["CHEEYOU_AZURE_OPENAI_CHAT_DEPLOYMENT"],
    )

# --- 1. MCQ QUIZ GENERATOR ---
def generate_quiz(vectorstore, llm, topic, number_of_questions):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    relevant_docs = retriever.invoke(topic)
    context = "\n\n".join([doc.page_content for doc in relevant_docs])

    quiz_prompt = PromptTemplate.from_template("""
You are a nursing educator.

Use ONLY the context below to generate a quiz.

Context:
{context}

Topic:
{topic}

Generate exactly {number_of_questions} multiple-choice questions.

Return ONLY valid JSON.
Do not include markdown.
Do not include extra text.

JSON format:
[
  {{
    "question": "Question text here",
    "options": {{
      "A": "Option A",
      "B": "Option B",
      "C": "Option C",
      "D": "Option D"
    }},
    "correct_answer": "A",
    "explanation": "Short explanation here"
  }}
]

Rules:
- Each question must have 4 options: A, B, C, D
- correct_answer must be only A, B, C, or D
- Use only the provided context
- Do not invent medical facts
- Explanation must be short and simple
""")

    final_prompt = quiz_prompt.format(
        context=context,
        topic=topic,
        number_of_questions=number_of_questions
    )

    response = llm.invoke(final_prompt)
    cleaned_response = clean_json_response(response.content)

    try:
        quiz = json.loads(cleaned_response)
        return quiz, relevant_docs
    except json.JSONDecodeError:
        st.error("The AI did not return valid JSON.")
        st.code(response.content)
        return None, relevant_docs

# --- 2. CLINICAL SCENARIO GENERATOR ---
def generate_clinical_scenarios(vectorstore, llm, number_of_scenarios):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 8})
    relevant_docs = retriever.invoke("clinical emergency nursing scenarios protocols guidelines")
    context = "\n\n".join([doc.page_content for doc in relevant_docs])

    scenario_prompt = PromptTemplate.from_template("""
You are a nursing educator.

Use ONLY the PDF context below to generate clinical scenario questions.

Context:
{context}

Generate exactly {number_of_scenarios} clinical scenario questions.

Return ONLY valid JSON. No markdown.

JSON format:
[
  {{
    "scenario": "Patient scenario here",
    "question": "What should the nurse do?",
    "model_answer": "Expected answer based only on the PDF",
    "marking_points": [
      "point 1",
      "point 2",
      "point 3"
    ]
  }}
]

Rules:
- Use only the PDF context
- Do not invent medical facts
- Scenario must be realistic for nursing practice
- Model answer must be short and clear
""")

    final_prompt = scenario_prompt.format(
        context=context,
        number_of_scenarios=number_of_scenarios
    )

    response = llm.invoke(final_prompt)
    cleaned_response = clean_json_response(response.content)

    try:
        scenarios = json.loads(cleaned_response)
        return scenarios, relevant_docs
    except json.JSONDecodeError:
        st.error("The AI did not return valid JSON for scenarios.")
        st.code(response.content)
        return None, relevant_docs
    
# --- 3. SCENARIO MARKING AI ---
def mark_scenario_answer(llm, scenario, question, model_answer, marking_points, user_answer):
    marking_prompt = f"""
You are a strict nursing educator.

Mark the user's answer based ONLY on the model answer and marking points.

Scenario:
{scenario}

Question:
{question}

Model answer:
{model_answer}

Marking points:
{marking_points}

User answer:
{user_answer}

Return in this format:

Score: X/5
Result: Correct / Partially Correct / Incorrect
Feedback:
- What the user did well
- What is missing
- Correct explanation based on the PDF

Do not add medical information outside the provided answer.
"""

    response = llm.invoke(marking_prompt)
    return response.content

# --- RESET LOGIC ---
def reset_quiz():
    st.session_state.quiz = None
    st.session_state.quiz_answers = {}
    st.session_state.quiz_submitted = False
    st.session_state.quiz_relevant_docs = []
    st.session_state.scenarios = None
    st.session_state.scenario_answers = {}
    st.session_state.scenario_feedback = {}
    st.session_state.scenario_relevant_docs = []

def render_quiz_page(text_main, text_sub, card_bg, divider_color):
    with st.sidebar:
        st.markdown(f"<h3 style='color:{text_main}; margin-top:-20px;'>🩺 NursBot</h3>", unsafe_allow_html=True)

        if st.session_state.logged_in:
            st.success(f"Logged in as {st.session_state.user_email}")

        if st.button("⬅ Back to Chatbot", use_container_width=True):
            st.session_state.current_page = "chat"
            st.rerun()

        if st.button("Logout", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.user_email = None
            st.session_state.user_full_name = None
            st.session_state.app_started = False
            st.session_state.current_page = "chat"
            st.rerun()

        st.divider()
        st.header("⚙️ Quiz Settings")

        topic = st.text_input(
            "Enter quiz topic",
            placeholder="Example: infant heart rate, shock, seizures, blood pressure",
            key="quiz_topic"
        )

        number_of_questions = st.number_input(
            "Number of questions",
            min_value=1,
            max_value=10,
            value=5,
            step=1,
            key="quiz_number_of_questions"
        )

        generate_button = st.button("Generate Quiz", type="primary", use_container_width=True)

        st.divider()
        st.header("🏥 Clinical Scenario Settings")

        number_of_scenarios = st.number_input(
            "Number of clinical scenario questions",
            min_value=1,
            max_value=10,
            value=3,
            step=1,
            key="scenario_number_of_scenarios"
        )

        generate_scenarios_button = st.button(
            "Generate Clinical Scenarios",
            type="primary",
            use_container_width=True
        )

        if st.button("Reset Quiz / Scenarios", use_container_width=True):
            reset_quiz()
            st.rerun()

    st.markdown(f"<h1 style='color:{text_main};'>📝 Nursing Quiz Generator</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:{text_sub};'>Generate MCQ quizzes and clinical scenarios based on your Medical Emergencies PDF.</p>", unsafe_allow_html=True)

    missing_keys = check_quiz_env()
    if missing_keys:
        st.error("Missing Azure OpenAI environment variables / secrets:")
        for key in missing_keys:
            st.code(key)
        st.info("Add the missing keys into `.streamlit/secrets.toml`, then restart Streamlit.")
        st.stop()

    pdf_path = get_quiz_pdf_path()
    # Handle the fact that get_quiz_pdf_path returns a list in app.py or a string in cheeyou
    # Assuming app.py get_quiz_pdf_path was modified to return a single path based on Chee You's code.
    if isinstance(pdf_path, list):
        pdf_path = pdf_path # Failsafe
        
    if not os.path.exists(pdf_path):
        st.error(f"PDF file not found: {pdf_path}")
        st.warning("Put your Medical Emergencies PDF in the same folder as your Streamlit app.")
        st.stop()

    if "quiz" not in st.session_state:
        st.session_state.quiz = None
    if "quiz_answers" not in st.session_state:
        st.session_state.quiz_answers = {}
    if "quiz_submitted" not in st.session_state:
        st.session_state.quiz_submitted = False
    if "quiz_relevant_docs" not in st.session_state:
        st.session_state.quiz_relevant_docs = []
    if "scenarios" not in st.session_state:
        st.session_state.scenarios = None
    if "scenario_answers" not in st.session_state:
        st.session_state.scenario_answers = {}
    if "scenario_feedback" not in st.session_state:
        st.session_state.scenario_feedback = {}
    if "scenario_relevant_docs" not in st.session_state:
        st.session_state.scenario_relevant_docs = []

    try:
        with st.spinner("Loading quiz vector database..."):
            vectorstore = create_quiz_vectorstore()
        with st.spinner("Loading Azure OpenAI quiz model..."):
            quiz_llm = create_quiz_llm()
    except Exception as e:
        st.error("Error while loading quiz generator.")
        st.code(str(e))
        st.stop()

    if generate_button:
        if not topic.strip():
            st.warning("Please enter a quiz topic first.")
        else:
            st.session_state.quiz = None
            st.session_state.quiz_answers = {}
            st.session_state.quiz_submitted = False
            st.session_state.quiz_relevant_docs = []
            with st.spinner("Generating quiz..."):
                quiz, relevant_docs = generate_quiz(
                    vectorstore,
                    quiz_llm,
                    topic,
                    number_of_questions
                )

            if quiz:
                st.session_state.quiz = quiz
                st.session_state.quiz_relevant_docs = relevant_docs
                st.success("Quiz generated successfully!")

    if generate_scenarios_button:
        st.session_state.scenarios = None
        st.session_state.scenario_answers = {}
        st.session_state.scenario_feedback = {}
        st.session_state.scenario_relevant_docs = []

        with st.spinner("Generating clinical scenarios from PDF..."):
            scenarios, relevant_docs = generate_clinical_scenarios(
                vectorstore,
                quiz_llm,
                number_of_scenarios
            )

        if scenarios:
            st.session_state.scenarios = scenarios
            st.session_state.scenario_relevant_docs = relevant_docs
            st.success("Clinical scenarios generated successfully!")

    if st.session_state.quiz:
        st.subheader("📝 Multiple-Choice Quiz")

        for i, q in enumerate(st.session_state.quiz):
            st.markdown(f"### Question {i + 1}")
            st.write(q["question"])

            options = q["options"]
            selected_answer = st.radio(
                label="Choose your answer:",
                options=list(options.keys()),
                format_func=lambda x: f"{x}. {options[x]}",
                key=f"quiz_question_{i}",
                disabled=st.session_state.quiz_submitted
            )

            st.session_state.quiz_answers[i] = selected_answer

            if st.session_state.quiz_submitted:
                correct_answer = q["correct_answer"].strip().upper()

                if selected_answer == correct_answer:
                    st.success("Correct!")
                else:
                    st.error(f"Wrong. Correct answer: {correct_answer}")

                st.info(f"Explanation: {q['explanation']}")

            st.divider()

        if not st.session_state.quiz_submitted:
            if st.button("Submit Quiz Answers", type="primary"):
                st.session_state.quiz_submitted = True
                st.rerun()
        else:
            score = 0
            for i, q in enumerate(st.session_state.quiz):
                correct_answer = q["correct_answer"].strip().upper()
                user_answer = st.session_state.quiz_answers.get(i)
                if user_answer == correct_answer:
                    score += 1

            st.subheader(f"Final Score: {score}/{len(st.session_state.quiz)}")

            if score == len(st.session_state.quiz):
                st.success("Excellent! You got all questions correct.")
            elif score >= len(st.session_state.quiz) / 2:
                st.warning("Good try! Review the explanations to improve.")
            else:
                st.error("Keep practising. Review the PDF content again.")

    if st.session_state.scenarios:
        st.subheader("🏥 Clinical Scenario Questions")
        st.caption("Answer in your own words. The AI will mark your answer based on the PDF-generated marking points.")

        for i, s in enumerate(st.session_state.scenarios):
            st.markdown(f"### Scenario {i + 1}")
            st.info(s["scenario"])
            st.write(f"**Question:** {s['question']}")

            user_answer = st.text_area(
                "Type your answer here:",
                key=f"scenario_answer_{i}",
                height=120
            )

            st.session_state.scenario_answers[i] = user_answer

            if st.button(f"Submit Scenario {i + 1}", key=f"submit_scenario_{i}"):
                if not user_answer.strip():
                    st.warning("Please type your answer first.")
                else:
                    with st.spinner("AI is marking your answer based on the PDF..."):
                        feedback = mark_scenario_answer(
                            quiz_llm,
                            s["scenario"],
                            s["question"],
                            s["model_answer"],
                            s["marking_points"],
                            user_answer
                        )

                    st.session_state.scenario_feedback[i] = feedback
                    st.rerun()

            if i in st.session_state.scenario_feedback:
                st.success("Marked by AI")
                st.write(st.session_state.scenario_feedback[i])

            with st.expander("View model answer / marking points"):
                st.write("**Model answer:**")
                st.write(s["model_answer"])
                st.write("**Marking points:**")
                for point in s["marking_points"]:
                    st.write(f"- {point}")

            st.divider()

    if st.session_state.quiz_relevant_docs:
        with st.expander("View retrieved quiz PDF sources"):
            for i, doc in enumerate(st.session_state.quiz_relevant_docs, start=1):
                page = doc.metadata.get("page", "Unknown")
                st.markdown(f"#### Source {i} | Page {page}")
                st.write(doc.page_content[:1000])

    if st.session_state.scenario_relevant_docs:
        with st.expander("View retrieved clinical scenario PDF sources"):
            for i, doc in enumerate(st.session_state.scenario_relevant_docs, start=1):
                page = doc.metadata.get("page", "Unknown")
                st.markdown(f"#### Source {i} | Page {page}")
                st.write(doc.page_content[:1000])

# ==========================================
# =========== FRONTEND UI ==================
# ==========================================

# --- STATE MANAGEMENT ---
if "app_started" not in st.session_state:
    st.session_state.app_started = False
if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False 
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
if "current_page" not in st.session_state:
    st.session_state.current_page = "chat"

# --- LOGIN SESSION STATES ---
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "user_email" not in st.session_state:
    st.session_state.user_email = None
if "user_full_name" not in st.session_state:
    st.session_state.user_full_name = None
if "show_login_popup" not in st.session_state:
    st.session_state.show_login_popup = False
if "show_register_popup" not in st.session_state:
    st.session_state.show_register_popup = False

def toggle_theme():
    st.session_state.dark_mode = not st.session_state.dark_mode

def get_chat_title(chat_id, messages):
    if not messages:
        return chat_id
    for m in messages:
        if m["role"] == "user":
            return m["content"][:20] + "..."
    return chat_id

@st.dialog("Login to NursBot")
def login_popup():
    st.markdown("### Welcome back 👋")

    email = st.text_input("Email", key="login_email")
    password = st.text_input("Password", type="password", key="login_password")

    login_col, register_col = st.columns(2)

    with login_col:
        if st.button("Login", type="primary", use_container_width=True):
            if not email or not password:
                st.error("Please enter your email and password.")
            else:
                success, user = login_user(email, password)

                if success:
                    st.session_state.logged_in = True
                    st.session_state.user_email = user["email"]
                    st.session_state.user_full_name = user["full_name"]
                    st.session_state.app_started = True
                    st.session_state.current_page = "chat"
                    st.session_state.show_login_popup = False
                    st.session_state.show_register_popup = False
                    st.success("Login successful!")
                    st.rerun()
                else:
                    st.error("Invalid email or password.")

    with register_col:
        if st.button("Register", use_container_width=True):
            st.session_state.show_login_popup = False
            st.session_state.show_register_popup = True
            st.rerun()

@st.dialog("Register New Account")
def register_popup():
    st.markdown("### Create your NursBot account")

    full_name = st.text_input("Full Name", key="register_full_name")
    email = st.text_input("Email", key="register_email")
    password = st.text_input("Password", type="password", key="register_password")
    confirm_password = st.text_input("Confirm Password", type="password", key="register_confirm_password")

    if st.button("Create Account", type="primary", use_container_width=True):
        if not full_name or not email or not password or not confirm_password:
            st.error("Please fill in all fields.")
        elif password != confirm_password:
            st.error("Passwords do not match.")
        else:
            success, message = register_user(full_name, email, password)

            if success:
                st.success(message)
                st.session_state.show_register_popup = False
                st.session_state.show_login_popup = True
                st.rerun()
            else:
                st.error(message)

    if st.button("Back to Login", use_container_width=True):
        st.session_state.show_register_popup = False
        st.session_state.show_login_popup = True
        st.rerun()

# --- DYNAMIC THEME COLORS ---
if st.session_state.dark_mode:
    bg_color = "#0B1120"
    text_main = "#FFFFFF"
    text_sub = "#94A3B8"
    nav_color = "#E2E8F0"
    divider_color = "#1E293B"
    card_bg = "#0F172A"
    card_hover = "rgba(59, 130, 246, 0.1)"
    badge_bg = "rgba(59, 130, 246, 0.1)"
    badge_text = "#3B82F6"
    badge_border = "rgba(59, 130, 246, 0.2)"
    primary_btn = "#3B82F6"
    primary_hover = "#2563EB"
    sec_btn_border = "#334155"
else:
    bg_color = "#F4F7F9"
    text_main = "#0F172A"
    text_sub = "#475569"
    nav_color = "#475569"
    divider_color = "#E2E8F0"
    card_bg = "#FFFFFF"
    card_hover = "rgba(29, 104, 189, 0.05)"
    badge_bg = "#EBF2FA"
    badge_text = "#1D68BD"
    badge_border = "#D6E4F4"
    primary_btn = "#1D68BD"
    primary_hover = "#15529A"
    sec_btn_border = "#E2E8F0"

# --- CUSTOM CSS ---
st.markdown(f"""
<style>
    /* 1. App & Background Layout */
    [data-testid="stAppViewContainer"] {{ background: {bg_color}; }}
    [data-testid="stHeader"] {{ background: transparent; }}
    
    /* 2. Sidebar & Dividers */
    [data-testid="stSidebar"] {{ background-color: {card_bg} !important; }}
    hr {{ border-bottom-color: {divider_color} !important; opacity: 1 !important; }}

    /* 3. 👉 PRIMARY BUTTONS 👈 */
    div[data-testid="stButton"] button[kind="primary"] {{
        background-color: {primary_btn} !important; 
        border-color: {primary_btn} !important; 
        color: white !important; 
        border-radius: 8px !important; 
        font-weight: 600;
    }}
    div[data-testid="stButton"] button[kind="primary"]:hover {{ 
        background-color: {primary_hover} !important; 
        border-color: {primary_hover} !important;
    }}

    /* 4. 👉 SECONDARY BUTTONS & POPOVERS 👈 */
    div[data-testid="stButton"] button[kind="secondary"],
    div[data-testid="stPopover"] button {{
        background-color: {card_bg} !important;
        color: {text_main} !important; 
        border: 1px solid {sec_btn_border} !important; 
        border-radius: 8px !important;
        box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05) !important;
    }}
    div[data-testid="stButton"] button[kind="secondary"]:hover,
    div[data-testid="stPopover"] button:hover {{
        border-color: {primary_btn} !important;
        color: {primary_btn} !important;
        background-color: {badge_bg} !important;
    }}

    /* 5. 👉 HIGH-CONTRAST ALERTS 👈 */
    [data-testid="stAlert"] {{
        background-color: {card_bg} !important; 
        border: 1px solid {sec_btn_border} !important;
        border-radius: 8px !important;
    }}
    [data-testid="stAlert"] > div {{ background-color: transparent !important; }}
    [data-testid="stAlert"] p, [data-testid="stAlert"] span {{ color: {text_main} !important; }}
    [data-testid="stAlert"] a {{ color: {primary_btn} !important; font-weight: 600 !important; text-decoration: none !important; }}
    [data-testid="stAlert"] a:hover {{ text-decoration: underline !important; }}

    /* 6. 👉 LANDING PAGE & DYNAMIC BADGE 👈 */
    .nav-links {{ display: flex; justify-content: center; gap: 30px; font-size: 14px; font-weight: 600; color: {nav_color}; margin-top: 10px; }}
    .badge {{ background-color: {badge_bg}; color: {badge_text}; padding: 6px 16px; border-radius: 20px; font-size: 13px; font-weight: 700; display: inline-block; margin-bottom: 1rem; border: 1px solid {badge_border}; }}
    .hero-title {{ font-size: 3.8rem; font-weight: 800; line-height: 1.1; margin-bottom: 1.5rem; color: {text_main}; }}
    .hero-title span {{ color: {primary_btn}; }} 
    .hero-subtitle {{ font-size: 1.2rem; color: {text_sub}; margin-bottom: 2rem; line-height: 1.6; }}
    
    /* 7. Stats Container */
    .stats-container {{ display: flex; align-items: center; gap: 30px; margin-top: 40px; padding-top: 30px; border-top: 1px solid {divider_color}; }}
    .stat-item {{ display: flex; flex-direction: column; }}
    .stat-value {{ font-size: 1.8rem; font-weight: 800; color: {text_main}; line-height: 1.1; }}
    .stat-label {{ font-size: 0.85rem; color: {text_sub}; font-weight: 500; margin-top: 4px; }}
    .stat-divider {{ height: 45px; width: 2px; background-color: {divider_color}; opacity: 0.5; }}
    
    /* 8. Chat UI Elements */
    .breadcrumb {{ color: {text_sub}; font-size: 12px; font-weight: 600; padding: 10px 0; border-bottom: 1px solid {divider_color}; margin-bottom: 20px; }}
    
    /* 9. Custom Disclaimer Box */
    .disclaimer-box {{ background-color: {card_bg}; border: 1px solid {sec_btn_border}; border-left: 3px solid #F59E0B; padding: 15px; border-radius: 8px; margin-top: 30px; text-align: left; color: {text_sub}; font-size: 11px; line-height: 1.5; }}
    .studio-btn-wrapper div[data-testid="stButton"] button {{ width: 100%; text-align: left; background-color: {bg_color}; border: 1px solid {sec_btn_border}; border-radius: 12px; padding: 15px; color: {text_main}; transition: all 0.2s ease; }}
    .studio-btn-wrapper div[data-testid="stButton"] button:hover {{ border-color: {primary_btn}; background: {card_hover}; transform: translateY(-2px); }}

    /* 10. Premium Image Styling */
    [data-testid="stImage"] img {{ border-radius: 24px !important; box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.05), 0 10px 10px -5px rgba(0, 0, 0, 0.02) !important; border: 1px solid {divider_color} !important; }}

    /* 11. 👉 FIX CHAT BUBBLES 👈 */
    [data-testid="stChatMessage"] {{ background-color: transparent !important; }}
    [data-testid="stChatMessageContent"] p, [data-testid="stChatMessageContent"] li, [data-testid="stChatMessageContent"] a, [data-testid="stChatMessageContent"] span {{ color: {text_main} !important; }}

    /* 12. 👉 CHAT INPUT BOX 👈 */
    /* Step 1: Force ALL inner wrappers and the text area to be fully transparent */
    [data-testid="stChatInput"] div, 
    [data-testid="stChatInput"] textarea, 
    [data-testid="stChatInput"] button {{
        background-color: transparent !important;
    }}
    
    /* Step 2: Apply our custom color ONLY to the top-level visual container */
    [data-testid="stChatInput"] > div {{
        background-color: {divider_color} !important;
        border: 1px solid {sec_btn_border} !important;
        border-radius: 12px !important;
    }}
    
    /* Step 3: Add the blue glow when typing and ensure text is visible */
    [data-testid="stChatInput"] > div:focus-within {{
        border-color: {primary_btn} !important;
    }}
    [data-testid="stChatInput"] textarea {{
        color: {text_main} !important;
    }}
</style>
""", unsafe_allow_html=True)

# --- POPUP TRIGGERS ---
if st.session_state.show_login_popup:
    login_popup()
elif st.session_state.show_register_popup:
    register_popup()

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
            if st.session_state.logged_in:
                st.session_state.app_started = True
            else:
                st.session_state.show_login_popup = True
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
                if st.session_state.logged_in:
                    st.session_state.app_started = True
                else:
                    st.session_state.show_login_popup = True
                st.rerun()
        with btn_col2: 
            st.button("Learn More", use_container_width=True)
            
        st.markdown(f"""
        <div class="stats-container">
            <div class="stat-item">
                <span class="stat-value">24/7</span>
                <span class="stat-label">Available</span>
            </div>
            <div class="stat-divider"></div>
            <div class="stat-item">
                <span class="stat-value">500+</span>
                <span class="stat-label">Protocols</span>
            </div>
            <div class="stat-divider"></div>
            <div class="stat-item">
                <span class="stat-value">98%</span>
                <span class="stat-label">Accuracy</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        try:
            st.image("nurse.png", use_container_width=True)
        except:
            st.info("Visual Placeholder: 'nurse.png' missing")


# --- VIEW 2: APP CHAT INTERFACE ---
else:
    if st.session_state.current_page == "quiz":
        render_quiz_page(text_main, text_sub, card_bg, divider_color)
        st.stop()

    current_messages = st.session_state.chat_sessions[st.session_state.current_chat]

    # 1. LEFT PANE: Sidebar History & Safety
    with st.sidebar:
        st.markdown(f"<h3 style='color:{text_main}; margin-top:-20px;'>🩺 NursBot</h3>", unsafe_allow_html=True)

        if st.session_state.logged_in:
            st.success(f"Logged in as {st.session_state.user_email}")

        if st.button("Logout", use_container_width=True):
            st.session_state.logged_in = False
            st.session_state.user_email = None
            st.session_state.user_full_name = None
            st.session_state.app_started = False
            st.rerun()
        
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

        # 👉 ZHEN RONG'S CODE: Sidebar Suggestions
        st.markdown(f"<p style='color:{text_sub}; font-size:12px; font-weight:700; letter-spacing: 1px;'>💡 EXAMPLE QUESTIONS</p>", unsafe_allow_html=True)
        st.caption("• What is the heart rate of an infant?")
        st.caption("• Show me an infant CPR video")
        st.caption("• Can I watch a blood pressure tutorial?")
        st.caption("• How to check heart rate video?")
        
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
        chat_col, studio_col = st.columns([2, 1], gap="large")
    else:
        chat_col = st.container()

    # 3. CENTER PANE: Chat
    with chat_col:
        c1, c2 = st.columns(2)
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
                    st.markdown(message["content"], unsafe_allow_html=True)
            
            if len(current_messages) == 0:
                st.markdown(f"<h2 style='color:{text_main}; text-align:center; margin-top:100px;'>How can I help you today?</h2>", unsafe_allow_html=True)

        # --- CHAT UI: MODE SELECTOR ---
        st.markdown("<p style='font-size: 13px; font-weight: 600; color: #475569; margin-bottom: 5px;'>🤖 Select AI Mode</p>", unsafe_allow_html=True)
        
        selected_mode = st.radio(
            "AI Mode",
            [
                "🧠 Azure (Clinical/Voice)", 
                "👁️ Gemini (Vision/Text)", 
                "🎥 Database (Video Search)",
                "📝 Quiz (Education Mode)"
            ],
            horizontal=True,
            label_visibility="collapsed"
        )

        uploaded_file = None
        spoken_text = None

        # Dynamically show tools based on the selected mode
        # Dynamically show tools based on the selected mode
        if "Gemini" in selected_mode:
            # Wrap the tools in a neat, modern bordered box
            with st.container(border=True):
                st.markdown("<p style='font-size: 14px; font-weight: 600; margin-bottom: 5px; color: #475569;'>👁️ Vision Analysis Panel</p>", unsafe_allow_html=True)
                
                # Split into two columns: 75% for the uploader, 25% for the preview
                upload_col, preview_col = st.columns([2, 1], gap="medium")
                
                with upload_col:
                    # Hide the default label to keep it clean
                    uploaded_file = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"], label_visibility="collapsed")
                    
                with preview_col:
                    if uploaded_file is not None:
                        # Center the preview text and fit the image perfectly inside its column
                        st.markdown("<p style='font-size: 12px; color: #475569; margin-bottom: 5px; text-align: center;'>Attached</p>", unsafe_allow_html=True)
                        st.image(uploaded_file, use_container_width=True)
                
        elif "Azure" in selected_mode:
            spoken_text = speech_to_text(language="en", use_container_width=True, just_once=True, key="STT")
        elif "Quiz" in selected_mode:
            st.info("💡 **Education Mode:** Generate multiple-choice clinical quizzes from your KKH protocols.")
            if st.button("Launch Quiz Generator ➔", type="primary", use_container_width=True):
                st.session_state.current_page = "quiz"
                st.rerun()

        # Get Text Input
        user_input = st.chat_input("Type your message here...")

        # Handle Studio Prompts, Voice, or Text
        if st.session_state.studio_prompt_trigger:
            actual_input = st.session_state.studio_prompt_trigger
            st.session_state.studio_prompt_trigger = None 
        elif spoken_text:
            actual_input = spoken_text
        else:
            actual_input = user_input

        # Append to history and rerun
        if actual_input:
            st.session_state.chat_sessions[st.session_state.current_chat].append({"role": "user", "content": actual_input})
            st.rerun()

    # Run AI Logic based on explicitly selected mode
    if len(current_messages) > 0 and current_messages[-1]["role"] == "user":
        latest_user_input = current_messages[-1]["content"]
        with chat_container:
            with st.chat_message("assistant"):
                try:
                    def stream_text(text):
                        for word in text.split(" "):
                            yield word + " "
                            time.sleep(0.02)

                    # ==========================================
                    # 👉 UNIFIED CHAT & AI ROUTING
                    # ==========================================

                    # 👉 TOKEN OPTIMIZATION: Only keep the last 4 messages in memory
                    if len(current_messages) > 4:
                        history_to_keep = current_messages[-5:-1]
                    else:
                        history_to_keep = current_messages[:-1]
                        
                    chat_history_lc = get_langchain_history(history_to_keep)

                    # 1. STRICT MODEL ROUTING 
                    if "Azure" in selected_mode:
                        if llm_azure is None:
                            st.error("Azure OpenAI is not configured in secrets. Please check Joeson's API keys.")
                            st.stop()
                        active_llm = llm_azure
                        loading_text = "Azure OpenAI (Joeson's Model)"
                        
                    elif "Gemini" in selected_mode:
                        active_llm = llm_gemini_bryan
                        loading_text = "Gemini 2.0 Flash (Bryan's Model)"
                        
                    elif "Database" in selected_mode:
                        active_llm = llm_gemini_zhenrong
                        loading_text = "Gemini 2.5 Flash & Video DB (Zhen Rong's Model)"
                        
                    elif "Quiz" in selected_mode:
                        try:
                            active_llm = create_quiz_llm()
                            loading_text = "Azure OpenAI (Chee You's Education Model)"
                        except Exception:
                            st.error("Azure OpenAI is not configured in secrets. Please check Chee You's API keys.")
                            st.stop()

                    # 2. IMAGE PRE-EXTRACTION (The Bulletproof Fix)
                    agent_input = latest_user_input

                    if uploaded_file is not None and "Gemini" in selected_mode:
                        img_bytes = uploaded_file.getvalue()
                        encoded_img = base64.b64encode(img_bytes).decode("utf-8")
                        mime_type = uploaded_file.type 
                        image_data = f"data:{mime_type};base64,{encoded_img}"
                        
                        with st.spinner("Extracting handwritten clinical data..."):
                            # Step A: Call the model directly to transcribe the image
                            vision_msg = HumanMessage(content=[
                                {"type": "text", "text": "Extract all text and numbers from this handwritten note perfectly. Do not solve, calculate, or explain anything. Just output the text."},
                                {"type": "image_url", "image_url": {"url": image_data}}
                            ])
                            transcription = active_llm.invoke([vision_msg]).content
                            
                        # Step B: Inject the pure text directly into the agent's prompt
                        agent_input = f"User Request: {latest_user_input}\n\n[Handwritten Note Contents]:\n{transcription}"

                    # 3. EXECUTE THE SELECTED AI
                    with st.spinner(f"Analyzing using {loading_text}..."):
                        agent = create_tool_calling_agent(active_llm, tools, prompt)
                        agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)

                        response = agent_executor.invoke({
                            "input": agent_input, 
                            "chat_history": chat_history_lc
                        })
                    
                    # --- CLEAN OUTPUT EXTRACTION ---
                    raw_output = response.get("output", "")
                    
                    # Handle Gemini's complex list format if it returns one
                    if isinstance(raw_output, list):
                        text_parts = []
                        for item in raw_output:
                            if isinstance(item, dict) and "text" in item:
                                text_parts.append(item["text"])
                            elif isinstance(item, str):
                                text_parts.append(item)
                        full_response = "".join(text_parts)
                    else:
                        # Standard fallback for normal text
                        full_response = str(raw_output)
                    
                    # Clean up weird regex remnants just in case
                    full_response = full_response.replace('\\n', '\n').replace('\\t', '\t').replace("\\'", "'")
                    
                    # 4. APPEND ZHEN RONG'S VIDEOS (Only if Database mode is active)
                    if "Database" in selected_mode and is_video_request(latest_user_input):
                        videos = search_video_tutorial(latest_user_input)

                        if videos:
                            full_response += "\n\n---\n\n📹 **Related Video Tutorials:**\n\n"

                            for video in videos:
                                embed_url = video["youtube_url"].replace("watch?v=", "embed/")

                                full_response += f"**Title:** {video['title']}  \n"
                                full_response += f"**Topic:** {video['topic']}  \n"
                                full_response += f"**Description:** {video['description']}  \n\n"
                                full_response += (
                                    f'<iframe width="100%" height="315" '
                                    f'src="{embed_url}" '
                                    f'frameborder="0" '
                                    f'allowfullscreen '
                                    f'style="border-radius: 12px; margin-bottom: 20px;">'
                                    f'</iframe>\n\n'
                                )

                        # 5. RENDER OUTPUT
                    if "Database" in selected_mode:
                            st.markdown(full_response, unsafe_allow_html=True)
                    else:
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