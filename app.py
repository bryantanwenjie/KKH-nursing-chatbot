import os
import re
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import streamlit as st

# --- THE MEMORY NUKE ---
if "nuke_complete" not in st.session_state:
    st.session_state.clear()
    st.session_state.nuke_complete = True
# -----------------------

# The Final Fix: Importing from the classic package
from langchain_classic.agents import AgentExecutor, create_tool_calling_agent
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool

st.success("✅ LangChain loaded successfully! The app is running.")

# ... the rest of your app.py code ...

# --- PAGE CONFIG ---
st.set_page_config(page_title="KKH Nursing Assistant", page_icon="🏥")
st.title("🏥 KKH Clinical Nursing Assistant")

# --- AUTHENTICATION ---
if "GOOGLE_API_KEY" in st.secrets:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

# --- TOOLS & RAG ---
# --- TOOLS & RAG ---
@st.cache_resource
def initialize_retriever():
    try:
        loader = PyPDFLoader("Section 01 - Medical Emergencies.pdf")
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        chunks = text_splitter.split_documents(docs)
        
        # 1. Safely grab the API key directly from secrets
        api_key = st.secrets.get("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY is completely missing from Streamlit Secrets!")

        # 2. Explicitly pass the key into the embedding function
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/gemini-embedding-001", # <--- THE FIX IS HERE
            google_api_key=api_key
        )
        
        vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)
        return vectorstore.as_retriever()
    
    except Exception as e:
        # 3. This forces Streamlit to show us the UNREDACTED error
        st.error(f"🚨 Google API Error: {str(e)}")
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

# --- AGENT SETUP ---
# 1. Update the model name to include the required suffix
llm = ChatGoogleGenerativeAI(model="gemini-3-flash-preview", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a strictly professional KKH Clinical Nursing Assistant. 
    
    Your ONLY purpose is to answer questions related to clinical protocols, nursing guidelines, and medical topics based on the provided KKH documents. 
    
    CRITICAL RULES:
    1. If a user asks a question unrelated to healthcare, nursing, or KKH (e.g., recipes, general technology, movies, casual chat), you MUST politely refuse to answer. 
    2. Do NOT use your general world knowledge to answer off-topic questions. 
    3. If refusing, gently remind the user that you are a clinical assistant and ask how you can help with medical protocols today."""),
    ("placeholder", "{chat_history}"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


# --- CHAT INTERFACE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# ADD THIS: A button to clear history if it gets messy
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
            response = agent_executor.invoke({
                "input": user_input,
                "chat_history": st.session_state.messages[:-1] 
            })
            
            raw_output = str(response.get("output", ""))
            
            # --- THE REGEX LASER ---
            # This hunts exactly for the text between 'text': ' and ', 'index':
            match = re.search(r"'text':\s*['\"](.*?)['\"],\s*'index':", raw_output, re.DOTALL)
            
            if match:
                # Extract it and fix any broken newlines or quotes
                full_response = match.group(1).replace('\\n', '\n').replace('\\t', '\t').replace("\\'", "'")
            else:
                full_response = raw_output
            # -----------------------

            st.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            
        except Exception as e:
            st.error(f"🚨 Error: {str(e)}")