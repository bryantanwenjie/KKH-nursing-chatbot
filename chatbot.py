import os
from dotenv import load_dotenv

# 1. Base LangChain and Google GenAI
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

# 2. Agent and Tooling (Modern 2026 Paths)
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate

# 1. SETUP & AUTHENTICATION
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# 2. DATA INGESTION (RAG)
# We move this to a tool so the Agent can choose to "search" the PDF
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

def get_retriever():
    loader = PyPDFLoader("Section 01 - Medical Emergencies.pdf")
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(docs)
    vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)
    return vectorstore.as_retriever()

retriever = get_retriever()

@tool
def search_nursing_protocols(query: str) -> str:
    """
    Search the KKH Medical Emergencies PDF for clinical guidelines and protocols.
    Use this when the nurse asks about specific emergency procedures or guidelines.
    """
    docs = retriever.invoke(query)
    return "\n\n".join([doc.page_content for doc in docs])

# 3. CUSTOM NURSING TOOLS
@tool
def calculate_fluid_requirement(weight_kg: float) -> str:
    """
    Calculates daily fluid requirements for a patient based on weight.
    Use this for weight-based clinical calculations.
    """
    # Holliday-Segar Formula
    if weight_kg <= 10:
        res = weight_kg * 100
    elif weight_kg <= 20:
        res = 1000 + (weight_kg - 10) * 50
    else:
        res = 1500 + (weight_kg - 20) * 20

    return f"The calculated fluid requirement is {res} mL/day."

# 4. INITIALIZE AGENT & GEMINI
# Use 'gemini-3-flash' or the model ID from your audit for 2026 stability
llm = ChatGoogleGenerativeAI(model="gemini-3-flash", temperature=0)
tools = [calculate_fluid_requirement, search_nursing_protocols]

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a KKH Clinical Nursing Assistant. You have access to medical emergency protocols and fluid calculators. Always be precise."),
    ("placeholder", "{chat_history}"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 5. REMOVED TERMINAL LOOP 
# (Streamlit handles the execution in app.py now)