import os
from dotenv import load_dotenv

# Core LangChain and Google GenAI imports
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

# Fixed imports for Agents and Tools
from langchain.agents.factory import create_agent
from langchain_core.tools import tool  # 'tool' is now here

# 1. SETUP & AUTHENTICATION
# Ensure your .env file contains: GOOGLE_API_KEY=your_key_here
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError(
        "Gemini API key not found. Set GOOGLE_API_KEY or GEMINI_API_KEY in your .env file."
    )

# 2. DATA INGESTION (RAG) - For Clinical Guidelines & Protocols
# This facilitates the retrieval of accurate healthcare information [cite: 61, 93]
def setup_knowledge_base():
    # Ensure this PDF exists in your project directory [cite: 53, 86]
    loader = PyPDFLoader("Section 01 - Medical Emergencies.pdf") 
    docs = loader.load()
    
    # Split text into manageable chunks for accurate retrieval
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_documents(docs)
    
    # Generate embeddings and store in a local vector database
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    return vectorstore.as_retriever()

# 3. CUSTOM NURSING TOOLS
@tool
def calculate_fluid_requirement(weight_kg: float) -> str:
    """
    Calculates daily fluid requirements for a patient based on weight.
    Use this when a nurse asks for fluid calculations or clinical guidance[cite: 66, 96].
    """
    # Standard formula for fluid requirement calculation 
    if weight_kg <= 10:
        res = weight_kg * 100
    elif weight_kg <= 20:
        res = 1000 + (weight_kg - 10) * 50
    else:
        res = 1500 + (weight_kg - 20) * 20
        
    return f"The calculated fluid requirement is {res} mL/day."

# 4. INITIALIZE AGENT & GEMINI
# Using Gemini 2.5 Flash for fast responses [cite: 58, 91]
llm = ChatGoogleGenerativeAI(api_key=api_key, model="gemini-2.5-flash", temperature=0)
tools = [calculate_fluid_requirement]

# Construct the agent
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a clinical nursing assistant. Answer nurse questions and use tools when needed."),
    ("placeholder", "{chat_history}"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# 5. EXECUTION LOOP
def main():
    print("--- KKH Nursing Chatbot Active ---")
    # This loop allows nurses to retrieve protocols and guidelines interactively [cite: 65, 95]
    
    while True:
        user_input = input("Nurse Query (type 'exit' to quit): ")
        if user_input.lower() in ["exit", "quit"]:
            break
            
        # Invoke the agent to determine if it should search protocols or use the tool
        try:
            response = agent.invoke({"messages": [{"role": "user", "content": user_input}]})
            
            # Extract the last message from the response
            if isinstance(response, dict) and "messages" in response:
                messages = response["messages"]
                if messages:
                    last_msg = messages[-1]
                    # Handle AIMessage objects with 'content' attribute
                    if hasattr(last_msg, 'content'):
                        content = last_msg.content
                        # If content is a list of dicts (block content), extract text
                        if isinstance(content, list):
                            output = " ".join([item.get('text', str(item)) for item in content if isinstance(item, dict)])
                        else:
                            output = str(content)
                    elif isinstance(last_msg, dict):
                        output = last_msg.get("content") or str(last_msg)
                    else:
                        output = str(last_msg)
                else:
                    output = "No response generated."
            else:
                output = str(response)
            print(f"\nChatbot: {output}\n")
        except Exception as e:
            print(f"\nError processing query: {str(e)}\n")

if __name__ == "__main__":
    main()