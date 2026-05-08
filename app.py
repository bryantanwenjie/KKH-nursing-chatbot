import streamlit as st
from chatbot import agent_executor, calculate_fluid_requirement  # Import your existing logic
from langchain_community.chat_message_histories import StreamlitChatMessageHistory

# 1. Page Configuration
st.set_page_config(page_title="KKH Nursing Chatbot", page_icon="🩺")
st.title("🩺 KKH Nursing Assistant")
st.markdown("Retrieval of Protocols & Clinical Calculations")

# 2. Initialize Chat History
# This stores messages in Streamlit's session_state automatically
msgs = StreamlitChatMessageHistory(key="chat_messages")

if len(msgs.messages) == 0:
    msgs.add_ai_message("Hello! I am the KKH Nursing Assistant. How can I help you today?")

# 3. Display Chat History
for msg in msgs.messages:
    st.chat_message(msg.type).write(msg.content)

# 4. Chat Input
if prompt := st.chat_input("Type your nursing query here..."):
    # Display user message
    st.chat_message("human").write(prompt)
    
    # Generate response using your existing agent_executor
    with st.chat_message("ai"):
        # The 'st_callback' shows the agent's "thinking" process (very cool for demos!)
        # from langchain_community.callbacks import StreamlitCallbackHandler
        # st_callback = StreamlitCallbackHandler(st.container())
        
        try:
            # Add user message first
            msgs.add_user_message(prompt)
            
            # Run your agent
            response = agent_executor.invoke({"input": prompt})
            output = response["output"]
            
            # Write and save the response
            st.write(output)
            msgs.add_ai_message(output)
            
        except Exception as e:
            st.error(f"Error: {e}")