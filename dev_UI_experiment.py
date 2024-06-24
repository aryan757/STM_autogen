import streamlit as st
import pandas as pd
from langchain_google_genai import GoogleGenerativeAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Streamlit UI
st.title('CSV Chatbot with Langchain')

# Upload CSV file
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Read the uploaded CSV file
    df = pd.read_csv(uploaded_file)

    # Get the API key from environment variables
    api_key = os.getenv('google_api_key')

    if api_key is None:
        st.error("API key not found. Please check your .env file.")
    else:
        # Initialize the LLM
        llm = GoogleGenerativeAI(model="models/gemini-1.5-pro-latest", google_api_key=api_key)
        
        # Create the Pandas DataFrame agent
        agent = create_pandas_dataframe_agent(
            llm,
            df,
            verbose=True,
            allow_dangerous_code=True  # Acknowledge the risks and enable the functionality
        )
        
        # Initialize session state for conversation history
        if 'conversation' not in st.session_state:
            st.session_state.conversation = []

        # Display the conversation history
        for chat in st.session_state.conversation:
            if chat["role"] == "user":
                st.markdown(f"**User:** {chat['text']}")
            else:
                st.markdown(f"**Agent:** {chat['text']}")

        # Form for user input
        with st.form(key='query_form', clear_on_submit=True):
            user_query = st.text_input("Enter your query:")
            submit_button = st.form_submit_button(label='Submit')

        if submit_button and user_query:
            # Add user query to conversation history
            st.session_state.conversation.append({"role": "user", "text": user_query})

            # Display loading animation
            with st.spinner('Processing...'):
                # Execute the agent with the user query
                response = agent(user_query)
            
            # Add agent response to conversation history
            st.session_state.conversation.append({"role": "agent", "text": response})

            # Rerun the script to update the conversation history
            st.experimental_rerun()
