import streamlit as st
import pandas as pd
from langchain_google_genai import GoogleGenerativeAI
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain.agents.agent_types import AgentType
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Streamlit UI
st.title('CSV Upload and Process with Langchain')

# Upload CSV file
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    # Read the uploaded CSV file
    df = pd.read_csv(uploaded_file)

    # Display the dataframe
    #st.write("DataFrame preview:")
    #st.write(df.head())

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
        
        # User input for query
        user_query = st.text_input("Enter your query:", "show me the columns")

        if st.button("Submit"):
            # Execute the agent with the user query
            response = agent(user_query)

            # Display the result
            st.write("Agent response:")
            st.write(response)
