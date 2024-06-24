from langchain_google_genai import GoogleGenerativeAI
import pandas as pd
from langchain.agents.agent_types import AgentType
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_experimental.agents import create_pandas_dataframe_agent
import pandas as pd
from langchain.agents import AgentType 
import os
from dotenv import load_dotenv


# Load environment variables
load_dotenv()


api_key = os.getenv('google_api_key')

df = pd.read_csv(".csv") #uploaded csv file .
llm = GoogleGenerativeAI(model="models/gemini-1.5-pro-latest", google_api_key=api_key)
agent = create_pandas_dataframe_agent(
    llm,
    df,
    verbose=True,
    allow_dangerous_code=True # Add this line to acknowledge the risks and enable the functionality
)

agent("shoe me the columns")
