import httpx
import pandas as pd
import os
import sqlite3

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain_community.agent_toolkits.sql.prompt import SQL_FUNCTIONS_SUFFIX
from langchain_community.utilities import SQLDatabase

from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent

from langchain_core.messages import AIMessage
from langchain_core.prompts import (
    ChatPromptTemplate,
    ChatMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)

from langchain.agents import create_sql_agent, AgentExecutor, create_openai_tools_agent
from langchain.agents.agent_types import AgentType
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts import PromptTemplate

from crewai import Agent, Task, Crew
from crewai.tools import BaseTool

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY")

if API_KEY:
    print("✅ Loaded OpenAI API Key")
else:
    print("❌ Failed to load OpenAI API Key")
    raise Exception("OpenAI API Key not set")


LLM_MODEL_NAME = "gpt-4o-mini"
DATA_PATH = "Hospital_General_Information.csv"


df = pd.read_csv(DATA_PATH)

df.head()


llm = ChatOpenAI(
    open_api_key=API_KEY,
    temperature=0,  # Set temp to 0 for deterministic response (no randomness)
    model=LLM_MODEL_NAME,
    max_tokens=500,  # Limit the response to XX tokens to control length
    http_client=httpx.Client(verify=False),  # Use HTTP with SSL disabled (dev)
)


# Give guidance to the model
system_message = SystemMessagePromptTemplate.from_template(
    """
You are a highly skilled healthcare assistant with expertise in comparing hospitals.
Your task is to assess various hospitals based on user's specific conditions, preferences, and needs.
You will evaluate hospitals considering factors such as medial specialties, patient reviews, location, cost, facilities, and the availability of treatment for specific conditions

When comparing hospitals, follow these guidelines:
- Condition-Specific Comparison: Focus on the hospital's expertise in treating the user's specific health condition (e.g., heart disease, cancer, etc.).
- Hospital FeaturesL Include details about the hospital's reputation, technology, facilities, specialized care, and any awards or recognitions.
- Location and Accessibility: Consider the proximity to the user's location and the convenience of travel.
- Cost and Insurance: Consider the proximity of treatment and insurance coverage options offered by the hospitals.
- Patient Feedback: Analyze reviews and ratings to gauge patient satisfaction and outcomes.
- Personalized Recommendation: Provide a clear, personalized suggestion based on the user's priorities, whether they are medical expertise, convenience, or cost.

Use "Hospital Type" column to look for good facilities of each hospital.
CAREFULLY look at each column name to understand what to output
"""
)

prompt = ChatPromptTemplate.from_messages([system_message])


# AI Toolkit
hospital_info_agent = create_pandas_dataframe_agent(
    llm=llm,  # the ChatOpenAI model
    df=df,  # the hospital dataset
    prompt=prompt,  # custom prompt template for hospital compare
    verbose=False,  # Disable detailed execution logs
    allow_dangerous_code=True,  # Enable execution of LLM-generated Python code (Use with Caution)
    agent_type=AgentType.OPENAI_FUNCTIONS,  # Use OpenAI's function-based agent type
)


print(hospital_info_agent.invoke("Which hospital has good medical imaging")["output"])
