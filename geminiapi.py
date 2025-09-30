import os
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI

# ======================
# API Keys Configuration
# ======================
# Set your Gemini API key (better to use environment variable in production)
os.environ["GOOGLE_API_KEY"] = ""  # Add your Google API key here

# LangSmith tracking (optional)
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = ""

# ======================
# Prompt Template
# ======================
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that helps people find information."),
    ("user", "Question: {question}")
])

# ======================
# Streamlit UI
# ======================
st.title("LLM - Gemini Project")
input_text = st.text_input("How may I help you?")

# Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",  # or "gemini-1.5-pro"
    temperature=0.2,
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# Output Parser
output_parser = StrOutputParser()

# Build Chain
chain = prompt | llm | output_parser

# ======================
# Run the App
# ======================
if input_text:
    response = chain.invoke({"question": input_text})
    st.write(response)
