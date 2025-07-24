from langchain_openai import ChatOpenAI
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0.2,
    max_tokens=100,
)

st.header('Research Tool')

user_input = st.text_input(
    "Enter your prompt"
)

if st.button('Summarize'):
    result = model.invoke(user_input)
    st.write(result.content)

