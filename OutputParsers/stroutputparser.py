""" We are using TinyLlama model from huggingface --> This model is not able to give by default 
structured output.  """

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id = "TinyLlama/TinyLlama-1.1B-chat-v1.0",
    task="text-generation",
)

model = ChatHuggingFace(llm=llm)

# 1st prompt --> detailed report

template1 = PromptTemplate(
    template='Write a detailed report on the {topic}',
    input_variables =['topic']
)

# 2nd prompt --> summary

template2 = PromptTemplate(
    template='Write a 5 line summary on {text}',
    input_variables=['text']
)

prompt1 = template1.invoke({'topic':'Qbits'})

result1 = model.invoke(prompt1)

prompt2 = template2.invoke({'text': result1})

result2 = model.invoke(prompt2)

print(result2.content)
