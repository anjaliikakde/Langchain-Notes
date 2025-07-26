from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser   # Often used with chains

from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI()

template1 = PromptTemplate(
    template ="Write a detailed report on {topic}",
    input_variables= ['topic']
)

template2 = PromptTemplate(
    template = "Write 3 line summary on {text}",
    input_variables= ['text']
)

parser = StrOutputParser()

chain  = template1 | model | parser | template2 | model | parser 

result = chain.invoke({'topic': 'Black holes'})

print(result)