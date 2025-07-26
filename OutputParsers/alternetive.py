from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate

load_dotenv()

model = ChatOpenAI()

# 1st prompt --> detailed report

template1 = PromptTemplate(
    template='Write a detailed report on the {topic}',
    input_variables =['topic']
)

# 2nd prompt --> summary

template2 = PromptTemplate(
    template='Write a 3 line summary on {text}',
    input_variables=['text']
)

prompt1 = template1.invoke({'topic':'Qbits'})

result1 = model.invoke(prompt1)

prompt2 = template2.invoke({'text': result1})

result2 = model.invoke(prompt2)

print(result1.content)
print("\n Summary generated : \n")
print(result2.content)
