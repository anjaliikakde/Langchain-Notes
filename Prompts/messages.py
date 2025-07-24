from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI()

messages =[
    SystemMessage(content='you are a helpful assistant'),
    HumanMessage(content='Tell me about langcahin ?'),
]

result = model.invoke(messages)

messages.append(AIMessage(content=result.content))

print(messages)