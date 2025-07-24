from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

chat_template = ChatPromptTemplate([
        ('system', 'You are a helpful {domain} expert.'),
        ('human', 'Explain in simple terms {topic} '),
])

prompt = chat_template.invoke({'domain': 'Quantum Computing', 'topic':' Quantum Entanglement'})

print(prompt)