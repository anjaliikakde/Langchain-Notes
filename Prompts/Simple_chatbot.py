from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI()

## To have previous conversation context.

chat_history = [
    SystemMessage(content= " You are a helpful AI assistant.")
    ]

while True:
    user_input = input("You: ")
    chat_history.append(HumanMessage(content= user_input))
    
    if user_input.lower() in ["exit", "quit"]:
        print("Exiting the chatbot. Goodbye!")
        break
    
    response = model.invoke(chat_history)
    chat_history.append(AIMessage(content= response.content))
    print(f"AI : {response.content}")
    
print(chat_history)