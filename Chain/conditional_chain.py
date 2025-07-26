from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain.schema.runnable import RunnableBranch, RunnableParallel, RunnableLambda
from pydantic import BaseModel, Field
from typing import Literal
from dotenv import load_dotenv
load_dotenv()

model = ChatOpenAI()

parser = StrOutputParser()

class Feedback(BaseModel):
    sentiment : Literal['positive', 'negative'] = Field(description='Give the sentiment of the feedback')
    
parser2 = PydanticOutputParser(pydantic_object= Feedback)


prompt1 = PromptTemplate(
    template='Classify the sentiment of the following feedback text into positive or negative \n {feedback} \n {format_instruction}',
    input_variables=['feedback'],
    partial_variables={'format_instruction':parser2.get_format_instructions()}
)

classifier_chain = prompt1 | model | parser2

result = classifier_chain.invoke({'feedback' : 'This is a wonderful smartphone'}).sentiment

# print(result)

# print(classifier_chain.invoke({'feedback' : 'This is a wonderful smartphone'}))

prompt2 = PromptTemplate(
    template='Wrte an appropriate response to this positive feedback \n {feedback}',
    input_variables=['feedback']
)

prompt3 = PromptTemplate(
    template='Wrte an appropriate response to this negative feedback \n {feedback}',
    input_variables=['feedback']
)

branch_chain = RunnableBranch(
    (lambda x:x.sentiment =='positive', prompt2 |model | parser ),
    (lambda x:x.sentiment =='negative', prompt3 |model | parser ),
    RunnableLambda(lambda x : "Could not find sentiment")
    
)

chain = classifier_chain | branch_chain

result = chain.invoke({'feedback' : 'This is a wonderful smartphone'})

print(result)