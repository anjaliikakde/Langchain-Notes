from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain.output_parsers import StructuredOutputParser,ResponseSchema
import os 

load_dotenv()

# Define model

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGUNGFACEHUB_ACCESS_TOKEN")
)

model = ChatHuggingFace(llm=llm)

schema = [
    ResponseSchema(name='fact1', description="Fact 1 obout the topic"),
    ResponseSchema(name='fact2', description="Fact 2 obout the topic"),
    ResponseSchema(name='fact2', description="Fact 3 obout the topic"),
]

parser = StructuredOutputParser.from_response_schemas(schema)

template = PromptTemplate(
    template = "give 3 facts about the {topic} \n {format_instruction}",
    input_variables=['topic'],
    partial_variables={'format_instruction': parser.get_format_instructions()}
)

prompt = template.invoke({'topic': 'black hole'})

result = model.invoke(prompt)

final_result = parser.parse(result.content)

print(final_result)

# We can't perform data validation using structured output