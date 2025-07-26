from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv
import os

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="google/gemma-2-2b-it",
    task="text-generation",
    huggingfacehub_api_token=os.getenv("HUGGUNGFACEHUB_ACCESS_TOKEN")
)

model = ChatHuggingFace(llm=llm)

parser = JsonOutputParser()

# Define the prompt
template = PromptTemplate(
    template="Give me the name, age, and city of a fictional person in the following format:\n{format_instruction}",
    input_variables=[],
    partial_variables={"format_instruction": parser.get_format_instructions()}
)

# Generate the prompt string
prompt_str = template.format()

response = model.invoke(prompt_str)

print("Raw model output:\n", response.content)

parsed_output = parser.invoke(response.content)

print("\nParsed JSON output:\n", parsed_output)
