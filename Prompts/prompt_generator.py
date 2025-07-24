from langchain_core.prompts import PromptTemplate

prompt_template =PromptTemplate( 
 template = """Please summarize the research paper titled "{paper_input}" with the following specifications:
Explanantion Style: {stlye_input}
Explanation Length: {length_input}
1.Mathematical Details:
   - Include mathematical details if applicable.
   - Explain complex concepts with easy interpretations where possible.
2. Analogies:
    - Use analogies to simplify complex concepts.
If Certain information not available in the paper, please mention that with "N/A" in the response.
Ensure the summary is clear, concise, and informative. 
""",
input_variables =['paper_input', 'stlye_input', 'length_input'],
validate_template=True
)

prompt_template.save('template.json')

## Run this file it will generate a template.json file with the prompt structure.