from langchain_openai import ChatOpenAI
import streamlit as st
from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate

load_dotenv()

model = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0.2,
)

st.header('Research Tool')

paper_input = st.selectbox("Select Reasearch Paper Name ", ["Select ....", "Attention is All You Need",
"BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding", 
"GPT-3: Language Models are Few-Shot Learners",
"T5: Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer",
"Defusion Models Beat GANs on Image Synthesis"])

stlye_input = st.selectbox("Select Explanation Style", ["Beginner - Friendly",
"Intermediate - Technical", "Advanced - In-Depth", "mathematical"])

length_input = st.selectbox("Select Explanation Length", ["Short (1- 2 Paragraphs)",
                            "Medium (3 -5 Paragraphs)", "Long(Detailed explanation)"])

## Template for dynamic prompt
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

prompt = prompt_template.invoke({
    'paper_input' : paper_input,
    'stlye_input': stlye_input,
    'length_input': length_input
})

if st.button('Summarize'):
    result = model.invoke(prompt)
    st.write(result.content)

