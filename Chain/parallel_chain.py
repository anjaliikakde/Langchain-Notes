"""
We process a large document using two LLMs in parallel — one from OpenAI and 
one from Google — to generate notes and quizzes.  
A third model is then used to combine their outputs into a final result.

"""

from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema.runnable import RunnableParallel

load_dotenv()

model1 = ChatOpenAI()

# llm = HuggingFaceEndpoint(
#     repo_id = "google/gema-2-2b-it",
#     task = "text-generation"
# )

# model2 = ChatHuggingFace(llm=llm)
 
prompt1 = PromptTemplate(
    template = 'Generate short notes on the {text}',
    input_variables= ['text']
)

prompt2 = PromptTemplate(
    template= 'generate 5 short quiz questions on the {text} with options .',
    input_variables=['text'] 
)

prompt3 = PromptTemplate(
    template= 'Merge the provided notes and quiz in a single documents. \n notes-->{notes} and quize--> {quiz}',
    input_variables= ['notes','quiz']
)

parser = StrOutputParser()

parallel_chain = RunnableParallel({
    'notes' : prompt1 | model1 | parser,
    'quiz' : prompt2 | model1 | parser
})

merge_chain = prompt3 | model1 |parser

chain = parallel_chain | merge_chain

text = """
Blockchain is a distributed, decentralized ledger technology that enables secure, transparent, and tamper-proof recording of data across a network of computers. Initially introduced as the foundation for Bitcoin in 2008 by an anonymous entity known as Satoshi Nakamoto, blockchain technology has since evolved beyond cryptocurrencies to revolutionize industries ranging from finance and healthcare to supply chain management and real estate.

At its core, a blockchain consists of a series of blocks, each containing a list of transactions. These blocks are chronologically linked using cryptographic hashes, forming a continuous chain. Once data is recorded in a block and added to the chain, it becomes nearly impossible to alter without changing all subsequent blocks and gaining consensus from the network. This immutability and transparency are key features that make blockchain trustworthy and secure.

One of the most important aspects of blockchain is decentralization. Unlike traditional centralized systems, where a single authority controls the database, blockchain operates on a peer-to-peer network. Every participant (or node) on the network has access to the full copy of the ledger and participates in validating transactions. This distributed nature removes the need for intermediaries, reduces the risk of single points of failure, and enhances system resilience.

To validate transactions, most blockchain networks use consensus mechanisms. One widely known method is Proof of Work (PoW), which requires nodes (called miners) to solve complex mathematical problems to add new blocks. Another method is Proof of Stake (PoS), where validators are chosen based on the number of tokens they hold and are willing to "stake" as collateral. These mechanisms ensure that only valid transactions are added to the ledger and protect the network against fraud and malicious actors.

Smart contracts are another powerful feature of blockchain. These are self-executing contracts with the terms directly written into code. They automatically execute actions when predetermined conditions are met, reducing the need for intermediaries and enhancing efficiency. Platforms like Ethereum have popularized smart contracts, enabling developers to build decentralized applications (dApps) on the blockchain.

Blockchain’s applications are far-reaching. In finance, it enables faster and cheaper cross-border payments, decentralized finance (DeFi), and tokenized assets. In supply chain management, it provides real-time tracking and verifiable authenticity of goods. In healthcare, blockchain can secure patient records and ensure data privacy. Governments are exploring its use for digital identity, voting systems, and transparent record-keeping.

Despite its potential, blockchain faces challenges. Scalability is a major concern, as networks can become slow and expensive as transaction volumes grow. Regulatory uncertainty also poses obstacles to widespread adoption, particularly in industries that require compliance with complex laws. Additionally, energy consumption in PoW-based blockchains has raised environmental concerns, pushing the development of more sustainable alternatives.

In conclusion, blockchain is a transformative technology offering enhanced security, transparency, and decentralization. As it continues to mature, it is expected to become an integral part of digital infrastructure across multiple sectors, fundamentally changing how data is stored, shared, and verified.

"""

result = chain.invoke({'text': text})

print(result)

chain.get_graph().print_ascii()