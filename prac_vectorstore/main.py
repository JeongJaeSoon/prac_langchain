from dotenv import load_dotenv
from langchain import OpenAI
from langchain.agents import AgentType, Tool, initialize_agent, load_tools
from langchain.chains import RetrievalQA
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.indexes import VectorstoreIndexCreator
from langchain.memory import (ConversationBufferMemory,
                              ConversationBufferWindowMemory)
from langchain.vectorstores import Chroma

load_dotenv()

llm = OpenAI(temperature=0)
loader = CSVLoader(file_path='sample.csv')
loaders = [loader]

embeddings = OpenAIEmbeddings()

index_creator = VectorstoreIndexCreator(
    vectorstore_cls=Chroma,
    embedding=OpenAIEmbeddings(),
    vectorstore_kwargs= {"collection_name": "collection"}
)
index_wrapper = index_creator.from_loaders(loaders)
# # COMPONENT TEST CODE
# temp = index_wrapper.query_with_sources("When is this data useful?")
# print(temp)


fnq_history_chain = RetrievalQA.from_chain_type(
  llm=llm,
  chain_type="stuff",
  retriever=index_wrapper.vectorstore.as_retriever(),
  verbose=True
)
# # COMPONENT TEST CODE
# temp = fnq_history_chain.run("When is this data useful?")
# print(temp)


tools = []
# tools = load_tools(["wikipedia", "llm-math"], llm=llm)
name = "Customer Support Knowledge Base"
description = """
useful for both customer service inquiries and providing information about business tools.
It includes details about customer support questions and answers, along with the status of each entry, relevant tools, manuals, and other business-related aspects.
"""
# description = """
# The Customer Support Knowledge Base is a comprehensive dataset that contains information related to customer inquiries and support.
# It encompasses frequently asked questions (FAQs) and their answers, along with various associated details.
# Each entry in the dataset includes entry and reminder dates, the status of the entry, the person responsible for data entry, question categories, the actual questions, provided answers, answer URLs, the individual who responded to the question, response statuses, references to relevant manual pages, and confirmation statuses from multiple sources.
# """
fnq_history_tool = Tool(
  name=name,
  description=description,
  func=fnq_history_chain.run,
)
tools.append(fnq_history_tool)

agent = initialize_agent(tools=tools, llm=llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True)
# COMPONENT TEST CODE
temp = agent.run("When is this data useful?")
print(temp)
