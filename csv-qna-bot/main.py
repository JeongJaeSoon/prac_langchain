from dotenv import load_dotenv
from fastapi import FastAPI
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

load_dotenv()
app = FastAPI()

loader = CSVLoader(file_path="sample.csv", encoding="utf-8", csv_args={'delimiter': ','})
data = loader.load()
embeddings = OpenAIEmbeddings()
vector_store = FAISS.from_documents(data, embeddings)

chain = ConversationalRetrievalChain.from_llm(
  llm = ChatOpenAI(temperature=0.0, model_name='gpt-3.5-turbo'),
  retriever=vector_store.as_retriever()
)

@app.get("/")
async def root():
  return {
    "Message": "Hello World",
  }

@app.get("/ask")
async def read_item(question: str):
  chat_history = []
  result = chain({
    "question": question,
    "chat_history": chat_history
  })
  answer = result["answer"]
  chat_history.append((question, answer))
  return {
    "question": question,
    "answer": answer,
  }
