import logging

logging.basicConfig(level=logging.DEBUG)

import os
from dotenv import load_dotenv

from slack_bolt.async_app import AsyncApp
from slack_bolt.adapter.socket_mode.aiohttp import AsyncSocketModeHandler

from fastapi import FastAPI

from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS

load_dotenv()

fastapi_app = FastAPI()
slack_app = AsyncApp(token=os.environ["SLACK_BOT_TOKEN"])
socket_handler = AsyncSocketModeHandler(slack_app, os.environ["SLACK_APP_TOKEN"])

loader = CSVLoader(file_path="sample.csv", encoding="utf-8", csv_args={'delimiter': ','})
data = loader.load()
embeddings = OpenAIEmbeddings()
vector_store = FAISS.from_documents(data, embeddings)

chain = ConversationalRetrievalChain.from_llm(
  llm = ChatOpenAI(temperature=0.0, model_name='gpt-3.5-turbo'),
  retriever=vector_store.as_retriever()
)

@slack_app.event("app_mention")
async def handle_events(body, event, say):
    question = body['event']['blocks'][0]['elements'][0]['elements'][1]['text'].strip()
    chat_history = []

    print("start")
    result = chain({
      "question": question,
      "chat_history": chat_history
    })
    answer = result["answer"]
    print("end")
    chat_history.append((question, answer))

    await say(f"Q: {question}\n\nA: {answer}")

@fastapi_app.get("/health")
async def health():
    return {"status": "ok"}

@fastapi_app.on_event("startup")
async def startup():
    await socket_handler.connect_async()