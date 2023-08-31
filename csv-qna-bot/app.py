import logging
import os

from dotenv import load_dotenv
from fastapi import FastAPI
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from slack_bolt.adapter.socket_mode.aiohttp import AsyncSocketModeHandler
from slack_bolt.async_app import AsyncApp

logging.basicConfig(level=logging.DEBUG)



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

bot_ids = os.environ["SLACK_BOT_IDS"].split(",")
whitelist_channels = os.environ["SLACK_WHITELIST_CHANNELS"].split(",")

@slack_app.event("app_mention")
async def handle_events(event, client, say):
    contents = event['blocks'][0]['elements'][0]['elements']

    bot_id = next((item["user_id"] for item in contents if item.get("user_id") in bot_ids), None)
    user_id = event['user']

    text = event['text']
    question:str = text.split(f'<@{bot_id}>')[-1].strip()

    if not question:
        await say(f"おいおい、内容がないぞ <@{user_id}> さん。", thread_ts=event['ts'])
        return

    if event['channel'] not in whitelist_channels:
        await say(f"すみません、<@{bot_id}> は特定のチャンネルでのみ動作します。", thread_ts=event['ts'])
        return

    await say(f":sun-spin:", thread_ts=event['ts'])

    chat_history = []

    print("start")
    result = chain({
      "question": question,
      "chat_history": chat_history
    })
    answer = result["answer"]
    print("end")

    chat_history.append((question, answer))

    await say(f"Q: {question}\n\nA: {answer}", thread_ts=event['ts'])

@fastapi_app.get("/health")
async def health():
    return {"status": "ok"}

@fastapi_app.on_event("startup")
async def startup():
    await socket_handler.connect_async()
