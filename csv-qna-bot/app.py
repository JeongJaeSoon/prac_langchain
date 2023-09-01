import asyncio
import logging
import os
import traceback

from dotenv import load_dotenv
from fastapi import FastAPI
from langchain.callbacks.base import AsyncCallbackHandler
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from slack_bolt.adapter.socket_mode.aiohttp import AsyncSocketModeHandler
from slack_bolt.async_app import AsyncApp
from slack_sdk.web.async_client import AsyncWebClient

logging.basicConfig(level=logging.DEBUG)



load_dotenv()

fastapi_app = FastAPI()
slack_app = AsyncApp(token=os.environ["SLACK_BOT_TOKEN"])
socket_handler = AsyncSocketModeHandler(slack_app, os.environ["SLACK_APP_TOKEN"])

loader = CSVLoader(file_path="sample.csv", encoding="utf-8", csv_args={'delimiter': ','})
documents = loader.load()

embeddings = OpenAIEmbeddings()

vectorstore = FAISS.from_documents(documents, embeddings)
retriever = vectorstore.as_retriever()

bot_ids = os.environ["SLACK_BOT_IDS"].split(",")
whitelist_channels = os.environ["SLACK_WHITELIST_CHANNELS"].split(",")

@slack_app.event("app_mention")
async def handle_events(event, client, say):
    progress = await say(f":sun-spin:", thread_ts=event['ts'])

    try:
        contents = event['blocks'][0]['elements'][0]['elements']

        bot_id = next((item["user_id"] for item in contents if item.get("user_id") in bot_ids), None)
        user_id = event['user']

        text = event['text']
        question:str = text.split(f'<@{bot_id}>')[-1].strip()

        if not question:
            await client.chat_update(
                channel=progress['channel'],
                ts=progress['ts'],
                text=f"おいおい、内容がないぞ <@{user_id}> さん。"
            )
            return

        if event['channel'] not in whitelist_channels:
            await client.chat_update(
                channel=progress['channel'],
                ts=progress['ts'],
                text=f"すみません、<@{bot_id}> は特定のチャンネルでのみ動作します。"
            )
            return

        llm = ChatOpenAI(
            # FIXME: Streaming is not working yet with the ConversationalRetrievalChain
            # streaming=True,
            # callbacks=[
            #     SlackUpdateAsyncHandler(
            #         client=client,
            #         channel=progress['channel'],
            #         ts=progress['ts']
            #     )
            # ],
            temperature=0,
            model_name='gpt-3.5-turbo',
        )

        chain = ConversationalRetrievalChain.from_llm(
            llm = llm,
            retriever=retriever
        )

        chat_history = []

        print("start")
        result = chain({
            "question": question,
            "chat_history": chat_history
        })
        answer = result['answer']

        await client.chat_update(
            channel=progress['channel'],
            ts=progress['ts'],
            text=f"{answer}"
        )
        chat_history.append((question, answer))

        print(result)
        return

    except Exception as e:
        err_msg = traceback.format_exc()
        await client.chat_update(
            channel=progress['channel'],
            ts=progress['ts'],
            text=f"エラーが発生しました。下記のエラーメッセージを担当開発者に共有してください。\n```{err_msg}```",
        )
        return

@fastapi_app.get("/health")
async def health():
    return {"status": "ok"}

@fastapi_app.on_event("startup")
async def startup():
    await socket_handler.connect_async()

# class SlackUpdateAsyncHandler(AsyncCallbackHandler):
#     def __init__(self, client: AsyncWebClient, channel, ts, initial_context="") -> None:
#         self.client = client
#         self.channel = channel
#         self.ts = ts
#         self.context = initial_context
#         self.tokens = []
#         self.update_frequency = 10
#         self.update_interval = 5

#     def on_llm_new_token(self, token: str, **kwargs) -> None:
#         self.tokens.append(token)

#         if len(self.tokens) >= self.update_frequency:
#             self.update_chat()

#         if not hasattr(self, 'update_task'):
#             self.update_task = asyncio.create_task(self.delayed_update())

#     async def delayed_update(self):
#         await asyncio.sleep(self.update_interval)
#         self.update_chat()

#     def update_chat(self):
#         self.context += ''.join(self.tokens)
#         self.tokens = []

#         asyncio.create_task(self.client.chat_update(
#             channel=self.channel,
#             ts=self.ts,
#             text=f"{self.context}"
#         ))

#         if hasattr(self, 'update_task'):
#             delattr(self, 'update_task')
