import os

import pandas as pd
from dotenv import load_dotenv
from fastapi import FastAPI
from langchain import OpenAI
from langchain.agents import create_pandas_dataframe_agent

load_dotenv()
app = FastAPI()

df = pd.read_csv("https://github.com/kairess/toy-datasets/raw/master/titanic.csv")
agent = create_pandas_dataframe_agent(OpenAI(temperature=0), df, verbose=True)

@app.get("/")
async def root():
  return {
    "Message": "Hello World",
  }

@app.get("/ask")
async def read_item(question: str):
  answer = agent.run(question)
  return {
    "question": question,
    "answer": answer,
  }
