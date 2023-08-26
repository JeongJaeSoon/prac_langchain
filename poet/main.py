import streamlit as st
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI

chat_model = ChatOpenAI()

load_dotenv()

st.title(":blue[AI] poet :sunglasses:")

content = st.text_input('Poet subject', '')

if st.button('Request a poem'):
  with st.spinner('AI is writing a poem...'):
    result = chat_model.predict(f"plz write a poem about {content}")
    st.write(result)
