from ast import List
from llama_index.llms.groq import Groq
from llama_index.core.llms import ChatMessage
import os
from dotenv import load_dotenv

from db import VectorStore
load_dotenv()
import numpy as np
from PIL import Image
import glob
import psycopg2
from mistralai.client import MistralClient

def main():
    vectorStore = VectorStore()
    llm = Groq(model="mixtral-8x7b-32768", api_key=os.environ.get("GROQ_API_KEY"))

    messages = [
        ChatMessage(
            role="system", content="You are a pirate with a colorful personality"
        ),
        ChatMessage(role="user", content="What is your name"),
    ]
    
    resp = llm.stream_chat(messages)

    for r in resp:
        print(r.delta, end="\n")

if __name__ == "__main__":
    main()
