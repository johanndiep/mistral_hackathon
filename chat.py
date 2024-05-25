from ast import List
from llama_index.llms.groq import Groq
from llama_index.core.llms import ChatMessage
import os
from dotenv import load_dotenv

from db import MistralAPI, VectorStore
load_dotenv()
import numpy as np
from PIL import Image
import glob
import psycopg2
from mistralai.client import MistralClient


default_prompt = "Where is the fire escape"
system_prompt = """
                    You will be provided with the following inputs:

                    User Query: A specific question or request for information related to a video. We have retrieved the most relevant image to the user query before I asked you. 
                    Image Caption: A brief description of the image.
                    Bounding Boxes: A list of bounding boxes, each containing specific parts of the image that may be relevant to the user query.
                    Your task is to analyze these inputs and generate a detailed response that addresses the user query, utilizing the information from the image caption and the contents within the bounding boxes. Follow these guidelines to ensure a comprehensive and accurate response:

                    Understand the User Query: Carefully read the user query to comprehend what information is being sought.
                    Analyze the Image Caption: Use the image caption to get an overall understanding of the image context.
                    Examine the Bounding Boxes: Review the contents within each bounding box for relevant details.
                    Integrate Information: Combine insights from the image caption and the bounding boxes to formulate a thorough response to the user query.
                    Be Detailed and Specific: Provide a detailed and specific answer, ensuring all aspects of the user query are addressed.
                    Clarify and Explain: If necessary, explain how the information from the bounding boxes relates to the query to provide clarity.
                    """

def main():
    prompt = "Where is the fire escape?"

    vs = VectorStore()
    mistral_client = MistralAPI()
    
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
