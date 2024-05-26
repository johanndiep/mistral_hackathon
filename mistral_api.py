from mistralai.client import MistralClient
from llama_index.llms.mistralai import MistralAI
import os
from dotenv import load_dotenv
load_dotenv()

class MistralAPI:
    def __init__(self):
        self.client = MistralClient(api_key=os.getenv("MISTRAL_API_KEY")) # Mistral directly
        self.llm = MistralAI(api_key=os.getenv("MISTRAL_API_KEY"), model="mistral-large-latest") # LLamaIndex

    def embed(self, txt):
        """ 
        Embedd a single string
        """
        embedding = self.client.embeddings("mistral-embed", txt)
        return embedding.data[0].embedding
    
    def chat(self, messages):
        """
        Returns a string of the LLM response
        """
        result = self.llm.chat(messages)
        return result.message.content