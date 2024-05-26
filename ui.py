import gradio as gr
import time
from chat import Chat

chat = Chat()
gr.Interface(
    fn=chat.chat,  # The function to process the prompt
    inputs="text",  # Text input
    outputs=[gr.Image(type="numpy"), "text"]  # Output both an image and a text
).launch(share=True)
