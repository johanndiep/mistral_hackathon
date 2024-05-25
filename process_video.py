import os
from db import VectorStore
from vision import LLAVAModel
from tqdm import tqdm

def process_video():
    vllm = LLAVAModel()
    vs = VectorStore()
    files = os.listdir("Mistral")
    for file in tqdm(files):
        if file.endswith('.jpg'):
            image_filename = os.path.join("Mistral", file)
            video_filename = 'demo'  # Assuming a placeholder video filename

            prompt = """
                Describe the image in detail, capturing all relevant information. Include the following aspects:

                Scene and Setting: Describe the overall environment, location, and time of day.
                Key Elements: Identify and describe all prominent objects, people, animals, and structures in the scene.
                Activities: Explain what the people and animals are doing.
                Colors and Lighting: Note the colors, lighting conditions, and any shadows or reflections.
                Emotions and Atmosphere: Convey the mood or feeling of the scene.
                Background and Foreground: Describe elements in both the background and foreground, providing a sense of depth.
                Textures and Details: Include finer details and textures of objects and surroundings.
                Context: Provide any contextual information that gives additional meaning to the scene.
            """
            caption = vllm.visual_inference(image_filename, prompt)
            print()
            print(caption)
            print()

            vs.insert(caption, image_filename, video_filename)

if __name__ == "__main__":
    process_video()