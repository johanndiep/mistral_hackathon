from llava.mm_utils import get_model_name_from_path
from llava.eval.run_llava import eval_model

class LLAVAModel:
    """A class representing a VLLM for generating text based on both text prompts and visual inputs."""

    def __init__(
        self,
        model_path="liuhaotian/llava-v1.6-mistral-7b",
        model_base=None,
        conv_mode=None,
        sep=",",
        temperature=0,
        top_p=None,
        num_beams=1,
        max_new_tokens=512,
    ):
        """
        Constructor for LLAVAModel class.

        Inputs:
            - model_path (str): Path to the VLLM model.
            - prompt (str): Prompt for generating text.
            - model_base (str, optional): Base model for VLLM, defaults to None.
            - conv_mode (str, optional): Convolution mode, defaults to None.
            - sep (str, optional): Separator used for VLLM input, defaults to ",".
            - temperature (float, optional): Sampling temperature for text generation, defaults to 0.
            - top_p (float, optional): Top-p sampling cutoff for text generation, defaults to None.
            - num_beams (int, optional): Number of beams for beam search, defaults to 1.
            - max_new_tokens (int, optional): Maximum number of new tokens to generate, defaults to 512.
        """
        self.model_path = model_path
        self.model_base = model_base
        self.conv_mode = conv_mode
        self.sep = sep
        self.temperature = temperature
        self.top_p = top_p
        self.num_beams = num_beams
        self.max_new_tokens = max_new_tokens

        self.model_name = get_model_name_from_path(model_path)

    def visual_inference(self, image_file, prompt):
        """
        Perform visual inference using VLLM model.

        Inputs:
            - image_file (str): Path to the image file.

        Outputs:
            - str: Generated text based on the image and prompt.
        """
        args = type(
            "Args",
            (),
            {
                "model_path": self.model_path,
                "model_base": self.model_base,
                "model_name": self.model_name,
                "query": prompt,
                "conv_mode": self.conv_mode,
                "image_file": image_file,
                "sep": self.sep,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "num_beams": self.num_beams,
                "max_new_tokens": self.max_new_tokens,
            },
        )()

        # Perform VLLM model evaluation
        return eval_model(args)

def test_LLAVAModel():
    # Example usage
    model_path = "liuhaotian/llava-v1.6-mistral-7b"
    prompt = "How many people are in this place?"

    vllm_model = LLAVAModel(model_path)

    image_path = "data/Untitled.jpeg"
    result = vllm_model.visual_inference(image_path, prompt)
    print(result)

    image_path = "data/Recording.jpeg"
    result = vllm_model.visual_inference(image_path)
    print(result)

if __name__ == "__main__":
    test_LLAVAModel()