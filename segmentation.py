from utils import extract_nouns
from PIL import Image
from lang_sam import LangSAM
from lang_sam.utils import draw_image
import torch
import numpy as np
import os

class ImageSegmentation:
    """A class representing the segmentation of objects based on the query."""

    def __init__(self, query, image_path):
        """
        Initialize the ImageSegmentation class.

        Inputs:
            - query (str): The query to extract objects from.
            - image_path (str): The path to the image file.
        """
        self.image_path = image_path
        self.segmented_image_path = self._get_segmented_image_path(image_path)
        self.model = LangSAM()
        self.image_pil = Image.open(image_path).convert("RGB")
        self.query = extract_nouns(query)
        self.masks = None
        self.objects = None
        self.boxes = None
        self.probabilities = None

    def filter_outliers(self):
        """
        Filter out outliers based on probabilities using standard deviation method.

        Outputs:
            - list: Filtered probabilities.
        """
        mean = torch.mean(self.probabilities)
        std_dev = torch.std(self.probabilities)
        threshold = 0.5 * std_dev
        lower_bound = mean - threshold
        print(self.objects)
        filtered_indices = self.probabilities >= lower_bound
        filtered_mask = self.masks[filtered_indices]
        filtered_boxes = self.boxes[filtered_indices]
        filtered_objects = [self.objects[i] for i in range(len(self.objects)) if filtered_indices[i]]
        filtered_probabilities = self.probabilities[filtered_indices]

        return filtered_mask, filtered_boxes, filtered_objects, filtered_probabilities

    def _get_segmented_image_path(self, image_path):
        """
        Create a new file path with '_segmented' added before the file extension.

        Inputs:
            image_path (str): The original image path.

        Outputs:
            str: The modified image path.
        """
        base, ext = os.path.splitext(image_path)
        return f"{base}_segmented{ext}"

    def get_filtered_results(self):
        """
        Get the final filtered results and generate a descriptive output string.

        Outputs:
            - str: Description of the number of objects detected.
        """
        self.masks, self.boxes, self.objects, self.probabilities = self.model.predict(
            self.image_pil, self.query
        )
        filtered_masks, filtered_boxes, filtered_objects, filtered_probabilities = self.filter_outliers()

        labels = [
            f"{filtered_object} {filtered_probability:.2f}"
            for filtered_object, filtered_probability in zip(
                filtered_objects, filtered_probabilities
            )
        ]

        image_array = np.asarray(self.image_pil)
        image = draw_image(image_array, filtered_masks, filtered_boxes, labels)
        Image.fromarray(np.uint8(image)).convert("RGB").save(self.segmented_image_path)

        return f"There are {len(filtered_probabilities)} {self.query}."


# Example usage
query = "Are there any people?"
image_path = "data/Recording.jpeg"
image_segmentation = ImageSegmentation(query, image_path)
result = image_segmentation.get_filtered_results()
print(result)

