from utils import extract_nouns
from PIL import Image
from lang_sam import LangSAM
from lang_sam.utils import draw_image
import torch
import numpy as np
import os

class ImageSegmentation:
    """A class representing the segmentation of objects based on the query."""

    def __init__(self):
        """
        Initialize the ImageSegmentation class. 
        """
        self.model = LangSAM()
        self.masks = None
        self.objects = None
        self.boxes = None
        self.probabilities = None

    def filter_outliers(self, masks, boxes, objects, probabilities):
        """
        Filter out outliers based on probabilities using standard deviation method.

        Outputs:
            - list: Filtered probabilities.
        """
        mean = torch.mean(probabilities)
        std_dev = torch.std(probabilities)
        threshold = 0.5 * std_dev
        lower_bound = mean - threshold
        #print(objects)
        filtered_indices = probabilities >= lower_bound
        filtered_mask = masks[filtered_indices]
        filtered_boxes = boxes[filtered_indices]
        filtered_objects = [objects[i] for i in range(len(objects)) if filtered_indices[i]]
        filtered_probabilities = probabilities[filtered_indices]

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

    def segment_simple(self, query, image_path):
        """
        Get the final filtered results and generate a descriptive output string.

        Outputs:
            - str: Description of the number of objects detected.
        """

        segmented_image_path = self._get_segmented_image_path(image_path)
        image_pil = image_pil = Image.open(image_path).convert("RGB")
        noun = extract_nouns(query)[0]
        masks, boxes, objects, probabilities = self.model.predict(
            image_pil, noun
        )
        filtered_masks, filtered_boxes, filtered_objects, filtered_probabilities = self.filter_outliers(masks, boxes, objects, probabilities)

        labels = [
            f"{filtered_object} {filtered_probability:.2f}"
            for filtered_object, filtered_probability in zip(
                filtered_objects, filtered_probabilities
            )
        ]

        image_array = np.asarray(image_pil)
        image = draw_image(image_array, filtered_masks, filtered_boxes, labels)
        Image.fromarray(np.uint8(image)).convert("RGB").save(segmented_image_path)

        return f"There are {len(filtered_probabilities)} {noun}.", image

    def segment(self, query, image_path, keywords=None):
        """
        Get the final filtered results and generate a descriptive output string.

        Outputs:
            - str: Description of the number of objects detected.
        """
        print("KEYWORDS", keywords)
        if keywords is None:
            keywords = extract_nouns(query)

        segmented_image_path = self._get_segmented_image_path(image_path)
        image_pil = image_pil = Image.open(image_path).convert("RGB")
        image_array = np.asarray(image_pil)

        seg_results = list()
        for word in keywords:
            masks, boxes, objects, probabilities = self.model.predict(
                image_pil, word
            )

            filtered_masks, filtered_boxes, filtered_objects, filtered_probabilities = self.filter_outliers(masks, boxes, objects, probabilities)

            labels = [
                f"{filtered_object} {filtered_probability:.2f}"
                for filtered_object, filtered_probability in zip(
                    filtered_objects, filtered_probabilities
                )
            ]

            detailed_results = [
                ("confidence: {:.2f}".format(filtered_probability), filtered_box.int().numpy().tolist())
                for filtered_probability, filtered_box in zip(filtered_probabilities, filtered_boxes)
            ]
            seg_results.append((word, len(filtered_probabilities), detailed_results))

            image_array = draw_image(image_array, filtered_masks, filtered_boxes, labels)

            #print(filtered_probabilities)
            #print(filtered_boxes)
        
        #Image.fromarray(np.uint8(image_array)).convert("RGB").save(segmented_image_path)
        
        ret_str = ""
        for res in seg_results:
            ret_str += f"In the keyword {res[0]} there are {res[1]} entities found with the following (confidence, bounding box): {res[2]}\n" 
        #ret_str =  f"From the keywords {nouns} we retrieve the following filtered_probabilites {filtered_probabilities} ."
        print("Segmentation output ", ret_str, "\n\n")
        return ret_str, image_array

def test_Segmentation():
    # Example usage
    query = "How many people and chairs?"
    image_path = "data/Untitled.jpeg"
    image_segmentation = ImageSegmentation()
    result, img = image_segmentation.segment(query, image_path)
    print(result)

if __name__ == "__main__":
    test_Segmentation()
