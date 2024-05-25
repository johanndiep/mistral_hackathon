
from db import VectorStore


def main():
    vs = VectorStore()

    import glob
    image_files = glob.glob("Mistral/*.jpg")  # Assuming images are in jpg format
    if image_files:
        first_image = image_files[0]
        vs.insert("video", first_image, "yolo")
    else:
        print("No images found in the Mistral folder.")

if __name__ == "__main__":
    main()

    