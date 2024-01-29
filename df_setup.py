import pandas as pd
import os
from PIL import Image
import io
import base64

# Function to encode image to base64
def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# Function to get all image paths along with labels
def get_image_paths_labels(base_path):
    data = []
    for label in os.listdir(base_path):
        label_path = os.path.join(base_path, label)
        if os.path.isdir(label_path):
            for file in os.listdir(label_path):
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    file_path = os.path.join(label_path, file)
                    # Add file path, base64 encoded image, and label to the list
                    dict = {
                        'file_name': file_path.replace("cats_ds/",""),
                        'labels': label
                    }
                    data.append(dict)
    return data



