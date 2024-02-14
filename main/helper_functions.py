import cv2
import numpy as np
import torch
from transformers import ViTImageProcessor, ViTForImageClassification

def change_detected(pframe, cframe):
    # Convert frames to grayscale
    gray_prev = cv2.cvtColor(pframe, cv2.COLOR_BGR2GRAY)
    gray_current = cv2.cvtColor(cframe, cv2.COLOR_BGR2GRAY)

    # Optional: Apply Gaussian blur
    gray_prev = cv2.GaussianBlur(gray_prev, (5, 5), 0)
    gray_current = cv2.GaussianBlur(gray_current, (5, 5), 0)

    # Compute absolute difference
    frame_diff = cv2.absdiff(gray_prev, gray_current)

    # Thresholding
    _, thresh = cv2.threshold(frame_diff, 30, 255, cv2.THRESH_BINARY)

    # Calculate the percentage of change
    change_percentage = np.sum(thresh == 255) / (thresh.shape[0] * thresh.shape[1]) * 100

    if change_percentage >= 2.0: 
        return True
    else: 
        return False 
    
def cat_or_not(): 
    # Initialize the image processor and model
    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
    model = ViTForImageClassification.from_pretrained('ChrisGuarino/model')  # Replace with your model
    model.eval()
    
    # Define your class labels
    class_labels = ['Prim', 'Rupe', 'No Cat']  # Replace with your actual labels

    #Confidence Threshold
    confidence_threshold = .8  # Define a threshold