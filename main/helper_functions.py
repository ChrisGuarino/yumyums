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
    
def cat_or_not(ret,frame): 
    # Initialize the image processor and model
    processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
    model = ViTForImageClassification.from_pretrained('ChrisGuarino/model')  # Replace with your model
    model.eval()
    
    # Define your class labels
    class_labels = ['Prim', 'Rupe', 'No Cat']  # Replace with your actual labels

    #Confidence Threshold
    confidence_threshold = .8  # Define a threshold 

    while True:
        
        if ret:
            # Preprocess the frame
            frame_resized = cv2.resize(frame, (400, 400))
            frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
            inputs = processor(images=frame_rgb, return_tensors="pt")

            # Get predicti?ons
            with torch.no_grad():
                predictions = model(**inputs).logits

                # Convert predictions to probabilities and get the highest probability class
                probabilities = torch.nn.functional.softmax(predictions, dim=-1)
                confidences, predicted_class_idx = torch.max(probabilities, dim=-1)
                predicted_class = class_labels[predicted_class_idx]#Something with +1 to shift the labels if we add a No Cat label
            print(probabilities)

            # Check if confidence is above the threshold
            if confidences.item() < confidence_threshold:
                label = 'No Cat'
                confidence = 0
            else:
                label = class_labels[predicted_class_idx.item()]  # +1 to account for 'No Cat'
                confidence = confidences.item()

            # Prepare the display text
            display_text = f'{label} ({confidence:.2f})'

            # Display the prediction on the frame
            cv2.putText(frame_resized, display_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Display the resulting frame
            cv2.imshow('Frame', frame_resized)

            # Break the loop with 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break

# Release the capture when done
cap.release()
cv2.destroyAllWindows()