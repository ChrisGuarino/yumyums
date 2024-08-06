# Catcha v2
## Setup:  Run the setup_env.sh for environment setup. 

### Model: https://huggingface.co/ChrisGuarino/model - Fine-tuned version of google/vit-base-patch16-224-in21k. 
### Dataset: https://huggingface.co/ChrisGuarino/cat_ds - It's very small. But it works for the most part. 

#### Base Model: https://huggingface.co/google/vit-base-patch16-224-in21k

### How to Run (Right Now...): run the main.ipynb cells. 

### Where it stands:<br>
I have the program using the laptop camera right now. It uses frame differencing to decide first if there is motion detected by the camera. Once motion is triggered the cat classification model is run and make an inference of where there is Prim, Rupe, or no cat in the frame. 
#### No idea how to apply this yet.

#### 08/06/24 
I am thinking of finishing this project.. One thing I want to do is just get the system to categorize what Rupert is doing in the frame. This would be using likely the VILA library to do Edge reasoning on the Raspi. 

https://medium.com/@ammani/python-usb-camera-tutorial-for-raspberry-pi-4-3098678a2464

