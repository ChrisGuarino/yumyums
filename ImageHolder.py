import cv2

class ImageHolder:
    def __init__(self):
        self.image = None

    def capture_image(self):
        cap = cv2.VideoCapture(0)
        ret, frame = cap.read()
        cap.release()
        if ret:
            self.image = frame