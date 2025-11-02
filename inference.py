# Contents of /Surname-MatricNumber/src/inference.py

import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from PIL import Image

class EmotionDetector:
    def __init__(self, model_path):
        self.model = load_model(model_path)

    def preprocess_image(self, image_path):
        image = Image.open(image_path)
        image = image.resize((48, 48))  # Resize to the input size of the model
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image /= 255.0  # Normalize the image
        return image

    def predict_emotion(self, image_path):
        processed_image = self.preprocess_image(image_path)
        predictions = self.model.predict(processed_image)
        emotion_index = np.argmax(predictions[0])
        return emotion_index  # Return the index of the predicted emotion

# Example usage:
# detector = EmotionDetector('face_emotionModel.h5')
# emotion = detector.predict_emotion('path_to_image.jpg')
# print(f'Predicted emotion index: {emotion}')