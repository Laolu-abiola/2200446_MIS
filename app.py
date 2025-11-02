from flask import Flask, request, render_template
import numpy as np
import cv2
from src.inference import predict_emotion

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file part"
    
    file = request.files['file']
    
    if file.filename == '':
        return "No selected file"
    
    # Read the image file
    img = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)
    
    # Predict the emotion
    emotion = predict_emotion(img)
    
    return f'Predicted Emotion: {emotion}'

if __name__ == '__main__':
    app.run(debug=True)