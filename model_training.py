import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint

# Load dataset
def load_data(data_path):
    # Assuming the dataset is in CSV format with image paths and labels
    data = pd.read_csv(data_path)
    return data

# Preprocess data
def preprocess_data(data):
    # Example preprocessing steps
    X = []  # Features
    y = []  # Labels
    for index, row in data.iterrows():
        image = load_image(row['image_path'])  # Function to load and preprocess image
        X.append(image)
        y.append(row['label'])
    
    X = np.array(X)
    y = np.array(y)
    
    # Encode labels
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    
    return X, y

# Build model
def build_model(input_shape):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(5, activation='softmax'))  # Assuming 5 emotion classes
    return model

# Train model
def train_model(X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = build_model(X_train.shape[1:])
    
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # Data augmentation
    datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
    datagen.fit(X_train)
    
    # Checkpoint to save the best model
    checkpoint = ModelCheckpoint('face_emotionModel.h5', monitor='val_accuracy', save_best_only=True)
    
    model.fit(datagen.flow(X_train, y_train, batch_size=32), validation_data=(X_val, y_val), epochs=50, callbacks=[checkpoint])

if __name__ == "__main__":
    data_path = os.path.join('data', 'raw', 'dataset.csv')  # Update with actual dataset path
    data = load_data(data_path)
    X, y = preprocess_data(data)
    train_model(X, y)