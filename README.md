# Emotion Detection Project

This project aims to develop a machine learning model that detects a person's emotional state based on input data, specifically images. The application is built using Flask for the web interface and utilizes deep learning techniques for emotion recognition.

## Project Structure

- **app.py**: Main entry point for the web application. Initializes the Flask app and handles user requests.
- **model_training.py**: Contains the logic for training the machine learning model, including data loading, preprocessing, and model saving.
- **requirements.txt**: Lists all dependencies required for the project.
- **database.db**: SQLite database for storing user data and image information.
- **face_emotionModel.h5**: The trained machine learning model for emotion detection.
- **link_web_app.txt**: Contains the deployment link for the web application.
- **templates/index.html**: Front-end interface for uploading images and displaying results.
- **data/raw**: Directory for the raw dataset used for training.
- **data/processed**: Directory for the processed dataset.
- **notebooks/exploratory.ipynb**: Jupyter notebook for exploratory data analysis.
- **src/inference.py**: Functions for making predictions using the trained model.
- **src/utils.py**: Utility functions for data preprocessing and model evaluation.
- **models/checkpoints**: Directory for storing model checkpoints during training.
- **tests/test_app.py**: Unit tests for the application.
- **.gitignore**: Specifies files and directories to be ignored by Git.
- **README.md**: Documentation for the project.

## Setup Instructions

1. Clone the repository to your local machine.
2. Navigate to the project directory.
3. Install the required dependencies using:
   ```
   pip install -r requirements.txt
   ```
4. Set up the SQLite database by running the necessary migrations (if applicable).
5. Train the model by executing:
   ```
   python model_training.py
   ```
6. Run the web application using:
   ```
   python app.py
   ```
7. Access the application in your web browser at `http://localhost:5000`.

## Usage

- Upload an image to detect the emotional state.
- The application will process the image and display the predicted emotion.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.