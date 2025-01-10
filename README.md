# Neural Intent Analysis System 🧠

This project is a web-based application for classifying intents in ATIS (Automatic Terminal Information Service) queries using a neural network model. The application is built using Flask and TensorFlow.

## Table of Contents 📚

- [Features](#features) ✨
- [Installation](#installation) 🛠️
- [Usage](#usage) 🖥️
- [Project Structure](#project-structure) 🗂️
- [Model Training](#model-training) 🎓
- [API Endpoints](#api-endpoints) 🌐
- [Project Outcomes](#project-outcomes) 🏆

- [Acknowledgements](#acknowledgements) 🙏

## Features ✨

- **Real-time Intent Classification**: Enter a query and get the predicted intent instantly.
- **Interactive UI**: A modern and responsive user interface with particle effects and animations.
- **Metrics Display**: Shows the primary intent, accuracy, and processing time for each query.

## Installation 🛠️

Follow these steps to set up the project on your local machine:

1. **Clone the repository**:
    ```bash
    git clone https://github.com/yourusername/intent-classification.git
    cd intent-classification
    ```

2. **Create and activate a virtual environment**:
    ```bash
    python -m venv myenv
    source myenv/bin/activate  # On Windows use `myenv\Scripts\activate`
    ```

3. **Install the required packages**:
    ```bash
    pip install -r requirements.txt
    ```

4. **Download the necessary models**:
    - Place the `model.h5`, `tokenizer.pkl`, and `label_encoder.pkl` files in the project root directory.
    - Download the SpaCy model:
      ```bash
      python -m spacy download en_core_web_sm
      ```

## Usage 🖥️

1. **Run the Flask application**:
    ```bash
    python app.py
    ```

2. **Open your browser** and navigate to `http://127.0.0.1:5000/`.

3. **Enter a query** in the input field and click "Analyze" to get the predicted intent.

## Project Structure 🗂️

- `app.py`: The main Flask application file.
- `templates/index.html`: The HTML template for the web interface.
- `requirements.txt`: The list of required Python packages.
- `.gitignore`: Git ignore file to exclude the virtual environment.

## Model Training 🎓

The model used for intent classification is a neural network trained on the ATIS dataset. The training process involves the following steps:

1. **Data Preprocessing**: Tokenization, padding, and lemmatization of the input queries.
2. **Model Architecture**: A neural network model built using TensorFlow/Keras.
3. **Training**: The model is trained on the ATIS dataset to classify different intents.

### Example Training Script

The following is a high-level overview of the training script used to train the model:

1. **Load and preprocess the dataset**.
2. **Define the model architecture** using TensorFlow/Keras.
3. **Compile and train the model**.
4. **Save the model and tokenizer** for later use.

## API Endpoints 🌐

### `GET /`

Renders the home page with the input form.

### `POST /predict`

Accepts a query from the form, processes it, and returns the predicted intent.

#### Request

- **Method**: POST
- **Content-Type**: application/x-www-form-urlencoded
- **Body**: `query=<user_query>`

#### Response

- **Content-Type**: text/html
- **Body**: Renders the home page with the prediction result.

## Project Outcomes 🏆

The project has achieved the following outcomes:

- **Enhanced Accuracy**: The model demonstrates a high level of accuracy in classifying intents for ATIS queries.
- **User-Friendly Interface**: Developed a responsive and engaging web interface that allows users to interact with the model seamlessly.
- **Efficient Processing**: Optimized the model and application for fast query processing, ensuring real-time responses.
- **Scalability**: Built an architecture that can be easily scaled for handling increased traffic or expanded functionality in the future.


## Acknowledgements 🙏

- [Flask](https://flask.palletsprojects.com/)
- [TensorFlow](https://www.tensorflow.org/)
- [SpaCy](https://spacy.io/)
- [Particles.js](https://vincentgarreau.com/particles.js/)
