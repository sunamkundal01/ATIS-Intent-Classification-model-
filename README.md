# Neural Intent Analysis System

This project is a web-based application for classifying intents in ATIS (Automatic Terminal Information Service) queries using a neural network model. The application is built using Flask and TensorFlow.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Training](#model-training)
- [API Endpoints](#api-endpoints)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Features

- **Real-time Intent Classification**: Enter a query and get the predicted intent instantly.
- **Interactive UI**: A modern and responsive user interface with particle effects and animations.
- **Metrics Display**: Shows the primary intent, accuracy, and processing time for each query.

## Installation

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

## Usage

1. **Run the Flask application**:
    ```bash
    python app.py
    ```

2. **Open your browser** and navigate to `http://127.0.0.1:5000/`.

3. **Enter a query** in the input field and click "Analyze" to get the predicted intent.

## Project Structure

- `app.py`: The main Flask application file.
- `templates/index.html`: The HTML template for the web interface.
- `requirements.txt`: The list of required Python packages.
- `.gitignore`: Git ignore file to exclude the virtual environment.

## Model Training

The model used for intent classification is a neural network trained on the ATIS dataset. The training process involves the following steps:

1. **Data Preprocessing**: Tokenization, padding, and lemmatization of the input queries.
2. **Model Architecture**: A neural network model built using TensorFlow/Keras.
3. **Training**: The model is trained on the ATIS dataset to classify different intents.

### Example Training Script

```python
import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder

# Load and preprocess the dataset
# ...existing code...

# Define the model architecture
model = Sequential()
model.add(Embedding(input_dim=vocab_size, output_dim=128, input_length=max_len))
model.add(LSTM(128, return_sequences=True))
model.add(LSTM(128))
model.add(Dense(num_classes, activation='softmax'))

# Compile and train the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Save the model and tokenizer
model.save('model.h5')
with open('tokenizer.pkl', 'wb') as file:
    pickle.dump(tokenizer, file)
with open('label_encoder.pkl', 'wb') as file:
    pickle.dump(label_encoder, file)
```

## API Endpoints

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

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [Flask](https://flask.palletsprojects.com/)
- [TensorFlow](https://www.tensorflow.org/)
- [SpaCy](https://spacy.io/)
- [Particles.js](https://vincentgarreau.com/particles.js/)
