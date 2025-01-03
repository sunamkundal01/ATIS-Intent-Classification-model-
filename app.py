from flask import Flask, request, render_template
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
import spacy

from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('model.h5')

# Load the tokenizer
with open('tokenizer.pkl', 'rb') as file:
    tokenizer = pickle.load(file)

# Load the label encoder
with open('label_encoder.pkl', 'rb') as file:
    label_encoder = pickle.load(file)


# Load SpaCy model
nlp = spacy.load("en_core_web_sm")

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the query from the form
    query = request.form['query']
    
    # Preprocess the query
    processed_query = " ".join([token.lemma_ for token in nlp(query)])
    sequence = tokenizer.texts_to_sequences([processed_query])
    padded_sequence = pad_sequences(sequence, maxlen=20)
    
    # Predict the intent
    prediction = model.predict(padded_sequence)
    predicted_intent = label_encoder.inverse_transform([np.argmax(prediction)])[0]

    return render_template('index.html', prediction_text=f'Predicted Intent: {predicted_intent}')

if __name__ == "__main__":
    app.run(debug=True)
