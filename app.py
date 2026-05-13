from flask import Flask, render_template, request, jsonify
import json
import pickle
import random
import nltk
from nltk.stem import PorterStemmer
nltk.download('punkt')

app = Flask(__name__)

stemmer = PorterStemmer()

# Load model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# Load intents
with open('intents.json') as file:
    data = json.load(file)

# Text preprocessing
def preprocess(text):

    words = text.lower().split()

    words = [stemmer.stem(word) for word in words]

    return " ".join(words)

# Get chatbot response
def get_response(user_input):

    processed_input = preprocess(user_input)

    X_test = vectorizer.transform([processed_input])

    probabilities = model.predict_proba(X_test)[0]

    confidence = max(probabilities)

    if confidence < 0.05:
        return "I'm not sure I understood that. Can you rephrase?"

    tag = model.predict(X_test)[0]

    for intent in data['intents']:

        if intent['tag'] == tag:

            return random.choice(intent['responses'])

    return "I don't understand."

# Homepage
@app.route("/")
def home():

    return render_template("index.html")

# Chat route
@app.route("/get")
def chatbot_response():

    user_text = request.args.get('msg')

    response = get_response(user_text)

    return jsonify({"response": response})

# Run app
if __name__ == "__main__":

    app.run(debug=True)