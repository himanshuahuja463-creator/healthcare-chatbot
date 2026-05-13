from flask import Flask, render_template, request, jsonify
import json
import pickle
import random

app = Flask(__name__)

# Load model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# Load intents
with open('intents.json') as file:
    data = json.load(file)

def get_response(user_input):

    X_test = vectorizer.transform([user_input])

    probabilities = model.predict_proba(X_test)[0]

    confidence = max(probabilities)

    if confidence < 0.40:
        return "I'm not sure I understood that. Can you rephrase?"

    tag = model.predict(X_test)[0]

    for intent in data['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])

    return "I don't understand."

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def chatbot_response():
    user_text = request.args.get('msg')
    response = get_response(user_text)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)