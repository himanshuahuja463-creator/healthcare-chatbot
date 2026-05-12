import json
import pickle
import random

# Load model and vectorizer
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))

# Load intents
with open('intents.json') as file:
    data = json.load(file)

print("Health Chatbot Started! Type 'quit' to stop.\n")

while True:
    user_input = input("You: ")

    if user_input.lower() == 'quit':
        print("Bot: Goodbye!")
        break

    # Convert input into vector
    X_test = vectorizer.transform([user_input])

    # Predict intent
    tag = model.predict(X_test)[0]

    # Generate response
    for intent in data['intents']:
        if intent['tag'] == tag:
            response = random.choice(intent['responses'])
            print("Bot:", response)