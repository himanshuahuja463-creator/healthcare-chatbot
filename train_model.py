import json
import pickle
import nltk
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

stemmer = PorterStemmer()

# Load intents
with open('intents.json') as file:
    data = json.load(file)

training_sentences = []
training_labels = []

def preprocess(text):

    words = text.lower().split()

    words = [stemmer.stem(word) for word in words]

    return " ".join(words)

# Prepare training data
for intent in data['intents']:

    for pattern in intent['patterns']:

        processed_pattern = preprocess(pattern)

        training_sentences.append(processed_pattern)

        training_labels.append(intent['tag'])

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(training_sentences)

# Train model
model = MultinomialNB()

model.fit(X, training_labels)

# Save model and vectorizer
pickle.dump(model, open('model.pkl', 'wb'))

pickle.dump(vectorizer, open('vectorizer.pkl', 'wb'))

print("Improved NLP model trained successfully!")