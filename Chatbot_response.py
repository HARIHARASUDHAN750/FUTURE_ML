import re
import pandas as pd
import pickle
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess(text):
    text = re.sub(r'\W', ' ', str(text))
    text = re.sub(r'\s+', ' ', text)
    text = text.lower()
    return ' '.join([stemmer.stem(word) for word in text.split() if word not in stop_words])

with open('chatbot_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

label_answer = pd.read_csv('label_answer_map.csv')
label_to_answer = dict(zip(label_answer['Category'], label_answer['Answer']))

def get_response(query):
    processed = preprocess(query)
    vec = vectorizer.transform([processed])
    predicted_label = model.predict(vec)[0]
    return label_to_answer.get(predicted_label, "Sorry, I don't understand.")
