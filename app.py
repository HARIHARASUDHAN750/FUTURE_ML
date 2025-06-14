from flask import Flask, render_template, request
import pickle
import re
import string

app = Flask(__name__)

# Load saved model and vectorizer
with open('chatbot_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Text preprocessing
def preprocess(text):
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    text = re.sub(r"\d+", "", text)
    tokens = text.split()
    return " ".join(tokens)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/get', methods=['POST'])
def get_bot_response():
    user_query = request.form['msg']
    processed = preprocess(user_query)
    vec = vectorizer.transform([processed])
    prediction = model.predict(vec)[0]
    return prediction

if __name__ == '__main__':
    app.run(debug=True)
