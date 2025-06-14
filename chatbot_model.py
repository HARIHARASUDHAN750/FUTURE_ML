import pandas as pd
import re
import string
import nltk
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Download stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords

# Load dataset
df = pd.read_csv("customer_support_tickets.csv")

# Combine Subject + Description for input
df['combined'] = df['Ticket Subject'].fillna('') + " " + df['Ticket Description'].fillna('')

# Preprocess function
def preprocess(text):
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    text = re.sub(r"\d+", "", text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return " ".join(tokens)

# Apply preprocessing
df['clean_text'] = df['combined'].apply(preprocess)

# Features and labels
X = df['clean_text']
y = df['Ticket Type']  # you can change this if needed

# TF-IDF Vectorization
vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

# Train model
model = LogisticRegression()
model.fit(X_vec, y)

# Save model and vectorizer
with open('chatbot_model.pkl', 'wb') as f:
    pickle.dump(model, f)

with open('tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

print("✅ Model trained and saved successfully.")
