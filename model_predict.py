# model_predict.py

import joblib
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Download NLTK data (only first time)
nltk.download('punkt')
nltk.download('stopwords')

# Load the trained model and vectorizer
model = joblib.load('baseline_model.pkl')
vectorizer = joblib.load('baseline_vectorizer.pkl')


# Preprocessing function (same as training)
def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english')]
    return ' '.join(tokens)


# Label map
label_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}


# Prediction function
def predict_sentiment(review_text):
    processed_text = preprocess_text(review_text)
    features = vectorizer.transform([processed_text])
    prediction = model.predict(features)[0]
    return label_map[prediction]


# Main interactive loop
if __name__ == "__main__":
    print("\nSentiment Predictor Ready.")
    print("Type your review and press Enter. (Type 'exit' to quit.)\n")

    while True:
        review = input("Enter review: ")
        if review.lower() == 'exit':
            print("Exiting Sentiment Predictor.")
            break

        sentiment = predict_sentiment(review)
        print(f"Predicted Sentiment: {sentiment}\n")
