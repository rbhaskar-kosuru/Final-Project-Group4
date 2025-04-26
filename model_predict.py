import joblib
import re
import nltk
import sys
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Download NLTK data if not already present
nltk.download('punkt')
nltk.download('stopwords')

# Load model and vectorizer
model = joblib.load('baseline_model.joblib')
vectorizer = joblib.load('baseline_vectorizer.joblib')

# Preprocessing function
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

# Main loop
if __name__ == "__main__":
    print("\nSentiment Predictor Ready.")
    print("Type or paste your review (multiple paragraphs allowed).")
    print("When done, type a single line with 'END' and press Enter to submit.")
    print("Type 'EXIT' anytime to quit.\n")

    while True:
        print("-" * 50)
        print("Enter your review (end with 'END' on a new line):")

        # Collect multiple lines
        lines = []
        while True:
            line = sys.stdin.readline().strip()

            if line.upper() == "EXIT":
                print("Exiting Sentiment Predictor.")
                sys.exit()

            if line.upper() == "END":
                break

            lines.append(line)

        review = "\n".join(lines).strip()

        if review == "":
            print("No review entered. Please try again.\n")
            continue

        sentiment = predict_sentiment(review)
        print(f"\nPredicted Sentiment: {sentiment}\n")
