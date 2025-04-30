import os
import subprocess
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from tqdm import tqdm
import joblib

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Paths
data_dir = "./amazon_electronics"
jsonl_gz_url = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Electronics.jsonl.gz"
gz_path = os.path.join(data_dir, "Electronics.jsonl.gz")
jsonl_path = os.path.join(data_dir, "Electronics.jsonl")

# Ensure data directory exists
os.makedirs(data_dir, exist_ok=True)

# Download and decompress data if needed
if not os.path.exists(jsonl_path):
    print("Downloading Electronics.jsonl.gz ...")
    subprocess.run(["wget", jsonl_gz_url, "-O", gz_path], check=True)
    print("Decompressing Electronics.jsonl.gz ...")
    subprocess.run(["gunzip", "-f", gz_path], check=True)
    print("Downloaded and decompressed Electronics.jsonl!")
else:
    print(f"Found {jsonl_path}, skipping download.")


def load_data(file_path: str, sample_size: int = 10000000) -> pd.DataFrame:
    """Load and sample the dataset with error handling."""
    abs_path = os.path.abspath(file_path)

    if not os.path.exists(abs_path):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        abs_path = os.path.join(current_dir, file_path)
        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"Could not find the data file at {file_path} or {abs_path}.")

    print(f"Loading data from: {abs_path}")
    data = []
    try:
        with open(abs_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(tqdm(f, total=sample_size, desc="Loading data")):
                if i >= sample_size:
                    break
                try:
                    entry = json.loads(line)
                    processed_entry = {
                        'rating': entry.get('rating', 0),
                        'text': entry.get('text', ''),
                        'title': entry.get('title', ''),
                        'helpful_votes': entry.get('helpful_votes', 0),
                        'verified_purchase': entry.get('verified_purchase', False)
                    }
                    data.append(processed_entry)
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping line {i + 1} due to JSON decode error: {e}")
                    continue
    except Exception as e:
        raise Exception(f"Error loading data file: {e}")

    if not data:
        raise ValueError("No data was loaded. The file might be empty or in an incorrect format.")

    print(f"Successfully loaded {len(data)} samples")
    return pd.DataFrame(data)


def preprocess_text(text):
    """Clean and preprocess text."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)


def map_ratings_to_sentiment(rating):
    """Map star ratings to sentiment labels."""
    if not isinstance(rating, (int, float)):
        return 1  # Neutral default
    if rating in [1, 2]:
        return 0  # Negative
    elif rating == 3:
        return 1  # Neutral
    else:
        return 2  # Positive


def main():
    try:
        print("Loading data...  :::")
        df = load_data(jsonl_path)

        df['sentiment'] = df['rating'].apply(map_ratings_to_sentiment)

        print("Preprocessing text...")
        df['processed_text'] = df['text'].apply(preprocess_text)

        print("Splitting data...")
        X_train, X_test, y_train, y_test = train_test_split(
            df['processed_text'].values, df['sentiment'].values,
            test_size=0.2, random_state=42
        )

        print("Vectorizing text...")
        vectorizer = TfidfVectorizer(
            max_features=50000,
            ngram_range=(1, 2),
            min_df=5,
            max_df=0.95
        )
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)

        print("Training model...")
        model = MultinomialNB(alpha=0.1)
        model.fit(X_train_tfidf, y_train)

        print("Evaluating model...")
        predictions = model.predict(X_test_tfidf)

        accuracy = accuracy_score(y_test, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, predictions, average='weighted')

        print("\nModel Evaluation:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")

        # ----------- FINAL EVALUATION METRICS PLOT ------------
        metrics = {
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1
        }

        plt.figure(figsize=(6, 5))
        sns.barplot(x=list(metrics.keys()), y=list(metrics.values()))
        plt.ylim(0, 1)
        plt.title('Final Evaluation Metrics')
        plt.ylabel('Score')
        for i, value in enumerate(metrics.values()):
            plt.text(i, value + 0.02, f"{value:.2f}", ha='center', va='bottom')
        plt.tight_layout()
        plt.savefig('baseline_final_evaluation_metrics.png')
        plt.close()
        print("Saved final evaluation metrics plot as baseline_final_evaluation_metrics.png")

        # Confusion matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, predictions)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('baseline_confusion_matrix.png')
        plt.close()
        print("Saved confusion matrix as baseline_confusion_matrix.png")

        # Save model and vectorizer
        joblib.dump(model, 'baseline_model.joblib')
        joblib.dump(vectorizer, 'baseline_vectorizer.joblib')
        print("Saved model and vectorizer.")

    except Exception as e:
        print(f"An error occurred: {e}")
        raise


if __name__ == "__main__":
    main()
