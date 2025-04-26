import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import json
import os
from tqdm import tqdm

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

def load_data(file_path: str, sample_size: int = 10000000) -> pd.DataFrame:
    """Load and sample the dataset with error handling."""
    abs_path = os.path.abspath(file_path)
    
    if not os.path.exists(abs_path):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        abs_path = os.path.join(current_dir, file_path)
        
        if not os.path.exists(abs_path):
            raise FileNotFoundError(
                f"Could not find the data file at {file_path} or {abs_path}. "
                "Please ensure the file exists and the path is correct."
            )
    
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
                    print(f"Warning: Skipping line {i+1} due to JSON decode error: {e}")
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
        
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation and special characters
    text = re.sub(r'[^\w\s]', '', text)
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def map_ratings_to_sentiment(rating):
    """Map star ratings to sentiment labels."""
    if not isinstance(rating, (int, float)):
        return 1  # Default to neutral if rating is invalid
        
    if rating in [1, 2]:
        return 0  # Negative
    elif rating == 3:
        return 1  # Neutral
    else:
        return 2  # Positive

def main():
    try:
        # Load and preprocess data
        print("Loading data...")
        df = load_data('Electronics.jsonl')
        
        # Map ratings to sentiment labels
        df['sentiment'] = df['rating'].apply(map_ratings_to_sentiment)
        
        # Preprocess review text
        print("Preprocessing text...")
        df['processed_text'] = df['text'].apply(preprocess_text)
        
        # Split data
        print("Splitting data...")
        X_train, X_test, y_train, y_test = train_test_split(
            df['processed_text'].values, df['sentiment'].values,
            test_size=0.2, random_state=42
        )
        
        # Vectorize text
        print("Vectorizing text...")
        vectorizer = TfidfVectorizer(
            max_features=50000,  # Increased from 10000
            ngram_range=(1, 2),  # Using both unigrams and bigrams
            min_df=5,            # Increased from 2
            max_df=0.95          # Increased from 0.9
        )
        X_train_tfidf = vectorizer.fit_transform(X_train)
        X_test_tfidf = vectorizer.transform(X_test)
        
        # Train model
        print("Training model...")
        model = LogisticRegression(
            C=1.0,              # Regularization strength
            max_iter=1000,      # Increased iterations
            n_jobs=-1,          # Use all available cores
            class_weight='balanced'  # Handle class imbalance
        )
        model.fit(X_train_tfidf, y_train)
        
        # Evaluate model
        print("Evaluating model...")
        predictions = model.predict(X_test_tfidf)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test, predictions, average='weighted')
        
        print("\nModel Evaluation:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, predictions)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('baseline_confusion_matrix.png')
        plt.close()
        
        # Save model and vectorizer
        import joblib
        joblib.dump(model, 'baseline_model.joblib')
        joblib.dump(vectorizer, 'baseline_vectorizer.joblib')
        
    except Exception as e:
        print(f"An error occurred: {e}")
        raise

if __name__ == "__main__":
    main() 