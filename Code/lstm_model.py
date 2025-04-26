import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import json

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

def load_data(file_path, sample_size=100000):
    """Load and sample the dataset."""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= sample_size:
                break
            data.append(json.loads(line))
    return pd.DataFrame(data)

def preprocess_text(text):
    """Clean and preprocess text."""
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
    if rating in [1, 2]:
        return 0  # Negative
    elif rating == 3:
        return 1  # Neutral
    else:
        return 2  # Positive

def create_model(vocab_size, max_length, embedding_dim=100):
    """Create and compile the LSTM model."""
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length),
        LSTM(128, dropout=0.2, recurrent_dropout=0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(3, activation='softmax')
    ])
    
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def main():
    # Load and preprocess data
    print("Loading data...")
    df = load_data('../Electronics.jsonl')
    
    # Map ratings to sentiment labels
    df['sentiment'] = df['star_rating'].apply(map_ratings_to_sentiment)
    
    # Preprocess review text
    print("Preprocessing text...")
    df['processed_text'] = df['review_body'].apply(preprocess_text)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df['processed_text'], df['sentiment'], test_size=0.2, random_state=42
    )
    
    # Tokenize and pad sequences
    print("Tokenizing and padding sequences...")
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X_train)
    
    vocab_size = len(tokenizer.word_index) + 1
    max_length = 300
    
    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_test_seq = tokenizer.texts_to_sequences(X_test)
    
    X_train_pad = pad_sequences(X_train_seq, maxlen=max_length, padding='post')
    X_test_pad = pad_sequences(X_test_seq, maxlen=max_length, padding='post')
    
    # Create and train model
    print("Creating and training model...")
    model = create_model(vocab_size, max_length)
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=3,
        restore_best_weights=True
    )
    
    history = model.fit(
        X_train_pad,
        y_train,
        epochs=10,
        batch_size=64,
        validation_split=0.1,
        callbacks=[early_stopping],
        verbose=1
    )
    
    # Evaluate model
    print("Evaluating model...")
    y_pred = model.predict(X_test_pad)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    accuracy = accuracy_score(y_test, y_pred_classes)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred_classes, average='weighted')
    
    print("\nModel Evaluation:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()
    
    # Plot confusion matrix
    cm = confusion_matrix(y_test, y_pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png')
    plt.close()

if __name__ == "__main__":
    main() 