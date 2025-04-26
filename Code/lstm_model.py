import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import json
import os
from typing import List, Dict, Tuple
from collections import Counter
import optuna
from tqdm import tqdm

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

class TextDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], vocab: Dict[str, int], max_length: int):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Convert text to indices
        tokens = word_tokenize(text.lower())
        indices = [self.vocab.get(token, self.vocab['<UNK>']) for token in tokens]
        
        # Pad or truncate
        if len(indices) < self.max_length:
            indices = indices + [self.vocab['<PAD>']] * (self.max_length - len(indices))
        else:
            indices = indices[:self.max_length]
        
        return torch.tensor(indices, dtype=torch.long), torch.tensor(label, dtype=torch.long)

class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, output_dim: int, n_layers: int,
                 dropout: float, pad_idx: int, bidirectional: bool = True):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=dropout, 
                           batch_first=True, bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
        embedded = self.dropout(self.embedding(text))
        output, (hidden, cell) = self.lstm(embedded)
        if self.lstm.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1))
        else:
            hidden = self.dropout(hidden[-1])
        return self.fc(hidden)

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

def build_vocabulary(texts: List[str], min_freq: int = 2) -> Dict[str, int]:
    """Build vocabulary from texts."""
    word_counts = Counter()
    for text in texts:
        tokens = word_tokenize(text.lower())
        word_counts.update(tokens)
    
    vocab = {'<PAD>': 0, '<UNK>': 1}
    idx = 2
    for word, count in word_counts.items():
        if count >= min_freq:
            vocab[word] = idx
            idx += 1
    
    return vocab

def train_epoch(model: nn.Module, iterator: DataLoader, optimizer: torch.optim.Optimizer,
                criterion: nn.Module, device: torch.device) -> float:
    """Train the model for one epoch."""
    model.train()
    epoch_loss = 0
    
    for batch in tqdm(iterator, desc="Training", leave=False):
        text, labels = batch
        text, labels = text.to(device), labels.to(device)
        
        optimizer.zero_grad()
        predictions = model(text)
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
    
    return epoch_loss / len(iterator)

def evaluate(model: nn.Module, iterator: DataLoader, criterion: nn.Module,
            device: torch.device) -> Tuple[float, np.ndarray, np.ndarray]:
    """Evaluate the model."""
    model.eval()
    epoch_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(iterator, desc="Evaluating", leave=False):
            text, labels = batch
            text, labels = text.to(device), labels.to(device)
            
            predictions = model(text)
            loss = criterion(predictions, labels)
            
            epoch_loss += loss.item()
            
            preds = torch.argmax(predictions, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    
    return epoch_loss / len(iterator), np.array(all_preds), np.array(all_labels)

def objective(trial):
    # Hyperparameters to tune
    embedding_dim = trial.suggest_int('embedding_dim', 50, 300)
    hidden_dim = trial.suggest_int('hidden_dim', 64, 256)
    n_layers = trial.suggest_int('n_layers', 1, 3)
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
    
    # Load and preprocess data
    df = load_data('Electronics.jsonl')
    df['sentiment'] = df['rating'].apply(map_ratings_to_sentiment)
    df['processed_text'] = df['text'].apply(preprocess_text)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df['processed_text'].values, df['sentiment'].values,
        test_size=0.2, random_state=42
    )
    
    # Build vocabulary
    vocab = build_vocabulary(X_train)
    
    # Create datasets
    train_dataset = TextDataset(X_train, y_train, vocab, max_length=300)
    test_dataset = TextDataset(X_test, y_test, vocab, max_length=300)
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LSTMClassifier(
        vocab_size=len(vocab),
        embedding_dim=embedding_dim,
        hidden_dim=hidden_dim,
        output_dim=3,
        n_layers=n_layers,
        dropout=dropout,
        pad_idx=vocab['<PAD>']
    ).to(device)
    
    # Training parameters
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    best_valid_loss = float('inf')
    patience = 3
    patience_counter = 0
    
    for epoch in range(10):  # Reduced epochs for hyperparameter tuning
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        valid_loss, predictions, labels = evaluate(model, test_loader, criterion, device)
        
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break
    
    return best_valid_loss

def main():
    try:
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # Hyperparameter optimization
        print("Starting hyperparameter optimization...")
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=20)  # Number of trials
        
        print("Best hyperparameters:")
        for key, value in study.best_params.items():
            print(f"{key}: {value}")
        
        # Train final model with best hyperparameters
        print("\nTraining final model with best hyperparameters...")
        best_params = study.best_params
        
        # Load and preprocess data
        print("Loading data...")
        df = load_data('Electronics.jsonl')
        df['sentiment'] = df['rating'].apply(map_ratings_to_sentiment)
        df['processed_text'] = df['text'].apply(preprocess_text)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            df['processed_text'].values, df['sentiment'].values,
            test_size=0.2, random_state=42
        )
        
        # Build vocabulary
        print("Building vocabulary...")
        vocab = build_vocabulary(X_train)
        
        # Create datasets
        print("Creating datasets...")
        train_dataset = TextDataset(X_train, y_train, vocab, max_length=300)
        test_dataset = TextDataset(X_test, y_test, vocab, max_length=300)
        
        # Create data loaders with larger batch size
        batch_size = 256  # Increased batch size for better performance
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        # Initialize model with best hyperparameters
        print("Initializing model...")
        model = LSTMClassifier(
            vocab_size=len(vocab),
            embedding_dim=best_params['embedding_dim'],
            hidden_dim=best_params['hidden_dim'],
            output_dim=3,
            n_layers=best_params['n_layers'],
            dropout=best_params['dropout'],
            pad_idx=vocab['<PAD>']
        ).to(device)
        
        # Training parameters
        optimizer = torch.optim.Adam(model.parameters(), lr=best_params['learning_rate'])
        criterion = nn.CrossEntropyLoss()
        n_epochs = 15  # Increased epochs
        
        # Training loop
        print("Training model...")
        best_valid_loss = float('inf')
        train_losses = []
        valid_losses = []
        
        for epoch in range(n_epochs):
            train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
            valid_loss, predictions, labels = evaluate(model, test_loader, criterion, device)
            
            train_losses.append(train_loss)
            valid_losses.append(valid_loss)
            
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), 'best_model.pt')
            
            print(f'Epoch: {epoch+1:02}')
            print(f'\tTrain Loss: {train_loss:.3f}')
            print(f'\tValid Loss: {valid_loss:.3f}')
        
        # Load best model and evaluate
        model.load_state_dict(torch.load('best_model.pt'))
        _, predictions, labels = evaluate(model, test_loader, criterion, device)
        
        # Calculate metrics
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
        
        print("\nModel Evaluation:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        
        # Plot training history
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Training Loss')
        plt.plot(valid_losses, label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot confusion matrix
        cm = confusion_matrix(labels, predictions)
        plt.subplot(1, 2, 2)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        plt.tight_layout()
        plt.savefig('training_history.png')
        plt.close()
        
    except Exception as e:
        print(f"An error occurred: {e}")
        raise

if __name__ == "__main__":
    main() 