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
import joblib
import json
import os
from typing import List, Dict, Tuple
from collections import Counter
from tqdm import tqdm

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# ----------- PATHS AND DOWNLOAD ------------
data_dir = "./amazon_electronics"
jsonl_gz_url = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Electronics.jsonl.gz"
gz_path = os.path.join(data_dir, "Electronics.jsonl.gz")
jsonl_path = os.path.join(data_dir, "Electronics.jsonl")

os.makedirs(data_dir, exist_ok=True)

if not os.path.exists(jsonl_path):
    import subprocess
    print("Downloading Electronics.jsonl.gz ...")
    subprocess.run(["wget", jsonl_gz_url, "-O", gz_path], check=True)
    print("Decompressing Electronics.jsonl.gz ...")
    subprocess.run(["gunzip", "-f", gz_path], check=True)
    print("Downloaded and decompressed Electronics.jsonl!")
else:
    print(f"Found {jsonl_path}, skipping download.")

# ----------- REST OF YOUR MODEL CODE ------------

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
        tokens = word_tokenize(text.lower())
        indices = [self.vocab.get(token, self.vocab['<UNK>']) for token in tokens]
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
            hidden = self.dropout(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1))
        else:
            hidden = self.dropout(hidden[-1])
        return self.fc(hidden)


def load_data(file_path: str, sample_size: int = 10000000) -> pd.DataFrame:
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(tqdm(f, total=sample_size, desc="Loading data")):
            if i >= sample_size:
                break
            try:
                entry = json.loads(line)
                processed_entry = {
                    'rating': entry.get('rating', 0),
                    'text': entry.get('text', '')
                }
                data.append(processed_entry)
            except json.JSONDecodeError:
                continue
    return pd.DataFrame(data)


def preprocess_text(text):
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
    if not isinstance(rating, (int, float)):
        return 1
    if rating in [1, 2]:
        return 0
    elif rating == 3:
        return 1
    else:
        return 2


def build_vocabulary(texts: List[str], min_freq: int = 2) -> Dict[str, int]:
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


def train_epoch(model, iterator, optimizer, criterion, device):
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


def evaluate(model, iterator, criterion, device):
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


def main():
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")

        print("Loading data...")
        df = load_data(jsonl_path)
        df['sentiment'] = df['rating'].apply(map_ratings_to_sentiment)
        df['processed_text'] = df['text'].apply(preprocess_text)

        X_train, X_test, y_train, y_test = train_test_split(
            df['processed_text'].values, df['sentiment'].values, test_size=0.2, random_state=42
        )

        print("Building vocabulary...")
        vocab = build_vocabulary(X_train)

        print("Creating datasets...")
        train_dataset = TextDataset(X_train, y_train, vocab, max_length=200)
        test_dataset = TextDataset(X_test, y_test, vocab, max_length=200)

        batch_size = 512
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4)

        print("Initializing model...")
        model = LSTMClassifier(
            vocab_size=len(vocab),
            embedding_dim=128,
            hidden_dim=128,
            output_dim=3,
            n_layers=1,
            dropout=0.2,
            pad_idx=vocab['<PAD>']
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
        criterion = nn.CrossEntropyLoss()

        n_epochs = 10
        best_valid_loss = float('inf')
        train_losses = []
        valid_losses = []

        print("Training model...")
        for epoch in range(n_epochs):
            train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
            valid_loss, predictions, labels = evaluate(model, test_loader, criterion, device)
            train_losses.append(train_loss)
            valid_losses.append(valid_loss)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), 'model_LSTM.pt')
                print(f"Saved best model at Epoch {epoch + 1}")

            print(f'Epoch {epoch + 1}: Train Loss={train_loss:.3f}, Valid Loss={valid_loss:.3f}')

        print("Evaluating best model...")
        model.load_state_dict(torch.load('model_LSTM.pt'))
        _, predictions, labels = evaluate(model, test_loader, criterion, device)

        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')

        print("\nFinal Evaluation:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")

        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train Loss')
        plt.plot(valid_losses, label='Valid Loss')
        plt.legend()
        plt.title('Training & Validation Loss')
        plt.xlabel('Epoch')

        plt.subplot(1, 2, 2)
        cm = confusion_matrix(labels, predictions)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')

        plt.tight_layout()
        plt.savefig('training_history.png')

        # SAVE THE VOCABULARY
        with open('LSTM_vectorizer.json', 'w') as f:
            json.dump(vocab, f)
        print("Saved LSTM_vectorizer.json")

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()

