import os
import subprocess
import json
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk

# Paths
data_dir = "./amazon_electronics"
jsonl_gz_url = "https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/raw/review_categories/Electronics.jsonl.gz"
gz_path = os.path.join(data_dir, "Electronics.jsonl.gz")
jsonl_path = os.path.join(data_dir, "Electronics.jsonl")

# Make sure amazon_electronics/ exists
os.makedirs(data_dir, exist_ok=True)

# Download Electronics.jsonl.gz if missing
if not os.path.exists(jsonl_path):
    print("Downloading Electronics.jsonl.gz ...")
    subprocess.run(["wget", jsonl_gz_url, "-O", gz_path], check=True)
    print("Decompressing Electronics.jsonl.gz ...")
    subprocess.run(["gunzip", "-f", gz_path], check=True)
    print("Downloaded and decompressed Electronics.jsonl!")
else:
    print(f"Found {jsonl_path}, skipping download.")

# Load Data
def load_data(file_path: str, sample_size: int = 10000000) -> pd.DataFrame:
    print(f"Loading data from: {file_path}")
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(tqdm(f, total=sample_size, desc="Loading data")):
            if i >= sample_size:
                break
            try:
                entry = json.loads(line)
                processed_entry = {
                    'rating': entry.get('rating', 0),
                    'text': entry.get('text', ''),
                }
                data.append(processed_entry)
            except json.JSONDecodeError:
                continue
    print(f"Successfully loaded {len(data)} samples")
    return pd.DataFrame(data)

# Map rating to 3 sentiment classes
def map_ratings_to_sentiment(rating):
    if not isinstance(rating, (int, float)):
        return 1
    if rating in [1, 2]:
        return 0
    elif rating == 3:
        return 1
    else:
        return 2

# Text Preprocessing
nltk.download('punkt')
nltk.download('stopwords')

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return tokens

# Build Vocabulary
def build_vocab(texts, min_freq=2):
    from collections import Counter
    counter = Counter()
    for text in texts:
        counter.update(text)
    vocab = {'<PAD>': 0, '<UNK>': 1}
    idx = 2
    for word, freq in counter.items():
        if freq >= min_freq:
            vocab[word] = idx
            idx += 1
    return vocab

# Dataset Class
class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=256):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        tokens = self.texts[idx]
        ids = [self.vocab.get(token, self.vocab['<UNK>']) for token in tokens]
        if len(ids) < self.max_len:
            ids += [self.vocab['<PAD>']] * (self.max_len - len(ids))
        else:
            ids = ids[:self.max_len]
        return torch.tensor(ids), torch.tensor(self.labels[idx])

# RNN Model for Text Classification
class TextRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, pad_idx, num_layers=2, bidirectional=True, dropout=0.5):
        super(TextRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.rnn = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=bidirectional,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: (batch_size, max_len)
        embedded = self.embedding(x)  # (batch_size, max_len, embed_dim)
        
        # LSTM output: (batch_size, max_len, hidden_dim * num_directions)
        # LSTM hidden: (num_layers * num_directions, batch_size, hidden_dim)
        output, (hidden, cell) = self.rnn(embedded)
        
        # If bidirectional, concatenate the final forward and backward hidden states
        if self.rnn.bidirectional:
            hidden = torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)
        else:
            hidden = hidden[-1, :, :]
            
        hidden = self.dropout(hidden)
        return self.fc(hidden)

# Main Training Script
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    df = load_data(jsonl_path)
    df['sentiment'] = df['rating'].apply(map_ratings_to_sentiment)
    df['tokens'] = df['text'].apply(preprocess_text)

    X_train, X_test, y_train, y_test = train_test_split(
        df['tokens'].values, df['sentiment'].values, test_size=0.2, random_state=42, stratify=df['sentiment'].values
    )

    vocab = build_vocab(X_train)

    train_dataset = TextDataset(X_train, y_train, vocab)
    test_dataset = TextDataset(X_test, y_test, vocab)

    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=512)

    model = TextRNN(
        vocab_size=len(vocab), 
        embed_dim=128, 
        hidden_dim=128, 
        num_classes=3, 
        pad_idx=vocab['<PAD>'],
        num_layers=2,
        bidirectional=True,
        dropout=0.5
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    n_epochs = 10
    best_loss = float('inf')

    train_losses, valid_losses = [], []

    for epoch in range(n_epochs):
        model.train()
        train_loss = 0
        for inputs, labels in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        model.eval()
        valid_loss = 0
        preds, true_labels = [], []

        with torch.no_grad():
            for inputs, labels in tqdm(test_loader, desc=f"Validation Epoch {epoch+1}"):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                valid_loss += loss.item()
                preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
                true_labels.extend(labels.cpu().numpy())

        avg_valid_loss = valid_loss / len(test_loader)
        valid_losses.append(avg_valid_loss)

        if avg_valid_loss < best_loss:
            best_loss = avg_valid_loss
            torch.save(model.state_dict(), 'best_textrnn_model.pt')

        print(f'Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Valid Loss={avg_valid_loss:.4f}')

    # Final evaluation
    model.load_state_dict(torch.load('best_textrnn_model.pt'))
    accuracy = accuracy_score(true_labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(true_labels, preds, average='weighted')

    print("\nFinal Model Evaluation:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Save Metrics Bar Chart
    metrics = {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'F1 Score': f1}
    plt.figure(figsize=(8,6))
    plt.bar(metrics.keys(), metrics.values(), color=['skyblue', 'lightgreen', 'salmon', 'orange'])
    plt.ylim(0, 1)
    plt.ylabel('Score')
    plt.title('Final Model Evaluation Metrics')
    for i, (metric, value) in enumerate(metrics.items()):
        plt.text(i, value + 0.02, f"{value:.2f}", ha='center', va='bottom', fontsize=10)
    plt.grid(axis='y')
    plt.savefig('final_evaluation_metrics_TextRNN_3class.png')
    plt.close()

    # Save Loss Curve
    plt.figure(figsize=(8,6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(valid_losses, label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('TextRNN_loss_curve_3class.png')
    plt.close()

    # Save Confusion Matrix
    cm = confusion_matrix(true_labels, preds)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Negative', 'Neutral', 'Positive'],
                yticklabels=['Negative', 'Neutral', 'Positive'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig('TextRNN_confusion_matrix_3class.png')
    plt.close()

if __name__ == "__main__":
    main()
