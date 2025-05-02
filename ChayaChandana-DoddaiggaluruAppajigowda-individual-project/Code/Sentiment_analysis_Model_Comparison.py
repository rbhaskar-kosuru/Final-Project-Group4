import os
import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
from transformers import BertTokenizer, BertModel

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Data loading function
def load_data(file_path, sample_size=50000):
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

# Convert ratings to sentiment classes
def map_ratings_to_sentiment(rating):
    if not isinstance(rating, (int, float)):
        return 1  # Neutral for invalid ratings
    if rating in [1, 2]:
        return 0  # Negative
    elif rating == 3:
        return 1  # Neutral
    else:
        return 2  # Positive

# Text preprocessing
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return tokens

# Build vocabulary
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

# Dataset classes
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

class BertDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = ' '.join(self.texts[idx])
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# Model definitions
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_filters, filter_sizes, num_classes, pad_idx, dropout=0.5):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.convs = nn.ModuleList([
            nn.Conv1d(in_channels=embed_dim, out_channels=num_filters, kernel_size=fs)
            for fs in filter_sizes
        ])
        self.fc = nn.Linear(len(filter_sizes) * num_filters, num_classes)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: (batch_size, max_len)
        embedded = self.embedding(x)  # (batch_size, max_len, embed_dim)
        embedded = embedded.permute(0, 2, 1)  # (batch_size, embed_dim, max_len)
        
        conved = [nn.functional.relu(conv(embedded)) for conv in self.convs]
        pooled = [nn.functional.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        
        concat = torch.cat(pooled, dim=1)
        concat = self.dropout(concat)
        return self.fc(concat)

class TextRNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, pad_idx, dropout=0.5):
        super(TextRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: (batch_size, max_len)
        embedded = self.embedding(x)  # (batch_size, max_len, embed_dim)
        
        output, hidden = self.rnn(embedded)
        # hidden: (1, batch_size, hidden_dim)
        
        hidden = hidden.squeeze(0)  # (batch_size, hidden_dim)
        hidden = self.dropout(hidden)
        return self.fc(hidden)

class TextLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_classes, pad_idx, dropout=0.5):
        super(TextLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x: (batch_size, max_len)
        embedded = self.embedding(x)  # (batch_size, max_len, embed_dim)
        
        output, (hidden, cell) = self.lstm(embedded)
        # hidden: (1, batch_size, hidden_dim)
        
        hidden = hidden.squeeze(0)  # (batch_size, hidden_dim)
        hidden = self.dropout(hidden)
        return self.fc(hidden)

class BertClassifier(nn.Module):
    def __init__(self, num_classes, dropout=0.5):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.bert.config.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output  # [CLS] token output
        pooled_output = self.dropout(pooled_output)
        return self.fc(pooled_output)

# Training and evaluation functions
def train_model(model, train_loader, optimizer, criterion, device, model_type='standard'):
    model.train()
    epoch_loss = 0
    epoch_acc = 0
    
    for batch in tqdm(train_loader, desc="Training"):
        optimizer.zero_grad()
        
        if model_type == 'bert':
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        else:
            texts, labels = batch
            texts, labels = texts.to(device), labels.to(device)
            
            outputs = model(texts)
            
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        
        _, predicted = torch.max(outputs, 1)
        epoch_acc += (predicted == labels).sum().item() / len(labels)
        
    return epoch_loss / len(train_loader), epoch_acc / len(train_loader)

def evaluate_model(model, data_loader, criterion, device, model_type='standard'):
    model.eval()
    epoch_loss = 0
    epoch_acc = 0
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            if model_type == 'bert':
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            else:
                texts, labels = batch
                texts, labels = texts.to(device), labels.to(device)
                
                outputs = model(texts)
            
            loss = criterion(outputs, labels)
            
            epoch_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            epoch_acc += (predicted == labels).sum().item() / len(labels)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted'
    )
    
    metrics = {
        'loss': epoch_loss / len(data_loader),
        'accuracy': epoch_acc / len(data_loader),
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': confusion_matrix(all_labels, all_preds)
    }
    
    return metrics

# Main function
def main():
    # Load and preprocess data
    data_dir = "./amazon_electronics"
    jsonl_path = os.path.join(data_dir, "Electronics.jsonl")
    
    # Make sure amazon_electronics/ exists
    os.makedirs(data_dir, exist_ok=True)
    
    df = load_data(jsonl_path, sample_size=100000)  # Adjust sample size as needed
    
    # Create sentiment labels
    df['sentiment'] = df['rating'].apply(map_ratings_to_sentiment)
    
    # Preprocess text
    df['tokens'] = df['text'].apply(preprocess_text)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        df['tokens'].values, df['sentiment'].values, 
        test_size=0.2, random_state=42, 
        stratify=df['sentiment'].values
    )
    
    # Build vocabulary
    vocab = build_vocab(X_train)
    print(f"Vocabulary size: {len(vocab)}")
    
    # Standard datasets (for CNN, RNN, LSTM)
    train_dataset = TextDataset(X_train, y_train, vocab)
    test_dataset = TextDataset(X_test, y_test, vocab)
    
    # BERT dataset
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert_train_dataset = BertDataset(X_train, y_train, tokenizer)
    bert_test_dataset = BertDataset(X_test, y_test, tokenizer)
    
    # DataLoaders
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    bert_train_loader = DataLoader(bert_train_dataset, batch_size=32, shuffle=True)
    bert_test_loader = DataLoader(bert_test_dataset, batch_size=32)
    
    # Model parameters
    embed_dim = 128
    hidden_dim = 128
    num_filters = 100
    filter_sizes = [3, 4, 5]
    num_classes = 3  # Negative, Neutral, Positive
    dropout = 0.5
    
    # Models
    models = {
        'CNN': {
            'model': TextCNN(
                vocab_size=len(vocab),
                embed_dim=embed_dim,
                num_filters=num_filters,
                filter_sizes=filter_sizes,
                num_classes=num_classes,
                pad_idx=vocab['<PAD>'],
                dropout=dropout
            ).to(device),
            'loader': (train_loader, test_loader),
            'type': 'standard'
        },
        'RNN': {
            'model': TextRNN(
                vocab_size=len(vocab),
                embed_dim=embed_dim,
                hidden_dim=hidden_dim,
                num_classes=num_classes,
                pad_idx=vocab['<PAD>'],
                dropout=dropout
            ).to(device),
            'loader': (train_loader, test_loader),
            'type': 'standard'
        },
        'LSTM': {
            'model': TextLSTM(
                vocab_size=len(vocab),
                embed_dim=embed_dim,
                hidden_dim=hidden_dim,
                num_classes=num_classes,
                pad_idx=vocab['<PAD>'],
                dropout=dropout
            ).to(device),
            'loader': (train_loader, test_loader),
            'type': 'standard'
        },
        'BERT': {
            'model': BertClassifier(
                num_classes=num_classes,
                dropout=dropout
            ).to(device),
            'loader': (bert_train_loader, bert_test_loader),
            'type': 'bert'
        }
    }
    
    # Training setup
    num_epochs = 5
    criterion = nn.CrossEntropyLoss()
    
    results = {}
    
    # Train and evaluate each model
    for model_name, model_info in models.items():
        print(f"\n{'='*50}")
        print(f"Training {model_name} model")
        print(f"{'='*50}")
        
        model = model_info['model']
        train_loader, test_loader = model_info['loader']
        model_type = model_info['type']
        
        # Optimizer - use different learning rates for BERT vs other models
        if model_name == 'BERT':
            optimizer = optim.Adam(model.parameters(), lr=2e-5)
        else:
            optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        train_losses, train_accs = [], []
        test_metrics = []
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            
            # Training
            train_loss, train_acc = train_model(
                model, train_loader, optimizer, criterion, device, model_type
            )
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            
            # Evaluation
            metrics = evaluate_model(
                model, test_loader, criterion, device, model_type
            )
            test_metrics.append(metrics)
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Test Loss: {metrics['loss']:.4f}, Test Acc: {metrics['accuracy']:.4f}")
            print(f"Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, F1: {metrics['f1']:.4f}")
        
        # Save final results
        results[model_name] = {
            'train_losses': train_losses,
            'train_accs': train_accs,
            'test_metrics': test_metrics,
            'final_metrics': test_metrics[-1]
        }
        
        # Save model
        torch.save(model.state_dict(), f'best_{model_name.lower()}_model.pt')
    
    # Visualization of results
    # 1. Final performance comparison
    fig, ax = plt.subplots(figsize=(12, 8))
    
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    x = np.arange(len(metrics))
    width = 0.2
    multiplier = 0
    
    for model_name, model_results in results.items():
        final_metrics = model_results['final_metrics']
        metric_values = [final_metrics[m] for m in metrics]
        
        offset = width * multiplier
        rects = ax.bar(x + offset, metric_values, width, label=model_name)
        multiplier += 1
    
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Scores')
    ax.set_title('Model Comparison - Final Performance')
    ax.set_xticks(x + width * (len(results) - 1) / 2)
    ax.set_xticklabels(metrics)
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('model_comparison_final_metrics.png')
    
    # 2. Training loss curves
    plt.figure(figsize=(10, 6))
    for model_name, model_results in results.items():
        plt.plot(model_results['train_losses'], label=f"{model_name} Train Loss")
    
    plt.title('Training Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig('model_comparison_loss_curves.png')
    
    # 3. Test accuracy curves
    plt.figure(figsize=(10, 6))
    for model_name, model_results in results.items():
        acc_values = [metrics['accuracy'] for metrics in model_results['test_metrics']]
        plt.plot(acc_values, label=f"{model_name} Test Accuracy")
    
    plt.title('Test Accuracy Curves')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.savefig('model_comparison_accuracy_curves.png')
    
    # 4. Confusion matrices (Last epoch)
    for model_name, model_results in results.items():
        cm = model_results['test_metrics'][-1]['confusion_matrix']
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Negative', 'Neutral', 'Positive'],
                   yticklabels=['Negative', 'Neutral', 'Positive'])
        plt.title(f'{model_name} - Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.savefig(f'{model_name.lower()}_confusion_matrix.png')
    
    # Print final results summary
    print("\n===== Final Results Summary =====")
    for model_name, model_results in results.items():
        final = model_results['final_metrics']
        print(f"\n{model_name} Model:")
        print(f"Accuracy: {final['accuracy']:.4f}")
        print(f"Precision: {final['precision']:.4f}")
        print(f"Recall: {final['recall']:.4f}")
        print(f"F1 Score: {final['f1']:.4f}")

if __name__ == "__main__":
    main()
