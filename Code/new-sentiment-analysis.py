import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import BertTokenizer, BertModel, BertForSequenceClassification
from transformers import get_linear_schedule_with_warmup
from torch.cuda.amp import autocast, GradScaler
import re
import nltk
import gc
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Set PyTorch memory allocation configuration
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Download NLTK resources if not already downloaded
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

# Set random seeds for reproducibility
SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
np.random.seed(SEED)
torch.backends.cudnn.deterministic = True

def print_gpu_memory_stats():
    """Print detailed GPU memory statistics."""
    if torch.cuda.is_available():
        print("\n----- GPU Memory Stats -----")
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"Total GPU Memory: {total_memory:.2f} GiB")
        allocated_memory = torch.cuda.memory_allocated() / 1024**3
        print(f"Allocated GPU Memory: {allocated_memory:.2f} GiB")
        reserved_memory = torch.cuda.memory_reserved() / 1024**3
        print(f"Reserved GPU Memory: {reserved_memory:.2f} GiB")
        free_memory = total_memory - allocated_memory
        print(f"Free GPU Memory: {free_memory:.2f} GiB")
        utilization = (allocated_memory / total_memory) * 100
        print(f"Memory Utilization: {utilization:.2f}%")
        print("---------------------------\n")
    else:
        print("CUDA not available")

def load_data(file_path, max_rows=None):
    """Load data from a JSONL file with robust error handling."""
    data = []
    error_count = 0
    
    try:
        with open(file_path, 'r', encoding='utf-8-sig') as f:
            for i, line in enumerate(f):
                if max_rows is not None and i >= max_rows:
                    break
                
                line = line.strip()
                if not line:  # Skip empty lines
                    continue
                    
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    error_count += 1
                    if error_count <= 5:  # Only show first few errors
                        print(f"Error parsing line {i+1}: {e}")
                        print(f"Problematic line: {line[:100]}...")  # Print start of the line
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        print("Please make sure the file exists and the path is correct.")
        raise
    except Exception as e:
        print(f"Unexpected error reading file: {e}")
        raise
    
    if error_count > 0:
        print(f"Total of {error_count} lines with JSON parsing errors were skipped.")
    
    if not data:
        print("Warning: No valid data was loaded from the file.")
        # Create a minimal dataframe with required columns to avoid errors
        return pd.DataFrame({"text": [], "rating": []})
    
    return pd.DataFrame(data)

def map_ratings_to_multiclass_sentiment(rating):
    """Map numerical ratings to multi-class sentiment (0-5)."""
    if not isinstance(rating, (int, float)):
        return 3  # Default to middle rating for missing values
    
    # Convert rating to integer in range 0-5
    # Assuming original ratings are 1-5, subtract 1 to get 0-4
    return int(rating) - 1 if 1 <= rating <= 5 else min(max(int(rating), 0), 4)

class BERTDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

def train_epoch(model, iterator, optimizer, criterion, device, scheduler=None, accumulation_steps=4):
    """Train the model for one epoch with gradient accumulation and mixed precision."""
    model.train()
    epoch_loss = 0
    steps = 0
    
    # Initialize gradient scaler for mixed precision training
    scaler = GradScaler()
    
    for i, batch in enumerate(tqdm(iterator, desc="Training", leave=False)):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        # Clear memory occasionally
        if i % 10 == 0:
            torch.cuda.empty_cache()
        
        # Forward pass with mixed precision
        with autocast():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss / accumulation_steps
        
        # Backward pass with gradient scaling
        scaler.scale(loss).backward()
        
        # Update weights every accumulation_steps
        if (i + 1) % accumulation_steps == 0 or (i + 1 == len(iterator)):
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            if scheduler is not None:
                scheduler.step()
                
            optimizer.zero_grad()
            steps += 1
        
        epoch_loss += loss.item() * accumulation_steps
        
        # Free memory
        del input_ids, attention_mask, labels, outputs, loss
        
        # Print memory stats occasionally
        if i % 100 == 0:
            print_gpu_memory_stats()
    
    return epoch_loss / steps

def evaluate(model, iterator, device):
    """Evaluate the model."""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(iterator, desc="Evaluating", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            # Clear memory before forward pass
            torch.cuda.empty_cache()
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
            
            # Free memory
            del input_ids, attention_mask, labels, outputs, logits
    
    return np.array(all_preds), np.array(all_labels)

def sample_balanced_dataset(df, max_samples=100000):
    """Sample a balanced subset of the data to reduce memory requirements."""
    print(f"Sampling balanced dataset with max {max_samples} samples...")
    
    # Get counts for each sentiment class
    class_counts = df['sentiment'].value_counts()
    min_count = min(class_counts.values)
    samples_per_class = min(min_count, max_samples // len(class_counts))
    
    # Sample equally from each class
    sampled_data = []
    for sentiment in class_counts.index:
        class_data = df[df['sentiment'] == sentiment]
        sampled_class = class_data.sample(samples_per_class, random_state=SEED)
        sampled_data.append(sampled_class)
    
    # Combine and shuffle
    sampled_df = pd.concat(sampled_data).sample(frac=1, random_state=SEED).reset_index(drop=True)
    print(f"Sampled dataset size: {len(sampled_df)}")
    print(f"Sentiment distribution: {sampled_df['sentiment'].value_counts().to_dict()}")
    
    return sampled_df

def main():
    try:
        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # Initial memory stats
        print_gpu_memory_stats()
        
        # Load and preprocess data
        print("Loading data...")
        try:
            # Load a limited number of rows for initial testing
            df = load_data('Electronics.jsonl', max_rows=1000000)  # Start with 1M rows
            if df.empty or len(df) < 100:
                print("Not enough valid data in file, creating sample data instead.")
                df = create_sample_data()
        except Exception as e:
            print(f"Error loading data: {e}")
            print("Creating sample data instead.")
            df = create_sample_data()
        
        # Check if required columns exist
        if 'rating' not in df.columns:
            print("Warning: 'rating' column not found in data. Adding default ratings.")
            df['rating'] = 3  # Default to neutral
        
        if 'text' not in df.columns:
            print("Warning: 'text' column not found in data. Adding empty text.")
            df['text'] = ""
            
        # Map ratings to multi-class sentiment (0-4)
        df['sentiment'] = df['rating'].apply(map_ratings_to_multiclass_sentiment)
        
        # For BERT, we can use the original text with minimal preprocessing
        df['processed_text'] = df['text'].apply(lambda x: x if isinstance(x, str) else "")
        
        # Print data statistics
        print(f"Data loaded: {len(df)} samples")
        print(f"Sentiment distribution: {df['sentiment'].value_counts().to_dict()}")
        
        # Sample a balanced subset to reduce memory requirements
        df = sample_balanced_dataset(df, max_samples=50000)
        
        # Memory check after data loading
        print_gpu_memory_stats()
        
        # Split data
        print("Splitting data...")
        X_train, X_test, y_train, y_test = train_test_split(
            df['processed_text'].values, df['sentiment'].values,
            test_size=0.2, random_state=SEED, stratify=df['sentiment'].values
        )
        
        # Free up memory
        del df
        gc.collect()
        torch.cuda.empty_cache()
        print_gpu_memory_stats()
        
        # Load BERT model pre-trained for sequence classification
        print("Loading BERT model and tokenizer...")
        model_name = "bert-base-uncased"
        tokenizer = BertTokenizer.from_pretrained(model_name)
        
        # Create datasets
        print("Creating datasets...")
        max_length = 128  # Reduce sequence length to save memory
        train_dataset = BERTDataset(X_train, y_train, tokenizer, max_length)
        test_dataset = BERTDataset(X_test, y_test, tokenizer, max_length)
        
        # Create data loaders with smaller batch size
        batch_size = 16  # Smaller batch size for BERT
        accumulation_steps = 4  # Accumulate gradients to simulate larger batch
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size)
        
        # Initialize model for sequence classification with 5 classes (0-4)
        print("Initializing BERT model for sequence classification...")
        num_labels = 5  # For ratings 0-4
        model = BertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        ).to(device)
        
        # Memory check after model loading
        print_gpu_memory_stats()
        
        # Training parameters
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
        n_epochs = 3  # Fewer epochs for large models
        
        # Create learning rate scheduler
        total_steps = len(train_loader) * n_epochs // accumulation_steps
        warmup_steps = int(total_steps * 0.1)  # 10% of total steps
        scheduler = get_linear_schedule_with_warmup(
            optimizer, 
            num_warmup_steps=warmup_steps, 
            num_training_steps=total_steps
        )
        
        # Training loop
        print("Starting training...")
        train_losses = []
        train_accs = []
        val_accs = []
        
        for epoch in range(n_epochs):
            print(f"Epoch {epoch+1}/{n_epochs}")
            
            # Train
            train_loss = train_epoch(
                model, 
                train_loader, 
                optimizer, 
                None,  # No criterion needed, loss is calculated in the model
                device, 
                scheduler, 
                accumulation_steps
            )
            train_losses.append(train_loss)
            
            # Evaluate on training data (use a subset to save memory)
            print("Evaluating on training data...")
            train_subset = torch.utils.data.Subset(
                train_dataset, 
                indices=np.random.choice(len(train_dataset), min(1000, len(train_dataset)), replace=False)
            )
            train_subset_loader = DataLoader(train_subset, batch_size=batch_size)
            train_preds, train_labels = evaluate(model, train_subset_loader, device)
            train_acc = accuracy_score(train_labels, train_preds)
            train_accs.append(train_acc)
            
            # Evaluate on test data
            print("Evaluating on test data...")
            val_preds, val_labels = evaluate(model, test_loader, device)
            val_acc = accuracy_score(val_labels, val_preds)
            val_accs.append(val_acc)
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Acc: {val_acc:.4f}")
            
            # Save model after each epoch
            torch.save(model.state_dict(), f'sentiment_model_epoch_{epoch+1}.pt')
            print(f"Model saved for epoch {epoch+1}")
        
        # Final evaluation
        print("\nFinal Evaluation:")
        test_preds, test_labels = evaluate(model, test_loader, device)
        test_acc = accuracy_score(test_labels, test_preds)
        print(f"Test Accuracy: {test_acc:.4f}")
        
        # Classification report
        print("\nClassification Report:")
        target_names = ['Rating 0', 'Rating 1', 'Rating 2', 'Rating 3', 'Rating 4']
        print(classification_report(test_labels, test_preds, target_names=target_names))
        
        # Plot confusion matrix
        cm = confusion_matrix(test_labels, test_preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.savefig('confusion_matrix.png')
        plt.close()
        
        # Save final model
        torch.save(model.state_dict(), 'multiclass_sentiment_model.pt')
        print("Final model saved successfully.")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
