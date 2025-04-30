import torch
import torch.nn as nn
import json
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Download NLTK data (only the first time)
nltk.download('punkt')
nltk.download('stopwords')

# Label map
label_map = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}

# Constants
MAX_LENGTH = 1000

# Preprocessing function
def preprocess_text(text):
    if not isinstance(text, str) or not text.strip():
        return []
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\d+', '', text)
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    return [word for word in tokens if word not in stop_words]

# Load vocabulary
with open('LSTM_vectorizer.json', 'r') as f:
    vocab = json.load(f)

PAD_IDX = vocab['<PAD>']
UNK_IDX = vocab['<UNK>']

# Convert tokens to padded tensor
def tokens_to_tensor(tokens, vocab, max_length):
    indices = [vocab.get(token, UNK_IDX) for token in tokens]
    if len(indices) < max_length:
        indices += [PAD_IDX] * (max_length - len(indices))
    else:
        indices = indices[:max_length]
    return torch.tensor([indices], dtype=torch.long)

# Model class (must match training)
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, dropout, pad_idx, bidirectional=True):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=dropout,
                            batch_first=True, bidirectional=bidirectional)
        self.fc = nn.Linear(hidden_dim * 2 if bidirectional else hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        embedded = self.dropout(self.embedding(text))
        output, (hidden, _) = self.lstm(embedded)
        if self.lstm.bidirectional:
            hidden = self.dropout(torch.cat((hidden[-2], hidden[-1]), dim=1))
        else:
            hidden = self.dropout(hidden[-1])
        return self.fc(hidden)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMClassifier(
    vocab_size=len(vocab),
    embedding_dim=128,
    hidden_dim=128,
    output_dim=3,
    n_layers=1,
    dropout=0.2,
    pad_idx=PAD_IDX
).to(device)

model.load_state_dict(torch.load('model_LSTM.pt', map_location=device))
model.eval()

# Prediction function
def predict_sentiment(text):
    tokens = preprocess_text(text)
    if not tokens:
        return "Invalid input: review is too short or empty."
    input_tensor = tokens_to_tensor(tokens, vocab, MAX_LENGTH).to(device)
    with torch.no_grad():
        logits = model(input_tensor)
        prediction = torch.argmax(logits, dim=1).item()
    return label_map[prediction]

# CLI interface (manual submit)
if __name__ == "__main__":
    print(80*"=")
    print("\nSentiment Predictor Ready (LSTM Model)\n")
    print("Write your review when prompted.")
    print("After writing, type 'Submit' to analyze sentiment or 'Exit' to quit.\n")

    while True:
        print("Write a review (Submit when you are done):")
        lines = []
        while True:
            line = input()
            if line.strip().lower() == "submit":
                break
            if line.strip().lower() == "exit":
                print("Exiting Sentiment Predictor.")
                exit()
            lines.append(line)

        review = "\n".join(lines).strip()
        if not review:
            print("Please enter a non-empty review before submitting.\n")
            continue

        sentiment = predict_sentiment(review)
        print(f"\nPredicted Sentiment: {sentiment}\n")
