# Instructions to run main scripts:

This repository contains two main scripts: 

- **`LSTM_train.py`**  
  Trains an LSTM-based sentiment classifier and saves:
  - Model weights (`model_LSTM.pt`)  
  - Vocabulary & tokenization data (`LSTM_vectorizer.json`)

- **`model_predict.py`**  
  Loads the trained model and vocabulary to perform inference on new reviews.

---

## Prerequisites

1. **Python 3.7+**  
2. Install required packages:
torch, numpypandas, nltk, scikit-learn


- Run **`LSTM_train.py`**   first 
- Make sure model_LSTM.pt and LSTM_vectorizer.json are in the same directory as model_predict.py.
- Run **`model_predict.py`**  
- Follow the instructions to enter the review. Enter 'Submit' at the end of the review and click Enter. 

---
- To run streamlit python script install streamlit and run the in the terminal.

---

**NOTE:** All the other code files are the work we have done.
