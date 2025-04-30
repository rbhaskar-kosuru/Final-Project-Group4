# Final-Project-Group4

# Sentimental Analysis of Amazon Electronics Reviews

## Team Members:

Chaya Chandana Doddaiggaluru Appajigowda

Ramana Bhaskar Kosuru

Adam Stuhltrager


## Overview of the Project:

The project focuses on advanced Natural Language Processing (NLP) techniques to develop a sentiment analysis system for Amazon Electronics product reviews. We selected this problem because understanding customer sentiment is critical for both manufacturers and retailers in the rapidly evolving electronics market. By automatically classifying reviews into negative, neutral, and positive sentiments, businesses can gain actionable insights about product reception without manual analysis of thousands of reviews.
Electronics reviews are particularly interesting for NLP analysis because they often contain technical jargon, complex opinions about multiple product features, and comparative references to other products. This complexity makes them an excellent candidate for applying advanced NLP techniques beyond simple binary classification.


## Dataset Source:

Amazon Electronics Reviews - https://amazon-reviews-2023.github.io. This comprehensive dataset contains millions of product reviews with various fields including:
•	Star ratings (1-5)

•	Review text

•	Review titles

•	Helpful votes

•	Verified purchase indicators

For sentiment classification:

•	Negative: 1-2 stars

•	Neutral: 3 stars

•	Positive: 4-5 stars

## Models Used:

1.	Baseline Model
2.	CNN Text Model
3.	RNN Model
4.	LSTM Model
   
## Framework and Libraries:

•	PyTorch

•	Scikit-learn

•	NLTK

•	Pandas/NumPy

•	Matplotlib/Seaborn

•	tqdm

## Code Structure:

1.	Text Preprocessing: Cleaning the review text by converting to lowercase, removing punctuation and numbers, and eliminating stopwords.
2.	Tokenization: Breaking the text into words or tokens for model processing.
3.	Vocabulary Building: Creating a mapping from tokens to indices with handling for out-of-vocabulary words.
4.	Feature Extraction:
   
   •	For baseline: TF-IDF vectorization with n-grams4.
   •	For deep learning models: Learning word embeddings from scratch.

5.	Sentiment Classification: Training models to predict the three sentiment classes.





