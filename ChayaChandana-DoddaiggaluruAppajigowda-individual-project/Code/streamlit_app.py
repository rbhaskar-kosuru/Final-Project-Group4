import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import nltk
#from model_predict import predict_sentiment
from model_predict import predict_sentiment, generate_response  

os.system('pip3 install streamlit')
          
# Setup page configuration
st.set_page_config(
    page_title="Amazon Review Sentiment Analysis",
    page_icon="📊",
    layout="wide"
)

# Ensure NLTK data is available
@st.cache_resource
def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('punkt')
        nltk.download('stopwords')

# Download required NLTK data
download_nltk_data()

# Check for required model files
required_files = ['LSTM_vectorizer.json', 'model_LSTM.pt']
missing_files = [f for f in required_files if not os.path.exists(f)]

if missing_files:
    st.error(f"Missing required files: {', '.join(missing_files)}. Please ensure these files are in the same directory as your app.")
    st.stop()

# Header
st.title("Amazon Review Sentiment Analysis")


# Function to analyze multiple reviews
def analyze_reviews(reviews):
    results = []
    for review in reviews:
        if isinstance(review, str) and review.strip():
            sentiment = predict_sentiment(review)
            results.append({
                "Review": review,
                "Sentiment": sentiment
            })
    return pd.DataFrame(results)

# Text input section
st.header("Enter Amazon Review")

# Set the review text based on selection
review_text = ""

# Text input area
review_text = st.text_area("Write a review", value=review_text, height=150)

# Submit
if st.button("Submit"):
    if review_text:
        with st.spinner("Analyzing sentiment..."):
            try:
                sentiment = predict_sentiment(review_text)
                response = generate_response(sentiment)
                
                # Create columns for results and explanation
                col1, col2 = st.columns(2)
                
                # Display result with color
                with col1:
                    st.subheader("Sentiment Analysis Result")
                    if sentiment == "Positive":
                        st.success(f"Sentiment: {sentiment}")
                    elif sentiment == "Negative":
                        st.error(f"Sentiment: {sentiment}")
                    else:
                        st.info(f"Sentiment: {sentiment}")
                
                # Add explanation based on sentiment
                with col2:
                    st.subheader("What This Means")
                    if sentiment == "Positive":
                        st.write("This review expresses satisfaction with the product. The customer is likely happy with their purchase.")
                    elif sentiment == "Negative":
                        st.write("This review expresses dissatisfaction. The customer is likely unhappy with their purchase.")
                    else:
                        st.write("This review is neutral or mixed. The customer may have both positive and negative opinions.")

                st.markdown("---")
                st.subheader("Automated Response")
                st.write(f"**Response to customer:** {response}")        
            
            except Exception as e:
                st.error(f"Error analyzing review: {str(e)}")
    else:
        st.warning("Please enter a review to analyze.")


            
# Model information in sidebar
st.sidebar.markdown("---")
st.sidebar.header("Model Information")
st.sidebar.info("""
This app uses an LSTM-based model for sentiment analysis.
- The model classifies reviews as Positive, Neutral, or Negative.
- Text preprocessing includes lowercasing, removal of punctuation and numbers, and stop word removal.
""")


# Footer
st.markdown("---")
st.markdown("Sentiment Analysis App | Created with Streamlit and PyTorch")
