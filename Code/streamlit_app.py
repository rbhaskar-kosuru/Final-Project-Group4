import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import os
import nltk
from model_predict import predict_sentiment

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
st.markdown("Analyze sentiment of Amazon product reviews using LSTM")

# Sidebar options
st.sidebar.header("Options")
input_method = st.sidebar.radio(
    "Choose input method:",
    ("Text Input", "File Upload")
)

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
if input_method == "Text Input":
    st.header("Enter Amazon Review")
    
    # Example reviews dropdown
    example_reviews = [
        "Select an example or enter your own review",
        "This product exceeded my expectations! It was easy to use and works perfectly.",
        "I'm very disappointed with this purchase. It broke after just two uses.",
        "The item arrived on time and functions as described. Nothing special but does the job."
    ]
    
    selected_example = st.selectbox("Or choose an example review:", example_reviews)
    
    # Set the review text based on selection
    review_text = selected_example if selected_example != example_reviews[0] else ""
    
    # Text input area
    review_text = st.text_area("Type or paste your review here:", value=review_text, height=150)
    
    # Analyze button
    if st.button("Analyze Sentiment"):
        if review_text:
            with st.spinner("Analyzing sentiment..."):
                try:
                    sentiment = predict_sentiment(review_text)
                    
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
                
                except Exception as e:
                    st.error(f"Error analyzing review: {str(e)}")
        else:
            st.warning("Please enter a review to analyze.")

# File upload section
else:
    st.header("Upload Reviews File")
    st.markdown("Upload a CSV or Excel file containing reviews")
    
    # Batch size configuration
    batch_size = st.sidebar.slider("Batch size for processing", 
                                 min_value=10, 
                                 max_value=1000, 
                                 value=100, 
                                 step=10,
                                 help="Larger batch sizes process faster but use more memory")
    
    # File uploader
    uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx"])
    
    if uploaded_file is not None:
        try:
            # Determine file type and read
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            # Show dataframe info
            st.write(f"Uploaded file contains {df.shape[0]} rows and {df.shape[1]} columns")
            
            # Display column names for selection
            st.subheader("Select the column containing reviews")
            review_col = st.selectbox("Review column", df.columns.tolist())
            
            # Show sample of the data
            st.subheader("Sample Reviews")
            st.dataframe(df.head())
            
            # Analyze button
            if st.button("Analyze Reviews"):
                # Get total number of reviews
                total_reviews = len(df)
                
                # Create a progress bar
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Process in batches to show progress
                results = []
                for i in range(0, total_reviews, batch_size):
                    end_idx = min(i + batch_size, total_reviews)
                    status_text.text(f"Processing reviews {i+1} to {end_idx} of {total_reviews}...")
                    
                    # Get batch of reviews
                    batch = df[review_col].iloc[i:end_idx].tolist()
                    
                    # Analyze batch
                    batch_results = analyze_reviews(batch)
                    results.append(batch_results)
                    
                    # Update progress
                    progress_bar.progress((end_idx) / total_reviews)
                
                # Combine all results
                results_df = pd.concat(results, ignore_index=True)
                
                # Clear status text and progress bar
                status_text.empty()
                progress_bar.empty()
                
                # Display results
                st.subheader("Analysis Results")
                st.dataframe(results_df)
                
                # Calculate sentiment distribution
                sentiment_counts = results_df['Sentiment'].value_counts()
                
                # Create columns for visualizations
                col1, col2 = st.columns(2)
                
                # Create pie chart
                with col1:
                    st.subheader("Sentiment Distribution")
                    fig1, ax1 = plt.subplots()
                    colors = {
                        'Positive': '#66b3ff', 
                        'Negative': '#ff9999', 
                        'Neutral': '#99ff99'
                    }
                    chart_colors = [colors.get(s, '#cccccc') for s in sentiment_counts.index]
                    ax1.pie(sentiment_counts, labels=sentiment_counts.index, autopct='%1.1f%%', 
                          startangle=90, colors=chart_colors)
                    ax1.axis('equal')
                    st.pyplot(fig1)
                
                # Create bar chart
                with col2:
                    st.subheader("Sentiment Counts")
                    fig2, ax2 = plt.subplots()
                    bars = ax2.bar(sentiment_counts.index, sentiment_counts.values, color=chart_colors)
                    ax2.set_ylabel('Count')
                    ax2.set_title('Number of Reviews by Sentiment')
                    # Add count labels on top of bars
                    for bar in bars:
                        height = bar.get_height()
                        ax2.text(bar.get_x() + bar.get_width()/2., height,
                                f'{height:,}', ha='center', va='bottom')
                    st.pyplot(fig2)
                
                # Summary statistics
                st.subheader("Summary Statistics")
                total = sum(sentiment_counts)
                stats_col1, stats_col2, stats_col3 = st.columns(3)
                
                with stats_col1:
                    pos_count = sentiment_counts.get('Positive', 0)
                    pos_percent = (pos_count / total) * 100 if total > 0 else 0
                    st.metric("Positive Reviews", f"{pos_count} ({pos_percent:.1f}%)")
                
                with stats_col2:
                    neg_count = sentiment_counts.get('Negative', 0)
                    neg_percent = (neg_count / total) * 100 if total > 0 else 0
                    st.metric("Negative Reviews", f"{neg_count} ({neg_percent:.1f}%)")
                
                with stats_col3:
                    neu_count = sentiment_counts.get('Neutral', 0)
                    neu_percent = (neu_count / total) * 100 if total > 0 else 0
                    st.metric("Neutral Reviews", f"{neu_count} ({neu_percent:.1f}%)")
                
                # Download results button
                csv = results_df.to_csv(index=False)
                st.download_button(
                    label="Download Results as CSV",
                    data=csv,
                    file_name="sentiment_analysis_results.csv",
                    mime="text/csv",
                )
                
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.exception(e)

# Model information in sidebar
st.sidebar.markdown("---")
st.sidebar.header("Model Information")
st.sidebar.info("""
This app uses an LSTM-based model for sentiment analysis.
- The model classifies reviews as Positive, Neutral, or Negative.
- Text preprocessing includes lowercasing, removal of punctuation and numbers, and stop word removal.
""")

# How to use section in sidebar
st.sidebar.markdown("---")
st.sidebar.header("How to Use")
st.sidebar.markdown("""
1. Choose input method (text or file upload)
2. For text input: Type or paste a review and click "Analyze"
3. For file upload: Select your CSV/Excel file and the review column
4. View sentiment results and visualizations
5. Download results as CSV if needed
""")

# Footer
st.markdown("---")
st.markdown("Sentiment Analysis App | Created with Streamlit and PyTorch")
