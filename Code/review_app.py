import streamlit as st
from review_generator import ReviewGenerator

st.title("Product Review Generator")

# Initialize the ReviewGenerator only once
@st.cache_resource
def get_generator():
    return ReviewGenerator()

generator = get_generator()

# Session state for reviews
if "reviews" not in st.session_state:
    st.session_state.reviews = []

# Input fields
product = st.text_input("Enter the product name:")
rating = st.slider("Select the rating (1-5):", 1, 5, 5)

if st.button("Generate Review"):
    if not product.strip():
        st.warning("Please enter a product name.")
    else:
        with st.spinner("Generating review..."):
            review = generator.generate_review(product, rating)
            if review:
                st.session_state.reviews.append({
                    "product": product,
                    "rating": rating,
                    "review": review
                })
            else:
                st.error("Failed to generate review. Please try again.")

# Display generated reviews
if st.session_state.reviews:
    st.subheader("Generated Reviews")
    for entry in st.session_state.reviews[::-1]:
        st.markdown(f"**Product:** {entry['product']}  \n**Rating:** {'★'*entry['rating']}{'☆'*(5-entry['rating'])}")
        st.info(entry["review"]) 