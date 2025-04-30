# Product Review Generator

This project uses a Hugging Face model to generate product reviews based on a product name and star rating.

## Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Run the review generator:
```bash
python review_generator.py
```

## Usage

1. When prompted, enter the name of the product you want to review
2. Enter a rating between 1 and 5 stars
3. The generator will create a review based on your input

## Features

- Uses GPT-2 model from Hugging Face for text generation
- Generates unique reviews based on product name and rating
- Adjustable parameters for review length and creativity
- Simple command-line interface

## Notes

- The first run will download the GPT-2 model (about 500MB)
- Reviews are generated using a temperature of 0.7 for a good balance of creativity and coherence
- The model is seeded for reproducibility
