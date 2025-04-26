import os
import sys
from transformers import pipeline, set_seed, AutoModelForCausalLM, AutoTokenizer
import torch

class ReviewGenerator:
    def __init__(self):
        try:
            # Check if CUDA is available
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            print(f"Using device: {self.device}")
            
            # Initialize model and tokenizer separately for better control
            print("Loading model and tokenizer...")
            model_name = "HuggingFaceH4/zephyr-7b-beta"
            
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto",
                trust_remote_code=True,
                use_safetensors=True
            )
            
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Initialize the pipeline with loaded model and tokenizer
            self.generator = pipeline(
                'text-generation',
                model=model,
                tokenizer=tokenizer,
                framework="pt"  # Explicitly use PyTorch
            )
            
            set_seed(42)  # For reproducibility
            print("Model initialized successfully!")
            
        except Exception as e:
            print("\nError initializing the model:", str(e))
            print("\nTroubleshooting steps:")
            print("1. Make sure you have the required packages installed:")
            print("   pip install --upgrade transformers torch accelerate safetensors")
            print("2. Check if you have enough system memory (at least 16GB recommended)")
            print("3. If using GPU, ensure you have CUDA installed correctly")
            print("\nFor detailed error information:")
            import traceback
            traceback.print_exc()
            sys.exit(1)
        
    def generate_review(self, product_name: str, rating: int) -> str:
        """
        Generate a product review based on the product name and rating.
        
        Args:
            product_name (str): The name of the product
            rating (int): The star rating (1-5)
            
        Returns:
            str: Generated review text
        """
        if not isinstance(rating, int) or not 1 <= rating <= 5:
            raise ValueError("Rating must be an integer between 1 and 5")
            
        # Create a more detailed prompt for better review generation
        sentiment = "positive" if rating >= 4 else "mixed" if rating == 3 else "negative"
        stars = "★" * rating + "☆" * (5 - rating)
        
        prompt = f"""<|system|>
You are a helpful assistant that writes detailed and natural product reviews.
<|user|>
Write a {sentiment} product review for the following product:
Product: {product_name}
Rating: {stars} ({rating}/5)

The review should be specific, balanced, and include both pros and cons where appropriate.
<|assistant|>"""
        
        try:
            # Generate the review with improved parameters
            generated_text = self.generator(
                prompt,
                max_new_tokens=200,
                num_return_sequences=1,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                repetition_penalty=1.2,
                truncation=True
            )
            
            # Extract and clean up the generated text
            review = generated_text[0]['generated_text']
            # Remove the prompt and get only the assistant's response
            review = review.split("<|assistant|>")[-1].strip()
            
            return review
        except Exception as e:
            print(f"Error generating review: {str(e)}")
            return None

def main():
    print("Product Review Generator")
    print("----------------------")
    
    try:
        print("\nInitializing model (this may take a moment)...")
        generator = ReviewGenerator()
        
        while True:
            try:
                # Get user input
                product = input("\nEnter the product name (or 'quit' to exit): ").strip()
                if not product:
                    print("Please enter a product name.")
                    continue
                if product.lower() == 'quit':
                    break
                    
                try:
                    rating = int(input("Enter the rating (1-5): "))
                    if not 1 <= rating <= 5:
                        print("Rating must be between 1 and 5")
                        continue
                except ValueError:
                    print("Please enter a valid number between 1 and 5")
                    continue
                
                # Generate and print the review
                print("\nGenerating review...")
                review = generator.generate_review(product, rating)
                
                if review:
                    print("\nGenerated Review:")
                    print("-----------------")
                    print(review)
                    print("-----------------")
                else:
                    print("Failed to generate review. Please try again.")
            except ValueError as ve:
                print(f"Error: {str(ve)}")
            except Exception as e:
                print(f"An error occurred: {str(e)}")
                
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {str(e)}")
    finally:
        print("\nThank you for using the Product Review Generator!")
    
if __name__ == "__main__":
    main() 