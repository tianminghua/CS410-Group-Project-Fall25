import json
import os
from typing import List, Dict, Any

# Define the exact path to the review file
# (Assuming your review data is at: dataset/Appliances_cleaned.jsonl)
REVIEW_FILE_PATH = "dataset/Appliances_cleaned.jsonl"

def get_reviews_by_product_id(product_id: str) -> List[Dict[str, Any]]:
    """
    Reads the full review corpus and filters records by a specific product ID (ASIN).
    
    Args:
        product_id: The ID (ASIN) of the product to search for.
        
    Returns:
        A list of dictionaries, each containing 'rating', 'title', and 'text' of a review.
    """
    reviews = []
    
    if not os.path.exists(REVIEW_FILE_PATH):
        print(f"Error: Review file not found at {REVIEW_FILE_PATH}")
        return []

    print(f"Searching for reviews of product ID: {product_id} in {REVIEW_FILE_PATH}...")
    
    # Read the JSONL file line by line for efficient processing of large files
    try:
        with open(REVIEW_FILE_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    record = json.loads(line)
                    
                    # --- CRITICAL CHANGE START: Using 'asin' for ID lookup ---
                    # Use 'asin' or 'parent_asin' for the product ID check
                    # We check both fields just in case the ID is stored differently per line.
                    review_asin = record.get("asin")
                    
                    if review_asin == product_id:
                        # --- CRITICAL CHANGE END ---
                        
                        # --- CRITICAL CHANGE START: Using 'rating', 'title', 'text' keys ---
                        reviews.append({
                            # New Key: 'rating' (float) -> was 'star_rating'
                            "rating": record.get("rating", "N/A"),
                            # New Key: 'title' (string) -> was 'review_title'
                            "title": record.get("title", "No Title"),
                            # New Key: 'text' (string) -> was 'review_text'
                            "text": record.get("text", "No Text"),
                        })
                        # --- CRITICAL CHANGE END ---

                except json.JSONDecodeError:
                    continue  # Skip corrupt lines
    except Exception as e:
        print(f"An error occurred while reading reviews: {e}")
        return []
        
    return reviews