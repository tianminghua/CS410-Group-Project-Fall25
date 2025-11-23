import logging
import re
from typing import Dict, Optional, List

from agent import create_agent, build_review_summary_prompt
from utils.retriever import create_retriever
from utils.config import Config
from utils.state import GraphState
from utils.review_loader import get_reviews_by_product_id

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global maps to link the list number (1, 2, 3...) to the actual data
PRODUCT_ID_MAP: Dict[int, str] = {}
PRODUCT_TITLE_MAP: Dict[int, str] = {}

class Colors:
    GREEN = '\033[92m'   # Good for the main Answer/Product list
    CYAN = '\033[96m'    # Good for Analysis/Review Summaries
    YELLOW = '\033[93m'  # Good for System Messages (loading...)
    BOLD = '\033[1m'
    RESET = '\033[0m'    # Resets color to default

def parse_llm_answer_for_products(llm_answer: str) -> None:
    """
    Parses the structured LLM answer to extract a map of 
    (list_number) -> (product_id). Populates the global maps.
    
    Correction: Updated title_regex to handle the LLM omitting "Product Title:".
    """
    global PRODUCT_ID_MAP, PRODUCT_TITLE_MAP
    
    # Clear previous run data
    PRODUCT_ID_MAP.clear()
    PRODUCT_TITLE_MAP.clear()
    
    # Regex to find lines like "1. [Title]" (Corrected to match actual output)
    # Captures the number (\d+) and the title (.+)
    title_regex = re.compile(r"^\s*(\d+)\.\s*(.+)$", re.MULTILINE)
    
    # Regex to find lines like "- ID: [B0...]" (This one was already working)
    # Captures the Product ID
    id_regex = re.compile(r"^\s*-\s*ID:\s*(B0[0-9A-Z]{8})$", re.MULTILINE)

    # Find all matches
    titles = [(int(m.group(1)), m.group(2).strip()) for m in title_regex.finditer(llm_answer)]
    ids = [m.group(1).strip() for m in id_regex.finditer(llm_answer)]
    
    # Map the titles (which have the list number) to the IDs (assuming order is preserved)
    if len(titles) == len(ids):
        for (num, title), pid in zip(titles, ids):
            # Clean up the title: remove rating info that sometimes leaks into the title
            clean_title = title.split('\n')[0].split('- ID:')[0].strip()
            PRODUCT_ID_MAP[num] = pid
            PRODUCT_TITLE_MAP[num] = clean_title
            
        logger.info(f"Parsed {len(titles)} products for secondary lookup.")
    else:
        logger.warning(f"Parsing failed: Found {len(titles)} titles and {len(ids)} IDs. Map not created.")


def main():
    try:
        # Load configuration... (omitted for brevity)
        config = Config()
        logger.info(f"Loaded configuration: {config}")

        # Initialize components... (omitted for brevity)
        retriever = create_retriever()
        agent = create_agent()

        while True:
            # Stage 1: Initial Question
            question = input("\nPlease enter your question (or type 'exit' to quit): ")

            if question.lower() == 'exit':
                print("Exiting the program. Goodbye!")
                break

            logger.info(f"Running agent with question: {question}")

            initial_state = GraphState(
                question=question,
                context=[],
                current_step="",
                final_answer="",
                retriever=retriever,
                web_search_tool=None,
                error=None,
                selected_namespaces=[],
                web_search_results=[]
            )

            result = agent.invoke(initial_state)
            logger.info(f"Agent result: {result}")
            
            # --- Document printing logic (Assume it's here) ---
            # ...
            # --- Document printing logic End ---

            # Print final answer
            if result.get("final_answer"):
                print(f"\n{Colors.GREEN}Answer: {result['final_answer']}{Colors.RESET}")
                
                # Prepare for Stage 2
                parse_llm_answer_for_products(result['final_answer'])
                
            elif result.get("error"):
                print(f"\nError occurred: {result['error']}")
            else:
                print("\nNo answer or error was returned.")

            print("\n" + "-" * 50 + "\n")  # Separator

            # Stage 2: Review Retrieval Prompt Loop
            if PRODUCT_ID_MAP:
                while True:
                    review_choice = input(
                        "Enter a product number (1, 2, 3...) to retrieve its reviews, "
                        "or type 'back' to ask a new question: "
                    )
                    
                    if review_choice.lower() == 'back':
                        print("Starting new query session.")
                        break
                    
                    try:
                        num = int(review_choice)
                        if num in PRODUCT_ID_MAP:
                            target_id = PRODUCT_ID_MAP[num]
                            target_title = PRODUCT_TITLE_MAP[num]
                            
                            reviews = get_reviews_by_product_id(target_id)
                            
                            # --- PRINT RAW REVIEWS (as before) ---
                            print(f"\n--- REVIEWS FOR: {target_title} (ID: {target_id}) ---")
                            
                            if reviews:
                                # Print up to 5 reviews to avoid overwhelming the user
                                for i, review in enumerate(reviews[:5]):
                                    print(f"Review {i+1} (Rating: {review['rating']}):")
                                    print(f"  Title: {review['title']}")
                                    print(f"  Text: {review['text'][:150]}...\n")
                                if len(reviews) > 5:
                                    print(f"Showing 5 of {len(reviews)} total reviews found.")
                            else:
                                print("No reviews found for this product ID in the external file.")
                                print("--- END REVIEWS ---\n")
                                continue # Skip summary generation if no reviews
                            
                            # 1. Limit the reviews to avoid LLM context overflow
                            MAX_REVIEWS_FOR_SUMMARY = 15
                            reviews_for_summary = reviews[:MAX_REVIEWS_FOR_SUMMARY]
                            
                            # 2. Format the reviews into a single string for the prompt
                            review_strings = []
                            for i, review in enumerate(reviews_for_summary):
                                review_strings.append(
                                    f"Review {i+1} (Rating: {review['rating']} - Title: {review['title']}):\n"
                                    f"Text: {review['text']}"
                                )
                            reviews_context = "\n---\n".join(review_strings)
                            
                            # 3. Call the LLM
                            llm = config.get_llm()
                            summary_prompt = build_review_summary_prompt(target_title)
                            summary_chain = summary_prompt | llm
                            
                            print(f"\n{Colors.BOLD}--- GENERATING OVERALL SUMMARY ---{Colors.RESET}")
                            
                            summary = summary_chain.invoke({
                                "reviews_context": reviews_context
                            })
                            
                            # Apply CYAN color to the summary
                            print(f"\n{Colors.CYAN}{summary}{Colors.RESET}")
                            print(f"{Colors.BOLD}--- END SUMMARY ---{Colors.RESET}\n")
                        else:
                            print("Invalid product number. Please enter a number from the list or 'back'.")
                    except ValueError:
                        print("Invalid input. Please enter a number (1, 2, 3...) or 'back'.")

    except Exception as e:
        logger.error(f"An unhandled error occurred: {str(e)}")


if __name__ == '__main__':
    main()