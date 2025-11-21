import logging
from agent import create_agent
from utils.retriever import create_retriever
from utils.config import Config
from utils.state import GraphState

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    try:
        # Load configuration
        config = Config()
        logger.info(f"Loaded configuration: {config}")

        # Initialize components
        retriever = create_retriever()

        # Create agent
        agent = create_agent()

        while True:
            # Prompt user for question
            question = input("Please enter your question (or type 'exit' to quit): ")

            if question.lower() == 'exit':
                print("Exiting the program. Goodbye!")
                break

            logger.info(f"Running agent with question: {question}")

            # Initialize the GraphState with all required fields
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

            # ===== PRINT RETRIEVED DOCUMENTS =====
            print("\n" + "="*30 + " RETRIEVED DOCUMENTS " + "="*30)
            
            context_docs = result.get("context", [])
            for i, doc in enumerate(context_docs):
                # Handle cases where doc might be a string (if legacy code runs)
                if hasattr(doc, "metadata"):
                    meta = doc.metadata
                    title = meta.get("title", "No Title")
                    rating = meta.get("average_rating", "N/A")
                    num_ratings = meta.get("rating_number", "N/A")
                    prod_id = meta.get("product_id", "N/A")
                    
                    print(f"\n[Doc {i+1}] {title}")
                    print(f"   ID: {prod_id} | Rating: {rating} ({num_ratings} reviews)")
                    # Print first 200 chars of content to keep it clean
                    content_preview = doc.page_content[:200].replace('\n', ' ') + "..."
                    print(f"   Content: {content_preview}")
                else:
                    print(f"\n[Doc {i+1}] {str(doc)[:200]}...")
            
            print("="*81 + "\n")
            # =====================================

            # ===== Task 2: this is where you can improve the user-facing output =====
            # You can show:
            # - retrieved passages (from result["context"])
            # - doc ids / scores (if you stored them in retrieval)
            # - then the final LLM answer
            # This makes it clear WHY the agent answered that way.

            # Print final answer
            if result.get("final_answer"):
                print(f"\nAnswer: {result['final_answer']}")
            elif result.get("error"):
                print(f"\nError occurred: {result['error']}")
            else:
                print("\nNo answer or error was returned.")

            print("\n" + "-" * 50 + "\n")  # Add a separator between questions

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")


if __name__ == '__main__':
    main()
