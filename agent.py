import logging
from typing import List

from langgraph.graph import StateGraph, END
from langchain.prompts import ChatPromptTemplate

from utils.llm import OllamaLLM
from utils.retriever import create_retriever
from utils.state import GraphState
from utils.config import Config

# ---------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
config = Config()


# ---------------------------------------------------------------------
# Graph construction (entry for main.py)
# ---------------------------------------------------------------------
def create_agent():
    """
    Build and compile the LangGraph workflow:
       retrieve -> generate_answer -> END
    """
    workflow = StateGraph(GraphState)

    # Define nodes
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("generate_answer", generate_answer)

    # Entry and edges
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate_answer")
    workflow.add_edge("generate_answer", END)

    return workflow.compile()


# ---------------------------------------------------------------------
# Node 1: Retrieval
# ---------------------------------------------------------------------
def retrieve(state: GraphState) -> GraphState:
    """
    Retrieve evidence passages for the given question.

    Notes:
    - This is where retrieval happens for the basic single-query case.
    - If you later want to show retrieved documents, scores, or titles, keep the
      retrieved metadata here so the output layer can format it.
    - If you later add long / multi-sentence support, you can first break the
      question into several sub-queries here and retrieve for each of them, then
      merge the results into state["context"].
    """
    try:
        logger.info("Starting retrieval")

        query = (state.get("question") or "").strip()
        if not query:
            state["error"] = "Empty question"
            state["current_step"] = "retrieve"
            logger.warning("Retrieval aborted: empty question")
            return state

        # Allow DI for tests; otherwise create from config
        retriever = state.get("retriever") or create_retriever()

        # Retrieve documents (ContextualCompressionRetriever or BaseRetriever)
        docs = retriever.invoke(query)
        if not docs and hasattr(retriever, "base_retriever"):
            logger.info("Compression yielded 0 docs; falling back to base retriever.")
            docs = retriever.base_retriever.invoke(query)

        # Build/extend context
        # Store the full Document objects, not just strings. 
        # This preserves metadata (rating, title, id).
        ctx = list(state.get("context") or [])
        ctx.extend(docs)

        state["context"] = ctx
        state["current_step"] = "retrieve"
        logger.info(f"Retrieved {len(docs)} documents")
        return state

    except Exception as e:
        logger.exception("Error in retrieve")
        state["error"] = str(e)
        state["current_step"] = "retrieve"
        return state


# ---------------------------------------------------------------------
# Helpers for prompting
# ---------------------------------------------------------------------
def build_prompt() -> ChatPromptTemplate:
    """
    Create the QA prompt with strict, mandatory formatting and filtering instructions.
    """
    template = """You are a shopping assistant that helps users select products.

Use ONLY the context below to answer the question. 

Context:
{context}

Question:
{question}

***MANDATORY SELECTION RULES***:
1. **Brand Diversity**: Prioritize listing products from DIFFERENT brands. 
2. **No Duplicates**: Do NOT list multiple variations (e.g., colors) of the same product model.
3. **Valid IDs Only**: Only list products that have a valid Product ID in the context. If a product has no ID, SKIP IT.
4. **Count**: List up to 5 distinct products. If you find fewer than 5 distinct brands/models, list fewer. Do not duplicate items just to reach 5.

***MANDATORY OUTPUT FORMAT***:
You must output the list in the exact format below. Do not add introductory text like "Here are the products...".

1. Product Title: [Title]
   - ID: [Product ID]
   - Rating: [Average Rating] ([Rating Number] reviews)
   - Description: [One-sentence summary]
2. Product Title: [Title]
...

Answer:"""
    return ChatPromptTemplate.from_template(template)

def build_review_summary_prompt(product_title: str) -> ChatPromptTemplate:
    """
    Create the prompt template for summarizing reviews.
    Optimized for a concise, scannable 'Scorecard' format.
    """
    template = f"""
You are an expert product reviewer. Your task is to summarize user reviews for: "{product_title}".

INSTRUCTIONS:
1. Read the reviews below.
2. Output a concise summary in the EXACT format provided.
3. Keep the "Verdict" to 1 sentence.
4. Limit "Pros" and "Cons" to the top 2 distinct points each.
5. Keep the "Analysis" to maximum 3 sentences.

REVIEWS CONTEXT:
{{reviews_context}}

***REQUIRED OUTPUT FORMAT***:

**Verdict:** [1-sentence bottom line, e.g., "A solid choice for beginners, but lacks durability."]

**Pros:**
* [Key Strength 1]
* [Key Strength 2]

**Cons:**
* [Key Weakness 1]
* [Key Weakness 2]

**Aspect Breakdown**

- **Performance/Cleaning**
  - Summary: [1–2 sentences about cleaning performance, power, effectiveness, etc.]
  - Example: "[Short quoted phrase from a review, if available.]"

- **Noise**
  - Summary: [1–2 sentences about how loud or quiet it is.]
  - Example: "[Short quoted phrase from a review, if available.]"

- **Ease of Use**
  - Summary: [1–2 sentences about installation, controls, daily use, learning curve.]
  - Example: "[Short quoted phrase from a review, if available.]"

- **Build Quality & Durability**
  - Summary: [1–2 sentences about materials, reliability, and how long it seems to last.]
  - Example: "[Short quoted phrase from a review, if available.]"

- **Price/Value for Money**
  - Summary: [1–2 sentences about whether the product feels worth the price.]
  - Example: "[Short quoted phrase from a review, if available.]"
"""
    return ChatPromptTemplate.from_template(template)

# ---------------------------------------------------------------------
# Node 2: Answer generation
# ---------------------------------------------------------------------
def generate_answer(state: GraphState) -> GraphState:
    try:
        logger.info("Starting answer generation")
        llm = OllamaLLM(model=config.OLLAMA_MODEL)

        # --- OPTIMIZED CONTEXT CONSTRUCTION ---
        # We format the context so the LLM sees the ID and Rating clearly.
        context_items = []
        for doc in state.get("context") or []:
            # Handle both Document objects (from retriever) and strings (legacy)
            if hasattr(doc, "metadata"):
                meta = doc.metadata
                # Create a structured block for the LLM to read
                item_str = (
                    f"Product Title: {meta.get('title', 'Unknown')}\n"
                    f"Product ID: {meta.get('product_id', 'Unknown')}\n"
                    f"Rating: {meta.get('average_rating', 'N/A')} stars "
                    f"({meta.get('rating_number', '0')} reviews)\n"
                    f"Description: {doc.page_content}\n"
                    "---"
                )
                context_items.append(item_str)
            else:
                context_items.append(str(doc))
        
        full_context = "\n".join(context_items)
        # ---------------------------------------

        prompt = build_prompt() # No longer need max_sentences argument
        chain = prompt | llm

        response = chain.invoke({
            "question": state.get("question", ""),
            "context": full_context
        })

        state["final_answer"] = response
        state["current_step"] = "generate_answer"
        logger.info("Answer generation completed")
        return state

    except Exception as e:
        logger.error(f"Error in generate_answer: {e}")
        state["error"] = str(e)
        return state
