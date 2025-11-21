"""
agent.py
=========================
LangGraph-based RAG agent for MP2.

Pipeline:
1) retrieve(state)        -> collects context via Pyserini BM25 (+ optional compression)
2) generate_answer(state) -> prompts an LLM (Ollama) with the retrieved context

State keys (GraphState):
- question: str                     # user question (required)
- context: List[str]                # accumulated evidence passages
- retriever: Optional[BaseRetriever]# allow injection (for testing)
- final_answer: Optional[str]       # model's answer
- error: Optional[str]              # error message if any
- current_step: str                 # "retrieve" | "generate_answer"
"""

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
    Create the QA prompt with strict, mandatory formatting instructions.
    """
    template = """You are a shopping assistant that helps users select products.

Use ONLY the context below to answer the question. If the answer is not in the context, say you don't know.
Crucially, you must filter the context to only include products relevant to the question.

Context:
{context}

Question:
{question}

***MANDATORY OUTPUT INSTRUCTIONS***:
You must list the products found that are relevant to the question (e.g., only "ice maker machines"). You MUST provide the Product ID, Average Rating, and Rating Number for each. DO NOT invent information or deviate from the structure below.

Example of Required Format:
1. Product Title: [Title of Product]
   - ID: [Product ID]
   - Rating: [Average Rating] ([Rating Number] reviews)
   - Description: [A very brief (one-phrase) summary of the product features]
2. Product Title: [Title of Next Product]
   - ID: [Product ID]
   - Rating: [Average Rating] ([Rating Number] reviews)
   - Description: [A very brief (one-phrase) summary of the product features]
...

Answer:"""
    return ChatPromptTemplate.from_template(template)

def build_review_summary_prompt(product_title: str) -> ChatPromptTemplate:
    """
    Create the prompt template for summarizing a collection of reviews.
    """
    template = f"""
You are an expert review summarizer. Your task is to read a collection of user reviews for the product: "{product_title}" and provide a balanced, overall summary.

INSTRUCTIONS:
1. Analyze the sentiment (positive, negative, neutral) across all reviews.
2. Identify the top 2 most common positive points (pros) and the top 2 most common negative points (cons).
3. Synthesize the findings into a clear, concise overall review.
4. Your final summary must be 4 to 6 sentences long.
5. Do NOT invent or hallucinate any details not present in the provided reviews.

REVIEWS CONTEXT:
{{reviews_context}}

Overall Product Review for "{product_title}":
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
