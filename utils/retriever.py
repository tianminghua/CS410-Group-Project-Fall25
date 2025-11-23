
import os
import json
import logging
import subprocess
import shutil
from tqdm import tqdm
from typing import List, Any, Optional

from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from pyserini.search.lucene import LuceneSearcher
from pydantic import PrivateAttr
from utils.config import Config

# ---------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------
logger = logging.getLogger(__name__)
config = Config()

# ---------------------------------------------------------------------
# 1. Define a LangChain-compatible BM25 Retriever
# ---------------------------------------------------------------------
class PyseriniBM25Retriever(BaseRetriever):

    index_dir: str                     # Path to the Lucene index
    k: int = 5                         # Number of results to return
    k1: float = 0.9                    # BM25 hyperparameter
    b: float = 0.4                     # BM25 hyperparameter
    _searcher: Any = PrivateAttr(default=None)

    def __init__(self, **data: Any):
        super().__init__(**data)

        # Initialize Pyserini searcher
        searcher = LuceneSearcher(self.index_dir)
        searcher.set_bm25(self.k1, self.b)
        self._searcher = searcher

        logger.info(
            f"PyseriniBM25Retriever ready — index={self.index_dir}, "
            f"k={self.k}, k1={self.k1}, b={self.b}"
        )
    
    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Optional[CallbackManagerForRetrieverRun] = None,
        **kwargs: Any,
    ) -> List[Document]:
        """Return top-k relevant documents for a given query."""
        hits = self._searcher.search(query, k=self.k)
        docs: List[Document] = []

        for h in hits:
            # Pyserini stores the raw JSON in .raw()
            raw_content = self._searcher.doc(h.docid).raw() or ""
            
            content_text = raw_content
            
            # specific fields to populate into metadata
            metadata = {
                "docid": h.docid, 
                "score": float(h.score)
            }

            # Attempt to parse the JSON we stored during preprocessing
            try:
                obj = json.loads(raw_content)
                
                # 1. Get Content
                # We prefer the 'contents' field we built in preprocessing
                content_text = obj.get("contents") or obj.get("text") or raw_content
                
                # 2. Extract Metadata fields
                if "title" in obj:
                    metadata["title"] = obj["title"]
                
                # Store product_id specifically for future review retrieval
                if "id" in obj:
                    metadata["product_id"] = obj["id"]
                    
                # Store rating info in metadata (useful for filtering logic later)
                if "average_rating" in obj:
                    metadata["average_rating"] = obj["average_rating"]
                if "rating_number" in obj:
                    metadata["rating_number"] = obj["rating_number"]

            except Exception:
                pass

            docs.append(
                Document(
                    page_content=content_text,
                    metadata=metadata,
                )
            )
        return docs


# ---------------------------------------------------------------------
# 2. Helper functions for preprocessing and indexing
# ---------------------------------------------------------------------

def preprocess_corpus(input_file: str, output_dir: str):
    """
    Convert a specific dataset (JSONL) into clean JSON documents
    that Pyserini can index.
    
    Enriches content with:
    - Category Hierarchy (Critical for finding products by type)
    - Brand
    - Price
    - Ratings
    """
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, "documents.jsonl")
    print(f"Preprocessing {input_file} -> {output_file}")
    
    with open(input_file, "r", encoding="utf-8") as fin, \
         open(output_file, "w", encoding="utf-8") as fout:
        
        for line in tqdm(fin, desc="Processing records"):
            line = line.strip()
            if not line:
                continue
            
            try:
                record = json.loads(line)
                doc_id = record.get("product_id")
                if not doc_id:
                    continue

                # 1. Extract Metadata
                title = record.get("title", "Unknown Product")
                categories = record.get("categories_str", "") or " ".join(record.get("categories", []))
                brand = record.get("brand", "")
                price = record.get("price")
                avg_rating = record.get("average_rating", "N/A")
                rating_num = record.get("rating_number", 0)
                
                # Handle price formatting
                price_str = f"${price}" if price else "N/A"

                # 2. Construct the Enriched Text (The "Searchable" Part)
                # This ensures a search for "Coffee Machine" hits this doc because 
                # 'categories' contains that phrase.
                base_text = record.get("all_text") or title
                
                enriched_text = (
                    f"{base_text}\n"
                    f"Category: {categories}\n"
                    f"Brand: {brand}\n"
                    f"Price: {price_str}\n"
                    f"[Rating: {avg_rating} stars | {rating_num} reviews]"
                )

                # 3. Construct Pyserini Document
                pyserini_doc = {
                    "id": doc_id,
                    "contents": enriched_text,
                    # Store raw metadata for the Agent to display nicely later
                    "title": title,
                    "average_rating": avg_rating,
                    "rating_number": rating_num,
                    "price": price_str,
                    "brand": brand,
                    "product_id": doc_id
                }
                
                fout.write(json.dumps(pyserini_doc) + "\n")
                
            except json.JSONDecodeError:
                continue


def build_index(input_dir: str, index_dir: str):
    """
    Build a BM25 index using Pyserini’s command-line interface.
    """
    if os.path.exists(index_dir) and os.listdir(index_dir):
        print(f"Index already exists at {index_dir}. Skipping index building.")
        return

    cmd = [
        "python", "-m", "pyserini.index.lucene",
        "--collection", "JsonCollection",
        "--input", input_dir,
        "--index", index_dir,
        "--generator", "DefaultLuceneDocumentGenerator",
        "--threads", "1",
        "--storePositions", "--storeDocvectors", "--storeRaw",
    ]

    print("Building index (this may take a few minutes)...")
    subprocess.run(cmd, check=True)
    print(f"Index built successfully at {index_dir}")


# ---------------------------------------------------------------------
# 3. High-level API for index creation and retriever construction
# ---------------------------------------------------------------------
def create_index(cname: str) -> str:
    """
    Create an index for the given corpus name (cname).
    The raw data should be under data/{cname}/{cname}.dat
    """
    base_dir = f"dataset"
    corpus_file = os.path.join(base_dir, f"{cname}.jsonl")
    processed_corpus_dir = f"processed_corpus/{cname}"
    index_dir = f"indexes/{cname}"

    # Step 1. Preprocess
    if not os.path.exists(processed_corpus_dir) or not os.listdir(processed_corpus_dir):
        preprocess_corpus(corpus_file, processed_corpus_dir)
    else:
        print(f"Preprocessed corpus already exists at {processed_corpus_dir}")

    # Step 2. Build index
    build_index(processed_corpus_dir, index_dir)
    return index_dir


def create_retriever():
    """
    Create a LangChain retriever pipeline:
    BM25 (Pyserini) → optional contextual compression (LLM).
    """
    try:
        index_dir = create_index(config.PYSERINI_CNAME)

        # 1. Create the base BM25 retriever
        base_retriever = PyseriniBM25Retriever(
            index_dir=index_dir,
            k=config.RETRIEVER_K,
            k1=config.PYSERINI_K1,
            b=config.PYSERINI_B,
        )
        logger.info(f"BM25 retriever created on {index_dir}")

        # 2. (Optional) Add contextual compression with an LLM
        # This step uses the model defined in utils/config.py
        llm = config.get_llm()  # make sure Config implements get_llm()
        compressor = LLMChainExtractor.from_llm(llm)

        retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=base_retriever,
        )

        logger.info("Full retriever created successfully.")
        return retriever

    except Exception as e:
        logger.error(f"Error creating retriever: {e}")
        raise
