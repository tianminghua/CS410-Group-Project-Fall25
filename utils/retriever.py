"""
retriever.py
=========================
Implements a BM25-based retriever using Pyserini for MP2 (LangGraph RAG Agent).

This module:
1. Preprocesses the raw text corpus into JSON files compatible with Pyserini.
2. Builds a Lucene index.
3. Wraps Pyserini’s BM25 retriever into a LangChain retriever.
4. Adds an optional contextual compression layer using an LLM.
"""

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
    """
    A LangChain retriever built on top of Pyserini’s BM25 searcher.
    It retrieves the top-k documents for a given query. You can change
    to your MP1 retriever here.
    """

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

    # def _get_relevant_documents(
    #     self,
    #     query: str,
    #     *,
    #     run_manager: Optional[CallbackManagerForRetrieverRun] = None,
    #     **kwargs: Any,
    # ) -> List[Document]:
    #     """Return top-k relevant documents for a given query.

    #     Notes:
    #     - This method is called once per query.
    #     - For multi-subquery retrieval in Task 3, call this method multiple times
    #       (one per sub-query) in the upper-level node and then fuse the results.
    #     """
    #     hits = self._searcher.search(query, k=self.k)
    #     docs: List[Document] = []

    #     for h in hits:
    #         raw = self._searcher.doc(h.docid).raw() or ""
    #         text, title = raw, None
    #         try:
    #             obj = json.loads(raw)
    #             text = obj.get("contents") or obj.get("text") or raw
    #             title = obj.get("title")
    #         except Exception:
    #             pass

    #         docs.append(
    #             Document(
    #                 page_content=text,
    #                 metadata={"docid": h.docid, "score": float(h.score), "title": title},
    #             )
    #         )
    #     return docs
    
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
# def preprocess_corpus(input_file: str, output_dir: str):
#     """
#     Convert a plain-text corpus (.dat or .txt) into JSON documents
#     that Pyserini can index.
#     """
#     os.makedirs(output_dir, exist_ok=True)
#     with open(input_file, "r") as f:
#         for i, line in enumerate(tqdm(f, desc="Preprocessing corpus")):
#             doc = {
#                 "id": str(i),  # Each doc needs a unique ID
#                 "contents": line.strip(),
#             }
#             with open(os.path.join(output_dir, f"doc{i}.json"), "w") as out:
#                 json.dump(doc, out)

def preprocess_corpus(input_file: str, output_dir: str):
    """
    Convert a specific dataset (JSONL) into clean JSON documents
    that Pyserini can index.
    
    Enriches content with rating info and preserves product_id.
    """
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # We will write a single large jsonl file
    output_file = os.path.join(output_dir, "documents.jsonl")

    print(f"Preprocessing {input_file} -> {output_file}")
    
    with open(input_file, "r", encoding="utf-8") as fin, \
         open(output_file, "w", encoding="utf-8") as fout:
        
        for line in tqdm(fin, desc="Processing records"):
            line = line.strip()
            if not line:
                continue
            
            try:
                # 1. Parse original data
                record = json.loads(line)
                
                doc_id = record.get("product_id")
                # Fallback to string hash if ID missing (rare)
                if not doc_id:
                    continue

                # Extract rating info
                avg_rating = record.get("average_rating", "N/A")
                rating_num = record.get("rating_number", 0)

                # 2. Construct the text content
                # We append the rating info to the text so the LLM "reads" it automatically.
                base_text = record.get("all_text") or record.get("title") or ""
                enriched_text = f"{base_text}\n[Average Rating: {avg_rating} | Ratings: {rating_num}]"

                # 3. Construct Pyserini-compatible document
                # We store extra fields in the JSON so we can retrieve them into metadata later
                pyserini_doc = {
                    "id": doc_id,
                    "contents": enriched_text,
                    "title": record.get("title", ""),
                    "average_rating": avg_rating,
                    "rating_number": rating_num,
                    # Store raw fields if you want strictly raw access later
                    "raw_category": record.get("main_category", "")
                }
                
                # 4. Write to output
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
