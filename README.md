# CS410 Group Project - Product Feature Retrieval and Performance Analysis from Amazon Reviews

We build a small but complete IR + text mining system on the Amazon **Appliances** collection. 
Users can type a natural language query (e.g., “quiet compact dishwasher under $400 for a small apartment”) and:

1. Get a ranked list of relevant products generated from a BM25 + Pyserini retriever and a local Ollama LLM.
2. Select a product to see rating statistics and star distribution computed from its reviews.
3. Read an aspect-based summary of real user opinions (performance/cleaning, noise, ease of use, durability, price/value).

This project connects classic IR (BM25, indexing, retrieval) with LLM-based summarization in a simple two-stage command-line interface.

## Tech Stack

- **Language & Runtime**
  - Python 3.10+

- **IR / Search**
  - [Pyserini](https://github.com/castorini/pyserini) (BM25, Lucene index)
  - Java runtime for Lucene (required by Pyserini)

- **LLM & Orchestration**
  - [Ollama](https://ollama.com/) (local LLM backend, e.g. `llama3.2:3b`)
  - `langchain` / `langchain-core` for LLM pipelines
  - `langgraph` for building the retrieval → generation agent

- **Data & Storage**
  - Cleaned Amazon Appliances metadata and reviews as JSONL files
  - Local `dataset/` folder with:
    - `meta_Appliances_cleaned.jsonl`
    - `Appliances_cleaned.jsonl`

- **Interface & Utilities**
  - Command-line interface (`main.py`) for two-stage interaction
  - Standard Python tooling (`venv`, `pip`, `requirements.txt`)

## Team Members

- **Tina Harter** (tharter2)
- **Dezhao Li** (dezhaol2)
- **Ziyuan Gu** (ziyuang3)

## Prerequisites

- **Python** ≥ 3.10
- **Conda** (recommended)

## Installation

1. Install Pyserini

   ```
   pip install pyserini
   ```

2. Install the required packages:

   ```
   pip install -r requirements.txt
   ```

3. Install Ollama 

   Follow the official instructions at https://ollama.com/download/. 
   For example, on linux:

   ```
   curl -fsSL https://ollama.com/install.sh | sh
   ```

4. Pull the Llama3.2 model (or another model of your choice):

   ```
   ollama pull llama3.2:3b
   ```

5. Download dataset from Google Drive:

   ```
   https://drive.google.com/drive/folders/1ZIVppO-I1QSTU-WzllgZ0dmFtQ1T3HB3
   ```
   Create a **dataset** directory under the project folder

   Save **Appliances_cleaned.jsonl** and **meta_appliances_cleaned.jsonl** under **dataset**


## Usage

Ensure your dataset is placed under the `dataset/` directory, then run:

   ``` 
   python main.py 
   ```

The agent will prompt you to enter a question. If you encounter an error such as `connection refused`, it means the Ollama service is not running.
Start it manually by running:

   ``` 
   ollama serve
   ```

Then, rerun the command above in a new terminal window.

## Main Features

- **Natural language product search**  
  - Users ask free-form questions about appliances (budget, size, noise level, etc.).
  - A Pyserini BM25 index over cleaned product metadata returns relevant candidates.

- **LLM-enhanced product ranking**  
  - A local Ollama model (e.g., `llama3.2:3b`) takes BM25 results and produces a short ranked list:
    - product title,
    - product ID (ASIN),
    - average rating and review count,
    - one-sentence description.

- **Single-product review analysis**  
  - Load all reviews for a selected product from `Appliances_cleaned.jsonl`.
  - Compute rating statistics and a 1★–5★ star distribution.
  - Generate an aspect-based summary of reviews, organized into:
    - Performance/Cleaning  
    - Noise  
    - Ease of Use  
    - Build Quality & Durability  
    - Price/Value for Money  

Together, these components let a user quickly go from a vague shopping need to concrete product options and a clear understanding of real user feedback.

## Project Structure

```
CS410 Group Project/
├── dataset/                         # Local data document
├── main.py                          # Entry point
├── agent.py                         # LangGraph agent logic
├── utils/
│   ├── config.py                    # Configuration management
│   ├── clean_appliances_reviews.py  # Clean up reviews
│   ├── clean_meta_appliances.py     # Clean up metadata
│   ├── review_loader.py             # Load reviews for specific product
│   ├── llm.py                       # Ollama LLM integration
│   ├── retriever.py                 # Document retrieval using Pyserini
│   └── state.py                     # State management for LangGraph
└── requirements.txt

```
