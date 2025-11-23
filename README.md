# CS410 Group Project - Product Feature Retrieval and Performance Analysis from Amazon Reviews

We propose developing FeatureSearch, a tool that enables users to search for specific product features within Amazon reviews and analyze related customer opinions. Existing review platforms only provide overall ratings, making it difficult to understand detailed product strengths and weaknesses. Our tool integrates information retrieval and sentiment analysis to extract and summarize feature-level feedback, offering structured insights instead of raw text. We plan to use Amazon’s public review dataset, applying TF-IDF or BM25 for retrieval and VADER for sentiment scoring, with results visualized through charts and word clouds. Evaluation will be based on retrieval relevance, sentiment accuracy, and user feedback. The team will divide work into data preprocessing, retrieval and analysis, and visualization/interface development.

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
