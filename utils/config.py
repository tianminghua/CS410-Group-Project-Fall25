import os
from dotenv import load_dotenv

class Config:
    def __init__(self):
        load_dotenv()  # Load environment variables from .env file

        # LLM Settings
        self.OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
        self.OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

        # Pyserini / Retriever Settings
        self.PYSERINI_CNAME = os.getenv("PYSERINI_CNAME", "meta_Appliances_cleaned")
        self.RETRIEVER_K = int(os.getenv("RETRIEVER_K", 20))
        self.PYSERINI_K1 = float(os.getenv("PYSERINI_K1", 2.0))
        self.PYSERINI_B = float(os.getenv("PYSERINI_B", 0.8))
        

    def __repr__(self):
        return (f"Config("
                f"OLLAMA_MODEL={self.OLLAMA_MODEL}, "
                f"RETRIEVER_K={self.RETRIEVER_K}), "
                f"PYSERINI_CNAME={self.PYSERINI_CNAME}, "
                f"PYSERINI_K1={self.PYSERINI_K1}, "
                f"PYSERINI_B={self.PYSERINI_B}, ")

    def get_llm(self):
        from langchain_ollama import ChatOllama
        from langchain_core.output_parsers import StrOutputParser
        llm = ChatOllama(
            model=self.OLLAMA_MODEL,
        )
        return llm | StrOutputParser()
