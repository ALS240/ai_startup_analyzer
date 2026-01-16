import os, json
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter

# ✅ Updated imports
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "data" / "embedding_config.json"
DOCS_PATH = BASE_DIR / "docs" / "startup_data.txt"
INDEX_DIR = BASE_DIR / "data" / "faiss_index"

load_dotenv()

def load_config():
    with open(CONFIG_PATH, "r") as f:
        return json.load(f)

def get_embeddings_from_config():
    cfg = load_config()
    provider = cfg.get("provider")

    if provider == "hf":
        return HuggingFaceEmbeddings(model_name=cfg.get("model"))
    elif provider == "openai":
        return OpenAIEmbeddings(model=cfg.get("model"), openai_api_key=os.getenv("OPENAI_API_KEY"))
    else:
        raise ValueError(f"Unsupported provider: {provider}")

def ingest():
    embedder = get_embeddings_from_config()
    loader = TextLoader(str(DOCS_PATH))
    docs = loader.load()

    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = text_splitter.split_documents(docs)

    faiss_index = FAISS.from_documents(split_docs, embedder)

    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faiss_index.save_local(str(INDEX_DIR))

    print(f"✅ FAISS index created and saved to {INDEX_DIR}")

if __name__ == "__main__":
    ingest()
