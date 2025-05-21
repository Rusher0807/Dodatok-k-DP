import os
import json
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document

# --- Config ---
json_files = ["malware_rag.json", "benign_rag.json"]  # Adjust if you put them elsewhere

# Initialize Ollama embeddings model
embedding_model = OllamaEmbeddings(model="nomic-embed-text")

# --- Load RAG Data ---
def load_rag_data(json_files):
    documents = []
    for file in json_files:
        with open(file, "r", encoding="utf-8") as f:
            data = json.load(f)
            for entry in data:
                # Embed: Assembly + Reasoning
                text = f"""
                Classification: {entry.get('classification', 'Unknown')}
                Assembly Code:
                {entry.get('assembly_code', '').strip()}
                
                Reasoning:
                {entry.get('reasoning', '').strip()}
                """
                documents.append(Document(page_content=text, metadata={"source": file}))
    return documents

# --- Create Vectorstore ---
documents = load_rag_data(json_files)
vectorstore = FAISS.from_documents(documents, embedding_model)
vectorstore.save_local("malware_index")
print("✅ Vectorstore saved as 'malware_index'")
