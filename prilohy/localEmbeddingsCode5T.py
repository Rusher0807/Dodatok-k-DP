import os, json, re, torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModel
from langchain_community.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.embeddings.base import Embeddings

# --- Custom Embedding Wrapper for CodeT5+ Encoder ---
class CodeT5pEmbedding(Embeddings):
    def __init__(self, model_name="Salesforce/codet5p-220m"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def embed_documents(self, texts):
        return [self._embed(text) for text in texts]

    def embed_query(self, text):
        return self._embed(text)

    def _embed(self, text):
        with torch.no_grad():
            tokens = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512, padding="max_length")
            tokens = {k: v.to(self.device) for k, v in tokens.items()}
            encoder_output = self.model.encoder(**tokens).last_hidden_state
            embedding = encoder_output.mean(dim=1)  # mean pooling
            return embedding.squeeze().cpu().tolist()

# --- Config ---
json_files = ["malware_finetune.jsonl"]
embedding_model = CodeT5pEmbedding()
output_dir = "malware_faiss_codet5p"

# --- Data Parser ---
def parse_rag_text(text):
    match = re.search(r"### Disassembled Code:\n(.*?)\n\n### Summary:\n(.*?)\n\n### Label:\n(.*)", text, re.DOTALL)
    if not match:
        return {"code": "", "summary": "", "label": "unknown"}
    return {
        "code": match.group(1).strip(),
        "summary": match.group(2).strip(),
        "label": match.group(3).strip().lower()
    }

def load_rag_data(json_files):
    documents = []
    for file in json_files:
        with open(file, "r", encoding="utf-8") as f:
            lines = [json.loads(line) for line in f if line.strip()] if file.endswith(".jsonl") else json.load(f)
            for entry in lines:
                parsed = parse_rag_text(entry.get("text", ""))
                content = f"""
### Label:
{parsed['label']}

### Disassembled Code:
{parsed['code']}

### Summary:
{parsed['summary']}
"""
                documents.append(Document(page_content=content.strip(), metadata={"source": file}))
    return documents

# --- Build and Save FAISS Vectorstore ---
documents = load_rag_data(json_files)
vectorstore = FAISS.from_documents(documents, embedding_model)
vectorstore.save_local(output_dir)
print(f"✅ Saved FAISS vectorstore to: {output_dir}")
