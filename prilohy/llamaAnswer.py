import os
import pandas as pd
import time
from tqdm import tqdm
from langchain.chains import RetrievalQA
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS
from langchain_ollama.embeddings import OllamaEmbeddings

# --- Config ---
input_folder = "input_texts2"
output_excel = "llama_outputs_RAG2.xlsx"

# --- Load FAISS vectorstore ---
embedding_model = OllamaEmbeddings(model="nomic-embed-text")
vectorstore = FAISS.load_local("malware_index", embedding_model, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever()

# --- Load Ollama LLM ---
llm = Ollama(model="llama3.2")

# --- Create RAG Chain ---
qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)

# --- Process all txt files ---
data = []

files = [f for f in os.listdir(input_folder) if f.endswith(".txt")]

for filename in tqdm(files, desc="Batch Processing Assembly Files"):
    filepath = os.path.join(input_folder, filename)
    with open(filepath, "r", encoding="utf-8") as f:
        c_code_snippet = f.read()

    # Query prompt
    query = f"""
    Analyze this Disassembly code and determine if it is malicious or benign based on stored malware knowledge:
    {c_code_snippet}
    Provide evidence and similarities to known malware.
    """

    try:
        print(f"\n➡️ Processing {filename} ...")
        start_time = time.time()
        result = qa_chain.invoke(query)
        elapsed = time.time() - start_time
        print(f"✅ Done {filename} in {elapsed:.2f} sec")

        data.append({
            "input_file": filename,
            "query": query,
            "output": result["result"]
        })

        # 🔄 Save after every file
        pd.DataFrame(data).to_excel(output_excel, index=False)
        print(f"💾 Saved progress to {output_excel}")

    except Exception as e:
        print(f"❌ Error on {filename}: {e}")
        data.append({
            "input_file": filename,
            "query": query,
            "output": f"ERROR: {e}"
        })
        pd.DataFrame(data).to_excel(output_excel, index=False)
        print(f"💾 Saved progress (with error) to {output_excel}")

print(f"\n✅ Final Excel file '{output_excel}' created successfully with {len(data)} results!")
