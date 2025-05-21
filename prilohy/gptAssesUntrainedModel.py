import os
import json
from pathlib import Path
import openai
import google.generativeai as genai

# === Configuration ===
client = openai.OpenAI(api_key="sk-proj-bJEVQAMprgvAOYoebMaGd2rw9aYjsQDeMoQ6z-tY5h7BnViuFlguggSmGkAn-KRvszIXp6SFiLT3BlbkFJt9VSuy4kDofQz2UDTM7xKytDChET4cADBPBC0crx9cf5_eeDiJrU91v9JPmes8R6kgfC9w9dQA")  # Replace with your actual OpenAI key
genai.configure(api_key="AIzaSyBo77iW0sQZB9pWUFy2D1fiH69iDWia4Vk")  # Replace with your Gemini key

disasm_dir = Path("./data_new/malicious_disas_cleaned")
metadata_dir = Path("./data_new/malicious_metadata_new")
output_dir = Path("./data_new/malicious_annotated_output")
output_dir.mkdir(exist_ok=True)

#models = genai.list_models()
#for model in models:
#    print(model.name)
#models = client.models.list()
#for model in models:
#    print(model.id)
#exit()

# === Match metadata for each disassembly ===
def match_metadata_file(disasm_file):
    base = disasm_file.stem
    metadata_file = metadata_dir / f"{base}_metadata.txt"
    return metadata_file if metadata_file.exists() else None

# === Load file contents ===
def load_sample(disasm_path, metadata_path):
    with open(disasm_path, encoding='utf-8') as f:
        disasm = f.read()
    with open(metadata_path, encoding='utf-8') as f:
        metadata = f.read()
    return disasm, metadata

# === Prompt builder ===
def build_prompt(disasm, metadata):
    return f"""
You are a malware analyst. Below is disassembled x86 code.

Your task:
1. Split the disassembly into logical code blocks (8–20 instructions).
2. For each block:
    - Summarize what the code block does.
    - Determine whether it is 'malicious' or 'benign'.

⚠️ IMPORTANT:
- If the disassembly is too long to fully process, you may skip some blocks or summarize only the first 100–200 instructions.
- Focus on blocks that show interesting behavior (e.g., API calls, memory manipulation, string handling, control flow).
- It's OK to only output a subset of the code blocks.

💡 Format:
Output one JSON object per line in this format:

{{
  "text": "### Disassembled Code:\\n<assembly block>\\n\\n### Summary:\\n<brief summary>\\n\\n### Label:\\nmalicious/benign"
}}

Each line should:
- Be a valid JSON object.
- Escape all newlines inside the string (`\\n`).
- Contain no extra commentary or explanation.

Disassembly:
{disasm}
""".strip()

# === Send to OpenAI GPT-4 ===
def annotate_with_gpt(prompt):
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "You are a malware analyst."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )
    return response.choices[0].message.content

# === Send to Gemini Pro ===
def annotate_with_gemini(prompt):
    model = genai.GenerativeModel("gemini-1.5-pro-latest")
    response = model.generate_content(prompt)
    return response.text

# === Save annotations to output folder ===
def save_annotation(output_file, content, model_name):
    annotated_path = output_dir / f"{output_file.stem}_{model_name}.jsonl"
    with open(annotated_path, "w", encoding='utf-8') as f:
        f.write(content)

# === Main Execution ===
samples = []
for disasm_file in disasm_dir.glob("*.txt"):
    if disasm_file.stat().st_size < 2048:  # Skip files < 2KB
        print(f"[!] Skipping {disasm_file.name} (too small)")
        continue

    metadata_file = match_metadata_file(disasm_file)
    if metadata_file:
        samples.append((disasm_file, metadata_file))

print(f"[+] Matched {len(samples)} samples")

for disasm_path, metadata_path in samples:
    print(f"[>] Processing: {disasm_path.name}")

    disasm, metadata = load_sample(disasm_path, metadata_path)
    prompt = build_prompt(disasm, metadata)

    # GPT
    try:
        gpt_output = annotate_with_gpt(prompt)
        save_annotation(disasm_path, gpt_output, "gpt4")
    except Exception as e:
        print(f"[!] GPT error on {disasm_path.name}: {e}")

    # Gemini
    #try:
    #    gemini_output = annotate_with_gemini(prompt)
    #    save_annotation(disasm_path, gemini_output, "gemini")
    #except Exception as e:
    #    print(f"[!] Gemini error on {disasm_path.name}: {e}")

    #exit() #try 1