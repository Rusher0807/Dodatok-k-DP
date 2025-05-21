import os
import re

INPUT_DIR = "./malicious_disas"
OUTPUT_DIR = "./malicious_disas_cleaned"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def clean_line(line):
    line = re.sub(r"^[\s|/\\:.\-,=`><]+", "", line)

    line = line.split(';')[0].rstrip()

    match = re.match(r"^(0x[0-9a-fA-F]+)\s+([0-9a-fA-F]{2,}(?:\s[0-9a-fA-F]{2,})*)\s+(.*)", line)
    if match:
        addr, _, instr = match.groups()
        return f"{addr} {instr.strip()}"

    return line if line.strip() else None

for filename in os.listdir(INPUT_DIR):
    if not filename.endswith(".txt"):
        continue

    input_path = os.path.join(INPUT_DIR, filename)
    output_path = os.path.join(OUTPUT_DIR, filename)

    with open(input_path, "r", encoding="utf-8", errors="ignore") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            cleaned = clean_line(line)
            if cleaned:
                fout.write(cleaned + "\n")
