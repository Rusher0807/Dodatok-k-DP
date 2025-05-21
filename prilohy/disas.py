import os, zipfile, lief, r2pipe
from pathlib import Path
from tqdm import tqdm

def extract_zip(zip_path, extract_to):
    try:
        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(path=extract_to, pwd=b'infected')
        return True
    except RuntimeError as e:
        print(f"[!] Failed to extract {zip_path}: {e}")
        return False

def analyze_file(filepath, out_txt, out_meta):
    try:
        # Disassembly with radare2
        r2 = r2pipe.open(filepath)
        r2.cmd('aaa')  # analyze all
        asm = r2.cmd('pdf @ entry0')  # disasm from entrypoint
        with open(out_txt, 'w') as f:
            f.write(asm)
        r2.quit()

        # Metadata with LIEF
        bin = lief.parse(filepath)
        with open(out_meta, 'w') as f:
            f.write(f"Format: {bin.format}\n")
            f.write(f"Entry point: {hex(bin.entrypoint)}\n")
            f.write("Imported libraries:\n")
            for lib in bin.libraries:
                f.write(f"  - {lib}\n")
    except Exception as e:
        print(f"[!] Error analyzing {filepath}: {e}")

def main(root_folder):
    for dirpath, _, filenames in os.walk(root_folder):
        print(f"Scanning folder: {dirpath}")  # 👈 DEBUG
        for f in filenames:
            if f.endswith('.zip'):
                zip_path = os.path.join(dirpath, f)
                print(f"[+] Found zip: {zip_path}")  # 👈 DEBUG
                malware_name = f[:-4]
                extract_path = os.path.join(dirpath, malware_name)
                os.makedirs(extract_path, exist_ok=True)

                if not extract_zip(zip_path, extract_path):
                    continue

                for file in os.listdir(extract_path):
                    if file.endswith(('.exe', '.bin', '.dll')):
                        full_path = os.path.join(extract_path, file)
                        print(f"[*] Analyzing: {full_path}")  # 👈 DEBUG
                        out_txt = os.path.join(extract_path, malware_name + ".txt")
                        out_meta = os.path.join(extract_path, malware_name + "_metadata.txt")
                        analyze_file(full_path, out_txt, out_meta)


if __name__ == "__main__":
    main("C:\\Users\\PCPC\\Downloads\\theZoo-master\\theZoo-master\\malware\\Binaries")
