# Dodatok-k-DP
Dodatok k práci Detekcia malvéru pomocou LLM

A.	Inštalačná príručka
Táto príručka popisuje inštaláciu potrebného softvéru a knižníc pre praktickú časť diplomovej práce: Detekcia správania malware s využitím LLM modelov.
Nastavenie virtuálneho stroja (pre sťahovanie a spracovanie malvérov)
1.	Nainštalujte Hyper-V alebo VirtualBox.
2.	Stiahnite si obraz systému Windows (ISO súbor) pomocou Windows Media Creation Tool.
3.	Vytvorte nový virtuálny stroj v Hyper-V alebo VirtualBox a nainštalujte Windows pomocou stiahnutého ISO súboru.
Inštalácia potrebného softvéru
•	Python: python.org 
•	CUDA Toolkit: developer.nvidia.com/cuda-downloads
•	Radare2: https://github.com/radareorg/radare2/releases
Inštalácia potrebných knižníc Python
•	Cez cmd použite: pip install torch transformers bitsandbytes peft datasets trl langchain langchain-community langchain-ollama ollama openpyxl lief r2pipe tqdm pandas matplotlib requests openai google-generativeai faiss-cpu 
Inštalácia Ollama
1.	Stiahnite a nainštalujte ollama.ai.
2.	Cez cmd použite: ollama pull llama3.2
Stiahnutie ostatných modelov
•	Mistral: https://huggingface.co/mistralai/Mistral-7B-v0.3
•	CodeLlama: https://huggingface.co/meta-llama/CodeLlama-7b-hf

-------------------------------------------------------------------------------------------------------------------------
B.	Používateľská príručka
Táto požívateľská príručka popisuje skripty a ich využitie v rámci praktickej časti diplomovej práce. Tieto skripty môžeme nájsť ako prílohu k práci, alebo tu na githube.
Stiahnutie vzoriek malvéru (odporúča sa vo virtuálnom prostredí)
•	Automaticky pomocou skriptu gitDownloadDike.py. Po spustení skriptu sa stiahnu vzorky malvérov zo stránky  DikeDataset.
Stiahnuť vzorky môžete aj manuálne zo stránok:
•	theZoo, MalwareBazaar alebo DikeDatas?t

Stiahnutie vzoriek neškodlivých súborov 
•	Automaticky pomocou skriptu gitDownloadDike.py. Pred spustením treba zmeniť link a výstupnú zložku.
api_url = https://api.github.com/repos/iosifache/DikeDataset/contents/files/malware na api_url = https://api.github.com/repos/iosifache/DikeDataset/contents/files/benign. Výstupnú zložku malicious napr. na benign v riadku 21 a 31.
Stiahnuť vzorky môžete aj manuálne z rovnakej stránky.
•	DikeDataset

Spracovanie súborov na assemblerský kód a metadáta
•	V skripte disas.py nastavte cestu k datasetu, ktorý následne spracujete (zložka malicious alebo benign podľa predošlého kroku)
•	Skript vytvorí z každého súboru názov.txt, a názov_metadata.txt.
•	Skript bol pôvodne vytvorený na automatické spracovanie archívov (.zip) obsahujúci malvéry zo zdroja theZoo, ktoré sú zaheslované heslom infected.

Čistenie assemblerských dát
•	Po vytvorení assemblerských dát použite skript disasCleaner.py na odstránenie nepotrebných znakov. Premenú INPUT_DIR zmeňte podľa Vašej vstupnej zložky, ktorá obsahuje súbory s assemblerským kódom a premennú OUTPUT_DIR podľa výstupnej zložky, do ktorej sa uložia súbory bez náhodných znakov

Vytváranie dát na RAG a dotrénovanie
•	Využitím skriptu gptAssesUntrainedModel.py. Je potrebné nastaviť api kľúč pre gpt alebo gemini. Defaultne sa používa gpt, pre použitie gemini sa môže zakomentovať riadky 114 až 118 a odkomentovať riadky 120 až 125.
•	Je potrebné nastaviť cestu pre vstupnú zložku s vytvorenými assemblerskými dátami disasm_dir na riadku 11 a následne pre zložku s metadátami na riadku 12 metadata_dir. 
•	Výstupné vytvorené súbory sa budú nachádzať v zložke s cestou output_dir, ktorý môžeme zmeniť na riadku 13.

Spustenie a testovanie llamy3B od Ollamy
•	Skript ollamaCreateEmbedings.py slúži na vytvorenie vektorovej databázy pre Llamu3B. Na riadku 8 zmeňme [“malware_rag.json“,“benign_rag.json“] na súbory v prílohe [“rag_benign_snipet”, “rag_malware_snippet”]. Alebo môžeme použiť vytvorenú sadu pomocou gpt/gemini podľa predošlého kroku.
•	Skript llamaAnswer.py môžeme následne využívať na generovanie odpovedí za použití vektorovej databázy. Stačí nám zvoliť cestu pre input_folder, ktorá obsahuje assemblerské kódy. Následne sa nám budú generovať odpovede do output_excel.
•	Pokiaľ chceme použiť Llamu3B bez RAG, tak nám stačí odstrániť embedding model a retriever na riadku 15,16,17 a 23. Následne na riadku 45 zmeniť result = qa_chain.invoke(query) na result = {"result": llm.invoke(query)}

Spustenie a testovanie lokálnych modelov Mistral a CodeLlama
•	Skript load7BwholeInput.py analyzuje celý súbor naraz a defaultne používa dotrénovaný model načítaním LoRA adaptérov. Pri testovaní nedotrénovaného modelu, na riadku 46 treba prepísať  model = PeftModel.from_pretrained(base_model, "lora-checkpoint-ll14") na model=base_model. 
•	Na riadku 39 a 49 je potrebné zmeniť cestu k nášmu zvolenému modelu.
•	Na riadku 58 cestu k priečinku s assemblerskými súbormi a na riadku 59 názov excel súboru, do ktorého sa zapíšu výsledky analýzy (sumarizácia a označenie Malicious  alebo Benign). Nedotrénovaný model môže generovať aj označenie unknown alebo môže byť aj bez označenia.
•	Môžeme si zvoliť aj dĺžku maximálne generovaných tokenov alebo parameter temperature na riadku 103.

•	Skript load7BchunkInput.py funguje podobne, akurát rozdieľ je v spracovaní údajov. Vstupný assemblerský kód sa rozdelí na segmenty o veľkosti CHUNK_SIZE (defaultne 20 riadkov). Je zvolená hranica MALICIOUS_THRESHOLD, pokiaľ LLM vyhodnotí aspoň 50% vstupných segmentov za škodlivé tak súbor je označený ako škodlivý. Toto môžeme vidieť vo výstupných excel súboroch (ako napr. FT_analysis_benign.xlsx)
•	Defaultne používa dotrénovaný model načítaním LoRA adaptérov. Pri testovaní nedotrénovaného modelu, na riadku 50 treba prepísať  model = PeftModel.from_pretrained(base_model, " lora-checkpoint-MistralN3epoch") na model=base_model. 
•	Cestu k nášmu modelu môžeme zvoliť na riadku 44 a 51.
•	Riadok 31 je pre zložku vstupných assemblerských súborov a riadok 32 je výstupný excel s vyhodnotením.
•	Na riadku 93 môžeme zmeniť maximálnu dĺžku generovaných tokenov a temperature.

•	Skript ChunkAnalysis.py je určený na vyhodnotenie celého excel súboru s označeniami. Stačí zmeniť na riadku 6 cestu pre súbor excelu. Výstupné excely z nedotrénovaných modelov môžu spôsobiť nepresné čísla kvôli chýbajúcemu Labelu.


Testovanie RAG pre Mistral a CodeLlama
•	Vektorovú databázu vytvoríme pomocou skriptu localEmbeddingsCode5T.py. Predvolene je nastavený anotovaný súbor z prílohy malware_finetune.json. Alebo nami vytvorené údaje podľa kroku vytváranie dát na RAG a dotrénovanie.
•	Po vytvorení databázy môžeme použiť skript load7BRAG.py, ktorý načíta vstupné súbory zo zložky definovanej v input_dir na riadku 57 a následne uloží výsledky do excelového súboru, ktorý môžeme zvoliť na riadku 58. Cestu k použitému modelu treba zmeniť v riadkoch 70 a 77.

Dotrénovanie Mistral a CodeLlama
•	Na dotrénovanie slúži skript fineTune7B.py. 
•	Zvolíme cestu pre náš vstupný model na riadku 26 model_path=”zvolená_cesta”
•	Na dataset na dotrénovanie v riadku 69 použijeme prílohu malware_finetune_data_snippet.jsonl. Alebo nami vytvorené údaje podľa kroku vytváranie dát na RAG a dotrénovanie.
•	Následne zvolíme výstupné zložky na riadku 83 output_dir a 108 model.save_pretrained(“zvolená cesta”)
