#  **Inštalačná príručka**

Táto príručka popisuje inštaláciu potrebného softvéru a knižníc pre praktickú časť diplomovej práce: Detekcia správania malware s využitím LLM modelov.

**Nastavenie virtuálneho stroja (pre sťahovanie a spracovanie malvérov)**

1. Nainštalujte Hyper-V alebo [VirtualBox](https://www.virtualbox.org/wiki/Downloads).
2. Stiahnite si obraz systému Windows (ISO súbor) pomocou [Windows Media Creation Tool](https://www.microsoft.com/sk-sk/software-download/windows10).
3. Vytvorte nový virtuálny stroj v Hyper-V alebo VirtualBox a nainštalujte Windows pomocou stiahnutého ISO súboru.

**Inštalácia potrebného softvéru**

- Python: [python.org](https://www.python.org/downloads/)
- CUDA Toolkit: [developer.nvidia.com/cuda-downloads](https://developer.nvidia.com/cuda-downloads)
- Radare2: <https://github.com/radareorg/radare2/releases>

**Inštalácia potrebných knižníc Python**

- Cez cmd použite: _pip install torch transformers bitsandbytes peft datasets trl langchain langchain-community langchain-ollama ollama openpyxl lief r2pipe tqdm pandas matplotlib requests openai google-generativeai faiss-cpu_

**Inštalácia Ollama**

1. Stiahnite a nainštalujte [ollama.ai](https://ollama.ai).
2. Cez cmd použite: _ollama pull llama3.2_

**Stiahnutie ostatných modelov**

- Mistral: <https://huggingface.co/mistralai/Mistral-7B-v0.3>
- CodeLlama: <https://huggingface.co/meta-llama/CodeLlama-7b-hf>



# **Používateľská príručka**

Táto požívateľská príručka popisuje skripty a ich využitie v rámci praktickej časti diplomovej práce. Tieto skripty môžeme nájsť ako prílohu k práci, alebo tu na githube.

**Stiahnutie vzoriek malvéru (odporúča sa vo virtuálnom prostredí)**

- Automaticky pomocou skriptu _gitDownloadDike.py. Po spustení skriptu sa stiahnu vzorky malvérov zo stránky_ [DikeDataset.](https://github.com/iosifache/DikeDataset/tree/main/files/malware)

Stiahnuť vzorky môžete aj manuálne zo stránok:

- [theZoo](https://github.com/ytisf/theZoo), [MalwareBazaar](https://bazaar.abuse.ch/browse/) alebo [DikeDatas?t](https://github.com/iosifache/DikeDataset/tree/main/files/malware)

**Stiahnutie vzoriek neškodlivých súborov**

- Automaticky pomocou skriptu _gitDownloadDike.py. Pred spustením treba zmeniť link a výstupnú zložku._

_api_url =_ [_https://api.github.com/repos/iosifache/DikeDataset/contents/files/malware_](https://api.github.com/repos/iosifache/DikeDataset/contents/files/malware) na _api_url =_ <https://api.github.com/repos/iosifache/DikeDataset/contents/files/benign>. Výstupnú zložku _malicious_ napr. na _benign_ v riadku 21 a 31.

Stiahnuť vzorky môžete aj manuálne z rovnakej stránky.

- [DikeDataset](https://github.com/iosifache/DikeDataset/tree/main/files/benign)

**Spracovanie súborov na assemblerský kód a metadáta**

- V skripte _disas.py_ nastavte cestu k datasetu, ktorý následne spracujete (zložka _malicious_ alebo _benign_ podľa predošlého kroku)
- Skript vytvorí z každého súboru _názov.txt_, a _názov_metadata.txt_.
- Skript bol pôvodne vytvorený na automatické spracovanie archívov (.zip) obsahujúci malvéry zo zdroja [theZoo](https://github.com/ytisf/theZoo), ktoré sú zaheslované heslom _infected_.

**Čistenie assemblerských dát**

- Po vytvorení assemblerských dát použite skript _disasCleaner.py_ na odstránenie nepotrebných znakov. Premenú _INPUT_DIR_ zmeňte podľa Vašej vstupnej zložky, ktorá obsahuje súbory s assemblerským kódom a premennú _OUTPUT_DIR_ podľa výstupnej zložky, do ktorej sa uložia súbory bez náhodných znakov

**Vytváranie dát na RAG a dotrénovanie**

- Využitím skriptu _gptAssesUntrainedModel.py_. Je potrebné nastaviť api kľúč pre gpt alebo gemini. Defaultne sa používa gpt, pre použitie gemini sa môže zakomentovať riadky 114 až 118 a odkomentovať riadky 120 až 125.
- Je potrebné nastaviť cestu pre vstupnú zložku s vytvorenými assemblerskými dátami _disasm_dir_ na riadku 11 a následne pre zložku s metadátami na riadku 12 _metadata_dir_.
- Výstupné vytvorené súbory sa budú nachádzať v zložke s cestou _output_dir, ktorý môžeme zmeniť na riadku 13._

**Spustenie a testovanie llamy3B od Ollamy**

- Skript _ollamaCreateEmbedings.py_ slúži na vytvorenie vektorovej databázy pre Llamu3B. Na riadku 8 zmeňme \[_“malware_rag.json“,“benign_rag.json“_\] na súbory v prílohe \[_“rag_benign_snipet”, “rag_malware_snippet”_\]_._ Alebo môžeme použiť vytvorenú sadu pomocou gpt/gemini podľa predošlého kroku.
- Skript _llamaAnswer.py_ môžeme následne využívať na generovanie odpovedí za použití vektorovej databázy. Stačí nám zvoliť cestu pre _input_folder_, ktorá obsahuje assemblerské kódy. Následne sa nám budú generovať odpovede do _output_excel_.
- Pokiaľ chceme použiť Llamu3B bez RAG, tak nám stačí odstrániť embedding model a retriever na riadku 15,16,17 a 23. Následne na riadku 45 zmeniť _result = qa_chain.invoke(query)_ na _result = {"result": llm.invoke(query)}_

**Spustenie a testovanie lokálnych modelov Mistral a CodeLlama**

- Skript _load7BwholeInput.py_ analyzuje celý súbor naraz a defaultne používa dotrénovaný model načítaním LoRA adaptérov. Pri testovaní nedotrénovaného modelu, na riadku 46 treba prepísať _model = PeftModel.from_pretrained(base_model, "lora-checkpoint-ll14")_ na _model=base_model_.
- Na riadku 39 a 49 je potrebné zmeniť cestu k nášmu zvolenému modelu.
- Na riadku 58 cestu k priečinku s assemblerskými súbormi a na riadku 59 názov excel súboru, do ktorého sa zapíšu výsledky analýzy (sumarizácia a označenie _Malicious_ alebo _Benign_). Nedotrénovaný model môže generovať aj označenie unknown alebo môže byť aj bez označenia.
- Môžeme si zvoliť aj dĺžku maximálne generovaných tokenov alebo parameter _temperature_ na riadku 103.
- Skript _load7BchunkInput.py_ funguje podobne, akurát rozdieľ je v spracovaní údajov. Vstupný assemblerský kód sa rozdelí na segmenty o veľkosti _CHUNK_SIZE_ (defaultne 20 riadkov). Je zvolená hranica _MALICIOUS_THRESHOLD_, pokiaľ LLM vyhodnotí aspoň 50% vstupných segmentov za škodlivé tak súbor je označený ako škodlivý. Toto môžeme vidieť vo výstupných excel súboroch (ako napr. _FT_analysis_benign.xlsx_)
- Defaultne používa dotrénovaný model načítaním LoRA adaptérov. Pri testovaní nedotrénovaného modelu, na riadku 50 treba prepísať _model = PeftModel.from_pretrained(base_model, "_ _lora-checkpoint-MistralN3epoch")_ na _model=base_model_.
- Cestu k nášmu modelu môžeme zvoliť na riadku 44 a 51.
- Riadok 31 je pre zložku vstupných assemblerských súborov a riadok 32 je výstupný excel s vyhodnotením.
- Na riadku 93 môžeme zmeniť maximálnu dĺžku generovaných tokenov a _temperature_.
- Skript _ChunkAnalysis.py je určený na vyhodnotenie celého excel súboru s označeniami._ Stačí zmeniť na riadku 6 cestu pre súbor excelu. Výstupné excely z nedotrénovaných modelov môžu spôsobiť nepresné čísla kvôli chýbajúcemu Labelu.

**Testovanie RAG pre Mistral a CodeLlama**

- Vektorovú databázu vytvoríme pomocou skriptu _localEmbeddingsCode5T.py_. Predvolene je nastavený anotovaný súbor z prílohy _malware_finetune.json_. Alebo nami vytvorené údaje podľa kroku vytváranie dát na RAG a dotrénovanie.
- Po vytvorení databázy môžeme použiť skript _load7BRAG.py_, ktorý načíta vstupné súbory zo zložky definovanej v _input_dir_ na riadku 57 a následne uloží výsledky do excelového súboru, ktorý môžeme zvoliť na riadku 58. Cestu k použitému modelu treba zmeniť v riadkoch 70 a 77.

**Dotrénovanie Mistral a CodeLlama**

- Na dotrénovanie slúži skript _fineTune7B.py_.
- Zvolíme cestu pre náš vstupný model na riadku 26 _model_path=”zvolená_cesta”_
- Na dataset na dotrénovanie v riadku 69 použijeme prílohu _malware_finetune_data_snippet.jsonl._ Alebo nami vytvorené údaje podľa kroku vytváranie dát na RAG a dotrénovanie.
- Následne zvolíme výstupné zložky na riadku 83 _output_dir_ a 108 _model.save_pretrained(“zvolená cesta”)_
