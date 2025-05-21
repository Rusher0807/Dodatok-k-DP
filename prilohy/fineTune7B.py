import torch
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
from datasets import load_dataset
from trl import SFTTrainer

# === Device & CUDA info ===
print(torch.__version__)
print(torch.cuda.is_available())
print("CUDA version:", torch.version.cuda)
print("GPU:", torch.cuda.get_device_name(0))

# === BitsAndBytes config ===
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

# === Load model & tokenizer ===
#D:\LLAMA7code
#D:\MistraGood\Mistra
model_path = r"D:\LLAMA7code"
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    device_map="auto",
    quantization_config=bnb_config,
    trust_remote_code=True
)
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)

# ✅ Patch missing pad_token (required for SFTTrainer / padding)
# === Ensure tokenizer has all required special tokens ===
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Patch BOS and EOS tokens if they're missing (Mistral might need this)
if tokenizer.bos_token is None or tokenizer.eos_token is None:
    tokenizer.add_special_tokens({
        "bos_token": "<s>",
        "eos_token": "</s>",
        "pad_token": tokenizer.pad_token or "</s>"
    })
    model.resize_token_embeddings(len(tokenizer))


# === LoRA config ===
lora_config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# === Prepare model ===
model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

# === Load and format dataset ===
#dataset = load_dataset("json", data_files=r"C:\Users\goodl\Downloads\SecureBert_Malware-Classification-main\malware_finetune.jsonl")["train"]
dataset = load_dataset("json", data_files=r"C:\Users\goodl\Downloads\SecureBert_Malware-Classification-main\malware_finetune.jsonl")["train"]

def format_example(example):
    return {"text": example["text"]}
    #prompt = f"### Instruction:\n{example['instruction']}\n"
    #if example.get("input"):
    #    prompt += f"### Input:\n{example['input']}\n"
    #prompt += f"### Response:\n{example['output']}"
    #return {"text": prompt}

dataset = dataset.map(format_example)

# === TrainingArguments ===
training_args = TrainingArguments(
    output_dir="lora-out-ll14",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=8,
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=5,  # Keep last 5 checkpoints
    evaluation_strategy="no",  # or "epoch" if you add validation
    learning_rate=2e-4,
    fp16=True,
    bf16=False,  # set True for A100/H100
    report_to="none"
)

# === Fine-tuning ===
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=dataset,
    args=training_args
)

trainer.train()

# === Save LoRA adapter ===
model.save_pretrained("lora-checkpoint-ll14")

# === [Optional] Load LoRA adapter later ===
# base_model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", trust_remote_code=True)
# merged_model = PeftModel.from_pretrained(base_model, "lora-checkpoint")
