import os
# Redirect caching
os.environ["HF_HOME"] = "/media/ssd/leonardofettucciari/cache1"
from dotenv import load_dotenv
import transformers
import torch
import csv
from evaluationCoT import evaluation_cot
from evaluationKG import evaluation_kg
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM
# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Authentication for gated models e.g. LLama
load_dotenv()  # Load environment variables from .env file
hf_token = os.getenv("HF_TOKEN")
login(hf_token)

model_list = ["meta-llama/Llama-3.2-3B-Instruct",
              "meta-llama/Llama-3.1-8B-Instruct",
              "Qwen/Qwen2.5-1.5B-Instruct",
              "Qwen/Qwen2.5-7B-Instruct",
              ]

full_metrics = []
for model_name in model_list:
    
    metrics_kg = evaluation_kg(
        model_name,
        "Tesi/KGvsCoT/data/trainKG.csv",
        "Tesi/KGvsCoT/data/evalKG.csv"
        )

    metrics_cot = evaluation_cot(
        model_name,
        "Tesi/KGvsCoT/data/trainCOT.csv",
        "Tesi/KGvsCoT/data/evalCOT.csv"
        )
    
    model_metrics = {
        'model_name': model_name,
        'ZT_KG': metrics_kg['accuracy_zs_t'],
        'ZT_CoT': metrics_cot['accuracy_zs_t'],

        'ZP_KG': metrics_kg['accuracy_zs_p'],
        'ZP_CoT': metrics_cot['accuracy_zs_p'],

        'FT_KG': metrics_kg['accuracy_fs_t'],
        'FT_CoT': metrics_cot['accuracy_fs_t'],

        'FP_KG': metrics_kg['accuracy_fs_p'],
        'FP_CoT': metrics_cot['accuracy_fs_p'],
    }

    full_metrics.append(model_metrics)
    
# Write metrics
with open(f"Tesi/KGvsCoT/output/metrics.tsv", mode="w", newline="", encoding="utf-8") as file:
    # Create a DictWriter with keys as fieldnames
    tsv_writer = csv.DictWriter(file, fieldnames=full_metrics[0].keys(), delimiter="\t")
    # Write the header (keys)
    tsv_writer.writeheader()
    # Write the row (values)
    tsv_writer.writerows(full_metrics)

