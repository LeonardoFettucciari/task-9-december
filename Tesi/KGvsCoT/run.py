import os
# Redirect caching
os.environ["HF_HOME"] = "/media/ssd/leonardofettucciari/cache1"
from dotenv import load_dotenv
import transformers
import torch
import csv
import tqdm
import re
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils.csv_utils import parse_csv, parse_csv_cot
from utils.prompt_utils import prepare_prompt
from utils.model_utils import get_answers, parse_answer_cot
from utils.metrics_utils import compute_metrics

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Authentication for gated models e.g. LLama
load_dotenv()
hf_token = os.getenv("HF_TOKEN")
login(hf_token)

# Models to run inference on
model_list = ["meta-llama/Llama-3.2-3B-Instruct",
              "meta-llama/Llama-3.1-8B-Instruct",
              "Qwen/Qwen2.5-1.5B-Instruct",
              "Qwen/Qwen2.5-7B-Instruct",
              ]

# I/O paths
shots_with_knowledge_path = "Tesi/KGvsCoT/data/train.csv"
shots_cot_path = "Tesi/KGvsCoT/data/train_cot.csv"
eval_data_with_knowledge_path = "Tesi/KGvsCoT/data/eval.csv"
eval_data_cot_path = "Tesi/KGvsCoT/data/eval_cot.csv"
metrics_output_path = "Tesi/KGvsCoT/output/metrics.tsv"

# Input parsing
shots_with_knowledge = parse_csv(shots_with_knowledge_path)
shots_cot = parse_csv_cot(shots_cot_path)
eval_data = parse_csv(eval_data_with_knowledge_path)

# Main
metrics_output = []
for model_index, model_name in enumerate(model_list, 1):
    print(f"Using model #{model_index}: {model_name}")

    # Load model and tokenizer
    model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
    tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, padding_side="left"
        )
    
    # Inizialize lists for computing metrics later
    ground_truths = []

    answers_zeroshot = []
    answers_zeroshot_with_knowledge = []
    answers_zeroshot_cot = []

    answers_fewshot = []
    answers_fewshot_with_knowledge = []
    answers_fewshot_cot = []
    
    # Iterate over the samples
    iterator = tqdm.tqdm(
        enumerate(eval_data),
        total=len(eval_data),
        desc="Generating Answers...",
    )
    
    full_output = []
    for i, sample in iterator:

        # Build prompts
        prompt_zeroshot =                prepare_prompt(sample, zero_shot=True)
        prompt_zeroshot_with_knowledge = prepare_prompt(sample, zero_shot=True, with_knowledge=True)
        prompt_zeroshot_cot =            prepare_prompt(sample, zero_shot=True, cot=True)

        prompt_fewshot =                 prepare_prompt(sample, few_shot=True, examples=shots_with_knowledge)
        prompt_fewshot_with_knowledge =  prepare_prompt(sample, few_shot=True, with_knowledge=True, examples=shots_with_knowledge)
        prompt_fewshot_cot =             prepare_prompt(sample, few_shot=True, cot=True, examples=shots_cot)

        # Generate answers
        raw_answer_zeroshot, answer_zeroshot =                               get_answers(model, tokenizer, prompt_zeroshot, model_name, max_new_tokens=1, device=device)
        raw_answer_zeroshot_with_knowledge, answer_zeroshot_with_knowledge = get_answers(model, tokenizer, prompt_zeroshot_with_knowledge, model_name, max_new_tokens=1, device=device)
        raw_answer_zeroshot_cot, _ =                                         get_answers(model, tokenizer, prompt_zeroshot_cot, model_name, max_new_tokens=512, device=device)

        raw_answer_fewshot, answer_fewshot =                                 get_answers(model, tokenizer, prompt_fewshot, model_name, max_new_tokens=1, device=device)
        raw_answer_fewshot_with_knowledge, answer_fewshot_with_knowledge =   get_answers(model, tokenizer, prompt_fewshot_with_knowledge, model_name, max_new_tokens=1, device=device)
        raw_answer_fewshot_cot, _ =                                          get_answers(model, tokenizer, prompt_fewshot_cot, model_name, max_new_tokens=512, device=device)


        # Apply regex to separate reasoning and answer's label
        cot_zeroshot_generated, answer_zeroshot_cot = parse_answer_cot(raw_answer_zeroshot_cot)
        cot_fewshot_generated, answer_fewshot_cot = parse_answer_cot(raw_answer_fewshot_cot)


        # Append answers for computing metrics later
        answers_zeroshot.append(answer_zeroshot)
        answers_zeroshot_with_knowledge.append(answer_zeroshot_with_knowledge)
        answers_zeroshot_cot.append(answer_zeroshot_cot)

        answers_fewshot.append(answer_fewshot)
        answers_fewshot_with_knowledge.append(answer_fewshot_with_knowledge)
        answers_fewshot_cot.append(answer_fewshot_cot)

        # Append the ground truth for computing metrics later
        ground_truths.append(sample["answerKey"])


        # Get information about the sample
        sample_output = {}
        sample_output['id'] = sample["id"]
        sample_output['question'] = sample["question"]
        sample_output['choices'] = "\n".join([f"{label}. {choice}" for label, choice in zip(sample['choices']['label'], sample['choices']['text'])])
        sample_output['gold_truth'] = sample['answerKey']
        sample_output['knowledge'] = "\n".join([f"{statement}" for statement in sample['statements']])
        sample_output['cot_zeroshot_generated'] = cot_zeroshot_generated   
        sample_output['cot_fewshot_generated'] = cot_fewshot_generated    
        
        sample_output['raw_answer_zeroshot'] = raw_answer_zeroshot   
        sample_output['raw_answer_zeroshot_with_knowledge'] = raw_answer_zeroshot_with_knowledge
        sample_output['raw_answer_zeroshot_cot'] = raw_answer_zeroshot_cot

        sample_output['raw_answer_fewshot'] = raw_answer_fewshot 
        sample_output['raw_answer_fewshot_with_knowledge'] = raw_answer_fewshot_with_knowledge
        sample_output['raw_answer_fewshot_cot'] = raw_answer_fewshot_cot

        sample_output['answer_zeroshot'] = answer_zeroshot
        sample_output['answer_zeroshot_with_knowledge'] = answer_zeroshot_with_knowledge
        sample_output['answer_zeroshot_cot'] = answer_zeroshot_cot

        sample_output['answer_fewshot'] = answer_fewshot
        sample_output['answer_fewshot_with_knowledge'] = answer_fewshot_with_knowledge
        sample_output['answer_fewshot_cot'] = answer_fewshot_cot

        full_output.append(sample_output)        

    # Save output
    with open(f"Tesi/KGvsCoT/output/{model_name.split('/')[1]}.tsv", mode="w", newline="", encoding="utf-8") as file:
        # Re-arrange output columns order as preferred
        fieldnames = ['id', 'question', 'choices', 'gold_truth',
                      
                      'answer_zeroshot', 'answer_zeroshot_with_knowledge', 'answer_zeroshot_cot',
                      'answer_fewshot', 'answer_fewshot_with_knowledge', 'answer_fewshot_cot',

                      'knowledge', 'cot_zeroshot_generated', 'cot_fewshot_generated',

                      'raw_answer_zeroshot', 'raw_answer_zeroshot_with_knowledge', 'raw_answer_zeroshot_cot',
                      'raw_answer_fewshot', 'raw_answer_fewshot_with_knowledge', 'raw_answer_fewshot_cot']
        
        tsv_writer = csv.DictWriter(file, fieldnames=fieldnames, delimiter="\t")
        tsv_writer.writeheader()
        tsv_writer.writerows(full_output)

    # Metrics
    metrics = compute_metrics(
        ground_truths,

        answers_zeroshot,
        answers_zeroshot_with_knowledge,
        answers_zeroshot_cot,

        answers_fewshot,
        answers_fewshot_with_knowledge,
        answers_fewshot_cot
        )

    # Free up resources
    del model
    torch.cuda.empty_cache()


    
    model_metrics = {
        'model_name': model_name,
        'accuracy_zeroshot': metrics['accuracy_zeroshot'],
        'accuracy_zeroshot_with_knowledge': metrics['accuracy_zeroshot_with_knowledge'],
        'accuracy_zeroshot_cot': metrics['accuracy_zeroshot_cot'],

        'accuracy_fewshot': metrics['accuracy_fewshot'],
        'accuracy_fewshot_with_knowledge': metrics['accuracy_fewshot_with_knowledge'],
        'accuracy_fewshot_cot': metrics['accuracy_fewshot_cot'],
    }

    metrics_output.append(model_metrics)
    
# Write metrics
with open(metrics_output_path, mode="w", newline="", encoding="utf-8") as file:
    tsv_writer = csv.DictWriter(file, fieldnames=metrics_output[0].keys(), delimiter="\t")
    tsv_writer.writeheader()
    tsv_writer.writerows(metrics_output)



