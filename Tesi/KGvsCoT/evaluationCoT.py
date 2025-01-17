import os
# Redirect caching
os.environ["HF_HOME"] = "/media/ssd/leonardofettucciari/cache1"
from dotenv import load_dotenv
import transformers
import torch
import tqdm
import csv
from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

from utils.model_utils import get_answers
from utils.prompt_utils import prepare_fewshot_prompt_cot, prepare_zeroshot_prompt_cot
from utils.csv_utils import parse_csv_cot
from utils.metrics_utils import compute_metrics

def evaluation_cot(model_name, train_data_path, eval_data_path):

    # Device
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = 'cpu'

    # Authentication for gated models e.g. LLama
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    login(hf_token)

    model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=device,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
    tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True, padding_side="left"
        )

    # CSV input data
    train_data_csv = parse_csv_cot(train_data_path)
    eval_data_csv = parse_csv_cot(eval_data_path)


    # Initialize result lists
    ground_truths = []
    answers_zeroshot_text_based = []
    answers_zeroshot_prob_based = []
    answers_fewshot_text_based = []
    answers_fewshot_prob_based = []


    # Iterate over the samples
    iterator = tqdm.tqdm(
        enumerate(eval_data_csv),
        total=len(eval_data_csv),
        desc="Generating Answers...",
    )

    final_list = []
    for i, sample in iterator:
        final = {}

        # Get the ground truth
        ground_truth = sample["answerKey"]
        ground_truths.append(ground_truth)

        # Build prompts
        zeroshot_prompt = prepare_zeroshot_prompt_cot(sample)
        fewshot_prompt = prepare_fewshot_prompt_cot(sample, train_data_csv)

        # Generate answers
        zeroshot_text_based_answer, zeroshot_prob_based_answer = get_answers(model, tokenizer, zeroshot_prompt, model_name)
        fewshot_text_based_answer, fewshot_prob_based_answer = get_answers(model, tokenizer, fewshot_prompt, model_name)

        # Get information about the sample
        final['id'] = sample["id"]
        final['question'] = sample["question"]
        final['choices'] = "\n".join([f"{choice}" for choice in sample["choices"]["text"]])

        zeroshot_text_based_answer = zeroshot_text_based_answer.strip('\n').split('Answer:')
        fewshot_text_based_answer = fewshot_text_based_answer.strip('\n').split('Answer:')
        final['zeroshot_raw_output'] = zeroshot_text_based_answer
        final['fewshot_raw_output'] = fewshot_text_based_answer
        final['gold_truth'] = sample['answerKey']

        # Zero shot results
        if(len(zeroshot_text_based_answer) > 1):
            final['zeroshot_text_answer'] = zeroshot_text_based_answer[1].split('.')[0].strip()
        else:
            final['zeroshot_text_answer'] = ""
        final['zeroshot_prob_answer'] = zeroshot_prob_based_answer.strip()

        # Few shot results
        if(len(fewshot_text_based_answer) > 1):
            final['fewshot_text_answer'] = fewshot_text_based_answer[1].split('.')[0].strip()
        else:
            final['fewshot_text_answer'] = ""
        final['fewshot_prob_answer'] = fewshot_prob_based_answer.strip()

        # Reasoning
        final['zeroshot_reasoning'] = zeroshot_text_based_answer[0]
        final['fewshot_reasoning'] = fewshot_text_based_answer[0]

        final_list.append(final)

        # Append answers
        answers_zeroshot_text_based.append(final['zeroshot_text_answer'])
        answers_zeroshot_prob_based.append(final['zeroshot_prob_answer'])
        answers_fewshot_text_based.append(final['fewshot_text_answer'])
        answers_fewshot_prob_based.append(final['fewshot_prob_answer'])

    # Save output
    with open(f"Tesi/KGvsCoT/output/{model_name.split('/')[1]}_COT.tsv", mode="w", newline="", encoding="utf-8") as file:
        tsv_writer = csv.DictWriter(file, fieldnames=final_list[0].keys(), delimiter="\t")
        tsv_writer.writeheader()
        tsv_writer.writerows(final_list)

    # Metrics
    metrics = compute_metrics(
        ground_truths,
        answers_zeroshot_text_based,
        answers_zeroshot_prob_based,
        answers_fewshot_text_based,
        answers_fewshot_prob_based
        )
    
    # Free up resources
    del model
    torch.cuda.empty_cache()

    return metrics