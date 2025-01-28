import torch
import re
def generate_text(model,
                  tokenizer,
                  prompt,
                  model_name,
                  max_new_tokens=512,
                  system=True,
                  device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    # Convert to chat template
    messages = []
    if system:
        messages.append({"role": "system", "content": prompt.pop(0)})
        
    for turn_id, turn in enumerate(prompt):
        if turn_id % 2 == 0:
            messages.append({"role": "user", "content": turn})
        else:
            messages.append({"role": "assistant", "content": turn})

    inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to(device)

    # Truncate depending on model used
    if model_name == "meta-llama/Llama-3.2-3B-Instruct":
        inputs = inputs[:, :-1]
    elif model_name == "meta-llama/Llama-3.1-8B-Instruct":
        inputs = inputs[:, :-1]
    elif model_name == "Qwen/Qwen2.5-1.5B-Instruct":
        inputs = inputs[:, :-2]
    elif model_name == "Qwen/Qwen2.5-7B-Instruct":
        inputs = inputs[:, :-2]
    else:
        inputs = inputs[:, :-1]


    # Create the attention mask.
    attention_mask = torch.ones_like(inputs).to(device)

    # Ensure pad_token_id is set
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Generate text using the model.
    model_outputs = model.generate(
        inputs,
        attention_mask=attention_mask,
        pad_token_id=tokenizer.pad_token_id,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.0,
        top_p=1.0,
        num_beams=1,
        output_scores=True,
        return_dict_in_generate=True,
    )

    # Decode the generated text.
    generated_text = tokenizer.decode(
        model_outputs.sequences[0][inputs[0].shape[0] :],
        skip_special_tokens=True,
    )

    return model_outputs, generated_text


def get_answers(model,
                tokenizer,
                prompt,
                model_name,
                max_new_tokens=512,
                system=True,
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    
    # Generate alternative labels with whitespaces in front.
    labels = ['A', 'B', 'C', 'D', 'E']
    labels.extend([f" {label}" for label in labels])

    # Generate text using the model.
    model_outputs, raw_generated_answer = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        model_name=model_name,
        max_new_tokens=max_new_tokens,
        system=system,
        device=device
    )

    # Get the probabilities of the first token.
    probabilities = torch.log_softmax(model_outputs.scores[-1], dim=-1)[0]

    # Check that the labels are in the tokenizer's vocabulary.
    labels = [label for label in labels if len(tokenizer.tokenize(label)) == 1]

    # Get the label IDs.
    label_ids = [tokenizer.encode(label, add_special_tokens=False)[0] for label in labels]

    # Get the probability of each label (A, B, C, D, E) and its variants.
    answer = [probabilities[label_id].item() for label_id in label_ids]


    # Get the label with the highest probability.
    answer = labels[answer.index(max(answer))]
    answer = answer.lstrip()

    return raw_generated_answer, answer

def parse_answer_cot(raw_answer):
    # Pattern to capture the reasoning text
    answer_pattern_list = []
    reasoning_pattern_list = []
    reasoning_pattern1 = r'(.*?)(?=\nAnswer:)'
    reasoning_pattern2 = r'(.*?)(?=\ncorrect answer is)'

    reasoning_pattern_list.append(reasoning_pattern1)
    reasoning_pattern_list.append(reasoning_pattern2)

    # Pattern to capture the answer label
    answer_pattern1 = r'Answer:\s*([A-Z])'
    answer_pattern2 = r'correct answer is\s*([A-Z])'

    answer_pattern_list.append(answer_pattern1)
    answer_pattern_list.append(answer_pattern2)

    # Extracting the reasoning text
    for reasoning_pattern in reasoning_pattern_list:
        reasoning_match = re.search(reasoning_pattern, raw_answer, re.MULTILINE)
        reasoning_text = reasoning_match.group(1).strip() if reasoning_match else ""
        if reasoning_match:
            break
        else:
            continue

    # Extracting the answer label
    for answer_pattern in answer_pattern_list:
        answer_match = re.search(answer_pattern, raw_answer, re.MULTILINE)
        answer_label = answer_match.group(1) if answer_match else ""
        if answer_match:
            break
        else:
            continue

    return reasoning_text, answer_label
    