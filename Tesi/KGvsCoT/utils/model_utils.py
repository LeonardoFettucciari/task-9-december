import torch
def generate_text(model,
                  tokenizer,
                  prompt,
                  labels=[],
                  max_new_tokens=512,
                  system=True,
                  truncate=True,
                  device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    messages = []

    # Convert to chat template
    if system:
        messages.append({"role": "system", "content": prompt.pop(0)})
        
    for turn_id, turn in enumerate(prompt):
        if turn_id % 2 == 0:
            messages.append({"role": "user", "content": turn})
        else:
            messages.append({"role": "assistant", "content": turn})

    inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to(device)
    if truncate:
      inputs = inputs[:, :-2] # TODO replace with handler function

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
                max_new_tokens=512,
                system=True,
                truncate=True,
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):
    # Generate alternative labels with whitespaces in front.
    labels = ['A', 'B', 'C', 'D', 'E']
    labels.extend([f" {label}" for label in labels])

    # Generate text using the model.
    model_outputs, text_based_answer = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        system=system,
        truncate=truncate,
        device=device
    )

    # Get the probabilities of the first token.
    probabilities = torch.log_softmax(model_outputs.scores[-1], dim=-1)[0]

    # Check that the labels are in the tokenizer's vocabulary.
    labels = [label for label in labels if len(tokenizer.tokenize(label)) == 1]

    # Get the label IDs.
    label_ids = [
        tokenizer.encode(label, add_special_tokens=False)[0] for label in labels
    ]
    # Get the probability of each label (A, B, C, D, E) and its variants.
    answer = [probabilities[label_id].item() for label_id in label_ids]


    # Get the label with the highest probability.
    answer = labels[answer.index(max(answer))]
    prob_based_answer = answer.lstrip()

    return text_based_answer, prob_based_answer

    