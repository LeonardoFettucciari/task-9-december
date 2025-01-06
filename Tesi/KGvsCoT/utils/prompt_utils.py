from prompts.prompts import template, example_template, SYSTEM_REQUEST_KG
from prompts.prompts import template_cot, example_template_cot, assistant_template_cot, SYSTEM_REQUEST_COT

def prepare_zeroshot_prompt_kg(sample):
  # system's shot
  system_shot = [SYSTEM_REQUEST_KG]

  # final user's shot
  question = sample['question']
  choices = sample['choices']['text']
  labels = sample['choices']['label']
  choices = "\n".join([f"{label}. {choice}" for label, choice in zip(labels, choices)])
  statements = sample['statements']
  knowledge = "\n".join([f"{statement}" for statement in statements])

  final_shot = template.format(
      question=question,
      choices=choices,
      knowledge=knowledge
      )

  prompt = (
      system_shot
      + [final_shot, "Answer: "]
      )

  return prompt


def prepare_fewshot_prompt_kg(sample, examples):
  # system's shot
  system_shot = [SYSTEM_REQUEST_KG]

  # few shot examples
  shots = []
  for i, example in enumerate(examples, 1):
    question = example['question']
    choices = example['choices']['text']
    labels = example['choices']['label']
    choices = "\n".join([f"{label}. {choice}" for label, choice in zip(labels, choices)])
    statements = example['statements']
    knowledge = "\n".join([f"{statement}" for statement in statements])
    answer = example['answerKey']

    # user's turn
    shot = example_template.format(
        count=i,
        question=question,
        choices=choices,
        knowledge=knowledge
        )
    shots.append(shot)

    # assistant's turn
    shots.append(f"Answer: {answer}")


  # final user's shot
  question = sample['question']
  choices = sample['choices']['text']
  labels = sample['choices']['label']
  choices = "\n".join([f"{label}. {choice}" for label, choice in zip(labels, choices)])
  statements = sample['statements']
  knowledge = "\n".join([f"{statement}" for statement in statements])

  final_shot = template.format(
      question=question,
      choices=choices,
      knowledge=knowledge
      )

  prompt = (
      system_shot
      + shots
      + [final_shot, "Answer: "]
      )

  return prompt

def prepare_zeroshot_prompt_cot(sample):
  # system's shot
  system_shot = [SYSTEM_REQUEST_COT]

  # final user's shot
  question = sample['question']
  choices = sample['choices']['text']
  labels = sample['choices']['label']
  choices = "\n".join([f"{label}. {choice}" for label, choice in zip(labels, choices)])

  final_shot = template_cot.format(
      question=question,
      choices=choices,
      )

  prompt = (
      system_shot
      + [final_shot, "Reasoning:"]
      )

  return prompt


def prepare_fewshot_prompt_cot(sample, examples):
  # system's shot
  system_shot = [SYSTEM_REQUEST_COT]

  # few shot examples
  shots = []
  for i, example in enumerate(examples, 1):
    question = example['question']
    choices = example['choices']['text']
    labels = example['choices']['label']
    choices = "\n".join([f"{label}. {choice}" for label, choice in zip(labels, choices)])
    reasoning = example['reasoning']
    answer = example['answerKey']

    # user's turn
    shot = example_template_cot.format(
        count=i,
        question=question,
        choices=choices,
        reasoning=reasoning
        )
    shots.append(shot)

    # assistant's turn
    assistant_shot = assistant_template_cot.format(
      reasoning=reasoning,
      answer=answer
    )
    shots.append(assistant_shot)

  # final user's shot
  question = sample['question']
  choices = sample['choices']['text']
  labels = sample['choices']['label']
  choices = "\n".join([f"{label}. {choice}" for label, choice in zip(labels, choices)])

  final_shot = template_cot.format(
      question=question,
      choices=choices,
      )

  prompt = (
      system_shot
      + shots
      + [final_shot, "Reasoning:"]
      )

  return prompt