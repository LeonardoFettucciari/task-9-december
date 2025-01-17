from prompts.prompts import (template, template_with_knowledge, template_cot,
                             template_fewshot, template_fewshot_with_knowledge, template_fewshot_cot, template_fewshot_cot_assistant)
from prompts.prompts import (SYSTEM_ZERO_SHOT, SYSTEM_ZERO_SHOT_WITH_KNOWLEDGE, SYSTEM_ZERO_SHOT_COT,
                             SYSTEM_FEW_SHOT, SYSTEM_FEW_SHOT_WITH_KNOWLEDGE, SYSTEM_FEW_SHOT_COT)

def prepare_prompt(sample,
                   zero_shot=False,
                   few_shot=False,
                   with_knowledge=False,
                   cot=False,
                   examples=[]):
  # Sample data
  question = sample['question']
  choices = "\n".join([f"{label}. {choice}" for label, choice in zip(sample['choices']['label'], sample['choices']['text'])])

  
  # 1/6 Zero-shot with knowledge
  if(zero_shot and with_knowledge):
    knowledge = "\n".join([f"{statement}" for statement in sample['statements']])
    final_shot = template_with_knowledge.format(
      question=question,
      choices=choices,
      knowledge=knowledge
      )
    
    return (
    [SYSTEM_ZERO_SHOT_WITH_KNOWLEDGE]
    + [final_shot]
    + ["Answer: "]
    )

  # 2/6 Zero-shot CoT
  elif(zero_shot and cot):
    final_shot = template_cot.format(
      question=question,
      choices=choices,
      )
    
    return (
    [SYSTEM_ZERO_SHOT_COT]
    + [final_shot]
    + ["Let's think step by step. "]
    )

  # 3/6 Zero-shot
  elif(zero_shot):
    final_shot = template.format(
      question=question,
      choices=choices,
      )
    
    return (
    [SYSTEM_ZERO_SHOT]
    + [final_shot]
    + ["Answer: "]
    )

  # 4/6 Few-shot with knowledge
  elif(few_shot and with_knowledge):
    shots = []
    for i, example in enumerate(examples, 1):
      example_question = example['question']
      example_choices = "\n".join([f"{label}. {choice}" for label, choice in zip(example['choices']['label'], example['choices']['text'])])
      example_answer = example['answerKey']
      example_knowledge = "\n".join([f"{statement}" for statement in example['statements']])

      shot = template_fewshot_with_knowledge.format(
          count=i,
          question=example_question,
          choices=example_choices,
          knowledge=example_knowledge,
          )
      
      shots += (
        [shot]
        + [f"Answer: {example_answer}"]
      )

    knowledge = "\n".join([f"{statement}" for statement in sample['statements']])
    final_shot = template_with_knowledge.format(
      question=question,
      choices=choices,
      knowledge=knowledge
      )
    
    return (
    [SYSTEM_FEW_SHOT_WITH_KNOWLEDGE]
    + shots
    + [final_shot]
    + ["Answer: "]
    )


  # 5/6 Few-shot CoT
  elif(few_shot and cot):
    shots = []
    for i, example in enumerate(examples, 1):
      example_question = example['question']
      example_choices = "\n".join([f"{label}. {choice}" for label, choice in zip(example['choices']['label'], example['choices']['text'])])
      example_answer = example['answerKey']
      example_reasoning = example['reasoning']

      shot = template_fewshot_cot.format(
          count=i,
          question=example_question,
          choices=example_choices
          )
      assistant_shot = template_fewshot_cot_assistant.format(
        reasoning=example_reasoning,
        answer=example_answer
      )
      
      shots += (
        [shot]
        + [assistant_shot]
      )

    final_shot = template_cot.format(
      question=question,
      choices=choices,
      )
    
    return (
    [SYSTEM_FEW_SHOT_COT]
    + shots
    + [final_shot]
    + ["Reasoning: "]
    )
  
  # 6/6 Few-shot
  elif(few_shot):
    shots = []
    for i, example in enumerate(examples, 1):
      example_question = example['question']
      example_choices = "\n".join([f"{label}. {choice}" for label, choice in zip(example['choices']['label'], example['choices']['text'])])
      example_answer = example['answerKey']

      shot = template_fewshot.format(
          count=i,
          question=example_question,
          choices=example_choices,
          )
      
      shots += (
        [shot]
        + [f"Answer: {example_answer}"]
      )

    final_shot = template.format(
      question=question,
      choices=choices,
      )
    
    return (
    [SYSTEM_FEW_SHOT]
    + shots
    + [final_shot]
    + ["Answer: "]
    )