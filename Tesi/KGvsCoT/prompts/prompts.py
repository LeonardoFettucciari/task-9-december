# 1
template_with_knowledge = '''
Question:
{question}

Choices:
{choices}

Knowledge:
{knowledge}
'''

# 2
template_cot = '''
Question:
{question}

Choices:
{choices}
'''

# 3
template = '''
Question:
{question}

Choices:
{choices}
'''

# 4
template_fewshot_with_knowledge = '''
Example {count}:

Question:
{question}

Choices:
{choices}

Knowledge:
{knowledge}
'''

# 5.1
template_fewshot_cot = '''
Example {count}:

Question:
{question}

Choices:
{choices}
'''

# 5.2
template_fewshot_cot_assistant = '''
Reasoning:
{reasoning}

Answer:
{answer}
'''

# 6
template_fewshot = '''
Example {count}:

Question:
{question}

Choices:
{choices}
'''

# System prompts
SYSTEM_ZERO_SHOT = '''
You are an expert in commonsense question answering. \
Given a question, along with its choices, \
provide only the label of the correct answer. \
'''

SYSTEM_ZERO_SHOT_WITH_KNOWLEDGE = '''
You are an expert in commonsense question answering. \
Given a question, along with its choices and a list of useful commonsense statements, \
provide only the label of the correct answer. \
'''

SYSTEM_ZERO_SHOT_COT = '''
You are an expert in commonsense question answering. \
Given a question, along with its choices, \
provide the reasoning process necessary to answer the question \
and then provide only the label of the correct answer. \
For the label, use the format 'Answer: <label>'. \
'''

SYSTEM_FEW_SHOT = '''
You are an expert in commonsense question answering. \
Given a question, along with its choices, \
provide only the label of the correct answer. \
'''

SYSTEM_FEW_SHOT_WITH_KNOWLEDGE = '''
You are an expert in commonsense question answering. \
Given a question, along with its choices and a list of useful commonsense statements, \
provide only the label of the correct answer. \
'''

SYSTEM_FEW_SHOT_COT = '''
You are an expert in commonsense question answering. \
Given a question, along with its choices, \
provide the reasoning process necessary to answer the question \
and then provide only the label of the correct answer. \
'''
