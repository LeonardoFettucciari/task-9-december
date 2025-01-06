example_template = '''
Example {count}:

Question:
{question}

Choices:
{choices}

Knowledge:
{knowledge}
'''

assistant_template_cot = '''
Reasoning:
{reasoning}
Answer:
{answer}
'''

example_template_cot = '''
Example {count}:

Question:
{question}

Choices:
{choices}

Reasoning:
{reasoning}
'''

template = '''
Question:
{question}

Choices:
{choices}

Knowledge:
{knowledge}
'''

template_cot = '''
Question:
{question}

Choices:
{choices}
'''

SYSTEM_REQUEST_KG = '''
You are an expert in commonsense question answering. \
Given a question, along with its choices and a list of useful commonsense statements, \
provide only the label of the correct answer. \
'''

SYSTEM_REQUEST_COT = '''
You are an expert in commonsense question answering. \
Given a question and its choices, provide the reasoning process \
necessary to answer the question and then provide only the label of the correct answer. \
Use format Reasoning: <reasoning>
Answer: <label>. \
'''
