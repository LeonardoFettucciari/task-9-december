import csv

def parse_csv(path):
    with open(path, mode='r') as file:
        csv_reader = csv.DictReader(file)
        
        question_size = 50
        choice_size = 10
        question = {}
        questions_list = []

        # Read rows in chunks
        for i, row in enumerate(csv_reader, 1):

            # Add first row
            if i % question_size == 1:
                question['id'] = row['question_id']
                question['question'] = row['question']
                question['choices'] = {}
                question['choices']['label'] = ['A', 'B', 'C', 'D', 'E']
                question['choices']['text'] = []        # Append later
                question['answerKey'] = row['answerKey']          
                question['statements'] = []             # Append later

            # Add all relevant statements
            if row['isRelevant'] == "TRUE":
                question['statements'].append(row['statements'])
                
            # Add choices
            if i % choice_size == 1:
                question['choices']['text'].append(row['choices'])
                if row['choices'] == question['answerKey']:
                    question['answerKey'] = question['choices']['label'][len(question['choices']['text']) - 1]
            
            # Add question to list when finished
            if i % question_size == 0:
                questions_list.append(question)
                question = {}

            
    return questions_list

def parse_csv_cot(path):
    with open(path, mode='r') as file:
        csv_reader = csv.DictReader(file)
        
        questions_list = []

        # Read rows in chunks
        for i, row in enumerate(csv_reader):
            question = {}
            # Add first row
            question['id'] = row['question_id']
            question['question'] = row['question']
            question['choices'] = {}
            question['choices']['label'] = ['A', 'B', 'C', 'D', 'E']
            question['choices']['text'] = row['choices'].splitlines()
            question['answerKey'] = row['answerKey']
            question['reasoning'] = row['reasoning']
            
            questions_list.append(question)

            
    return questions_list