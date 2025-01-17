def compute_metrics(
        ground_truths,
        answers_zeroshot = [],
        answers_zeroshot_with_knowledge = [],
        answers_zeroshot_cot = [],
        answers_fewshot = [],
        answers_fewshot_with_knowledge = [],
        answers_fewshot_cot = [],
):
    correct_zeroshot = 0
    correct_zeroshot_with_knowledge = 0
    correct_zeroshot_cot = 0
    correct_fewshot = 0
    correct_fewshot_with_knowledge = 0
    correct_fewshot_cot = 0

    for (
        gt,
        zs,
        zs_wk,
        zs_cot,
        fs,
        fs_wk,
        fs_cot
    ) in zip(
        ground_truths,
        answers_zeroshot,
        answers_zeroshot_with_knowledge,
        answers_zeroshot_cot,
        answers_fewshot,
        answers_fewshot_with_knowledge,
        answers_fewshot_cot
    ):
        if gt == zs.strip():
            correct_zeroshot += 1
        if gt == zs_wk.strip():
            correct_zeroshot_with_knowledge += 1
        if gt == zs_cot.strip():
            correct_zeroshot_cot += 1

        if gt == fs.strip():
            correct_fewshot += 1
        if gt == fs_wk.strip():
            correct_fewshot_with_knowledge += 1
        if gt == fs_cot.strip():
            correct_fewshot_cot += 1

    accuracy_zeroshot = correct_zeroshot/len(ground_truths)
    accuracy_zeroshot_with_knowledge = correct_zeroshot_with_knowledge/len(ground_truths)
    accuracy_zeroshot_cot = correct_zeroshot_cot/len(ground_truths)

    accuracy_fewshot = correct_fewshot/len(ground_truths)
    accuracy_fewshot_with_knowledge = correct_fewshot_with_knowledge/len(ground_truths)
    accuracy_fewshot_cot = correct_fewshot_cot/len(ground_truths)

    return {'accuracy_zeroshot': accuracy_zeroshot,
            'accuracy_zeroshot_with_knowledge': accuracy_zeroshot_with_knowledge,
            'accuracy_zeroshot_cot': accuracy_zeroshot_cot,
            'accuracy_fewshot': accuracy_fewshot,
            'accuracy_fewshot_with_knowledge': accuracy_fewshot_with_knowledge,
            'accuracy_fewshot_cot': accuracy_fewshot_cot,
            }