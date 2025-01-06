def compute_metrics(
        ground_truths,
        answers_zeroshot_text_based=[],
        answers_zeroshot_prob_based=[],
        answers_fewshot_text_based=[],
        answers_fewshot_prob_based=[]
):
    correct_zs_t = 0
    correct_zs_p = 0
    correct_fs_t = 0
    correct_fs_p = 0

    for (
        gt,
        zs_t,
        zs_p,
        fs_t,
        fs_p
    ) in zip(
        ground_truths,
        answers_zeroshot_text_based,
        answers_zeroshot_prob_based,
        answers_fewshot_text_based,
        answers_fewshot_prob_based
    ):
        if gt == zs_t.strip():
            correct_zs_t += 1
        if gt == zs_p.strip():
            correct_zs_p += 1
        if gt == fs_t.strip():
            correct_fs_t += 1
        if gt == fs_p.strip():
            correct_fs_p += 1

    accuracy_zs_t = correct_zs_t/len(ground_truths)
    accuracy_zs_p = correct_zs_p/len(ground_truths)
    accuracy_fs_t = correct_fs_t/len(ground_truths)
    accuracy_fs_p = correct_fs_p/len(ground_truths)

    return {'accuracy_zs_t': accuracy_zs_t,
            'accuracy_zs_p': accuracy_zs_p,
            'accuracy_fs_t': accuracy_fs_t,
            'accuracy_fs_p': accuracy_fs_p
            }