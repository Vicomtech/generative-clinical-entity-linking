import pandas as pd


def build_summary_scores(df_scores, split_name):
    metrics = ['ACCURACY', 'ROUGE-L-F1', 'BLEU', 'METEOR', 'BERTSCORE', 'SEMSCORE', 'PL1', 'PLII']
    available_metrics = [metric for metric in metrics if metric in df_scores.columns]

    grouped = df_scores.groupby('model')[available_metrics].agg(['mean', 'std'])

    formatted = pd.DataFrame(index=grouped.index)
    for metric in available_metrics:
        mean_values = grouped[(metric, 'mean')]
        std_values = grouped[(metric, 'std')]
        formatted[metric] = [f"{mean:.2f} ± {std:.2f}" for mean, std in zip(mean_values, std_values)]

    formatted = formatted.reset_index()
    formatted['split'] = split_name
    ordered_columns = ['split', 'model'] + available_metrics
    formatted = formatted[ordered_columns]

    print(f"\n{split_name.upper()} SUMMARY")
    print(formatted.to_string(index=False))
    return formatted


df_test_scores = pd.read_csv("models_test_scores_seeds.tsv", sep='\t')
df_test_scores_unseen = pd.read_csv("models_test_scores_seeds_unseen.tsv", sep='\t')

summary_seen = build_summary_scores(df_test_scores, 'seen')
summary_unseen = build_summary_scores(df_test_scores_unseen, 'unseen')

final_summary_df = pd.concat([summary_seen, summary_unseen], ignore_index=True)
final_summary_df.to_csv("models_test_scores_summary.tsv", sep='\t', index=False)

print("\nFINAL SUMMARY (seen + unseen)")
print(final_summary_df.to_string(index=False))
print("\nSaved: models_test_scores_summary.tsv")

