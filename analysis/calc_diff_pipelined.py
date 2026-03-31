import pandas as pd 
import os 


file_codiesp = 'generative_performance.csv'
df = pd.read_csv(os.path.join('tables', file_codiesp))


metrics = ["PL I", "PL II"]

for metric in metrics:
    df[f'{metric}_diff'] = round(df[f'{metric} Test'] - df[f'{metric} Unseen'], 2)

print(df.head())

means = {}
stds = {}
for diff in [f'{metric}_diff' for metric in metrics]:
    mean = df[diff].mean()
    std = df[diff].std()
    means[diff] = round(mean, 2)
    stds[diff] = round(std, 2)


print('Means:', means)
print('Stds: ', stds)
df_means = pd.DataFrame(means, index=[0])
df_means['Model'] = 'Mean'
df_stds = pd.DataFrame(stds, index=[0])
df_stds['Model'] = 'Std'

df0 = pd.concat([df, df_means, df_stds], ignore_index=True)


df0 = df0[["Model", "PL I_diff",  "PL II_diff"]]
df0.to_csv('tables/generative_performance_diff.csv', index=False)

