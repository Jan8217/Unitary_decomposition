import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('C:/Users/yanzh/PycharmProjects/Unitary_decomposition/dqas/csv_files_for_matrix/dimension_qubit_3/combined_csv/new_dataset_for_full_plots_full.csv')

df_first_group = df.iloc[1:3343]
df_second_group = df.iloc[3344:6716]
df_third_group = df.iloc[6717:9847]
df_forth_group = df.iloc[9848:13036]
df_fifth_group = df.iloc[13037:16359]
df_sixth_group = df.iloc[16360:19612]
df_seveth_group = df.iloc[19613:22820]
df_eighth_group = df.iloc[22821:26316]
df_ninth_group = df.iloc[26317:29538]
df_tenth_group = df.iloc[29539:32721]
df_eleventh_group = df.iloc[32722:35748]


plt.figure(figsize=(10, 6))

plt.scatter(df_first_group['steps'], df_first_group['avg_loss'], label='Group 1', color='blue')
plt.scatter(df_second_group['steps'], df_second_group['avg_loss'], label='Group 2', color='red')
plt.scatter(df_third_group['steps'], df_third_group['avg_loss'], label='Group 3', color='green')
plt.scatter(df_forth_group['steps'], df_forth_group['avg_loss'], label='Group 4', color='yellow')
plt.scatter(df_fifth_group['steps'], df_fifth_group['avg_loss'], label='Group 5', color='black')
plt.scatter(df_sixth_group['steps'], df_sixth_group['avg_loss'], label='Group 6', color='gray')
plt.scatter(df_seveth_group['steps'], df_seveth_group['avg_loss'], label='Group 7', color='purple')
plt.scatter(df_eighth_group['steps'], df_eighth_group['avg_loss'], label='Group 8', color='brown')
plt.scatter(df_ninth_group['steps'], df_ninth_group['avg_loss'], label='Group 9', color='skyblue')
plt.scatter(df_tenth_group['steps'], df_tenth_group['avg_loss'], label='Group 10', color='lightgreen')
plt.scatter(df_eleventh_group['steps'], df_eleventh_group['avg_loss'], label='Group 11', color='aqua')

plt.xlabel('Steps')
plt.ylabel('Loss')
plt.title('Loss for 3 qubit with 25 placeholders')
plt.legend()
plt.grid(True)

plt.show()
