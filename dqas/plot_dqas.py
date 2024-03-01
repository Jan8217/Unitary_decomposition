import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('C:/Users/yanzh/PycharmProjects/Unitary_decomposition/dqas/csv_files_for_matrix/dimension_qubit_3/new_operation_pool_02/combined_csv/new_dataset_for_full_plots_full.csv')

df_first_group = df.iloc[2:10001]
df_second_group = df.iloc[10002:20001]
df_third_group = df.iloc[20002:30001]
df_forth_group = df.iloc[30002:40001]
df_fifth_group = df.iloc[40002:50001]
#df_sixth_group = df.iloc[50002:60001]
#df_seveth_group = df.iloc[60002:70001]
#df_eighth_group = df.iloc[70002:80001]
#df_ninth_group = df.iloc[80002:90001]
#df_tenth_group = df.iloc[90002:100001]

plt.figure(figsize=(10, 6))

plt.plot(df_first_group['steps'], df_first_group['avg_loss'], label='Group 1', color='blue')
plt.plot(df_second_group['steps'], df_second_group['avg_loss'], label='Group 2', color='red')
plt.plot(df_third_group['steps'], df_third_group['avg_loss'], label='Group 3', color='green')
plt.plot(df_forth_group['steps'], df_forth_group['avg_loss'], label='Group 4', color='yellow')
plt.plot(df_fifth_group['steps'], df_fifth_group['avg_loss'], label='Group 5', color='black')
#plt.plot(df_sixth_group['steps'], df_sixth_group['avg_loss'], label='Group 6', color='gray')
#plt.plot(df_seveth_group['steps'], df_seveth_group['avg_loss'], label='Group 7', color='purple')
#plt.plot(df_eighth_group['steps'], df_eighth_group['avg_loss'], label='Group 8', color='brown')
#plt.plot(df_ninth_group['steps'], df_ninth_group['avg_loss'], label='Group 9', color='skyblue')
#plt.plot(df_tenth_group['steps'], df_tenth_group['avg_loss'], label='Group 10', color='lightgreen')

plt.xlabel('Steps')
plt.ylabel('Loss')
plt.title('Loss for 3 qubit with 20 placeholders')
plt.legend()
plt.grid(True)

plt.show()
