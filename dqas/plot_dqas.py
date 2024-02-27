import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv('C:/Users/yanzh/PycharmProjects/Unitary_decomposition/dqas/csv_files_for_matrix/dimension_qubit_5/combined_csv/new_dataset_for_full_plots_full.csv')

df_first_group = df.iloc[1:1001]
df_second_group = df.iloc[1001:]

plt.figure(figsize=(10, 6))

plt.plot(df_first_group['steps'], df_first_group['avg_loss'], label='Group 1', color='blue')

# 第二组数据
plt.plot(df_second_group['steps'], df_second_group['avg_loss'], label='Group 2', color='red')

plt.xlabel('Steps')
plt.ylabel('Loss')
plt.title('Loss vs. Steps for Two Groups')
plt.legend()
plt.grid(True)

plt.show()

'''
import pandas as pd
import matplotlib.pyplot as plt

csv_file_path = 'C:/Users/yanzh/PycharmProjects/Unitary_decomposition/dqas/csv_files_for_matrix/dimension_qubit_5/' \
                'combined_csv/new_dataset_for_full_plots_full.csv'

df = pd.read_csv(csv_file_path)

x = df['steps']
y = df['avg_loss']

plt.figure(figsize=(10, 6))
plt.plot(x, y, marker='o', linestyle='-', color='blue')
plt.title('Average Loss vs. Steps')
plt.xlabel('Steps')
plt.ylabel('Average Loss')
plt.grid(True)
plt.show()
'''
