import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv('C:/Users/yanzh/PycharmProjects/Unitary_decomposition/dqas/csv_files_for_matrix/dimension_qubit_3/new_operation_pool_09/combined_csv/new_dataset_for_full_plots_full.csv')


df_first_group = df.iloc[1:5051]
#df_first_group = df_first_group.sort_values('steps')

df_second_group = df.iloc[5052:8928]
#df_second_group = df_second_group.sort_values('steps')

df_third_group = df.iloc[8929:10337]
#df_third_group = df_third_group.sort_values('steps')

df_forth_group = df.iloc[10338:12187]
#df_forth_group = df_forth_group.sort_values('steps')

df_fifth_group = df.iloc[12188:13332]
#df_fifth_group = df_fifth_group.sort_values('steps')

#df_sixth_group = df.iloc[30155:40154]
#df_sixth_group = df_sixth_group.sort_values('steps')

#df_sevth_group = df.iloc[40155:42473]
#df_sevth_group = df_sevth_group.sort_values('steps')

#df_eighth_group = df.iloc[42474:44757]
#df_eighth_group = df_eighth_group.sort_values('steps')

#df_ninth_group = df.iloc[44758:46115]
#df_ninth_group = df_ninth_group.sort_values('steps')

#df_tenth_group = df.iloc[46116:47577]
#df_tenth_group = df_tenth_group.sort_values('steps')

plt.figure(figsize=(10, 6))

plt.plot(df_first_group['steps'], df_first_group['avg_loss'], label='Group 1_lr_0.01', color='blue')
plt.plot(df_second_group['steps'], df_second_group['avg_loss'], label='Group 2_lr_0,02', color='red')
plt.plot(df_third_group['steps'], df_third_group['avg_loss'], label='Group 3_lr_0.03', color='green')
plt.plot(df_forth_group['steps'], df_forth_group['avg_loss'], label='Group 4_lr_0.04', color='yellow')
plt.plot(df_fifth_group['steps'], df_fifth_group['avg_loss'], label='Group 5_lr_0.05', color='black')
#plt.plot(df_sixth_group['steps'], df_sixth_group['avg_loss'], label='Group 6_lr_0.01', color='gray')
#plt.plot(df_sevth_group['steps'], df_sevth_group['avg_loss'], label='Group 7_lr_0.02', color='purple')
#plt.plot(df_eighth_group['steps'], df_eighth_group['avg_loss'], label='Group 8_lr_0.02', color='brown')
#plt.plot(df_ninth_group['steps'], df_ninth_group['avg_loss'], label='Group 9_lr_0.03', color='skyblue')
#plt.plot(df_tenth_group['steps'], df_tenth_group['avg_loss'], label='Group 10_lr_0.03', color='lightgreen')

plt.xlabel('Steps')
plt.ylabel('Loss')
plt.title('Loss for 3 qubit with 20 placeholders')
plt.legend()
plt.grid(True)

plt.show()
