import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv('C:/Users/yanzh/PycharmProjects/Unitary_decomposition/dqas/csv_files_for_matrix/dimension_qubit_3/new_operation_pool_10/combined_csv/new_dataset_for_full_plots_full.csv')


df_first_group = df.iloc[1:1523]
#df_first_group = df_first_group.sort_values('steps')

df_second_group = df.iloc[1524:2620]
#df_second_group = df_second_group.sort_values('steps')

df_third_group = df.iloc[2621:3849]
#df_third_group = df_third_group.sort_values('steps')

df_forth_group = df.iloc[3850:4924]
#df_forth_group = df_forth_group.sort_values('steps')

df_fifth_group = df.iloc[4925:6090]
#df_fifth_group = df_fifth_group.sort_values('steps')

df_sixth_group = df.iloc[6091:7380]
#df_sixth_group = df_sixth_group.sort_values('steps')

df_sevth_group = df.iloc[7381:8648]
#df_sevth_group = df_sevth_group.sort_values('steps')

df_eighth_group = df.iloc[8649:9795]
#df_eighth_group = df_eighth_group.sort_values('steps')

df_ninth_group = df.iloc[9796:10999]
#df_ninth_group = df_ninth_group.sort_values('steps')

df_tenth_group = df.iloc[11000:11850]
#df_tenth_group = df_tenth_group.sort_values('steps')

'''
df_11_group = df.iloc[11851:12716]
#df_first_group = df_first_group.sort_values('steps')

df_12_group = df.iloc[12717:22716]
#df_second_group = df_second_group.sort_values('steps')

df_13_group = df.iloc[22717:24173]
#df_third_group = df_third_group.sort_values('steps')

df_14_group = df.iloc[24174:34173]
#df_forth_group = df_forth_group.sort_values('steps')

df_15_group = df.iloc[34174:35103]
#df_fifth_group = df_fifth_group.sort_values('steps')

df_16_group = df.iloc[35104:36161]
#df_sixth_group = df_sixth_group.sort_values('steps')

df_17_group = df.iloc[36162:46161]
#df_sevth_group = df_sevth_group.sort_values('steps')

df_18_group = df.iloc[46162:46707]
#df_eighth_group = df_eighth_group.sort_values('steps')

df_19_group = df.iloc[46708:48068]
#df_ninth_group = df_ninth_group.sort_values('steps')

df_20_group = df.iloc[48069:52259]

df_21_group = df.iloc[52260:57294]
#df_eighth_group = df_eighth_group.sort_values('steps')
'''

plt.figure(figsize=(10, 6))

plt.plot(df_first_group['steps'], df_first_group['avg_loss'], label='Group 1_lr_0.04', color='blue')
plt.plot(df_second_group['steps'], df_second_group['avg_loss'], label='Group 2_lr_0.04', color='red')
plt.plot(df_third_group['steps'], df_third_group['avg_loss'], label='Group 3_lr_0.04', color='green')
plt.plot(df_forth_group['steps'], df_forth_group['avg_loss'], label='Group 4_lr_0.05', color='yellow')
plt.plot(df_fifth_group['steps'], df_fifth_group['avg_loss'], label='Group 5_lr_0.05', color='black')
plt.plot(df_sixth_group['steps'], df_sixth_group['avg_loss'], label='Group 6_lr_0.05', color='gray')
plt.plot(df_sevth_group['steps'], df_sevth_group['avg_loss'], label='Group 7_lr_0.06', color='purple')
plt.plot(df_eighth_group['steps'], df_eighth_group['avg_loss'], label='Group 8_lr_0.06', color='brown')
plt.plot(df_ninth_group['steps'], df_ninth_group['avg_loss'], label='Group 9_lr_0.06', color='peru')
plt.plot(df_tenth_group['steps'], df_tenth_group['avg_loss'], label='Group 10_lr_0.07', color='olive')
'''
plt.plot(df_11_group['steps'], df_11_group['avg_loss'], label='Group 11_lr_0.07', color='navy')
plt.plot(df_12_group['steps'], df_12_group['avg_loss'], label='Group 12_lr_0,07', color='lavender')
plt.plot(df_13_group['steps'], df_13_group['avg_loss'], label='Group 13_lr_0.08', color='plum')
plt.plot(df_14_group['steps'], df_14_group['avg_loss'], label='Group 14_lr_0.08', color='coral')
plt.plot(df_15_group['steps'], df_15_group['avg_loss'], label='Group 15_lr_0.08', color='tan')
plt.plot(df_16_group['steps'], df_16_group['avg_loss'], label='Group 16_lr_0.09', color='lightgreen')
plt.plot(df_17_group['steps'], df_17_group['avg_loss'], label='Group 17_lr_0.09', color='greenyellow')
plt.plot(df_18_group['steps'], df_18_group['avg_loss'], label='Group 18_lr_0.09', color='aqua')
plt.plot(df_19_group['steps'], df_19_group['avg_loss'], label='Group 19_lr_0.10', color='crimson')
plt.plot(df_20_group['steps'], df_20_group['avg_loss'], label='Group 20_lr_0.10', color='pink')
plt.plot(df_21_group['steps'], df_21_group['avg_loss'], label='Group 21_lr_0.10', color='limegreen')
'''
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.title('Loss for 3 qubit with 20 placeholders')
plt.legend()
plt.grid(True)

plt.show()
