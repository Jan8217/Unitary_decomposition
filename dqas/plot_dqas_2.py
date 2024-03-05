import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

df = pd.read_csv('C:/Users/yanzh/PycharmProjects/Unitary_decomposition/dqas/csv_files_for_matrix/dimension_qubit_3/new_operation_pool_03_20_placeholder/combined_csv/new_dataset_for_full_plots_full.csv')



plt.figure(figsize=(10, 6))

# 为了使图表清晰，为每条线选择一个颜色
colors = plt.cm.viridis(np.linspace(0, 1, 10))

# 绘制10条线，每条线代表10000个数据点
group_size = 10000  # 每组数据的大小
data_length = len(df)
for i in range(10):
    start_idx = i * group_size
    end_idx = start_idx + group_size
    # 由于数据长度可能不够，需要检查end_idx不超过data_length
    end_idx = min(end_idx, data_length)
    plt.plot(df['steps'][start_idx:end_idx], df['avg_loss'][start_idx:end_idx],
             label=f'Group {i+1}', color=colors[i])

plt.title('Loss for 3 qubit with 20 placeholders - Grouped Lines')
plt.xlabel('Steps')
plt.ylabel('Loss')
plt.legend(loc='upper right')  # 修改图例的位置，以防止与线重叠
plt.grid(True)
plt.show()