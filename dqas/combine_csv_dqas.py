import glob
import pandas

csv_list = glob.glob('C:/Users/yanzh/PycharmProjects/Unitary_decomposition/dqas/csv_files_for_matrix/dimension_qubit_5/*.csv')

def format_pandas(df):
    df_new = df.iloc[:1].copy()
    df_new['end loss'] = df['avg_loss'].iloc[-1]
    df_new['total steps'] = df['steps'].iloc[-1]
    return df_new


appended_csv = format_pandas(pandas.read_csv(csv_list[0]))
appended_csv['filename'] = csv_list[0]
appended_csv = format_pandas(appended_csv)
full_append = pandas.read_csv(csv_list[0])
full_append['filename'] = csv_list[0]

for file_i in csv_list[1:]:
    new_csv = pandas.read_csv(file_i)
    new_csv['filename'] = file_i

    appended_csv = pandas.concat([appended_csv, format_pandas(new_csv)])
    full_append = pandas.concat([full_append, new_csv])

print(appended_csv)
combined_csv_file_name = 'new_dataset_for_full_plots.csv'
appended_csv.to_csv('C:/Users/yanzh/PycharmProjects/Unitary_decomposition/dqas/csv_files_for_matrix/dimension_qubit_5/combined_csv/'+combined_csv_file_name,index = False)

full_csv_file_name = 'new_dataset_for_full_plots_full.csv'
full_append.to_csv('C:/Users/yanzh/PycharmProjects/Unitary_decomposition/dqas/csv_files_for_matrix/dimension_qubit_5/combined_csv/'+full_csv_file_name,index = False)




'''
import pandas as pd
import os
import datetime

# 指定包含CSV文件的目录
csv_dir = 'C:/Users/yanzh/PycharmProjects/Unitary_decomposition/dqas/csv_files_for_matrix/dimension_qubit_5/'

# 读取目录下的所有CSV文件
csv_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]

# 合并所有CSV文件到一个DataFrame
df_list = []
for file in csv_files:
    df = pd.read_csv(os.path.join(csv_dir, file))
    df_list.append(df)

combined_df = pd.concat(df_list, ignore_index=True)

# 生成一个包含日期时间的唯一文件名
timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
output_filename = f"combined_csv_{timestamp}.csv"

# 定义最终的完整路径
csv_dir_d =  'C:/Users/yanzh/PycharmProjects/Unitary_decomposition/dqas/csv_files_for_matrix/dimension_qubit_5/combined_csv/'
output_path = os.path.join(csv_dir_d, output_filename)

# 保存合并后的DataFrame到CSV文件
combined_df.to_csv(output_path, index=False)

print(f"Combined CSV file saved as: {output_path}")
'''


