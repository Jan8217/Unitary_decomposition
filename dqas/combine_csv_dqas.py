import glob
import pandas
import os
import pandas as pd


csv_list = glob.glob('C:/Users/yanzh/PycharmProjects/Unitary_decomposition/dqas/csv_files_for_matrix/dimension_qubit_3/new_operation_pool_22/*.csv')

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

def check_and_create_combined_csv_folder(path):
    # 检查指定路径下是否存在combined_csv子文件夹
    combined_csv = os.path.join(path, "combined_csv")
    if not os.path.exists(combined_csv):
        # 如果combined_csv子文件夹不存在，则创建
        os.makedirs(combined_csv)
        print("子文件夹 'combined_csv' 已创建.")
    else:
        print("子文件夹 'combined_csv' 已存在.")

# 指定路径
folder_path = "C:/Users/yanzh/PycharmProjects/Unitary_decomposition/dqas/csv_files_for_matrix/dimension_qubit_3/new_operation_pool_22"

# 检查并创建combined_csv子文件夹
check_and_create_combined_csv_folder(folder_path)

combined_csv_file_name = 'new_dataset_for_full_plots.csv'
appended_csv.to_csv(os.path.join(folder_path, "combined_csv", combined_csv_file_name), index=False)

full_csv_file_name = 'new_dataset_for_full_plots_full.csv'
full_append.to_csv(os.path.join(folder_path, "combined_csv", full_csv_file_name), index=False)


#combined_csv_file_name = 'new_dataset_for_full_plots.csv'
#appended_csv.to_csv('C:/Users/yanzh/PycharmProjects/Unitary_decomposition/dqas/csv_files_for_matrix/dimension_qubit_3/new_operation_pool_16/combined_csv/'+combined_csv_file_name,index = False)

#full_csv_file_name = 'new_dataset_for_full_plots_full.csv'
#full_append.to_csv('C:/Users/yanzh/PycharmProjects/Unitary_decomposition/dqas/csv_files_for_matrix/dimension_qubit_3/new_operation_pool_16/combined_csv/'+full_csv_file_name,index = False)