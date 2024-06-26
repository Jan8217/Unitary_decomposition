import glob
import pandas

csv_list = glob.glob('C:/Users/yanzh/PycharmProjects/Unitary_decomposition/MIT_code/csv_files_for_matrix/dimension_32/*.csv')

def format_pandas(df):

	df_new = df.iloc[:1].copy()
	df_new['end loss'] = df['loss'].iloc[-1]
	df_new['total steps'] = df['gradient descent step'].iloc[-1]
	return df_new
	#df_new = df[0:1]
	#df_new['end loss'] = df['loss'][df.shape[0]-1]
	#df_new['total steps'] = df['gradient descent step'][df.shape[0]-1]
	#return df_new

appended_csv = format_pandas(pandas.read_csv(csv_list[0]))
appended_csv['filename'] = csv_list[0]
appended_csv = format_pandas(appended_csv)
full_append = pandas.read_csv(csv_list[0])
full_append['filename'] = csv_list[0]

#appended_csv = pandas.read_csv(csv_list[0])
#appended_csv['filename'] = csv_list[0]
#appended_csv = format_pandas(appended_csv)
#full_append = pandas.read_csv(csv_list[0])
#full_append['filename'] = csv_list[0]

for file_i in csv_list[1:]:
	new_csv = pandas.read_csv(file_i)
	new_csv['filename'] = file_i

	appended_csv = pandas.concat([appended_csv, format_pandas(new_csv)])
	full_append = pandas.concat([full_append, new_csv])
	#appended_csv = appended_csv.append(format_pandas(new_csv))
	#full_append = full_append.append(new_csv)

# appended_csv.drop(appended_csv.columns[0], axis=1)

print(appended_csv)
combined_csv_file_name = 'new_dataset_for_full_plots.csv'
appended_csv.to_csv('C:/Users/yanzh/PycharmProjects/Unitary_decomposition/MIT_code/csv_files_for_matrix/dimension_32/combined_csv/'+combined_csv_file_name,index = False)

full_csv_file_name = 'new_dataset_for_full_plots_full.csv'
full_append.to_csv('C:/Users/yanzh/PycharmProjects/Unitary_decomposition/MIT_code/csv_files_for_matrix/dimension_32/combined_csv/'+full_csv_file_name,index = False)

