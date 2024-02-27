import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from collections import OrderedDict


def return_steps_and_loss_vecs(df):
	filenames = pd.unique(df['filename'])
	x_vec = []
	y_vec = []
	for file_i in filenames:
		rows = df['filename'] == file_i
		x_vec.append(df[rows]['gradient descent step'].to_numpy())
		y_vec.append(df[rows]['loss'].to_numpy())
	return filenames, x_vec, y_vec


def build_param_colormap(df):

	params = pd.unique(df.sort_values(by=['number of time parameters'])['number of time parameters'])
	print(params)
	colormap = {}
	n = 0
	for i in params:
		colormap[i] = n
		n += 1
	return colormap

def format_param_str(params, dim):
	new_params = []
	dim = dim[0]
	for i in params:
		# new_params.append('{}: ${:0.2}d^2$'.format(i,float(i)/dim/dim))
		new_params.append('${:0.2f}d^2$'.format(float(i)/dim/dim))

	return new_params

def build_and_save_plot(df, filename, 
	legend_title = 'Number of parameters', 
	x_axis_label = "Gradient Descent Steps"):
	filenames, x_vec, y_vec = return_steps_and_loss_vecs(df)
	colormap = build_param_colormap(df)

	# style
	plt.style.use('seaborn-v0_8-darkgrid')
	 
	# create a color palette
	palette = plt.get_cmap('Set1')

	for i, file_i in enumerate(filenames):
		n_params = pd.unique(df[df['filename'] == file_i]['number of time parameters'])
		n_params = n_params[0]
		plt.semilogy(x_vec[i], y_vec[i], marker='', color=palette(colormap[n_params]), linewidth=1.5, alpha=0.7, label=n_params)

	handles, labels = plt.gca().get_legend_handles_labels()
	by_label = OrderedDict(zip(labels, handles))

	if by_label:
		labels, handles = zip(*sorted(by_label.items(), key=lambda t: int(t[0])))
		unique_dims = pd.unique(df['dimension of unitary matrix'])

		if unique_dims.size > 0:
			labels = format_param_str(labels, unique_dims)
			plt.legend(handles, labels, bbox_to_anchor=(1.02, 1), title=legend_title)

		else:
			print("Warning: 'dimension of unitary matrix' is empty.")


	plt.subplots_adjust(right=0.75)

	plt.ylabel("Loss (log axis)")
	plt.xlabel(x_axis_label)

	fig = plt.gcf()
	fig.set_size_inches(5,4)
	plt.savefig("./figures/"+filename)	
	#plt.close()



if __name__ == '__main__':

	df = pd.read_csv('C:/Users/yanzh/PycharmProjects/Unitary_decomposition/MIT_code/csv_files_for_matrix/dimension_32/combined_csv/new_dataset_for_full_plots_full.csv')
	build_and_save_plot(df, 'figure_d4_unitary.pdf', '# params (2K)')


	build_and_save_plot(df[df['number of target parameters'] == 8],
						'figure_d4_unitary_adam.pdf',
						legend_title='# params (2K)',
						x_axis_label='Num. of Adam optimizer steps')
	plt.show()

