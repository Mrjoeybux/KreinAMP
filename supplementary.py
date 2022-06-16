from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score
import seaborn as sns
import os
import glob
import pickle
from kreinamp.loaders import init_dataset, kernels, algos
from kreinamp.globals import ROOTPATH
from kreinamp.utils import load_test_data
import numpy as np

strains = ["PA", "SA"]

NUM_DECIMALS = 3
delimeter = " | "


def plot_kmat_eig_vals(result, title):
	eig_vals = []
	for fold in result.folds:
		eig_vals.append(sorted(fold["kmat_eig_vals"], reverse=True))
	plt.plot(np.arange(len(eig_vals[0])), np.mean(np.asarray(eig_vals), axis=0))
	plt.yscale("symlog")
	plt.title(title)
	plt.show()


def plot_lengths():
	sns.set_style("white")
	plt.rcParams.update({
		"text.usetex": True})
	datasets = ["deepamp", "ampscan"]
	for key in datasets:
		dataset = init_dataset(key)
		X = dataset.load()[0].reshape(-1).tolist()
		lengths = [len(x) for x in X]
		plt.xlabel("Length")
		g = sns.histplot(lengths)
		g.set_yticks([x*50 for x in range(10)])
		save_path = "./images/{0}_lengths.png".format(key)
		plt.savefig(save_path)
		print("Image saved at: {}".format(save_path))
		plt.clf()


def dbaasp_predictions():
	table_column_width = max([len("DBAASP {}".format(strain)) for strain in strains])
	table_column_width = max(table_column_width, NUM_DECIMALS + 2)
	column_format = "{0: >" + str(table_column_width) + "}"
	numeric_format = "{0:." + str(NUM_DECIMALS) + "f}"
	headers = ["Model", "AUC", "ACC"]
	row_length = len(headers)*table_column_width + (len(headers) - 1)*len(delimeter)
	print("DBAASP Model performance on test set:\n")
	print("-"*row_length)
	print(delimeter.join([column_format.format(header) for header in headers]))
	print("-"*row_length)
	for strain in strains:
		path = os.path.join(ROOTPATH, "data", "independent_test",
		                    "dbaasp_{}_peptide_predictions.txt".format(strain.lower()))
		row = ["DBAASP {}".format(strain)]
		score_pred = []
		class_pred = []
		with open(path, "r") as f:
			for line in f:
				activity, confidence = line.split("\t")[-2:]
				confidence = float(confidence.split(" (")[0])
				if activity == "Not Active":
					score_pred.append(1 - confidence)
					class_pred.append(-1)
				else:
					score_pred.append(confidence)
					class_pred.append(1)
		seqs, names, true = load_test_data(strain)
		row.append(numeric_format.format(roc_auc_score(y_true=true, y_score=score_pred)))
		row.append(numeric_format.format(accuracy_score(y_true=true, y_pred=class_pred)))
		print(delimeter.join([column_format.format(val) for val in row]))
		print("-"*row_length)
	print("")


def load_saved_model(dataset):
	results = glob.glob("{0}/experiment_results/{1}/train_val_test/*".format(ROOTPATH, dataset))
	if not results:
		raise ValueError("No models found for {} dataset!".format(dataset))
	elif len(results) == 1:
		path = results[0]
	else:
		print("-------------------------------------------------------------------------------------------------------")
		print("{:5s}  |  Name".format("ID"))
		print("-------------------------------------------------------------------------------------------------------")
		for i, result in enumerate(results):
			print("{:5s}  |  {}".format(str(i), result.split("/")[-1]))
		print("-------------------------------------------------------------------------------------------------------")
		ID = int(input("Please input ID of model to load: "))
		try:
			path = results[ID]
		except:
			raise ValueError("Result {} does not exist!".format(ID))
	with open(path + "/results_file.dat", "rb") as f:
		result_obj = pickle.load(f)
	print("Loaded results at {}/results_file.dat".format(path))
	return path, result_obj


def load_kernel_function(key, params):
	kernel = kernels[key]
	kernel.set_params(**params)
	return kernel


def load_model(key, params):
	model = algos[key]
	model.set_params(**params)
	return model


def predict(args):
	results_path, results = load_saved_model(args.dataset)
	save_path = results_path + "/peptide_predictions.dat"
	dataset = init_dataset(args.dataset)
	X, y = dataset.load()
	with open(save_path, "w") as f:
		for strain in strains:
			for key in results.keys():
				result = results[key]
				kernel_key, model_key = key.split("_")
				kernel = load_kernel_function(kernel_key, result["best_kernel_params"])
				model = load_model(model_key, result["best_model_params"])
				train_kmat = kernel.fit_transform(X)
				test_data, names, amps = load_test_data(strain)
				test_kmat = kernel.transform(test_data)
				model.fit(train_kmat, y)
				pred = model.predict(test_kmat)
				pred[pred > 0] = 1
				pred[pred < 1] = -1
				acc = np.mean(pred == amps)
				f.write(
					"#######################################################################################################\n")
				f.write("{0} {1}\n".format(key, strain))
				f.write(
					"#######################################################################################################\n")
				f.write("\n")
				f.write("Accuracy, {0:.3f}\n".format(acc))
				f.write("\n")
				for i in range(pred.shape[0]):
					if np.sign(pred[i]) >= 0:
						pred_i = "AMP"
					else:
						pred_i = "non-AMP"
					f.write("{},{}\n".format(names[i], pred_i))
	print("\nResults saved at: {}".format(save_path))