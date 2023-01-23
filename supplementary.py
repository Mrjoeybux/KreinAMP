from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score, matthews_corrcoef
import seaborn as sns
import os
import glob
import pickle
from kreinamp.loaders import init_dataset, kernels, algos
from kreinamp.globals import ROOTPATH
from kreinamp.utils import load_test_data, symbol2weight
import numpy as np
import pandas as pd

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
		X, y = dataset.load()
		pos_idx = np.where(y == 1)
		neg_idx = np.where(y == -1)
		X = X.reshape(-1)
		X_pos = X[pos_idx].tolist()
		X_neg = X[neg_idx].tolist()
		data = []
		for x in X_pos:
			data.append({"Classification": "AMP", "Length": len(x)})
		for x in X_neg:
			data.append({"Classification": "non-AMP", "Length": len(x)})
		data = pd.DataFrame(data)

		plt.xlabel("Length")

		if key == "deepamp":
			g = sns.histplot(data, x="Length", hue="Classification", multiple="dodge", kde=True, alpha=0.6, kde_kws={"cut": 10, "clip": [0, 40]})
			g.set_yticks([x*50 for x in range(6)])
			g.set_xticks([x*5 for x in range(9)])
		else:
			g = sns.histplot(data, x="Length", hue="Classification", multiple="dodge", kde=True, alpha=0.6, kde_kws={"cut": 10, "clip": [0, 180]})
			g.set_yticks([x*50 for x in range(6)])
		save_path = "kreinamp/images/{0}_all_lengths.png".format(key)
		plt.savefig(save_path)
		print("Image saved at: {}".format(save_path))
		plt.clf()
	plot_aa_dist()


def _aa_dist(X):
	aa_counts = {}
	for aa in symbol2weight.keys():
		aa_counts[aa] = 0
	for x in X:
		unique = set(x)
		for aa in unique:
			aa_counts[aa] += x.count(aa)
	return aa_counts


def plot_aa_dist():
	sns.set_style("white")
	plt.rcParams.update({
		"text.usetex": True})
	datasets = ["deepamp", "ampscan"]
	data = []
	keymap = {
		"deepamp": "DeepAMP",
		"ampscan": "AMPScan"
		}
	for key in datasets:
		dataset = init_dataset(key)
		X, y = dataset.load()
		X = X.reshape(-1)

		for x in X.tolist():
			for aa in x:
				data.append({"Amino Acid": aa, "Dataset": keymap[key]})
	data = pd.DataFrame(data)
	data = data.sort_values(by=["Amino Acid"])

	ampscan_proportions = pd.DataFrame(data[data["Dataset"] == "AMPScan"].value_counts(normalize=True).reset_index())
	ampscan_proportions.columns = ["Amino Acid", "Dataset", "Proportion"]

	deepamp_proportions = pd.DataFrame(data[data["Dataset"] == "DeepAMP"].value_counts(normalize=True).reset_index())
	deepamp_proportions.columns = ["Amino Acid", "Dataset", "Proportion"]
	data = pd.concat([deepamp_proportions, ampscan_proportions]).reset_index(drop=True)
	data = data.sort_values(by="Amino Acid")
	print(data)
	g = sns.barplot(data=data, x="Amino Acid", y="Proportion", hue="Dataset", alpha=0.6, saturation=1)
	save_path = "kreinamp/images/all_counts.png"
	plt.savefig(save_path)
	print("Image saved at: {}".format(save_path))
	plt.clf()


def dbaasp_predictions():
	table_column_width = max([len("DBAASP {}".format(strain)) for strain in strains])
	table_column_width = max(table_column_width, NUM_DECIMALS + 2)
	column_format = "{0: >" + str(table_column_width) + "}"
	numeric_format = "{0:." + str(NUM_DECIMALS) + "f}"
	headers = ["Model", "AUC", "ACC", "MCC"]
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
		row.append(numeric_format.format(matthews_corrcoef(y_true=true, y_pred=class_pred)))
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