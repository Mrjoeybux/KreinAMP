from kreinamp.learning_algorithms import EigenDecomposition
from kreinamp.data import DBAASP
from tempfile import TemporaryDirectory
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import numpy as np
import os
import pickle
from datetime import datetime
import logging
from kreinamp.utils import init_loggers, ttest_errs
from kreinamp.globals import ROOTPATH, HPC_PYTHON_PATH, SHUFFLE_DATA, RANDOM_STATE
import shutil
from kreinamp.loaders import init_dataset, load_parameter_grid, algos, kernels, init_scoring_function
import subprocess

__logger = logging.getLogger("Experiment")

algos_and_kernels = [
	("LA", ["KSVM"]),
	("EDIT", ["KSVM"]),
	("GKM", ["SVM"]),
	]

# Formatting options for log file
NUM_DECIMALS = 3
delimeter = " | "
table_column_width = max([len(algo) for algo in algos]) + max([len(kernel) for kernel in kernels])
table_column_width = max(table_column_width, (NUM_DECIMALS + 2)*2 + 5)
column_format = "{0: >" + str(table_column_width) + "}"
numeric_format = "{0:." + str(NUM_DECIMALS) + "f}"
precision_numeric_format = "{0:." + str(NUM_DECIMALS*2) + "f}"


def check_dataset_dir(dataset_key):
	result_dir = os.path.join(ROOTPATH, "experiment_results")
	if not os.path.exists(result_dir):
		os.mkdir(result_dir)
	dataset_dir = os.path.join(result_dir, dataset_key)
	if not os.path.exists(dataset_dir):
		os.mkdir(dataset_dir)
	return dataset_dir


def create_save_dir(dataset_dir, num_outer_folds, name):
	dir_name = datetime.now().strftime("%d-%m-%Y--%H-%M-%S")
	if name is not None:
		dir_name += "_" + name
	if num_outer_folds == 1:
		exp_type_dir = os.path.join(dataset_dir, "train_val_test")
	else:
		exp_type_dir = os.path.join(dataset_dir, "nested_cv")
	if not os.path.exists(exp_type_dir):
		os.mkdir(exp_type_dir)
	current_dir = os.path.join(exp_type_dir, dir_name)
	os.mkdir(current_dir)
	return current_dir


def setup_logging(log_level, log_path, print_log):
	if print_log:
		path = None
	else:
		path = log_path
	init_loggers(log_level, path)


def gen_cv_folds(X, y, cv_func, num_folds):
	cv = cv_func(n_splits=num_folds, shuffle=SHUFFLE_DATA, random_state=RANDOM_STATE)
	return [(train, test) for train, test in cv.split(X, y)]


def tune_parameters(X, y, inner_cv_folds, model, kernel_function, num_jobs, scoring_func, parameter_grid):
	with TemporaryDirectory(prefix="kernel_matrix_cache_") as tmpdir:
		if model.requires_eigen_decomposition:
			pipe = Pipeline(steps=[
				("kernel", kernel_function),
				("eigen", EigenDecomposition()),
				("model", model),
				],
					memory=tmpdir)
		else:
			pipe = Pipeline(steps=[
				("kernel", kernel_function),
				("model", model),
				],
					memory=tmpdir)
		grid_search = GridSearchCV(estimator=pipe,
		                           cv=inner_cv_folds,
		                           param_grid=parameter_grid,
		                           scoring=scoring_func.cross_val_score(),
		                           refit=False,
		                           verbose=1,
		                           n_jobs=num_jobs)
		grid_search.fit(X, y)
		pipe.set_params(**grid_search.best_params_)
		return {"best_params"       : grid_search.best_params_,
		        "best_kernel_params": pipe.named_steps["kernel"].get_params(),
		        "best_model_params" : pipe.named_steps["model"].get_params()}


def run_one(kernel_key, model_key, X, y, train_args, test_args, inner_cv_folds, num_jobs, scoring_func, param_grid,
            store_eig_vals=False):
	kernel_function = kernels[kernel_key]
	model = algos[model_key]

	best_params = tune_parameters(X[train_args], y[train_args], inner_cv_folds, model, kernel_function, num_jobs,
	                              scoring_func, param_grid)

	best_model = algos[model_key]
	best_model.set_params(**best_params["best_model_params"])

	best_kernel = kernels[kernel_key]
	best_kernel.set_params(**best_params["best_kernel_params"])

	kmat_train = best_kernel.fit_transform(X[train_args], y[train_args])
	if store_eig_vals:
		new_kernel = kernels[kernel_key]
		new_kernel.set_params(**best_params["best_kernel_params"])
		kmat_eig_vals, _ = np.linalg.eigh(new_kernel.fit_transform(X, y))
	else:
		kmat_eig_vals = None

	best_model.fit(kmat_train, y[train_args])
	train_pred = best_model.predict(kmat_train)
	test_pred = best_model.predict(best_kernel.transform(X[test_args]))

	train_score = scoring_func.score(y[train_args], train_pred)
	test_score = scoring_func.score(y[test_args], test_pred)

	train_secondary_score = scoring_func.secondary_score(y[train_args], train_pred)
	test_secondary_score = scoring_func.secondary_score(y[test_args], test_pred)

	score_name = scoring_func.name
	second_name = scoring_func.secondary_name
	pred = np.zeros((y.shape[0],))
	pred[train_args] = train_pred[:]
	pred[test_args] = test_pred[:]
	results = {
		"train_args"          : train_args,
		"test_args"           : test_args,
		"inner_cv_folds"      : inner_cv_folds,
		"weights"             : best_model.w_,
		"pred"                : pred,
		"kmat_eig_vals"       : kmat_eig_vals,
		"Train " + score_name : train_score,
		"Test " + score_name  : test_score,
		"Train " + second_name: train_secondary_score,
		"Test " + second_name : test_secondary_score,
		}
	results.update(best_params)
	return results


def save_results(results_path, all_results):
	print("Results saved at: {}".format(results_path))
	with open(results_path, "wb") as f:
		pickle.dump(all_results, f)


def log_general_info(dataset_key, problem, X, y):
	__logger.info("############################################################################################")
	__logger.info("Experiment: {}".format(dataset_key))
	__logger.info("############################################################################################")
	__logger.info("")
	__logger.info("Problem: {}".format(problem))
	__logger.info("Size: {}".format(y.shape[0]))
	if problem == "regression":
		__logger.info("Label average: {0:.3f} +/- {1:.3f}".format(np.mean(y), np.std(y)))
	elif problem == "binary-classification":
		__logger.info("Class ratio: {0:.3f}".format(np.argwhere(y == 1).shape[0]/y.shape[0]))
	lengths = [len(X[i, 0]) for i in range(X.shape[0])]
	__logger.info("Sequence length range: [{}, {}]".format(min(lengths), max(lengths)))
	__logger.info("")


def train_val_test(X, y, dataset, num_inner_folds, scorer, num_jobs, headers):
	# Total row length of log file table
	row_length = len(headers)*table_column_width + (len(headers) - 1)*len(delimeter)
	all_results = {}
	train, test = dataset.predefined_splits["train_args"], dataset.predefined_splits["test_args"]
	inner_cv_folds = gen_cv_folds(X[train], y[train], dataset.cross_validation, num_inner_folds)

	for comb in algos_and_kernels:
		kernel_key, all_algos = comb
		for algo in all_algos:
			# For given algo/kernel combo: tune parameters and predict and test set
			param_grid = load_parameter_grid(kernel_key, algo)
			results = run_one(kernel_key, algo, X, y, train, test, inner_cv_folds, num_jobs, scorer, param_grid)
			# Save results to all_results
			experiment_key = kernel_key + "_" + algo
			all_results[experiment_key] = results

	# Print to log results on train and test set
	__logger.info("Results")
	__logger.info("-"*row_length)
	__logger.info(delimeter.join([column_format.format(header) for header in headers]))
	__logger.info("-"*row_length)
	for comb in algos_and_kernels:
		kernel_key, all_algos = comb
		for algo in all_algos:
			experiment_key = kernel_key + "_" + algo
			row = [experiment_key]
			for key in headers[1:]:
				val = numeric_format.format(all_results[experiment_key][key])
				row.append(val)
			__logger.info(delimeter.join([column_format.format(val) for val in row]))
	__logger.info("-"*row_length)
	__logger.info("")
	return all_results


def significance_test(results):

	# Compute welch t-test with bonferroni correction
	baseline_model_key = "GKM_SVM"
	comparison_model_keys = ["LA_KSVM", "EDIT_KSVM"]
	metric = "Test AUC"
	headers = ["Baseline", "Comparison", "P value", "Result"]
	baseline_results = results[baseline_model_key][metric]
	n = len(comparison_model_keys)
	bonferroni_alpha = 0.05/n
	alpha_string = precision_numeric_format.format(bonferroni_alpha)
	row_length = len(headers)*table_column_width + (len(headers) - 1)*len(delimeter)

	__logger.info("{0} significance tests at alpha = {1}".format(metric, alpha_string))
	__logger.info("-"*row_length)
	__logger.info(delimeter.join([column_format.format(header) for header in headers]))
	__logger.info("-"*row_length)
	for i in range(n):
		comparison_results = results[comparison_model_keys[i]][metric]
		row = [baseline_model_key, comparison_model_keys[i]]
		pval, is_significant = ttest_errs(baseline_results, comparison_results, alpha=bonferroni_alpha)
		row.append(precision_numeric_format.format(pval))
		if is_significant:
			row.append("significant")
		else:
			row.append("not significant")
		__logger.info(delimeter.join([column_format.format(val) for val in row]))
	__logger.info("-"*row_length)


def nested_cv(X, y, dataset, num_outer_folds, num_inner_folds, scorer, num_jobs, headers):
	outer_cv_folds = gen_cv_folds(X, y, dataset.cross_validation, num_outer_folds)
	# Total row length of log file table
	row_length = len(headers)*table_column_width + (len(headers) - 1)*len(delimeter)

	all_results = {}
	for i, (train, test) in enumerate(outer_cv_folds):
		__logger.info("Fold {0}/{1}:".format(i + 1, num_outer_folds))
		__logger.info("-"*row_length)
		__logger.info(delimeter.join([column_format.format(header) for header in headers]))
		__logger.info("-"*row_length)
		inner_cv_folds = gen_cv_folds(X[train], y[train], dataset.cross_validation, num_inner_folds)
		for comb in algos_and_kernels:
			kernel_key, all_algos = comb
			for algo in all_algos:
				# For one outer cv fold and given algo/kernel combo: tune parameters and predict and test set
				param_grid = load_parameter_grid(kernel_key, algo)
				results = run_one(kernel_key, algo, X, y, train, test, inner_cv_folds, num_jobs, scorer, param_grid)
				# Save results to all_results
				experiment_key = kernel_key + "_" + algo
				if experiment_key not in all_results.keys():
					all_results[experiment_key] = {key: [] for key in results}
				for key in results:
					all_results[experiment_key][key].append(results[key])

				# Print results to log file
				scores = [all_results[experiment_key][key][-1] for key in headers[1:]]
				row = [experiment_key] + [numeric_format.format(val) for val in scores]
				__logger.info(delimeter.join([column_format.format(val) for val in row]))

		__logger.info("-"*row_length)
		__logger.info("")

	# Print to log average results over all outer folds
	__logger.info("Average Results")
	__logger.info("-"*row_length)
	__logger.info(delimeter.join([column_format.format(header) for header in headers]))
	__logger.info("-"*row_length)
	for comb in algos_and_kernels:
		kernel_key, all_algos = comb
		for algo in all_algos:
			experiment_key = kernel_key + "_" + algo
			row = [experiment_key]
			for key in headers[1:]:
				mean = numeric_format.format(np.mean(all_results[experiment_key][key]))
				std = numeric_format.format(np.std(all_results[experiment_key][key]))
				row.append(mean + " +/- " + std)
			__logger.info(delimeter.join([column_format.format(val) for val in row]))
	__logger.info("-"*row_length)
	__logger.info("")
	significance_test(all_results)
	__logger.info("")
	return all_results


def run(dataset_key, score_key, num_inner_folds, num_outer_folds, num_jobs):
	start_time = datetime.now()
	dataset = init_dataset(dataset_key)
	X, y = dataset.load()

	log_general_info(dataset_key, dataset.problem, X, y)
	scorer = init_scoring_function(score_key, dataset.problem)
	headers = ["Model",
	           "Train " + scorer.name,
	           "Test " + scorer.name,
	           "Train " + scorer.secondary_name,
	           "Test " + scorer.secondary_name
	           ]
	if num_outer_folds == 1:
		all_results = train_val_test(X, y, dataset, num_inner_folds, scorer, num_jobs, headers)
	else:
		# Test args refer to independent test set, not part of original dataset
		if issubclass(type(dataset), DBAASP):
			X = X[dataset.predefined_splits["train_args"]]
			y = y[dataset.predefined_splits["train_args"]]
		all_results = nested_cv(X, y, dataset, num_outer_folds, num_inner_folds, scorer, num_jobs, headers)

	tdelta = datetime.now() - start_time
	hours, rem = divmod(tdelta.seconds, 3600)
	minutes, seconds = divmod(rem, 60)
	__logger.info("Walltime: {0:02d} hrs {1:02d} mins {2:02d} secs".format(hours, minutes, seconds))
	return all_results


def sbatch_call(args_, dataset_dir, experiment_name_):
	# If running on slurm HPC, this function prepares the job submission command
	mem_per_node = args_.memory
	time_hours = "{}:00:00".format(args_.time)
	cpus_per_task = args_.num_cpus
	partition = str(args_.partition)
	return ["sbatch",
	        "--nodes", "1",
	        "--mem", "{}g".format(mem_per_node),
	        "--time", time_hours,
	        "--cpus-per-task", str(cpus_per_task),
	        "--partition", str(partition),
	        "--job-name", "{}.run".format(experiment_name_),
	        "--output", "{}".format(os.path.join(dataset_dir, "out_file.out")),
	        "--tasks-per-node", "1"]


def submit_to_HPC(args_, dataset_dir):
	local_args__ = ["dataset", "print_log", "log_level", "score", "num_outer", "num_inner"]
	submit_script__ = os.path.join(ROOTPATH, "slurm_submit_script.sh")
	experiment_command = " ".join(
			["--{0} {1}".format(arg, getattr(args_, arg)) for arg in vars(args_) if arg in local_args__])
	if args_.name is not None:
		experiment_command += " --name {}".format(args_.name)
	sbatch_command = sbatch_call(args_, dataset_dir, "{0}_experiment".format(args_.dataset))
	sbatch_command.append(submit_script__)
	experiment_command = "{0} {1} ".format(HPC_PYTHON_PATH, os.path.realpath(__file__)) + experiment_command
	sbatch_command.append(experiment_command)
	subprocess.check_call(sbatch_command)


def main(args):
	if args.num_outer < 1:
		raise ValueError("Number of outer folds must be greater than or equal to 1")
	dataset_dir = check_dataset_dir(args.dataset)
	if bool(args.HPC):
		if HPC_PYTHON_PATH is None:
			raise ValueError("Please set 'HPC_PYTHON_PATH' to the location of your python interpretter")
		if not os.path.exists(HPC_PYTHON_PATH):
			raise ValueError("Path to python interpretter does not exist: {}".format(HPC_PYTHON_PATH))
		submit_to_HPC(args, dataset_dir)
	else:
		current_dir = create_save_dir(dataset_dir, args.num_outer, args.name)
		results_path = os.path.join(current_dir, "results_file.dat")
		log_path = os.path.join(current_dir, "log_file.dat")
		setup_logging(args.log_level, log_path, bool(args.print_log))
		results = run(args.dataset, args.score, args.num_inner, args.num_outer, args.num_cpus)
		save_results(results_path, results)
		if os.path.exists(os.path.join(dataset_dir, "out_file.out")):
			shutil.move(os.path.join(dataset_dir, "out_file.out"), os.path.join(current_dir, "out_file.out"))
