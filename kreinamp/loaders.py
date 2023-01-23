from kreinamp.data import AmpScanner, DeepAMP, SA25923_MAX, SA25923_MEAN, SA25923_MEDIAN, \
	SA29213_MAX, SA29213_MEAN, SA29213_MEDIAN, PA27853_MAX, PA27853_MEAN, PA27853_MEDIAN
from kreinamp.grid import SVM, KSVM, GKM, EDIT, LA, AAC
from kreinamp.sklearn_kernel_function import LocalAlignment, NormalisedEditDistance, GappedKMer, GlobalAlignment, AminoAcidComposition
from kreinamp.scoring_function import scoring_funcs__
from kreinamp.learning_algorithms import SquareHingeKreinSVM, SquareHingeKernelSVM

algos = {
	"KSVM": SquareHingeKreinSVM(),
	"SVM" : SquareHingeKernelSVM()
	}

kernels = {
	"LA"  : LocalAlignment(),
	"GA"  : GlobalAlignment(),
	"EDIT": NormalisedEditDistance(),
	"GKM" : GappedKMer(),
	"AAC": AminoAcidComposition()
	}

datasets = [
	AmpScanner,
	DeepAMP,

	SA25923_MAX,
	SA25923_MEAN,
	SA25923_MEDIAN,

	SA29213_MAX,
	SA29213_MEAN,
	SA29213_MEDIAN,

	PA27853_MAX,
	PA27853_MEAN,
	PA27853_MEDIAN
	]

model_grids__ = [SVM, KSVM]
kernel_grids__ = [GKM, EDIT, LA, AAC]


def init_dataset(dataset_key):
	dataset__ = None
	for dataset in datasets:
		d = dataset()
		if d.check_key(dataset_key):
			dataset__ = d
	if dataset__ is None:
		raise ValueError("Unknown Dataset: {}".format(dataset_key))
	return dataset__


def load_parameter_grid(kernel_key, model_key):
	kernel_grid__ = None
	for kernel in kernel_grids__:
		k = kernel()
		if k.name == kernel_key:
			kernel_grid__ = k
	if kernel_grid__ is None:
		raise ValueError("Unknown Kernel: {}".format(kernel_key))

	model_grid__ = None
	for model in model_grids__:
		m = model()
		if m.name == model_key:
			model_grid__ = m
	if model_grid__ is None:
		raise ValueError("Unknown Model: {}".format(model_key))

	grid = [
		{**k, **m}
		for k in kernel_grid__(name_to_append="kernel" + "__")
		for m in model_grid__(name_to_append="model" + "__")
		]
	return grid


def init_scoring_function(score_key, problem):
	if score_key == "default":
		val_to_check = problem
		default = True
	else:
		val_to_check = score_key
		default = False
	scoring_func__ = None
	for func in scoring_funcs__:
		s = func()
		if s.check_key(val_to_check, default):
			scoring_func__ = s
	if scoring_func__ is None:
		raise ValueError("Unknown Scoring Function: {}".format(val_to_check))
	else:
		return scoring_func__
