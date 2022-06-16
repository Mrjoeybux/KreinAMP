import numpy as np
import logging
import pickle
from logging import handlers
from scipy import stats
from kreinamp.globals import TEST_DATA_PATH

logger_ids = ["Experiment"]


def init_loggers(lvl, log_fpath=None):
	name_size = max([len(id) for id in logger_ids])
	if log_fpath is None:
		for logger_id in logger_ids:
			add_stream_log_handler(logger_id, name_size=name_size, log_level=lvl)
	else:
		for logger_id in logger_ids:
			add_file_log_handlers(logger_id, log_fpath, name_size=name_size, log_level=lvl)


def configure_file_log_handler(fpath, name_size, level=logging.INFO, max_file_size=5000000, num_backups=10):
	handler = logging.handlers.RotatingFileHandler(fpath, maxBytes=max_file_size, backupCount=num_backups)
	formatter = logging.Formatter(
			'[%(asctime)s] --- [%(levelname)5s] --- [%(name){}s] --- %(message)s'.format(name_size))
	handler.setFormatter(formatter)
	handler.setLevel(level)
	return handler


def add_stream_log_handler(logger_id, name_size, log_level=logging.INFO):
	logger = logging.getLogger(logger_id)
	logger.setLevel(log_level)
	ch = logging.StreamHandler()
	formatter = logging.Formatter(
			'[%(asctime)s] --- [%(levelname)5s] --- [%(name){}s] --- %(message)s'.format(name_size))
	ch.setFormatter(formatter)
	logger.addHandler(ch)


def add_file_log_handlers(logger_id, fpath, name_size, log_level=logging.INFO, max_file_size=5000000, num_backups=10):
	logger = logging.getLogger(logger_id)
	logger.setLevel(log_level)
	logger.addHandler(configure_file_log_handler(fpath, name_size, log_level, max_file_size, num_backups))


def init_stream_loggers(log_level=logging.INFO):
	for logger_id in logger_ids:
		add_stream_log_handler(logger_id, log_level=log_level)


def init_file_loggers(fpath, log_level=logging.INFO):
	for logger_id in logger_ids:
		add_file_log_handlers(logger_id, fpath, log_level=log_level)


def parse_fasta(path, pos_label=None, return_names=False):
	seqs = []
	partial_sequence = ""
	if pos_label is None:
		with open(path, "r") as f:
			for i, line in enumerate(f):
				if i == 0:
					continue
				if ">" in line:
					seqs.append(partial_sequence.replace(" ", ""))
					partial_sequence = ""
					continue
				else:
					partial_sequence += line[:-1]
			seqs.append(partial_sequence.replace(" ", ""))
		return seqs, len(seqs)
	else:
		labels = []
		with open(path, "r") as f:
			for i, line in enumerate(f):
				if i == 0:
					if ">" in line:
						if pos_label in line:
							labels.append(1)
						else:
							labels.append(-1)
					continue
				if ">" in line:
					if pos_label in line:
						labels.append(1)
					else:
						labels.append(-1)
					seqs.append(partial_sequence.replace(" ", ""))
					partial_sequence = ""
					continue
				else:
					partial_sequence += line[:-1]
			seqs.append(partial_sequence.replace(" ", ""))
		return seqs, len(seqs), labels


symbol2weight = {
	"A": 89.094,
	"C": 121.154,
	"D": 133.104,
	"E": 147.131,
	"F": 165.192,
	"G": 75.067,
	"H": 155.156,
	"I": 131.175,
	"K": 146.189,
	"L": 131.175,
	"M": 149.208,
	"N": 132.119,
	"O": 255.313,
	"P": 115.132,
	"Q": 146.146,
	"R": 174.203,
	"S": 105.093,
	"T": 119.119,
	"U": 168.064,
	"V": 117.148,
	"W": 204.228,
	"Y": 181.191,
	}

wildcard2members = {
	"B": ("D", "N"),
	"J": ("I", "L"),
	"X": tuple(symbol2weight.keys()),
	"Z": ("E", "Q"),
	}

non_wildcard_symbols = list(symbol2weight.keys())
wildcard_symbols = list(wildcard2members.keys())

for wildcard, members in wildcard2members.items():
	symbol2weight[wildcard] = np.mean([symbol2weight[x] for x in members])


def molecular_weight(seq):
	return sum(symbol2weight[symbol] for symbol in seq)


def uM_to_ug_per_ml(conc, seq):
	"""
	Converts between micro-Moles per Liter (micro-Molar concentration) and
	micrograms per milliliter using the estimated molecular weight of an amino
	acid sequence.
	Estimated molecular weight is given in Daltons, or grams per mole.

	Dimensional Arithmetic:
	    micro-moles per liter
	        = 10**-6 mole / liter

	    micro-grams per milliliter
	        = 10**-6 gram / 10**-3 liter

	    (micro-mole / liter) * (gram / mole) * 10**-3
	        = (10**-6 mole / liter) * (gram / mole) * 10**-3
	        = (10**-6 gram / liter) * 10**-3
	        = (10**-6 gram / 10**3 * 1 liter)
	        = micro-grams / milliliter

	Args:
	    conc: float, Micro-molar concentration.
	    seq: str, Amino acid sequence in FASTA format.

	Returns: float, concentration in micrograms per milliliter.
	"""
	return conc*molecular_weight(seq)*10**-3


is_range = lambda val: "-" in val
is_greater_than = lambda val: ">" in val
is_less_than = lambda val: "<" in val
is_equality = lambda val: "=" in val


def is_number(s):
	if s.isnumeric():
		return True
	else:
		try:
			float(s)
			return True
		except ValueError:
			return False


def load_test_data(strain="SA", mic_negative_cutoff=100):
	names = []
	seqs = []
	amp = []
	with open(TEST_DATA_PATH, "r") as f:
		for i, line in enumerate(f):
			if i == 0:
				continue
			name, seq, MIC_SA, MIC_PA = line.split(",")
			if strain == "SA":
				MIC = MIC_SA
			elif strain == "PA":
				MIC = MIC_PA
			else:
				raise ValueError("Unknown strain: {}".format(strain))
			names.append(name)
			seqs.append(seq)
			if float(MIC.split(">")[-1]) >= mic_negative_cutoff:
				amp.append(-1)
			else:
				amp.append(1)
	return np.array(seqs), names, np.array(amp)


def ttest_errs(e1, e2, alpha=0.05):
	ttest = stats.ttest_ind(e1, e2, equal_var=False)
	pval = ttest[1]
	if pval < alpha:
		return pval, True
	return pval, False


names = {"GKM_SVM"  : "GKM",
         "LA_KSVM"  : "Local Alignment",
         "EDIT_KSVM": "Edit Distance"}


def extract_vals(results, model_name, metric):
	vals = []
	for val in results[model_name][metric]:
		d = {"Test Accuracy": val, "Model": names[model_name]}
		vals.append(d)
	return vals
