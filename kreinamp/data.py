import warnings

warnings.filterwarnings("ignore", category=UserWarning)
from kreinamp.globals import ROOTPATH
from sklearn.model_selection import StratifiedKFold, KFold
import numpy as np
import os
from kreinamp.utils import parse_fasta, is_number, is_range, is_greater_than, is_less_than, is_equality, uM_to_ug_per_ml, load_test_data
import pandas as pd

"""
All derived classes use the same link/reference as the parent class, unless explicitly stated.
"""


class AbstractData:
	def __init__(self):
		self.keyname = None
		self.path = None
		self.problem = None
		self.data_transformer = self.__identity_data_transformer
		self.index_transformer = self.__identity_index_transformer
		self.base_name = None
		self.split_type = "nested-cv"
		self.predefined_splits = None

	def __identity_data_transformer(self, X, y, params=None):
		return X, y

	def __identity_index_transformer(self, X, y, cv_args, params=None):
		return cv_args

	def load_func(self):
		raise NotImplementedError

	def load(self):
		return self.load_func()

	def cross_validation(self, n_splits, shuffle, random_state):
		return None

	def check_key(self, key):
		if key == self.keyname:
			return True


class AmpScanner(AbstractData):
	# https://www.dveltri.com/ascan/v2/about.html - Dataset section
	def __init__(self):
		super().__init__()
		self.keyname = "ampscan"
		self.path = os.path.join(ROOTPATH, "data", "ampscanner")
		self.problem = "binary-classification"
		self.split_type = "train-test"

	def load_func(self):
		train_pos, num_train_pos = parse_fasta(os.path.join(self.path, "AMP.tr.fa"))
		train_neg, num_train_neg = parse_fasta(os.path.join(self.path, "DECOY.tr.fa"))

		val_pos, num_val_pos = parse_fasta(os.path.join(self.path, "AMP.eval.fa"))
		val_neg, num_val_neg = parse_fasta(os.path.join(self.path, "DECOY.eval.fa"))

		train_pos += val_pos
		train_neg += val_neg
		num_train_pos += num_val_pos
		num_train_neg += num_val_neg

		test_pos, num_test_pos = parse_fasta(os.path.join(self.path, "AMP.te.fa"))
		test_neg, num_test_neg = parse_fasta(os.path.join(self.path, "DECOY.te.fa"))
		self.predefined_splits = {
			"train_args": np.arange(num_train_pos + num_train_neg),
			"test_args" : np.arange(start=num_train_pos + num_train_neg,
			                        stop=num_train_pos + num_train_neg + num_test_pos + num_test_neg)
			}
		X = np.array(train_pos + train_neg + test_pos + test_neg)
		y = np.array([1]*num_train_pos + [-1]*num_train_neg + [1]*num_test_pos + [-1]*num_test_neg)
		return np.expand_dims(X, axis=1), y

	def cross_validation(self, n_splits, shuffle, random_state):
		return StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)


class DeepAMP(AbstractData):
	# https://app.cbbio.online/ampep/dataset - DeepAmPEP30 section
	def __init__(self):
		super().__init__()
		self.keyname = "deepamp"
		self.path = os.path.join(ROOTPATH, "data", "deepAMP")
		self.problem = "binary-classification"
		self.split_type = "train-test"

	def load_func(self):
		train_pos, num_train_pos = parse_fasta(os.path.join(self.path, "train_po.fasta"))
		train_neg, num_train_neg = parse_fasta(os.path.join(self.path, "train_ne.fasta"))
		test_pos, num_test_pos = parse_fasta(os.path.join(self.path, "test_po.fasta"))
		test_neg, num_test_neg = parse_fasta(os.path.join(self.path, "test_ne.fasta"))
		self.predefined_splits = {
			"train_args": np.arange(num_train_pos + num_train_neg),
			"test_args" : np.arange(start=num_train_pos + num_train_neg,
			                        stop=num_train_pos + num_train_neg + num_test_pos + num_test_neg)
			}

		X = np.array(train_pos + train_neg + test_pos + test_neg)
		y = np.array([1]*num_train_pos + [-1]*num_train_neg + [1]*num_test_pos + [-1]*num_test_neg)
		return np.expand_dims(X, axis=1), y

	def cross_validation(self, n_splits, shuffle, random_state):
		return StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)


class DBAASPDataParser():
	def __init__(self, targets, mic_positive_cutoff, mic_negative_cutoff, measure="MIC", min_length=6, max_length=27,
	             problem="classification"):
		self.targets = targets
		self.mic_positive_cutoff = mic_positive_cutoff
		self.mic_negative_cutoff = mic_negative_cutoff
		self.measure = measure
		self.min_length = min_length
		self.max_length = max_length
		self.problem = problem
		self.relevant_column_names = ["SEQUENCE",
		                              "TARGET OBJECT",
		                              "TARGET ACTIVITY - TARGET SPECIES",
		                              "TARGET ACTIVITY - ACTIVITY MEASURE GROUP",
		                              "TARGET ACTIVITY - CONCENTRATION",
		                              "TARGET ACTIVITY - UNIT",
		                              "TARGET ACTIVITY - MEDIUM"]
		self.convert_units = {"µM"   : uM_to_ug_per_ml,
		                      "µg/ml": lambda conc, seq: conc}

	def relevant_targets(self, df):
		print(df.index[df["TARGET ACTIVITY - TARGET SPECIES"].str.contains(self.targets, case=False, na=False)].shape)
		return df.index[df["TARGET ACTIVITY - TARGET SPECIES"].str.contains(self.targets, case=False, na=False)]

	def relevant_measure(self, df):
		return df.index[df["TARGET ACTIVITY - ACTIVITY MEASURE GROUP"] == self.measure]

	@staticmethod
	def parse_concentration(conc):
		# print(conc)
		if is_number(conc):
			parsed_conc = float(conc)

		elif is_range(conc):
			min_conc, max_conc = conc.split("-")
			min_conc = DBAASPDataParser.parse_concentration(min_conc)
			max_conc = DBAASPDataParser.parse_concentration(max_conc)
			parsed_conc = (max_conc - min_conc)/2

		elif is_greater_than(conc):
			if is_equality(conc):
				parsed_conc = float(conc.split(">=")[-1])

			else:
				parsed_conc = float(conc.split(">")[-1]) + 1

		elif is_less_than(conc):
			if is_equality(conc):
				parsed_conc = float(conc.split("<=")[-1])

			else:
				parsed_conc = float(conc.split("<")[-1]) - 1

		else:
			return None
		return parsed_conc

	def aggregate_concentrartions(self, concentrations):
		raise NotImplementedError

	def parse_one_sequence(self, seq_df):
		# n is the number of seperate measurements
		n = seq_df.shape[0]
		concentrations = []
		seq = seq_df.iloc[0]["SEQUENCE"]
		for i in range(n):
			unit = seq_df.iloc[i]["TARGET ACTIVITY - UNIT"]
			if not unit in self.convert_units.keys():
				continue
			conc = seq_df.iloc[i]["TARGET ACTIVITY - CONCENTRATION"]
			parsed_conc = DBAASPDataParser.parse_concentration(conc)
			if parsed_conc is not None:
				concentrations.append(self.convert_units[unit](parsed_conc, seq))
		if not concentrations:
			return None
		aggr_conc = self.aggregate_concentrartions(concentrations)
		if self.problem == "binary-classification":
			if aggr_conc <= self.mic_positive_cutoff:
				return 1
			elif aggr_conc >= self.mic_negative_cutoff:
				return -1
			else:
				return None
		elif self.problem == "regression":
			return aggr_conc
		else:
			raise ValueError("Unknown problem: {}".format(self.problem))

	def parse(self, path, test_data):
		df = pd.read_csv(path)[self.relevant_column_names]
		relevant_indices = [self.relevant_measure,
		                    self.relevant_targets]
		total_index = df.index[np.arange(df.shape[0])]
		for func in relevant_indices:
			index = func(df)
			total_index = total_index.intersection(index)
		df = df.iloc[total_index]
		seqs = df["SEQUENCE"].unique()
		seq_list = []
		labels = []
		for seq in seqs:
			if seq in test_data:
				continue
			if self.min_length <= len(seq) <= self.max_length:
				label = self.parse_one_sequence(df[df["SEQUENCE"] == seq])
				if label is None:
					continue
				else:
					seq_list.append(seq)
					labels.append(label)
			else:
				continue
		return seq_list, labels


class DBAASPMeanParser(DBAASPDataParser):
	def aggregate_concentrartions(self, concentrations):
		return np.mean(concentrations)


class DBAASPMaxParser(DBAASPDataParser):
	def aggregate_concentrartions(self, concentrations):
		return max(concentrations)


class DBAASPMedianParser(DBAASPDataParser):
	def aggregate_concentrartions(self, concentrations):
		return np.median(concentrations)


class DBAASP(AbstractData):
	def __init__(self):
		super().__init__()
		self.keyname = "dbaasp"
		self.path = os.path.join(ROOTPATH, "data", "dbaasp")
		self.problem = "binary-classification"
		self.split_type = "train-test"
		self.targets = None
		self.mic_positive_cutoff = 25
		self.mic_negative_cutoff = 100
		self.min_length = 6
		self.max_length = 18
		self.extension = None
		self.pos_label_ID = None
		self.strain = None
		self.MIC_aggregate_func = None

	def load_func(self):
		if self.MIC_aggregate_func == "mean":
			parser = DBAASPMeanParser(self.targets, self.mic_positive_cutoff, self.mic_negative_cutoff,
			                          min_length=self.min_length, max_length=self.max_length, problem=self.problem)
		elif self.MIC_aggregate_func == "max":
			parser = DBAASPMaxParser(self.targets, self.mic_positive_cutoff, self.mic_negative_cutoff,
			                         min_length=self.min_length, max_length=self.max_length, problem=self.problem)
		elif self.MIC_aggregate_func == "median":
			parser = DBAASPMedianParser(self.targets, self.mic_positive_cutoff, self.mic_negative_cutoff,
			                            min_length=self.min_length, max_length=self.max_length, problem=self.problem)
		else:
			raise ValueError("Unknown aggregate key: {}".format(self.MIC_aggregate_func))
		test_seq, _, test_labels = load_test_data(self.strain, self.mic_negative_cutoff)
		seq, labels = parser.parse(self.path, test_seq)
		data = np.concatenate((np.array(seq), test_seq))
		y = np.concatenate((np.array(labels), test_labels))
		self.predefined_splits = {
			"train_args": np.arange(len(seq)),
			"test_args" : np.arange(start=len(seq), stop=data.shape[0])
			}
		return np.expand_dims(data, axis=1), y

	def cross_validation(self, n_splits, shuffle, random_state):
		if self.problem == "classification":
			return StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
		else:
			return KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)


# Staphylococcus aureus


class SA(DBAASP):
	def __init__(self):
		super(SA, self).__init__()
		self.keyname = "SA"
		self.strain = "SA"
		self.targets = "Staphylococcus aureus"
		self.min_length = 6
		self.max_length = 18
		self.path = os.path.join(self.path, "SA_5-100.csv")


class SA25923(SA):
	def __init__(self):
		super(SA25923, self).__init__()
		self.keyname = "SA25923"
		self.targets = "Staphylococcus aureus ATCC 25923"


class SA25923_MAX(SA25923):
	def __init__(self):
		super().__init__()
		self.keyname = "SA25923_MAX"
		self.MIC_aggregate_func = "max"


class SA25923_MEAN(SA25923):
	def __init__(self):
		super().__init__()
		self.keyname = "SA25923_MEAN"
		self.MIC_aggregate_func = "mean"


class SA25923_MEDIAN(SA25923):
	def __init__(self):
		super().__init__()
		self.keyname = "SA25923_MEDIAN"
		self.MIC_aggregate_func = "median"


# Staphylococcus aureus ATCC 29213


class SA29213(SA):
	def __init__(self):
		super(SA29213, self).__init__()
		self.keyname = "SA29213"
		self.targets = "Staphylococcus aureus ATCC 29213"


class SA29213_MAX(SA29213):
	def __init__(self):
		super().__init__()
		self.keyname = "SA29213_MAX"
		self.MIC_aggregate_func = "max"


class SA29213_MEAN(SA29213):
	def __init__(self):
		super().__init__()
		self.keyname = "SA29213_MEAN"
		self.MIC_aggregate_func = "mean"


class SA29213_MEDIAN(SA29213):
	def __init__(self):
		super().__init__()
		self.keyname = "SA29213_MEDIAN"
		self.MIC_aggregate_func = "median"


# Pseudomonas aeruginosa ATCC 27853


class PA27853(DBAASP):
	def __init__(self):
		super(PA27853, self).__init__()
		self.keyname = "PA27853"
		self.strain = "PA"
		self.targets = "Pseudomonas aeruginosa ATCC 27853"
		self.min_length = 6
		self.max_length = 18
		self.path = os.path.join(self.path, "PA_ATCC_27853_6-18.csv")


class PA27853_MAX(PA27853):
	def __init__(self):
		super().__init__()
		self.keyname = "PA27853_MAX"
		self.MIC_aggregate_func = "max"


class PA27853_MEAN(PA27853):
	def __init__(self):
		super().__init__()
		self.keyname = "PA27853_MEAN"
		self.MIC_aggregate_func = "mean"


class PA27853_MEDIAN(PA27853):
	def __init__(self):
		super().__init__()
		self.keyname = "PA27853_MEDIAN"
		self.MIC_aggregate_func = "median"
