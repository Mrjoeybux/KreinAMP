import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from abc import ABCMeta, abstractmethod
from fastsk import FastSK
from kreinamp.utils import symbol2weight
from fastsk.utils import Vocabulary
import editdistance
import parasail


class KernelFunction(BaseEstimator, TransformerMixin):
	__metaclass__ = ABCMeta

	def __init__(self):
		self.training_data = None

	def fit(self, X):
		self.training_data = self.to_list(X)
		self.fit_routine(self.training_data)

	def fit_routine(self, X):
		pass

	def transform(self, X):
		return self.compute_rectangular_kernel_matrix(self.to_list(X), self.training_data)

	def fit_transform(self, X, y=None, **fit_params):
		self.fit(X)
		return self.compute_square_kernel_matrix(self.training_data)

	def set_params(self, **parameters):
		for parameter, value in parameters.items():
			setattr(self, parameter, value)
		return self

	def to_list(self, X):
		if X.ndim == 1:
			return X.tolist()
		else:
			return np.squeeze(X).tolist()

	@abstractmethod
	def compute_square_kernel_matrix(self, X):
		pass

	@abstractmethod
	def compute_rectangular_kernel_matrix(self, X1, X2):
		pass

	@abstractmethod
	def get_params(self, deep=True):
		pass


class ParasailAlignment(KernelFunction):
	__metaclass__ = ABCMeta

	def __init__(self, open_penalty=5, extend_penalty=2, substitution_matrix="blosum62", strategy="striped"):
		super().__init__()
		self.open_penalty = open_penalty
		self.extend_penalty = extend_penalty
		self.substitution_matrix = substitution_matrix
		self.strategy = strategy
		self.function_key = self.get_function_key()

	def get_params(self, deep=True):
		return {"open_penalty"       : self.open_penalty,
		        "extend_penalty"     : self.extend_penalty,
		        "substitution_matrix": self.substitution_matrix,
		        "strategy"           : self.strategy}

	def prepare_aligner(self, query):
		if self.strategy in ["striped", "scan"]:
			function_name = "{0}_trace_{1}_profile_16".format(self.function_key, self.strategy)
			aligner = getattr(parasail, function_name)
			profile = parasail.profile_create_16(query, getattr(parasail, self.substitution_matrix))
			return lambda target: aligner(profile, target, self.open_penalty, self.extend_penalty)
		else:
			raise ValueError("Unknown strategy \"{0}\", choose from \"striped\" or \"scan\".".format(self.strategy))

	def dot(self, seq1, seq2):
		aligner = self.prepare_aligner(seq1)
		alignment = aligner(target=seq2)
		return alignment.score

	def compute_self_alignments(self, X):
		self_alignments = []
		for i in range(len(X)):
			aligner = self.prepare_aligner(X[i])
			self_alignment = aligner(target=X[i])
			self_alignments.append(self_alignment.score)
		return self_alignments

	def fit_routine(self, X):
		self.train_self_alignments = self.compute_self_alignments(X)

	def compute_square_kernel_matrix(self, X):
		n = len(X)
		K = np.zeros((n, n))
		for i in range(n):
			aligner = self.prepare_aligner(X[i])
			K[i, i] = self.train_self_alignments[i]
			for j in range(i + 1, n):
				alignment = aligner(target=X[j])
				if alignment.score < 0:
					K[i, j] = self.handle_negative_alignments(alignment.score)
				else:
					K[i, j] = alignment.score
				K[j, i] = K[i, j]
		norm_mat = np.sqrt(np.outer(self.train_self_alignments, self.train_self_alignments))
		return np.divide(K, norm_mat)

	def handle_negative_alignments(self, score):
		return score

	def compute_rectangular_kernel_matrix(self, X1, X2):

		n = len(X1)
		m = len(X2)
		K = np.zeros((n, m))
		for i in range(n):
			aligner = self.prepare_aligner(X1[i])
			for j in range(m):
				alignment = aligner(target=X2[j])
				if alignment.score < 0:
					K[i, j] = self.handle_negative_alignments(alignment.score)
				else:
					K[i, j] = alignment.score
		test_self_alignments = self.compute_self_alignments(X1)
		norm_mat = np.sqrt(np.outer(test_self_alignments, self.train_self_alignments))
		return np.divide(K, norm_mat)

	@abstractmethod
	def get_function_key(self):
		pass


class LocalAlignment(ParasailAlignment):

	def get_function_key(self):
		return "sw"

	def handle_negative_alignments(self, score):
		return 0


class SemiGlobalAlignment(ParasailAlignment):
	def get_function_key(self):
		return "sg"


class GlobalAlignment(ParasailAlignment):
	def get_function_key(self):
		return "nw"


class GappedKMer(KernelFunction):
	def __init__(self, g=1, m=1):
		super().__init__()
		self._vocab = None
		self.g = g
		self.m = m

	def _init_vocab(self):
		self._vocab = Vocabulary()

	def _convert_data(self, sequences):
		new_seq = []
		for i in range(len(sequences)):
			new_seq.append([self._vocab.add(token) for token in sequences[i]])
		return new_seq

	def fit_routine(self, X):
		self._init_vocab()

	def compute_square_kernel_matrix(self, X):
		converted = self._convert_data(X)
		kernel = FastSK(g=self.g, m=self.m, approx=True)
		kernel.compute_train(converted)
		return np.array(kernel.get_train_kernel())

	def compute_rectangular_kernel_matrix(self, X1, X2):
		if not isinstance(X1, list):
			X1 = [X1]
		converted_X1 = self._convert_data(X1)
		converted_X2 = self._convert_data(X2)
		kernel = FastSK(g=self.g, m=self.m, approx=True)
		kernel.compute_kernel(converted_X1, converted_X2)
		return np.array(kernel.get_test_kernel()).T

	def get_params(self, deep=True):
		return {"g": self.g, "m": self.m}


class NormalisedEditDistance(KernelFunction):
	def compute_square_kernel_matrix(self, X):
		n = len(X)
		K = np.ones((n, n))
		for i in range(n):
			for j in range(i + 1, n):
				K[i, j] = self.dot(X[i], X[j])
				K[j, i] = K[i, j]
		return K

	def compute_rectangular_kernel_matrix(self, X1, X2):
		n = len(X1)
		m = len(X2)
		K = np.zeros((n, m))
		for i in range(n):
			for j in range(m):
				K[i, j] = self.dot(X1[i], X2[j])
		return K

	def dot(self, x1, x2):
		ed = editdistance.eval(x1, x2)
		return 1 - (2*ed/(len(x1) + len(x2) + ed))

	def get_params(self, deep=True):
		return {}


class AminoAcidComposition(KernelFunction):

	def _feature_vec(self, peptide):
		return [peptide.count(amino_acid) for amino_acid in symbol2weight.keys()]

	def feature_mat(self, X):
		return np.array([self._feature_vec(x) for x in X])

	def compute_square_kernel_matrix(self, X):
		F = self.feature_mat(X)
		K = F @ F.T
		return K.astype("float")

	def compute_rectangular_kernel_matrix(self, X1, X2):
		# X1 - test
		# X2 - train
		F1 = self.feature_mat(X1)
		F2 = self.feature_mat(X2)
		K = F1 @ F2.T
		return K.astype("float")

	def get_params(self, deep=True):
		return {}