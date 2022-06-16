import warnings

warnings.filterwarnings("ignore", category=UserWarning)
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, make_scorer, roc_auc_score


def mzoe_error(predictions, y, mzoe_cut_off=0):
	return 1 - np.mean(np.sign(predictions - mzoe_cut_off) == np.sign(y))


class Scorer:
	def __init__(self):
		self.name = None
		self.secondary_name = None
		self.key = None
		self.default_key = None

	def check_key(self, key, default=True):
		if default:
			return key == self.default_key
		else:
			return key == self.key

	def cross_val_score(self):
		raise NotImplementedError

	def score(self, y_true, y_pred):
		raise NotImplementedError

	def secondary_score(self, y_true, y_pred):
		raise NotImplementedError

	def convert(self, score):
		return score


class BinaryClassificationScore(Scorer):
	def __init__(self):
		super(BinaryClassificationScore, self).__init__()
		self.name = "AUC"
		self.secondary_name = "ACC"
		self.key = "auc"
		self.default_key = "binary-classification"

	def cross_val_score(self):
		def my_auc(actual, pred):
			return roc_auc_score(actual, pred)

		return make_scorer(my_auc)

	def score(self, y_true, y_pred):
		return roc_auc_score(y_true, y_pred)

	def secondary_score(self, y_true, y_pred):
		return 1.0 - mzoe_error(y_pred, y_true)


class MultiClassificationScore(Scorer):
	def __init__(self):
		super(MultiClassificationScore, self).__init__()
		self.name = "MEAN-AUC"
		self.secondary_name = "ACC"
		self.key = "mc-auc"
		self.default_key = "multiclass-classification"

	def cross_val_score(self):
		def my_auc(actual, pred):
			auc_ = 0
			labels = np.unique(actual)
			for label in labels:
				converted = np.copy(actual)
				pos_idx = np.argwhere(converted == label)
				neg_idx = np.argwhere(converted != label)
				converted[pos_idx] = 1
				converted[neg_idx] = -1
				auc_ += roc_auc_score(converted, pred[:, label])
			return auc_/labels.shape[0]

		return make_scorer(my_auc)

	def score(self, y_true, y_pred):
		auc_ = 0
		labels = np.unique(y_true)
		for label in labels:
			converted = np.copy(y_true)
			pos_idx = np.argwhere(converted == label)
			neg_idx = np.argwhere(converted != label)
			converted[pos_idx] = 1
			converted[neg_idx] = -1
			auc_ += roc_auc_score(converted, y_pred[:, label])
		return auc_/labels.shape[0]

	def secondary_score(self, y_true, y_pred):
		return np.argwhere(y_true == np.argmax(y_pred, axis=1)).shape[0]/y_true.shape[0]


class RegressionScore(Scorer):
	def __init__(self):
		super(RegressionScore, self).__init__()
		self.name = "RMSE"
		self.secondary_name = "r^2"
		self.key = "rmse"
		self.default_key = "regression"

	def cross_val_score(self):
		def rmse(actual, pred):
			return mean_squared_error(actual, pred, squared=False)

		return make_scorer(rmse, greater_is_better=False)

	def score(self, y_true, y_pred):
		return np.sqrt(mean_squared_error(y_true, y_pred))

	def secondary_score(self, y_true, y_pred):
		return r2_score(y_true, y_pred)

	def convert(self, rmse_score):
		return rmse_score


class AccuracyScore(Scorer):
	def __init__(self):
		super().__init__()
		self.name = "ACC"
		self.secondary_name = None
		self.key = "ACC"
		self.default_key = None

	def cross_val_score(self):
		def acc(actual, pred):
			return 1.0 - mzoe_error(pred, actual)

		return make_scorer(acc)

	def score(self, y_true, y_pred):
		return 1.0 - mzoe_error(y_pred, y_true)


scoring_funcs__ = [BinaryClassificationScore, MultiClassificationScore, RegressionScore, AccuracyScore]
