import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
import numpy as np
import logging
from cvxopt.solvers import qp
from cvxopt import matrix

__logger = logging.getLogger("Learners")


def kernel_matrix(eig_vals, eig_vecs):
	return np.multiply(eig_vecs, eig_vals.reshape(1, -1)).dot(eig_vecs.T)


class EigenDecomposition(BaseEstimator, TransformerMixin):
	""" Computes the sign of the eigenvalues of kernel matrix. Defining it as a transformer and using caching
	reduces the number of eigendecompositions required, speeding up a grid search."""

	def transform(self, X):
		return X

	def fit_transform(self, X, y=None, **fit_params):
		eigvals, eigvecs = np.linalg.eigh(X)
		return {"kmat": X, "eigvals": eigvals, "eigvecs": eigvecs}


class Model(BaseEstimator, ClassifierMixin):
	def __init__(self, **kwargs):
		self.w_ = None
		self.requires_eigen_decomposition = False

	def set_weights(self, w):
		self.w_ = w

	def weights(self):
		return self.w_


def solve_quad_prog(kmat, y, diagonal_scaler):
	"""
	Solves the SVM quadratic program
	@param kmat: kernel matrix
	@param y: label vector
	@param diagonal_scaler: constant to add to diagonal of K @ y.T y
	@return: weight vector - solution of quadratic program
	"""
	n = kmat.shape[0]
	P = np.multiply(kmat, np.outer(y, y))
	np.fill_diagonal(P, P.diagonal() + diagonal_scaler)
	P = matrix(P)
	q = matrix(np.full((n,), -1.0))
	G = matrix(-1*np.eye(n))
	h = matrix(np.zeros((n,)))
	sol = qp(P=matrix(P), q=q, G=G, h=h, options={'show_progress': False})
	return np.array(sol['x']).reshape(-1, )


class SquareHingeKernelSVM(Model):
	def __init__(self, C=1.0):
		"""
		@param C: regularisation hyperparameter
		"""
		super().__init__()
		self.C = C

	def fit(self, X, y):
		"""
		solves the SVM problem via quadratic programming
		@param X: kernel matrix
		@param y: label vector
		@return: instance of self
		"""
		self._fit(X, y)

	def predict(self, X):
		return self._predict(X)

	def _fit(self, X, y):
		if 0 in y:
			raise ValueError("labels must be 1 and -1: found 0")
		beta = solve_quad_prog(X, y, 1.0/self.C)
		self.w_ = np.multiply(beta, y)
		return self

	def _predict(self, X):
		return np.squeeze(X).dot(self.w_).reshape(-1)

	def get_params(self, deep=True):
		return {"C": self.C}

	def set_params(self, **parameters):
		for parameter, value in parameters.items():
			setattr(self, parameter, value)
		return self

	@property
	def _pairwise(self):
		return True


class SquareHingeKreinSVM(Model):
	def __init__(self, mse_lambda_p=1.0, mse_lambda_m=1.0):
		"""
		@param mse_lambda_p: regularisation hyperparameter
		@param mse_lambda_m: regularisation hyperparameter
		"""
		super().__init__()
		self.mse_lambda_p = mse_lambda_p
		self.mse_lambda_m = mse_lambda_m
		self.requires_eigen_decomposition = True

	def fit(self, X, y):
		"""
		solves the KreinSVM problem via quadratic programming
		@param X: kernel matrix
		@param y: label vector
		@return: instance of self
		"""
		if 0 in y:
			raise ValueError("labels must be 1 and -1: found 0")
		if isinstance(X, dict):
			eigvals = X["eigvals"]
			eigvecs = X["eigvecs"]
		else:
			kmat = X
			eigvals, eigvecs = np.linalg.eigh(kmat)

		n = eigvals.shape[0]
		signs = np.sign(eigvals)
		pm_args = (np.argwhere(signs >= 0).reshape(-1), np.argwhere(signs < 0).reshape(-1))
		scale_vec = np.ones((n,))
		scale_vec[pm_args[0]] /= self.mse_lambda_p
		scale_vec[pm_args[1]] /= self.mse_lambda_m
		regularised_kmat = kernel_matrix(np.multiply(np.absolute(eigvals), scale_vec), eigvecs)
		beta = solve_quad_prog(regularised_kmat, y, 1.0)
		K = kernel_matrix(eigvals, eigvecs)
		pred = (regularised_kmat*y).dot(beta)
		self.w_ = np.linalg.lstsq(a=K, b=pred, rcond=None)[0]
		self.b_ = 0
		return self

	def predict(self, X):
		fx = np.squeeze(X).dot(self.w_).reshape(-1)
		return fx + self.b_

	def get_params(self, deep=True):
		return {"mse_lambda_p": self.mse_lambda_p, "mse_lambda_m": self.mse_lambda_m}

	def set_params(self, **parameters):
		for parameter, value in parameters.items():
			setattr(self, parameter, value)
		return self

	@property
	def _pairwise(self):
		return True
