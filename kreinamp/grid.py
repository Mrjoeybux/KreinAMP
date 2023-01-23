class GridClass(object):

	@property
	def name(self):
		return self.__class__.__name__

	def remove_invalid_combinations(self, grid_dict, txt):
		return [grid_dict]

	def gen_grid(self, txt):
		grid_ = {}
		for attribute, value in self.__dict__.items():
			if not attribute.startswith("__"):
				grid_[txt + attribute] = value
		return grid_

	def __call__(self, name_to_append=None):
		if name_to_append is None:
			txt = ""
		else:
			txt = name_to_append
		grid_dict = self.gen_grid(txt)
		return self.remove_invalid_combinations(grid_dict, txt)


""" 
Model Grids -
Classname must be the same as the key found in algos from loaders.py
i.e. the SquareHingeKernelSVM class has key "SVM", so the grid for this class is SVM
"""


class SVM(GridClass):
	def __init__(self):
		self.C = [0.01, 0.1, 1, 10, 100]


class KSVM(GridClass):
	def __init__(self):
		self.mse_lambda_p = [0.01, 0.1, 1, 10, 100]
		self.mse_lambda_m = [0.01, 0.1, 1, 10, 100]


""" 
Kernel Grids - 
Classname must be the same as the key found in kernels from loaders.py
i.e. the LocalAlignment class has key "LA", so the grid for this class is LA

"""


class GKM(GridClass):
	def __init__(self):
		self.g = [1, 2, 3, 4, 5]
		self.m = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

	def remove_invalid_combinations(self, grid_dict, txt):
		"""
		g must be greater than m
		"""
		valid_params = []
		for m in grid_dict[txt + "m"]:
			allowed_gs = []
			for g in grid_dict[txt + "g"]:
				if g <= m:
					continue
				else:
					allowed_gs.append(g)
			if not allowed_gs:
				continue
			else:
				valid_params.append({txt + "m": [m], txt + "g": allowed_gs})
		return valid_params


class EDIT(GridClass):
	def __init__(self):
		pass


class LA(GridClass):
	def __init__(self):
		self.substitution_matrix = ["blosum62"]
		self.strategy = ["striped"]


class AAC(GridClass):
	def __init__(self):
		pass
