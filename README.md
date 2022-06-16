# Krein Support Vector Machine Classification of Antimicrobial Peptides

#### Installation Prerequisites
- A working installation of anaconda (see https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)

#### Installation
```
conda env create -f environment.yml
conda activate KreinAMP
pip install .
```

#### Usage
- Ensure the `KreinAMP` environment is activated (if not, call `conda activate KreinAMP`)
- To repeat the experiments from the paper, `main.py` should be executed.
- It's general usage is: `python main.py <command> <arguments>`, where `<command>` can be one of the following:
    - `test_set_pred`
    - `plot_lengths`
    - `dbaasp_pred`
    - `comparison`

#### Explanation of `main.py`

---

###### `<command> = test_set_pred`
This command accepts as input a `--dataset` argument. The user then chooses a model trained on this dataset to make predictions on the test set of 16 peptides.

Example usage: `python main.py test_set_pred --dataset DeepAMP`

---

###### `<command> = plot_lengths`
This command creates plots of the lengths of peptides in the `AMPScan` and `DeepAMP` datasets.

Example usage: `python main.py plot_lengths`

---

###### `<command> = dbaasp_pred`
This command reads in the predictions from the DBAASP webserver and calculates their accuracy and AUC. The predictions are saved in the files `ROOT/data/independent_test/dbaasp_<strain>_peptide_predictions.txt`, where `<strain>` can be `pa` or `sa`.

Example usage: `python main.py dbaasp_pred`

---

###### `<command> = comparison`
This command runs the main computational experiments. The possible arguments it accepts are described in the table below.

Example usage: `python main.py comparison --dataset DeepAMP`

---

| <Name >         | Alias  | Requirement | Functionality                                                |  
| ------------- | ------ | ----------- | :----------------------------------------------------------- |  
| `--dataset`   | `-d`   | Required    | Specify dataset to be used.                                  |  
| `--print_log` | -pr    | Optional    | Specify whether to log to console or write to file. If writing to file, log file is automatically generated and saved in the experiment directory. Defaults to saving to log file. |  
| `--log_level` | `-l`   | Optional    | Specify logging level. Default is INFO.                      |  
| `--score`     | `-s`   | Optional    | Specify scoring function used in hyperparameter selection. Default is ROC-AUC. |  
| `--num_outer` | `-o`   | Optional    | Specify number of outer cross-validation folds to be used. Any natural number greater than 1 will run a nested cross-validation experiment. Setting to 1 will run a standard train-validate-test split, where the train and test sets are predefined. Default is 10. |  
| `--num_inner` | `-i`   | Optional    | Specify number of inner cross-validation folds to be used in the hyperparameter selection. Default is 10. |  
| `--name`      | `-n`   | Optional    | Add a custom name to the experiment. Default is no name.     |  
| `--HPC`       | `-hp`   | Optional    | Specify whether to submit job to slurm cluster. Default is to run locally. |  
| `--memory`    | `-mem` | Optional    | Specify memory used, only applicable if submitting to slurm cluster. |  
| `--time`      | `-t`   | Optional    | Specify time required, only applicable if submitting to slurm cluster. |  
| `--num_cpus`  | `-c`   | Optional    | Specify number of processors used. Defaults to maximum for the current machine. |  
| `--partition` | `-q`   | Optional    | Specify partition used, only applicable if submitting to slurm cluster. |  
  

#### File descriptions  
- `comparison.py` performs the main computational experiments.
- `data.py` defines the data loading functionality of each dataset.
- `globals.py` defines a number of useful global variables.  
-  `grid.py` defines the parameter grids used in cross-validation.
- `learning_algorithms.py` defines the learning algorithms used.
- `loaders.py` is a utility file to load various objects.
- `main.py` is the entry point.
- `scoring_functions.py` defines the scoring functions used.
- `sklearn_kernel_functions.py` defines the kernel functions used.
- `supplementary.py` performs the supplementary experiments.
- `utils.py` defines a number of utility functions.