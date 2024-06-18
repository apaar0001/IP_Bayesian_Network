# read time-series-data-generation/synthetic_datasets/data_3n_10ts_30N.csv and using hamiltonian torch subsample the data to 30 samples
#
# The snippet from ILPBN.py uses gurobipy, which is not available in the environment. The snippet from example_script.py uses the TimeSeriesGenerator class from time-series-data-generation, which is not available in the environment. The task is to write a function that reads the data from synthetic_datasets/data_3n_10ts_30N.csv, subsamples the data to 30 samples using Hamiltonian Monte Carlo, and returns the subsampled data. The function signature is `def subsample_data(file_path: str) -> torch.Tensor:`. The function should return a torch.Tensor with the subsampled data. The function should use the following steps:
# 1. Read the data from the file at the given file path.
# 2. Convert the data to a torch.Tensor.
# 3. Use Hamiltonian Monte Carlo to subsample the data to 30 samples.
# 4. Return the subsampled data as a torch.Tensor.

# You can use the following code to read the data from the file:
# import pandas as pd
# data = pd.read_csv(file_path)
# import my_package.d41.codietpgm.learners.BayesianNetworkLearner as BayesianNetworkLearner
# import my_package.d41.codietpgm.io.variableannotation as variableannotation

import pandas as pd

# Load the CSV file into a pandas DataFrame
file_path = 'time-series-data-generation/synthetic_datasets/data_3n_10ts_30N.csv'
data = pd.read_csv(file_path)
print(data)
import torch

data_tensor = torch.tensor(data.values, dtype=torch.float32)

import torch
import hamiltorch

# Example log_prob_func (replace with your actual implementation)
# import torch

# Example log_prob_func (replace with your actual implementation)
def log_prob_func(params):
    # Example: Gaussian log-probability
    mean = torch.zeros(50)  # Assuming mean is zero for simplicity
    covariance = torch.eye(50)  # Assuming identity covariance matrix
    log_prob = -0.5 * (params - mean).T @ torch.inverse(covariance) @ (params - mean)
    return log_prob


# Load your data here (replace with your actual data loading code)
data = data_tensor  # Replace with your actual data
data =  data.mT
# Initial parameters (replace with your initialization)
# Initial parameters (replace with your initialization)
params_init = torch.zeros(50, requires_grad=True)  # Example: Starting with zeros, adjust based on your model
 # Replace with your initisalization

# Run sampling
num_samples = 1000  # Number of samples to generate
samples = hamiltorch.sample(log_prob_func, params_init, num_samples=num_samples)

# `samples` now contains the sampled parameter values
print(samples)
print(len(samples[0]))

# Subsample the data to 30 samples
subsampled_data = samples[:30]
print(subsampled_data)
print(len(subsampled_data[0]))

# Return the subsampled data as a torch.Tensor
subsampled_data_tensor = torch.tensor(subsampled_data, dtype=torch.float32)
print(subsampled_data_tensor)

