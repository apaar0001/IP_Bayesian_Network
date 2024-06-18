import pandas as pd
import torch
import hamiltorch
import numpy as np

import pandas as pd
import torch

def read_data(file_path):
    # Read the CSV file
    data = pd.read_csv(file_path, header=None)
    # Select only the first 5 columns
    data_subset = data.iloc[:, :5]
    # Convert all values to numeric, setting errors='coerce' will replace non-numeric values with NaN
    data_subset = data_subset.apply(pd.to_numeric, errors='coerce')
    # Check if there are any NaN values and handle them, for example, by filling them with 0
    data_subset = data_subset.fillna(0)
    # Convert to tensor
    data_tensor = torch.tensor(data_subset.values, dtype=torch.float32)
    return data_tensor

file_path = 'synthetic_datasets/data_3n_10ts_30N.csv'
data_tensor = read_data(file_path)
print(data_tensor.shape)  # Verify the shape of the tensor

# Subsample the Data using Hamiltonian Monte Carlo
def log_prob_func(params):
    mean = torch.zeros(params.shape[0])
    covariance = torch.eye(params.shape[0])
    # Handle the case where params is one-dimensional
    if params.dim() == 1:
        diff = (params - mean).unsqueeze(0)  # Add a batch dimension
        log_prob = -0.5 * (diff @ torch.inverse(covariance) @ diff.T).squeeze()
    else:
        log_prob = -0.5 * (params - mean).mT @ torch.inverse(covariance) @ (params - mean)
    
    return log_prob

def subsample_data(data_tensor, num_samples=15):
    params_init = data_tensor.mean(dim=0)  # Initializing around the mean of the data
    params_init.requires_grad = True
    samples = hamiltorch.sample(log_prob_func, params_init, num_samples=num_samples)
    return torch.stack(samples)

subsampled_data = subsample_data(data_tensor, num_samples=30)
print(subsampled_data.shape)  # Verify the shape of the subsampled tensor

# ILPBN Model Definition
from collections import defaultdict
from pgmpy.models import BayesianNetwork
from gurobipy import GRB, Model

class ILPBN(BayesianNetwork):
    def __init__(self):
        super().__init__()
        self.variables = None
        self._model = None
        self.adjacency = None

    def learn_weights(self, data, lambda_n=0.5):
        data = data.T
        df = pd.DataFrame(data.numpy()).T
        n, m = df.shape
        x = df.to_numpy()
        model = Model("ILPBN")
        beta = model.addMVar((m, m), vtype=GRB.CONTINUOUS, name="beta")
        g = model.addMVar((m, m), vtype=GRB.BINARY, name="g")
        layer = model.addMVar(m, lb=1.0, ub=m, vtype=GRB.CONTINUOUS, name="layer")
        model.addConstrs((1 - m + m * g[j, k] <= layer[k] - layer[j] for j in range(m) for k in range(m)), name="(14a)")
        model.addConstrs((beta[j, k] * (1 - g[j, k]) == 0 for j in range(m) for k in range(m)), name="(13c)")
        model.setObjective(
            sum((x[d, k] - x[d, :] @ beta[:, k]) * (x[d, k] - x[d, :] @ beta[:, k]) for d in range(n) for k in range(m)) + lambda_n * g.sum(),
            GRB.MINIMIZE)
        model.optimize()
        self._model = model
        self.variables = df.columns
        self.adjacency = g

    def get_edges(self):
        edge_list = set()
        for i in range(len(self.variables)):
            for j in range(len(self.variables)):
                if int(self.adjacency[i, j].item().X) == 1:
                    edge_list.add((self.variables[i], self.variables[j]))
        return edge_list

# Initialize ILPBN model and add edges
obj = ILPBN()
edges = [('X1_t_0', 'X2_t_0'), ('X3_t_0', 'X4_t_0'),('X2_t_0','X5_t_0'),('X1_t_0','X4_t_0')]  # Reduce the number of edges for example
# obj.add_edges_from(edges)
for edge in edges:
    obj.add_edge(*edge)
# Learn weights from subsampled data
subsampled_data_reduced = subsample_data(data_tensor, num_samples=10) 
print(subsampled_data_reduced.shape)
print(subsampled_data_reduced)
print(subsampled_data_reduced[0])

 # Reduce num_samples for example
obj.learn_weights(subsampled_data_reduced, lambda_n=0.5)  # Use reduced data
# print(subsample_data_reduced)

# Retrieve and print the edges
learned_edges = obj.get_edges()
print("Learned Edges:", learned_edges)