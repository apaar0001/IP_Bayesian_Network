"""
Copyright (c) 2020 by Impulse Innovations Ltd. Private and confidential. Part of the causaLens product.
"""

import matplotlib.pyplot as plt
import pickle
import networkx as nx
import random 
import os 

from dgp.src.data_generation_configs import (
    CausalGraphConfig, DataGenerationConfig, FunctionConfig, NoiseConfig, RuntimeConfig
)
from dgp.src.time_series_generator import TimeSeriesGenerator

path = os.getcwd()

if __name__ == '__main__':

    # Set general attributes.
    complexity = 0

    # Set attributes for causal graph. More are set directly in the configuration below.
    min_lag = 0 #to allow instanteaneous links 
    max_lag = 1
    num_targets = 0
    num_features = 5
    num_latent = 0

    # Set attributes for data generation.
    #num_samples = [100, 100, 100]
    num_samples = [100] * 30

    # complexity is only used to initialise any unprovided configs. Here they are all initialised so it is ignored.
    config = DataGenerationConfig(random_seed=1, complexity=complexity, percent_missing=0.0,
                                  causal_graph_config=CausalGraphConfig(
                                      graph_complexity=complexity,
                                      include_noise=True,  
                                      max_lag=max_lag,
                                      min_lag=min_lag,
                                      num_targets=num_targets,
                                      num_features=num_features,
                                      num_latent=num_latent,
                                      prob_edge=0.3,
                                      max_parents_per_variable=5,
                                      max_target_parents=2, max_target_children=0,
                                      max_feature_parents=3, max_feature_children=2,
                                      max_latent_parents=2, max_latent_children=2,
                                      allow_latent_direct_target_cause=False,
                                      allow_target_direct_target_cause=False,
                                      prob_target_autoregressive=0.1,
                                      prob_feature_autoregressive=0.5,
                                      prob_latent_autoregressive=0.2,
                                      prob_noise_autoregressive=0.0,
                                  ),
                                  function_config=FunctionConfig(
                                      function_complexity=complexity
                                  ),
                                  noise_config=NoiseConfig(
                                      noise_complexity=complexity,
                                      noise_variance=[0.01, 0.1]
                                  ),
                                  runtime_config=RuntimeConfig(
                                      num_samples=num_samples, data_generating_seed=random.sample(range(30), 30) 
                                  )
                                  )

    # Instantiate a time series generator.
    ts_generator = TimeSeriesGenerator(config=config)

    # Query for the completed config now that causal graph and SCM have been created.
    full_config_dict = ts_generator.get_full_config()

    # Generate data sets from this configuration.
    data = ts_generator.generate_datasets()
    datasets, causal_graph = data
        
    file_path = os.path.join(path, "synthetic_datasets/data_3n_10ts_30N_datasets.pickle")
    graph_path = os.path.join(path, "synthetic_datasets/data_3n_10ts_30N_graphs.gpickle")
    with open(file_path, 'wb') as file:
        pickle.dump(datasets, file)
    with open(graph_path, 'wb') as f:
        pickle.dump(causal_graph, f, pickle.HIGHEST_PROTOCOL)
    
##############################
# To view data
   # df = datasets[0]
    # View data.
   # for node in df.columns:
    #    plt.figure()
    #    plt.plot(df.index, df[node])
    #    plt.xlabel('Time')
     #   plt.ylabel(node)
     #   plt.title(node)

    # Compare target against its parents.
    #for parent in causal_graph.get_parents('Y1_t'):
    #    if causal_graph.is_feature_node(parent):
    #        plt.figure()
    #        var, lag = causal_graph.get_var_and_lag(parent)
    #        if lag > 0:
    #            plt.scatter(df[var][:-lag], df['Y1'][lag:])
    #            plt.title(f'Y1 vs. {var} lag {lag}')
    #        else:
    #            plt.scatter(df[var], df['Y1'])
    #            plt.title(f'Y1 vs. {var}')
    #        plt.xlabel(var)
    #        plt.ylabel('Y1')

    # View causal graph
    #causal_graph.display_graph(include_noise=False)  
    #gr = causal_graph.display_graph(include_noise=False)  # Set to True to graph noise nodes.
    #print(gr)

    # Show plots.
    #plt.show()
