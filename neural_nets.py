from sklearn.neural_network import MLPClassifier
from ucimlrepo import fetch_ucirepo
from base_model_evaluator import run_grid_searches

# Define your datasets list
dataset_list = {"census": 20, "RT-IoT2022": 942, "onlineRetail": 352, "Diabetes": 296, "CDCHealthIndicator": 891}

# Loop through each dataset
for name, dataset_id in dataset_list.items():
    # Fetch the dataset
    dataset = fetch_ucirepo(id=dataset_id)

    # Data (as pandas dataframes)
    X = dataset.data.features
    y = dataset.data.targets

    # Define the parameter grid for the neural network
    param_grids = [
        {'hidden_layer_sizes': [(100,)]},
        {'hidden_layer_sizes': [(50,), (100,), (150,)]},
        {'hidden_layer_sizes': [(50,), (100,), (150,) (200,)]},
        {'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 100)],
         'alpha': [0.0001, 0.001, 0.01]},
        {'hidden_layer_sizes': [(50,), (100,), (50, 50), (100, 100), (50, 100, 50)],
         'alpha': [0.0001, 0.001, 0.01, 0.1],
         'activation': ['relu', 'tanh', 'logistic'],
         'solver': ['sgd', 'adam'],
         'learning_rate_init': [0.001, 0.01]}
    ]

    # Run grid searches using the custom run_grid_searches function
    run_grid_searches(name, MLPClassifier(max_iter=100), X, y, param_grids)
