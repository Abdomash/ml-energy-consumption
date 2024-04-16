from sklearn.neural_network import MLPClassifier
from base_model_evaluator import run_grid_searches


# Loop through each dataset
def run_neural_nets(name, x, y):

    # Define the parameter grid for the neural network
    param_grids = [
        {'hidden_layer_sizes': [(100,), (50, 50)]},
        {'hidden_layer_sizes': [(100,), (50, 50), (100, 100), (200, 200)]},
        {'hidden_layer_sizes': [(100,), (50, 50), (100, 100), (200, 200), (300, 300)]},
        {'hidden_layer_sizes': [(100,), (50, 50), (100, 100), (200, 200), (300, 300)], 'alpha': [0.0001, 0.001], 'activation': ['relu', 'tanh', 'logistic']}
    ]

    # Run grid searches using the custom run_grid_searches function
    run_grid_searches(name, MLPClassifier(max_iter=100), x, y, param_grids)