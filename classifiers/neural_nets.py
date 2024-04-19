from sklearn.neural_network import MLPClassifier
from classifiers.base_model_evaluator import run_grid_searches


def run_neural_nets(name, x, y):

    # Define the parameter grid for the neural network
    param_grids = [
        {'hidden_layer_sizes': [(100,), (50, 50)]},
        {'hidden_layer_sizes': [(100,), (50, 50), (100, 100), (200, 200)]},
        {'hidden_layer_sizes': [(100,), (50, 50), (100, 100), (200, 200), (300, 300)]},
        {'hidden_layer_sizes': [(100,), (50, 50), (100, 100), (200, 200), (300, 300)], 'alpha': [0.0001, 0.001], 'activation': ['relu', 'tanh', 'logistic']}
    ]

    run_grid_searches(name, MLPClassifier(max_iter=100, random_state=21), x, y, param_grids)