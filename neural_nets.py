from sklearn.neural_network import MLPClassifier
from base_model_evaluator import run_grid_searches


# Loop through each dataset
def run_neural_nets(name, x, y):

    # Define the parameter grid for the neural network
    param_grids = [
        {'hidden_layer_sizes': [range(1, 201, 100)]},
        {'hidden_layer_sizes': [range(1, 201, 50)]},
        {'hidden_layer_sizes': [range(1, 201, 25)]},
        {'hidden_layer_sizes': [range(1, 201, 25)], 'alpha': [0.001, 0.01]},
        {'hidden_layer_sizes': [range(1, 201, 25)], 'alpha': [0.001, 0.01], 'activation': ['relu', 'tanh', 'logistic']}
    ]

    # Run grid searches using the custom run_grid_searches function
    run_grid_searches(name, MLPClassifier(max_iter=100), x, y, param_grids)
