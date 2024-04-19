from sklearn.neighbors import KNeighborsClassifier
from base_model_evaluator import run_grid_searches 

# Fetch dataset
def run_knn(name, x, y):

    # Define multiple hyperparameter grids
    param_grids = [
        {'n_neighbors': range(5, 20, 6)},
        {'n_neighbors': range(5, 20, 6), 'weights': ['uniform', 'distance']},
        {'n_neighbors': range(5, 20, 3), 'weights': ['uniform', 'distance'], 'metric': ['minkowski', 'euclidean']},
        {'n_neighbors': range(5, 20, 3), 'weights': ['uniform', 'distance'], 'metric': ['minkowski', 'euclidean'], 'p': [1, 2, 3]}
    ]

    run_grid_searches(name, KNeighborsClassifier(), x, y, param_grids)
