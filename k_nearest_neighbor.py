from sklearn.neighbors import KNeighborsClassifier
from base_model_evaluator import run_grid_searches 

# Fetch dataset
def run_knn(name, x, y):

    # Define multiple hyperparameter grids
    param_grids = [
        {'n_neighbors': range(1, 31, 5)},
        {'n_neighbors': range(1, 31, 5), 'weights': ['uniform', 'distance']},
        {'n_neighbors': range(1, 31, 3), 'weights': ['uniform', 'distance'], 'metric': ['euclidean', 'manhattan']},
        {'n_neighbors': range(1, 31, 3), 'weights': ['uniform', 'distance'], 'metric': ['minkowski', 'euclidean'], 'p': [1, 2]},
        # {'n_neighbors': range(1, 31, 2), 'weights': ['uniform', 'distance'], 'metric': ['euclidean', 'manhattan', 'minkowski'], 'p': [1, 2, 3], 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']},
    ]

    # Run grid searches
    run_grid_searches(name, KNeighborsClassifier(), x, y, param_grids)
