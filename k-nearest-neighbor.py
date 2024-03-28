from sklearn.neighbors import KNeighborsClassifier
from ucimlrepo import fetch_ucirepo
from base_model_evaluator import run_grid_searches 
# possible other datasets -> "OnlineNewsPopularity": 332, "PhishingWebsites": 327
# Fetch dataset
dataset_list = {"census": 20, "RT-IoT2022": 942, "onlineRetail": 352, "Diabetes": 296, "CDCHealthIndicator": 891}
for name, dataset_id in dataset_list.items():
    censusIncome = fetch_ucirepo(id=dataset_id)

    # Data (as pandas dataframes)
    x = censusIncome.data.features
    y = censusIncome.data.targets

    # Define multiple hyperparameter grids
    e,f,g = 10, 10, 10
    param_grids = [
        {'n_neighbors': range(1, 31, 5)},
        {'n_neighbors': range(1, 31, 5), 'weights': ['uniform', 'distance']},
        {'n_neighbors': range(1, 31, 3), 'weights': ['uniform', 'distance'], 'metric': ['euclidean', 'manhattan']},
        {'n_neighbors': range(1, 31, 3), 'weights': ['uniform', 'distance'], 'metric': ['minkowski'], 'p': [1, 2]},
        {'n_neighbors': range(1, 31, 2), 'weights': ['uniform', 'distance'], 'metric': ['euclidean', 'manhattan', 'minkowski'], 'p': [1, 2, 3], 'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']},
    ]

    # Run grid searches
    run_grid_searches(name, KNeighborsClassifier(), x, y, param_grids)
