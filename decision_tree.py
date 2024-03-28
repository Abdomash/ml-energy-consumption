from sklearn.tree import DecisionTreeClassifier
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
        {'max_depth': range(1, e, 5), 'min_samples_split': range(2, f, 6), 'min_samples_leaf': range(1, g, 5)},
        {'max_depth': range(1, e, 3), 'min_samples_split': range(2, f, 4), 'min_samples_leaf': range(1, g, 3)},
        {'max_depth': range(1, e, 2), 'min_samples_split': range(2, f, 2), 'min_samples_leaf': range(1, g, 2)},
        # Continue adding more grids as needed, each time increasing the range/values
    ]

    # Run grid searches
    run_grid_searches(name, DecisionTreeClassifier(), x, y, param_grids)
