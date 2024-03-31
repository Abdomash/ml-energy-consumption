from sklearn.tree import DecisionTreeClassifier
from base_model_evaluator import run_grid_searches 
# possible other datasets -> "OnlineNewsPopularity": 332, "PhishingWebsites": 327
# Fetch dataset
def run_decision_tree(name, x, y):
    
    # Define multiple hyperparameter grids
    max_value = 30
    param_grids = [
        {'max_depth': range(1, max_value, 5), 'min_samples_split': range(2, max_value, 6), 'min_samples_leaf': range(1, max_value, 5)},
        {'max_depth': range(1, max_value, 3), 'min_samples_split': range(2, max_value, 4), 'min_samples_leaf': range(1, max_value, 3)},
        {'max_depth': range(1, max_value, 2), 'min_samples_split': range(2, max_value, 2), 'min_samples_leaf': range(1, max_value, 2)},
        # Continue adding more grids as needed, each time increasing the range/values
    ]

    # Run grid searches
    run_grid_searches(name, DecisionTreeClassifier(), x, y, param_grids)
