from sklearn.tree import DecisionTreeClassifier
from base_model_evaluator import run_grid_searches 
# possible other datasets -> "OnlineNewsPopularity": 332, "PhishingWebsites": 327
# Fetch dataset
def run_decision_tree(name, x, y):
    
    # Define multiple hyperparameter grids
    param_grids = [
        {'max_depth': range(1, 31, 12)},
        {'max_depth': range(1, 31, 6), 'min_samples_split': range(2, 17, 4)},
        {'max_depth': range(1, 31, 3), 'min_samples_split': range(2, 17, 2)},
        {'max_depth': range(1, 31, 3), 'min_samples_split': range(2, 17, 2), 'min_samples_leaf': range(1, 17, 2)}
    ]

    # Run grid searches
    run_grid_searches(name, DecisionTreeClassifier(), x, y, param_grids)
