from sklearn.tree import DecisionTreeClassifier
from classifiers.base_model_evaluator import run_grid_searches 

def run_decision_tree(name, x, y):
    
    # Define multiple hyperparameter grids
    param_grids = [
        {'max_depth': range(1, 31, 12)},
        {'max_depth': range(1, 31, 6), 'min_samples_split': range(2, 17, 4)},
        {'max_depth': range(1, 31, 3), 'min_samples_split': range(2, 17, 2)},
        {'max_depth': range(1, 31, 3), 'min_samples_split': range(2, 17, 2), 'min_samples_leaf': range(1, 17, 2)}
    ]

    run_grid_searches(name, DecisionTreeClassifier(random_state=21), x, y, param_grids)
