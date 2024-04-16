from sklearn.ensemble import RandomForestClassifier
from base_model_evaluator import run_grid_searches

# Loop through each dataset
def run_random_forest(name, x, y):

    # Define the parameter grid for the Random Forest
    param_grids = [
        {'n_estimators': [10, 50, 100]},
        {'n_estimators': [10, 50, 100, 200], 'max_depth': [None, 10, 20]},
        {'n_estimators': [10, 50, 100, 200], 'max_depth': [None, 10, 20], 'min_samples_split': [2, 5]},
        {'n_estimators': [10, 50, 100, 200, 300], 'max_depth': [None, 10, 20], 'min_samples_split': [2, 5], 'min_samples_leaf': [1, 2]}
    ]

    # Run grid searches using the custom run_grid_searches function
    run_grid_searches(name, RandomForestClassifier(), x, y, param_grids)
