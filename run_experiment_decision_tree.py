from sklearn.tree import DecisionTreeClassifier
from ucimlrepo import fetch_ucirepo
from modelevaluation import run_grid_searches

# Fetch dataset
census_income = fetch_ucirepo(id=20)

# Data (as pandas dataframes)
X = census_income.data.features
y = census_income.data.targets

# Define multiple hyperparameter grids
param_grids = [
    {'max_depth': range(1, 30, 5), 'min_samples_split': [2, 4, 6], 'min_samples_leaf': [1, 2]},
    {'max_depth': range(1, 30, 3), 'min_samples_split': [2, 4, 6, 8], 'min_samples_leaf': [1, 2, 3]},
    {'max_depth': range(1, 30), 'min_samples_split': [2, 4, 6, 8, 10], 'min_samples_leaf': [1, 2, 3, 4]},
    # Continue adding more grids as needed, each time increasing the range/values
]

# Run grid searches
results = run_grid_searches(DecisionTreeClassifier(), X, y, param_grids)
