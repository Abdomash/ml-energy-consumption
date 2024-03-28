from sklearn.ensemble import RandomForestClassifier
from ucimlrepo import fetch_ucirepo
from base_model_evaluator import run_grid_searches

# Define your datasets list
dataset_list = {"census": 20, "RT-IoT2022": 942, "onlineRetail": 352, "Diabetes": 296, "CDCHealthIndicator": 891}

# Loop through each dataset
for name, dataset_id in dataset_list.items():
    # Fetch the dataset
    dataset = fetch_ucirepo(id=dataset_id)

    # Data (as pandas dataframes)
    X = dataset.data.features
    y = dataset.data.targets

    # Define the parameter grid for the Random Forest
    param_grids = [
        {'n_estimators': [10, 50, 100]},
        {'n_estimators': [100, 200],
         'max_depth': [None, 10, 20]},
        {'n_estimators': [100, 200],
         'max_depth': [None, 10, 20],
         'min_samples_split': [2, 5]},
        {'n_estimators': [100, 200, 300],
         'max_depth': [None, 10, 20],
         'min_samples_split': [2, 5],
         'min_samples_leaf': [1, 2]},
        {'n_estimators': [100, 200, 300],
         'max_depth': [None, 10, 20, 30],
         'min_samples_split': [2, 5, 10],
         'min_samples_leaf': [1, 2, 4],
         'bootstrap': [True, False]}
    ]

    # Run grid searches using the custom run_grid_searches function
    run_grid_searches(name, RandomForestClassifier(), X, y, param_grids)
