# Machine Learning Model Energy Consumption Analysis Tool

## Overview

This repository contains code for measuring the energy consumption of different machine learning models. The code is developed for the associated paper titled "A Comparative Analysis of Machine-Learning Model Energy Consumption".

## Code Description
The provided Python script `base_model_evalutor.py` utilizes scikit-learn and CodeCarbon libraries to evaluate the energy consumption of a given classifier trained on a given dataset, fine-tuned on given hyperparamter options. The script explores all different combinations of hyperparameters to assess their impact on both accuracy and energy consumption. The different `<model_name>.py` files run different machine learning algorithms with the given dataset and hyperparameter options on `base_model_evalutor.py`.

## Prerequisites
- Python: `>=3.7`
- Libraries: `scikit-learn`, `codecarbon`, `setuptools`

## Usage
1. Clone this repository to your local machine.
2. Ensure you have the necessary prerequisites installed.
3. Run the `<model_name>.py` script.

### Example Usage
```
# Example usage
# Note: Replace `fetch_ucirepo` and `param_grids` with actual data and parameters.

# Fetch dataset
censusIncome = fetch_ucirepo(id=20)  # Placeholder
x = censusIncome.data.features  # Placeholder
y = censusIncome.data.targets  # Placeholder

# Define a classifier, e.g., RandomForestClassifier from sklearn ensemble
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier()

# Define multiple hyperparameter grids
param_grids = [
    {'max_depth': range(1, 10), 'min_samples_split': [2, 4], 'min_samples_leaf': [1, 2]},
    # Add more param_grids as needed
]

# Run grid searches and save results to CSV
results = run_grid_searches('censusIncome', classifier, x, y, param_grids)
```

## Code Explanation
- The script loads the dataset, hyperparameters, and classifer from `<model_name>.py` into `base_model_evaluator.py`.
- For each option of hyperparameters, the script initializes the classifier, trains its model, and evaluates its performance on the given dataset.
- The CodeCarbon library is used to track energy consumption during model training.
- Results, including hyperparameters, accuracy, and energy consumption, are saved to their corresponding file in the results folder.

## Results
The `results` folder contains the results for each experiment run. The file is formatted as `<dataset_name>_<classifier>_results.csv`. These findings are further discussed in the associated paper.

## Authors
Abdulrahman Alshahrani, Abdallah Al-Sukhni