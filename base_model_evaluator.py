import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from codecarbon import EmissionsTracker
import os
import time

def run_grid_searches(dataset_name, classifier, x, y, param_grids, test_size=0.2, random_state=42, cv=5, scoring='accuracy', n_jobs=-1):
    classifier_name = classifier.__class__.__name__


    emissions_temp_file_name = f'temp_{dataset_name}_{classifier_name}_emissions.csv'
    results_file_path = os.path.join("results", f'{dataset_name}_{classifier_name}_results.csv')
    
    if os.path.exists(results_file_path):
        os.remove(results_file_path)
    
    if os.path.exists(emissions_temp_file_name):
        os.remove(emissions_temp_file_name)
    
    # Encode the categorical variables if needed
    if isinstance(x, pd.DataFrame):
        x_encoded = pd.get_dummies(x)
    else:
        x_encoded = x

    if isinstance(y, pd.Series) and y.dtype == 'object':
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
    else:
        y_encoded = y

    # Split dataset
    x_train, x_test, y_train, y_test = train_test_split(x_encoded, y_encoded, test_size=test_size, random_state=random_state)

    # Initialize the emissions tracker
    tracker = EmissionsTracker(project_name=f'{dataset_name}_{classifier_name}',
                               output_file=emissions_temp_file_name,
                               measure_power_secs=5)

    results = []

    for i, param_grid in enumerate(param_grids):
        print(f"Starting Grid Search {i+1} with parameters: {param_grid}")

        # Start tracking emissions and time
        start_time = time.time()
        tracker.start()

        # Create GridSearchCV object
        grid_search = GridSearchCV(classifier, param_grid, cv=cv, scoring=scoring, n_jobs=n_jobs)

        # Perform grid search
        grid_search.fit(x_train, y_train)

        # Stop tracking and get energy consumption data and time elapsed
        energy_consumed = tracker.stop()
        elapsed_time = time.time() - start_time

        # Best parameters and best score
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_

        # Test the best model
        best_clf = grid_search.best_estimator_
        predictions = best_clf.predict(x_test)
        test_accuracy = accuracy_score(y_test, predictions)

        # Save the results
        results.append({
            'dataset_name': dataset_name,
            'classifier': classifier_name,
            'grid_search_number': i+1,
            'param_grid': str(param_grid),  # Convert dict to string for CSV compatibility
            'best_params': str(best_params),  # Convert dict to string for CSV compatibility
            'best_cv_score': best_score,
            'test_accuracy': test_accuracy,
            'energy_consumed': energy_consumed,
            'elapsed_time': elapsed_time,
        })

        # # Output the results for this grid search
        # print(f"")
        # print(f"Results for Grid Search {i+1}:")
        # print(f"Best Parameters: {best_params}")
        # print(f"Test Accuracy: {test_accuracy}")
        # print(f"Energy Consumed: {energy_consumed} kWh")
        # print(f"Elapsed Time: {elapsed_time} seconds")

    print(f"\n----------------")
    print(f"All Grid Searches completed.")

    # Convert results to a DataFrame
    results_df = pd.DataFrame(results)

    while not os.path.exists(emissions_temp_file_name): 
        os.sleep(1)
    
    emissions_df = pd.read_csv(emissions_temp_file_name)
    full_results_df = pd.concat([results_df, emissions_df], axis=1)
    
    # Save the DataFrame to a CSV file
    results_df.to_csv(results_file_path, index=False)

    # Remove Temporary file
    os.remove(emissions_temp_file_name)
    
    print(f"Saved results to '{results_file_path}'")

'''
# Example usage
# Note: Replace `fetch_ucirepo` and `param_grids` with actual data and parameters.

# Fetch dataset
censusIncome = fetch_ucirepo(id=20)  # Placeholder
x = censusIncome.data.features  # Placeholder
y = censusIncome.data.targets  # Placeholder

# Define a classifier, e.g., RandomForestClassifier from sklearn.ensemble
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier()

# Define multiple hyperparameter grids
param_grids = [
    {'max_depth': range(1, 10), 'min_samples_split': [2, 4], 'min_samples_leaf': [1, 2]},
    # Add more param_grids as needed
]

# Run grid searches and save results to CSV
results = run_grid_searches('censusIncome', classifier, x, y, param_grids)
'''