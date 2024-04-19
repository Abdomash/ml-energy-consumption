import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from codecarbon import EmissionsTracker
import os
import time

def run_grid_searches(dataset_name, classifier, x, y, param_grids, test_size=0.2, random_state=42, cv=5, scoring='accuracy', n_jobs=-1):
    """
    Perform grid search on a given classifier using the provided parameter grids.

    Args:
        dataset_name (str): The name of the dataset.
        classifier (object): The classifier object to be used for grid search.
        x (array-like or DataFrame): The input features.
        y (array-like or Series): The target variable.
        param_grids (list of dict): The parameter grids to be searched.
        test_size (float, optional): The proportion of the dataset to include in the test split. Defaults to 0.2.
        random_state (int, optional): The seed used by the random number generator. Defaults to 42.
        cv (int, optional): The number of folds in cross-validation. Defaults to 5.
        scoring (str, optional): The scoring metric to evaluate the models. Defaults to 'accuracy'.
        n_jobs (int, optional): The number of jobs to run in parallel. Defaults to -1.

    Returns:
        None
    """
    classifier_name = classifier.__class__.__name__

    
    # Define file names and paths
    emissions_temp_file_name = f'temp_{dataset_name}_{classifier_name}_emissions.csv'
    results_file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", os.environ['DEVICE_NAME'], f'{dataset_name}_{classifier_name}_results.csv')
    emissions_back_file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", os.environ['DEVICE_NAME'], 'backup', f'backup_{dataset_name}_{classifier_name}_emissions.csv')
    results_back_file_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results", os.environ['DEVICE_NAME'], 'backup', f'backup_{dataset_name}_{classifier_name}_results.csv')

    # Remove any existing files
    if os.path.exists(results_file_path):
        os.remove(results_file_path)
    if os.path.exists(emissions_temp_file_name):
        os.remove(emissions_temp_file_name)

    # Initialize the emissions tracker
    tracker = EmissionsTracker(project_name=f'{dataset_name}_{classifier_name}',
                               output_file=emissions_temp_file_name,
                               measure_power_secs=5,
                               )

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

    x_train, x_test, y_train, y_test = train_test_split(x_encoded, y_encoded, test_size=test_size, random_state=random_state)

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

    print(f"\n----------------")
    print(f"All Grid Searches completed.")

    # Convert results to a DataFrame
    results_df = pd.DataFrame(results)

    # Wait until emissions temp file is created
    while not os.path.exists(emissions_temp_file_name):
        time.sleep(1)
    
    # Read emissions temp file
    emissions_df = pd.read_csv(emissions_temp_file_name)

    # Backup the original datasets before concatenating them
    results_df.to_csv(results_back_file_path, index=False)
    emissions_df.to_csv(emissions_back_file_path, index=False)

    # Concatenate results and emissions dataframes
    full_results_df = pd.concat([results_df, emissions_df], axis=1)
    
    # Save the DataFrame to a CSV file
    full_results_df.to_csv(results_file_path, index=False)
    
    # Remove emissions temp file
    os.remove(emissions_temp_file_name)
    
    print(f"Saved results to '{results_file_path}'")

