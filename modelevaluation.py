import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from codecarbon import EmissionsTracker


def run_grid_searches(classifier, X, y, param_grids, test_size=0.2, random_state=42, cv=5, scoring='accuracy', n_jobs=-1):
    classifier_name = classifier.__class__.__name__
    
    # Encode the categorical variables if needed
    if isinstance(X, pd.DataFrame):
        X_encoded = pd.get_dummies(X)
    else:
        X_encoded = X

    if isinstance(y, pd.Series) and y.dtype == 'object':
        label_encoder = LabelEncoder()
        y_encoded = label_encoder.fit_transform(y)
    else:
        y_encoded = y

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=test_size, random_state=random_state)

    # Initialize the emissions tracker
    tracker = EmissionsTracker()

    results = []

    for i, param_grid in enumerate(param_grids):
        print(f"Starting Grid Search {i+1} with parameters: {param_grid}")
        # Start tracking
        tracker.start()

        # Create GridSearchCV object
        grid_search = GridSearchCV(classifier, param_grid, cv=cv, scoring=scoring, n_jobs=n_jobs)

        # Perform grid search
        grid_search.fit(X_train, y_train)

        # Stop tracking and get energy consumption data
        energy_consumed = tracker.stop()

        # Best parameters and best score
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_

        # Test the best model
        best_clf = grid_search.best_estimator_
        predictions = best_clf.predict(X_test)
        test_accuracy = accuracy_score(y_test, predictions)

        # Save the results
        results.append({
            'classifier': classifier_name,
            'grid_search_number': i+1,
            'param_grid': str(param_grid),  # Convert dict to string for CSV compatibility
            'best_params': str(best_params),  # Convert dict to string for CSV compatibility
            'best_cv_score': best_score,
            'test_accuracy': test_accuracy,
            'energy_consumed': energy_consumed,
        })

        # Output the results for this grid search
        print(f"Results for Grid Search {i+1}:")
        print(f"Best Parameters: {best_params}")
        print(f"Best CV Score: {best_score}")
        print(f"Test Accuracy: {test_accuracy}")
        print(f"Energy Consumed: {energy_consumed} kWh")

    # Convert results to a DataFrame
    results_df = pd.DataFrame(results)
    
    # Save the DataFrame to a CSV file
    results_df.to_csv('grid_search_results.csv', index=False)
    
    print("\nAll Grid Searches completed. Summary of Results:")
    for result in results:
        print(result)

    return results

# Example usage
# Note: Replace `fetch_ucirepo` and `param_grids` with actual data and parameters.

# Fetch dataset
# census_income = fetch_ucirepo(id=20)  # Placeholder
# X = census_income.data.features  # Placeholder
# y = census_income.data.targets  # Placeholder

# Define a classifier, e.g., RandomForestClassifier from sklearn.ensemble
# from sklearn.ensemble import RandomForestClassifier
# classifier = RandomForestClassifier()

# Define multiple hyperparameter grids
# param_grids = [
#     {'max_depth': range(1, 10), 'min_samples_split': [2, 4], 'min_samples_leaf': [1, 2]},
#     # Add more param_grids as needed
# ]

# Run grid searches and save results to CSV
# results = run_grid_searches(classifier, X, y, param_grids)
