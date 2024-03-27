from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from codecarbon import EmissionsTracker
import pandas as pd
from ucimlrepo import fetch_ucirepo

# Fetch dataset
census_income = fetch_ucirepo(id=20)

# Data (as pandas dataframes)
X = census_income.data.features
y = census_income.data.targets

# Encode the categorical variables. Assuming 'X' is a pandas DataFrame.
X_encoded = pd.get_dummies(X)

# If 'y' is also categorical (e.g., 'yes', 'no'), it needs to be encoded as well.
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, test_size=0.2, random_state=42)

# Define multiple hyperparameter grids
param_grids = [
    {'max_depth': range(1, 30, 5), 'min_samples_split': [2, 4, 6], 'min_samples_leaf': [1, 2]},
    {'max_depth': range(1, 30, 3), 'min_samples_split': [2, 4, 6, 8], 'min_samples_leaf': [1, 2, 3]},
    {'max_depth': range(1, 30), 'min_samples_split': [2, 4, 6, 8, 10], 'min_samples_leaf': [1, 2, 3, 4]},
    # Continue adding more grids as needed, each time increasing the range/values
]

# Initialize the emissions tracker
tracker = EmissionsTracker()

results = []

for i, param_grid in enumerate(param_grids):
    print(f"Starting Grid Search {i+1} with parameters: {param_grid}")
    # Start tracking
    tracker.start()

    # Initialize the decision tree model
    clf = DecisionTreeClassifier()

    # Create GridSearchCV object
    grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)

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
        'grid_search_number': i+1,
        'best_params': best_params,
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

# Output the overall results
print("\nAll Grid Searches completed. Summary of Results:")
for result in results:
    print(result)
