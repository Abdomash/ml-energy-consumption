from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from codecarbon import EmissionsTracker
import itertools

# Load dataset (example: Iris dataset)
X, y = load_iris(return_X_y=True)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define hyperparameters to test
max_depths = [None, 10, 20, 30]
min_samples_splits = [2, 4, 6]

# Create combinations of all hyperparameters
hyperparameter_combinations = list(itertools.product(max_depths, min_samples_splits))

# Initialize the emissions tracker
tracker = EmissionsTracker()

results = []

for max_depth, min_samples_split in hyperparameter_combinations:
    # Start tracking
    tracker.start()
    
    # Initialize and train the decision tree model
    clf = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split)
    clf.fit(X_train, y_train)
    
    # Predictions and accuracy
    predictions = clf.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    
    # Stop tracking and get energy consumption data
    energy_consumed = tracker.stop()
    
    results.append({
        'max_depth': max_depth,
        'min_samples_split': min_samples_split,
        'accuracy': accuracy,
        'energy_consumed': energy_consumed,
    })

# Output the results
for result in results:
    print(result)
