from classifiers.decision_tree import run_decision_tree
from classifiers.neural_nets import run_neural_nets
from classifiers.random_forest import run_random_forest
from classifiers.k_nearest_neighbor import run_knn
from sklearn import datasets
from ucimlrepo import fetch_ucirepo
import pandas as pd
import os

# clean olivetti
olivetti_data = datasets.fetch_olivetti_faces()
olivetti = olivetti_data.data, olivetti_data.target

# clean census
census_data = fetch_ucirepo(id=20)
combined = pd.concat([pd.DataFrame(census_data.data.features), pd.DataFrame(census_data.data.targets)], axis=1)
cleaned_combined = combined.dropna(subset=combined.columns[:-1])
cleaned_features = cleaned_combined.iloc[:, :-1]
cleaned_targets = cleaned_combined.iloc[:, -1]
census = cleaned_features, cleaned_targets

# clean RT-IoT2022
rt_data = fetch_ucirepo(id=942).data
rt = rt_data.features.drop(columns=['id.orig_p', 'id.resp_p']), rt_data.targets

# clean CDC
cdc_data = fetch_ucirepo(id=891).data
cdc = cdc_data.features, cdc_data.targets


dataset = {
    "olivetti": olivetti,
    "census": census,
    "RT-IoT2022": rt,
    "CDCHealthIndicator": cdc
}

# Set the folder name to where the results will be saved as an environment variable
os.environ['DEVICE_NAME'] = 'my-pc' # TODO: Change this to the name of your device

if not os.path.exists(os.path.join('results',os.environ['DEVICE_NAME'])):
    os.makedirs(os.path.join('results',os.environ['DEVICE_NAME']))
    os.makedirs(os.path.join('results',os.environ['DEVICE_NAME'],'backup'))

# for each dataset, run the decision tree, neural nets, random forest, and k-nearest-neighbor models
for name, (data, target) in dataset.items():
    run_decision_tree(name, data, target)
    run_neural_nets(name, data, target)
    run_random_forest(name, data, target)
    run_knn(name, data, target)
