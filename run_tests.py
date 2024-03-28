from decision_tree import run_decision_tree
from neural_nets import run_neural_nets
from random_forest import run_random_forest
from k_nearest_neighbor import run_knn

dataset = {"census": 20, "RT-IoT2022": 942, "onlineRetail": 352, "Diabetes": 296, "CDCHealthIndicator": 891}

# for each dataset, run the decision tree, neural nets, random forest, and k-nearest-neighbor models
for name, dataset_id in dataset.items():
    run_decision_tree(name, dataset_id)
    run_neural_nets(name, dataset_id)
    run_random_forest(name, dataset_id)
    run_knn(name, dataset_id)
