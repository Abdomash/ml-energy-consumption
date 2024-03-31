from decision_tree import run_decision_tree
from neural_nets import run_neural_nets
from random_forest import run_random_forest
from k_nearest_neighbor import run_knn
from sklearn import datasets
import os
# "census": 20, "RT-IoT2022": 942,"onlineRetail": 352 "Diabetes": 296, "CDCHealthIndicator": 891
# possible other datasets -> "OnlineNewsPopularity": 332, "PhishingWebsites": 327

olivetti = datasets.fetch_olivetti_faces()
newsgroups = datasets.fetch_20newsgroups()
lfw_people = datasets.fetch_lfw_people()
covtype = datasets.fetch_covtype()


dataset = {
    "olivetti": olivetti,
    "newsgroups": newsgroups,
    "lfw_people": lfw_people,
    "covtype": covtype
}
# Set the folder name to where the results will be saved as an environment variable
os.environ['DEVICE_NAME'] = 'abdo-mac' # Change this to the name of your device

if not os.path.exists(os.path.join('results',os.environ['DEVICE_NAME'])):
    os.makedirs(os.path.join('results',os.environ['DEVICE_NAME']))

# for each dataset, run the decision tree, neural nets, random forest, and k-nearest-neighbor models
for name, dataset_id in dataset.items():
    data = dataset_id.data
    target = dataset_id.target
    run_decision_tree(name, data, target)
    run_neural_nets(name, data, target)
    run_random_forest(name, data, target)
    run_knn(name, data, target)
