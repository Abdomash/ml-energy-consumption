# Machine Learning Model Energy Consumption Analysis

## Overview
This repository contains code for conducting a comparative analysis of machine-learning model energy consumption, as described in the associated paper titled "A Comparative Analysis of Machine-Learning Model Energy Consumption".

## Code Description
The provided Python script (`base_model_evalutor.py`) utilizes scikit-learn and CodeCarbon libraries to evaluate the energy consumption of a given classifier trained on a given dataset, fine-tuned on given hyperparamter options. The script explores all different combinations of hyperparameters to assess their impact on both accuracy and energy consumption.

## Prerequisites
- Python >=3.7
- Required libraries: scikit-learn, codecarbon, setuptools

## Usage
1. Clone this repository to your local machine.
2. Ensure you have the necessary prerequisites installed.
3. Run the `<model_name>.py` script.

## Code Explanation
- The script loads the dataset, hyperparameters, and classifer from `<model_name>.py` into `base_model_evaluator.py`.
- For each option of hyperparameters, the script initializes the classifier, trains it, and evaluates its performance on the given dataset.
- The CodeCarbon library is used to track energy consumption during model training.
- Results, including hyperparameters, accuracy, and energy consumption, are saved to their corresponding file in the results folder.

## Results
The results folder contains the results for each experiment run. The file is formatted as `<dataset_name>_<classifier>_results.csv`. These findings are further discussed in the associated paper.

## Authors
Abdulrahman Alshahrani, Abdallah Al-Sukhni
