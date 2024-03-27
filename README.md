# Machine Learning Model Energy Consumption Analysis

## Overview
This repository contains code for conducting a comparative analysis of machine-learning model energy consumption, as described in the associated paper titled "A Comparative Analysis of Machine-Learning Model Energy Consumption".

## Code Description
The provided Python script (`energy_consumption_analysis.py`) utilizes scikit-learn and CodeCarbon libraries to evaluate the energy consumption of a Decision Tree classifier trained on the Iris dataset. The script explores different combinations of hyperparameters (max_depth and min_samples_split) to assess their impact on both accuracy and energy consumption.

## Prerequisites
- Python 3.x
- Required libraries: scikit-learn, codecarbon

## Usage
1. Clone this repository to your local machine.
2. Ensure you have the necessary prerequisites installed.
3. Run the `energy_consumption_analysis.py` script.

## Code Explanation
- The script loads the Iris dataset and splits it into training and testing sets.
- It defines a set of hyperparameters to test for the Decision Tree classifier.
- Using itertools, it generates combinations of hyperparameters.
- For each combination, the script initializes the classifier, trains it, and evaluates its performance.
- The CodeCarbon library is used to track energy consumption during model training.
- Results, including hyperparameters, accuracy, and energy consumption, are printed at the end of execution.

## Results
The results obtained from running the script provide insights into the relationship between hyperparameters, model accuracy, and energy consumption. These findings are further discussed in the associated paper.

## Contributing
Contributions to this project are welcome. Please feel free to submit pull requests or open issues for any improvements or suggestions.

## Authors
[Insert Names of Authors]

## License
This project is licensed under the [Insert License] License - see the LICENSE.md file for details.
