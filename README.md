If this work is of use to you, please consider citing the following:

Wolfe, Sean, and M. Reza Emami. "A Data-driven Approach to the Classification of Temporary Captures in the Earth-Moon System." 2024 IEEE Aerospace Conference. IEEE, 2024.

# Classifier Hyperparameter Tuning and Evaluation

This repository contains a Python implementation for training and evaluating various classifiers, performing hyperparameter tuning, and analyzing multiple dataset forms. The primary goal is to apply machine learning classifiers to a variety of tasks with custom dataset preprocessing and tuning strategies.

## Classifiers

The code employs a variety of classifiers to evaluate their performance on different target tasks. These classifiers include:

- **Neural Network Classifier (MLPClassifier)**
- **Support Vector Machine (SVC)**
- **Random Forest Classifier (RandomForestClassifier)**
- **Gradient Boosting Classifier (GradientBoostingClassifier)**
- **Stochastic Gradient Descent Classifier (SGDClassifier)**
- **K Nearest Neighbor Classifier (KNeighborsClassifier)**
- **Gaussian Naive Bayes (GaussianNB)**
- **Decision Tree Classifier (DecisionTreeClassifier)**
- **Bagging Classifier (BaggingClassifier)**
- **Extremely Randomized Trees (ExtraTreesClassifier)**
- **AdaBoost Classifier (AdaBoostClassifier)**
- **Histogram Gradient Boosting Classifier (HistGradientBoostingClassifier)**
- **Voting Classifier** combining Random Forest and Histogram Gradient Boosting

### Example Classifier Configuration

```python
classifiers = [
    VotingClassifier(
        estimators=[
            ('rf', RandomForestClassifier(n_estimators=200, max_features=6, max_depth=None)),
            ('hgb', HistGradientBoostingClassifier(max_depth=8, max_leaf_nodes=60, learning_rate=0.3))
        ],
        voting='soft'
    )
]
labels = ['Voting Classifier']

Target Classes and Labels

The target classes and corresponding labels represent different classification tasks that the classifiers are trained to predict. Some examples of the target classes and their labels include:

    Non-STC / STC → STC
    Prograde / Retrograde → Retrograde
    TCF / TCo → Became Minimoon
    Not / Crossed 1 Hill → Crossed 1 Hill
    Not / 100+ Days in 1 Hill → 100+ Days in 1 Hill
    Taxonomy → Taxonomy
    Classed 1 Hill Duration → 1 Hill Duration
    Classed Minimum Distance → Minimum Distance

Target Classes and Labels Configuration

target_classes = ['1 Hill Duration', 'Minimum Distance']
target_labels = ['Classed 1 Hill Duration', 'Classed Minimum Distance']

Dataset Forms

The code works with various forms of datasets, including both oversampled and undersampled datasets. Some available dataset forms are:

    Quantile Transform
    Standard Scalar
    Normalized
    Oversampled
    Original
    Undersampled
    Quantile Oversampled

Dataset Form Example

dataset_forms = ['Oversampled']

Hyperparameter Tuning

The code includes a hyperparameter tuning section to optimize classifier parameters. A custom scorer based on accuracy is used for evaluation, and a parameter grid defines the range of hyperparameters to search.
Example Hyperparameter Tuning

custom_scorer = make_scorer(custom_accuracy, greater_is_better=True)
param_grid = {
    'hgb__max_depth': [7, 8, None],
}

Preprocessing and Dataset Handling

The datasets are preprocessed and transformed using various techniques, such as:

    Cluster data preprocessing
    Feature scaling
    Data oversampling and undersampling

Example Preprocessing Step

datasets = preprocessing_main(cluster_data_ini.loc[:, ["Helio x at Capture", ...]])

Requirements

    scikit-learn
    pandas
    numpy
    matplotlib
    seaborn
    joblib (optional, for saving models)

How to Run

    Clone the repository:

git clone https://github.com/your-username/your-repository.git
cd your-repository

Install the necessary dependencies:

pip install -r requirements.txt

Run the script to train and evaluate classifiers:

    python classifier_script.py

    Results will be saved and can be viewed in the output folder.

License

This project is licensed under the MIT License - see the LICENSE file for details.
Acknowledgments

    Scikit-learn for providing the machine learning models and utilities.
    Any relevant contributors or libraries.
