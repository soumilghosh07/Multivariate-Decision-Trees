import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn import datasets
import numpy as np
from trial6Tree import DecisionTree6
from trial5Tree import DecisionTree5
from trial4Tree import DecisionTree4
from trial3Tree import DecisionTree3

# Load breast cancer dataset
data = datasets.load_breast_cancer()
X, y = data.data, data.target

# Define the number of folds for cross-validation
n_folds = 5

# Calculate the size of each fold
fold_size = len(X) // n_folds

# Shuffle the data and labels
indices = np.arange(len(X))
np.random.shuffle(indices)
X_shuffled = X[indices]
y_shuffled = y[indices]

# Initialize an empty list to store cross-validation accuracies
cv_scores = []

# Perform cross-validation
for i in range(n_folds):
    # Split data into training and test sets for this fold
    test_start = i * fold_size
    test_end = (i + 1) * fold_size
    X_test_fold = X_shuffled[test_start:test_end]
    y_test_fold = y_shuffled[test_start:test_end]
    X_train_fold = np.concatenate([X_shuffled[:test_start], X_shuffled[test_end:]])
    y_train_fold = np.concatenate([y_shuffled[:test_start], y_shuffled[test_end:]])

    # Initialize your DecisionTree classifier
    clf = DecisionTree6(max_depth=10)

    # Train your classifier on the training fold
    clf.fit(X_train_fold, y_train_fold)

    # Make predictions on the test fold
    fold_predictions = clf.predict(X_test_fold)

    # Calculate accuracy for this fold
    fold_acc = accuracy_score(y_test_fold, fold_predictions)
    cv_scores.append(fold_acc)

# Calculate the mean accuracy across all folds
mean_cv_score = np.mean(cv_scores)
print("Mean cross-validation accuracy for 6:", mean_cv_score)

# Perform cross-validation
for i in range(n_folds):
    # Split data into training and test sets for this fold
    test_start = i * fold_size
    test_end = (i + 1) * fold_size
    X_test_fold = X_shuffled[test_start:test_end]
    y_test_fold = y_shuffled[test_start:test_end]
    X_train_fold = np.concatenate([X_shuffled[:test_start], X_shuffled[test_end:]])
    y_train_fold = np.concatenate([y_shuffled[:test_start], y_shuffled[test_end:]])

    # Initialize your DecisionTree classifier
    clf = DecisionTree5(max_depth=10)

    # Train your classifier on the training fold
    clf.fit(X_train_fold, y_train_fold)

    # Make predictions on the test fold
    fold_predictions = clf.predict(X_test_fold)

    # Calculate accuracy for this fold
    fold_acc = accuracy_score(y_test_fold, fold_predictions)
    cv_scores.append(fold_acc)

# Calculate the mean accuracy across all folds
mean_cv_score = np.mean(cv_scores)
print("Mean cross-validation accuracy for 5:", mean_cv_score)
# Perform cross-validation
for i in range(n_folds):
    # Split data into training and test sets for this fold
    test_start = i * fold_size
    test_end = (i + 1) * fold_size
    X_test_fold = X_shuffled[test_start:test_end]
    y_test_fold = y_shuffled[test_start:test_end]
    X_train_fold = np.concatenate([X_shuffled[:test_start], X_shuffled[test_end:]])
    y_train_fold = np.concatenate([y_shuffled[:test_start], y_shuffled[test_end:]])

    # Initialize your DecisionTree classifier
    clf = DecisionTree4(max_depth=10)

    # Train your classifier on the training fold
    clf.fit(X_train_fold, y_train_fold)

    # Make predictions on the test fold
    fold_predictions = clf.predict(X_test_fold)

    # Calculate accuracy for this fold
    fold_acc = accuracy_score(y_test_fold, fold_predictions)
    cv_scores.append(fold_acc)

# Calculate the mean accuracy across all folds
mean_cv_score = np.mean(cv_scores)
print("Mean cross-validation accuracy for 4:", mean_cv_score)
