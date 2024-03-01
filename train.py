
#%%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the datasets
good_posture = pd.read_csv('good_posture.csv')
bad_posture = pd.read_csv('bad_posture.csv')

# Add a target column to each dataset
good_posture['target'] = 1  # 1 for good posture
bad_posture['target'] = 0  # 0 for bad posture

# Combine the datasets
data = pd.concat([good_posture, bad_posture], ignore_index=True)

# Shuffle the dataset
data = data.sample(frac=1).reset_index(drop=True)

# Split the dataset into features and target variable
X = data.drop('target', axis=1)
y = data['target']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the Random Forest classifier
clf = RandomForestClassifier(n_estimators=3, random_state=42)

# Train the classifier
clf.fit(X_train, y_train)

# Predict on the test set
y_pred = clf.predict(X_test)

# Calculate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Display a classification report
print(classification_report(y_test, y_pred, target_names=['Bad Posture', 'Good Posture']))

# %%
# Save the model as pkl

import joblib

joblib.dump(clf, 'posture_model.pkl')

# %%
#visualize the model
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

plt.figure(figsize=(20, 10))
plot_tree(clf.estimators_[0], feature_names=X.columns, class_names=['Bad Posture', 'Good Posture'], filled=True)
plt.show()



## cross validation alttaki

# %%
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import numpy as np

# Assuming you've already prepared your datasets with diverse and augmented data

# Load the datasets
good_posture = pd.read_csv('good_posture.csv')
bad_posture = pd.read_csv('bad_posture.csv')

# Combine and shuffle datasets
data = pd.concat([good_posture.assign(target=1), bad_posture.assign(target=0)], ignore_index=True).sample(frac=1).reset_index(drop=True)

# Split the dataset
X = data.drop('target', axis=1)
y = data['target']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Initialize the classifier with basic parameters for hyperparameter tuning
clf = RandomForestClassifier(random_state=42)

# Hyperparameters grid
param_grid = {
    'n_estimators': [10, 50, 100],
    'max_depth': [None, 10, 20, 30],
    'min_samples_leaf': [1, 2, 4]
}

# Grid search with cross-validation
grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Best estimator after grid search
clf_best = grid_search.best_estimator_

# Evaluate on the test set
y_pred = clf_best.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print(classification_report(y_test, y_pred, target_names=['Bad Posture', 'Good Posture']))

# Save the optimized model
joblib.dump(clf_best, 'posture_model_optimized.pkl')

# %%
