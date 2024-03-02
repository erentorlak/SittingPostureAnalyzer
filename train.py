
#%%
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the datasets
good_posture = pd.read_csv('goodPosture.csv')
bad_posture = pd.read_csv('badPosture.csv')

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
#%%
X

#%%
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
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

def load_and_prepare_data(good_posture_path, bad_posture_path):
    # Load the datasets
    good_posture = pd.read_csv(good_posture_path)
    bad_posture = pd.read_csv(bad_posture_path)

    # Add a target column to each dataset
    good_posture['target'] = 1  # 1 for good posture
    bad_posture['target'] = 0  # 0 for bad posture

    # Combine and shuffle the datasets
    data = pd.concat([good_posture, bad_posture], ignore_index=True).sample(frac=1).reset_index(drop=True)
    return data

def train_random_forest(X_train, y_train, n_estimators=3, random_state=42):
    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    clf.fit(X_train, y_train)
    return clf

def evaluate_model(clf, X_test, y_test):
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(classification_report(y_test, y_pred, target_names=['Bad Posture', 'Good Posture']))

def save_model(clf, model_path):
    joblib.dump(clf, model_path)

def visualize_model(clf, feature_names):
    plt.figure(figsize=(20, 10))
    plot_tree(clf.estimators_[0], feature_names=feature_names, class_names=['Bad Posture', 'Good Posture'], filled=True)
    plt.show()

def main():
    data = load_and_prepare_data('goodPosture.csv', 'badPosture.csv')
    X = data.drop('target', axis=1)
    y = data['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    clf = train_random_forest(X_train, y_train)
    evaluate_model(clf, X_test, y_test)
    save_model(clf, 'posture_asdmodel.pkl')
    visualize_model(clf, list(X_train.columns))

if __name__ == "__main__":
    main()

# %%
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# Load the datasets
good_posture = pd.read_csv('goodPosture.csv')
bad_posture = pd.read_csv('badPosture.csv')

# Add a target column to each dataset
good_posture['target'] = 1  # 1 for good posture
bad_posture['target'] = 0  # 0 for bad posture

# Combine and shuffle the datasets
data = pd.concat([good_posture, bad_posture], ignore_index=True).sample(frac=1).reset_index(drop=True)

# Split the dataset into features and target variable
X = data.drop('target', axis=1)
y = data['target']

# Initialize the Random Forest classifier
clf = RandomForestClassifier(n_estimators=3, random_state=42)

# Perform cross-validation
cv_scores = cross_val_score(clf, X, y, cv=5)  # 5-fold cross-validation

# Print the cross-validation scores
print(f"Cross-validation scores: {cv_scores}")
print(f"Mean CV Score: {cv_scores.mean():.2f}")

# Split the dataset into training and testing sets for final model training
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the classifier on the full training data
clf.fit(X_train, y_train)

# Predict on the test set
y_pred = clf.predict(X_test)

# Display a classification report
print(classification_report(y_test, y_pred, target_names=['Bad Posture', 'Good Posture']))

# Save the trained model
joblib.dump(clf, 'posture_model.pkl')

# %%

import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

plt.figure(figsize=(20, 10))
plot_tree(clf.estimators_[0], feature_names=list(X.columns), class_names=['Bad Posture', 'Good Posture'], filled=True)
plt.show()

# %%
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

class PostureModelTrainer:
    def __init__(self, good_posture_path, bad_posture_path, model_path):
        self.good_posture_path = good_posture_path
        self.bad_posture_path = bad_posture_path
        self.model_path = model_path
        self.clf = RandomForestClassifier(n_estimators=3, random_state=42)
    
    def load_and_prepare_data(self):
        good_posture = pd.read_csv(self.good_posture_path)
        bad_posture = pd.read_csv(self.bad_posture_path)
        good_posture['target'] = 1
        bad_posture['target'] = 0
        data = pd.concat([good_posture, bad_posture], ignore_index=True).sample(frac=1).reset_index(drop=True)
        return data
    
    def perform_cross_validation(self, X, y):
        cv_scores = cross_val_score(self.clf, X, y, cv=5)
        print(f"Cross-validation scores: {cv_scores}")
        print(f"Mean CV Score: {cv_scores.mean():.2f}")
    
    def train_and_evaluate(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        self.clf.fit(X_train, y_train)
        y_pred = self.clf.predict(X_test)
        print(classification_report(y_test, y_pred, target_names=['Bad Posture', 'Good Posture']))
    
    def save_model(self):
        joblib.dump(self.clf, self.model_path)
    
    def visualize_tree(self):
        plt.figure(figsize=(20, 10))
        plot_tree(self.clf.estimators_[0], feature_names=list(X.columns), class_names=['Bad Posture', 'Good Posture'], filled=True)
        plt.show()
    
    def run(self):
        data = self.load_and_prepare_data()
        X = data.drop('target', axis=1)
        y = data['target']
        self.perform_cross_validation(X, y)
        self.train_and_evaluate(X, y)
        self.save_model()
        self.visualize_tree()

if __name__ == "__main__":
    trainer = PostureModelTrainer('goodPosture.csv', 'badPosture.csv', 'posture_model.pkl')
    trainer.run()

# %%
