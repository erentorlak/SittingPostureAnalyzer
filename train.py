# %%
import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

class PostureModel:
    def __init__(self, good_posture_path, bad_posture_path, model_path):
        self.good_posture_path = good_posture_path
        self.bad_posture_path = bad_posture_path
        self.model_path = model_path
        self.clf = RandomForestClassifier(n_estimators=10, max_depth=2, random_state=42)  
        self.columns = []  

    def load_data(self):
        good_posture = pd.read_csv(self.good_posture_path)
        bad_posture = pd.read_csv(self.bad_posture_path)
        good_posture['target'] = 1
        bad_posture['target'] = 0
        self.columns = good_posture.columns
        return pd.concat([good_posture, bad_posture], ignore_index=True).sample(frac=1).reset_index(drop=True)

    def train_and_evaluate(self, X, y):
        cv_scores = cross_val_score(self.clf, X, y, cv=5)
        print(f"Cross-validation scores: {cv_scores}")
        print(f"Mean CV Score: {cv_scores.mean():.2f}")

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        self.clf.fit(X_train, y_train)
        y_pred = self.clf.predict(X_test)
        print(classification_report(y_test, y_pred, target_names=['Bad Posture', 'Good Posture']))

    def save_model(self):
        joblib.dump(self.clf, self.model_path)

    def visualize_tree(self):
        plt.figure(figsize=(20, 10))
        plot_tree(self.clf.estimators_[0], feature_names=list(self.columns), class_names=['Bad Posture', 'Good Posture'], filled=True)
        plt.show()

    def run(self):
        data = self.load_data()
        X = data.drop('target', axis=1)
        y = data['target']
        self.train_and_evaluate(X, y)
        self.save_model()
        self.visualize_tree()

if __name__ == "__main__":
    model = PostureModel('goodPosture.csv', 'badPosture.csv', 'posture_model.pkl')
    model.run()

# %%
