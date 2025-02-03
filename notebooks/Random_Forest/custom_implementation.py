import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load dataset
df = pd.read_csv('src/cleaned_data/cleaned_data.csv')
X = df.drop('Label', axis=1).values
y = df['Label'].values

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define Gini impurity
def gini_impurity(y):
    counts = np.bincount(y)
    probabilities = counts / len(y)
    return 1 - np.sum(probabilities ** 2)

# Bootstrap sampling
def bootstrap_sample(X, y):
    n_samples = X.shape[0]
    idxs = np.random.choice(n_samples, size=n_samples, replace=True)
    return X[idxs], y[idxs]

# Decision Tree Node
class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
    def is_leaf(self):
        return self.value is not None

# Decision Tree Class
class DecisionTree:
    def __init__(self, max_depth=10, min_samples_split=2, n_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.root = None
    
    def fit(self, X, y):
        self.n_features = X.shape[1] if not self.n_features else min(self.n_features, X.shape[1])
        self.root = self._grow_tree(X, y)
    
    def _grow_tree(self, X, y, depth=0):
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))
        if depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split:
            return Node(value=Counter(y).most_common(1)[0][0])
        feat_idxs = np.random.choice(n_feats, self.n_features, replace=False)
        best_feat, best_thresh = self._best_split(X, y, feat_idxs)
        left_idxs = X[:, best_feat] <= best_thresh
        return Node(best_feat, best_thresh, self._grow_tree(X[left_idxs], y[left_idxs], depth + 1),
                    self._grow_tree(X[~left_idxs], y[~left_idxs], depth + 1))
    
    def _best_split(self, X, y, feat_idxs):
        best_gini, split_idx, split_thresh = float('inf'), None, None
        for feat_idx in feat_idxs:
            for thresh in np.unique(X[:, feat_idx]):
                left_idxs = X[:, feat_idx] <= thresh
                gini = self._gini_gain(y, left_idxs)
                if gini < best_gini:
                    best_gini, split_idx, split_thresh = gini, feat_idx, thresh
        return split_idx, split_thresh
    
    def _gini_gain(self, y, left_idxs):
        n, n_left, n_right = len(y), np.sum(left_idxs), len(y) - np.sum(left_idxs)
        return (n_left * gini_impurity(y[left_idxs]) + n_right * gini_impurity(y[~left_idxs])) / n if n_left and n_right else float('inf')
    
    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])
    
    def _traverse_tree(self, x, node):
        return node.value if node.is_leaf() else self._traverse_tree(x, node.left if x[node.feature] <= node.threshold else node.right)

# Random Forest Class
class RandomForest:
    def __init__(self, n_trees=100, max_depth=10, min_samples_split=2, n_features=None):
        self.n_trees, self.max_depth, self.min_samples_split, self.n_features = n_trees, max_depth, min_samples_split, n_features
        self.trees = []
    
    def fit(self, X, y):
        self.trees = [DecisionTree(self.max_depth, self.min_samples_split, self.n_features) for _ in range(self.n_trees)]
        for tree in self.trees:
            X_sample, y_sample = bootstrap_sample(X, y)
            tree.fit(X_sample, y_sample)
    
    def predict(self, X):
        return np.round(np.mean([tree.predict(X) for tree in self.trees], axis=0)).astype(int)

# Train model
rf = RandomForest(n_trees=100, max_depth=10, min_samples_split=2, n_features=int(np.sqrt(X.shape[1])))
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Streamlit UI
st.title("🌳 Random Forest Classifier")
st.write("Enter the feature values below to classify the sample.")

# User input
feature_names = df.drop(columns=["Label"]).columns
user_input = [st.number_input(f"{feature}", min_value=0.0, step=0.1) for feature in feature_names]
user_data = np.array(user_input).reshape(1, -1)

if st.button("Predict Class"):
    prediction = rf.predict(user_data)
    st.subheader(f"Prediction: {'Class 1' if prediction[0] == 1 else 'Class 0'}")

st.write(f"### Model Accuracy: {accuracy * 100:.2f}%")
st.write("### Confusion Matrix")
fig, ax = plt.subplots()
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
st.pyplot(fig)
