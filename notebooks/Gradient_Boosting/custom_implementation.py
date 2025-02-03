import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import accuracy_score, confusion_matrix

# Load dataset
df = pd.read_csv('src/cleaned_data/cleaned_data.csv')

# Helper functions
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def compute_residuals(y_true, y_pred_log_odds):
    y_pred_prob = sigmoid(y_pred_log_odds)
    return y_true - y_pred_prob

# Gradient Boosting Classifier
class SimpleGradientBoostingClassifier:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.trees = []
        self.initial_log_odds = None

    def fit(self, X, y):
        self.initial_log_odds = np.log(np.mean(y) / (1 - np.mean(y))) if np.mean(y) != 0 else 0
        y_pred_log_odds = np.full_like(y, self.initial_log_odds, dtype=float)

        for _ in range(self.n_estimators):
            residuals = compute_residuals(y, y_pred_log_odds)
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, residuals)
            y_pred_log_odds += self.learning_rate * tree.predict(X)
            self.trees.append(tree)

    def predict_proba(self, X):
        y_pred_log_odds = np.full(X.shape[0], self.initial_log_odds, dtype=float)
        for tree in self.trees:
            y_pred_log_odds += self.learning_rate * tree.predict(X)
        return sigmoid(y_pred_log_odds)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)

# Train model
X = df.drop('Label', axis=1).values
y = df['Label'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

gb = SimpleGradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)
gb.fit(X_train, y_train)

# Streamlit UI
st.title("🚰 Water Quality Classification with Gradient Boosting")
st.write("Enter the water quality parameters below to predict water quality.")

# Get feature names
feature_names = df.drop(columns=["Label"]).columns

# Create user input fields
user_input = []
for feature in feature_names:
    value = st.number_input(f"{feature}", min_value=0.0, step=0.1)
    user_input.append(value)

# Convert input to numpy array
user_data = np.array(user_input).reshape(1, -1)

# Predict
if st.button("Predict Water Quality"):
    prediction = gb.predict(user_data)
    class_label = "Safe" if prediction[0] == 1 else "Unsafe"
    st.subheader(f"Prediction: {class_label}")

# Evaluate on test set
y_pred = gb.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

st.write(f"### Model Accuracy: {accuracy*100:.2f}%")
st.write("### Confusion Matrix")
fig, ax = plt.subplots()
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
st.pyplot(fig)
