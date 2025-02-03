import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# Load dataset
data = pd.read_csv('src/cleaned_data/cleaned_data.csv')

# Features and labels
X = data.drop('Label', axis=1).values
y = data['Label'].values

# Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Add bias term (column of ones)
X_train = np.c_[np.ones((X_train.shape[0], 1)), X_train]
X_test = np.c_[np.ones((X_test.shape[0], 1)), X_test]

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Cost function
def compute_cost(X, y, theta):
    m = len(y)
    h = sigmoid(X @ theta)
    cost = (1/m) * (-y.T @ np.log(h) - (1 - y).T @ np.log(1 - h))
    return cost

# Gradient Descent
def gradient_descent(X, y, theta, learning_rate, num_iterations):
    m = len(y)
    cost_history = []

    for _ in range(num_iterations):
        h = sigmoid(X @ theta)
        gradient = (1/m) * (X.T @ (h - y))
        theta -= learning_rate * gradient
        cost_history.append(compute_cost(X, y, theta))

    return theta, cost_history

# Train model
theta = np.zeros(X_train.shape[1])
learning_rate = 0.01
num_iterations = 1000
theta, cost_history = gradient_descent(X_train, y_train, theta, learning_rate, num_iterations)

# Prediction function
def predict(X, theta):
    return (sigmoid(X @ theta) >= 0.5).astype(int)

# Evaluate model
y_pred = predict(X_test, theta)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Streamlit UI
st.title("🚰 Water Quality Classification (Logistic Regression)")
st.write("Enter the water quality parameters below to predict water quality.")

# Get feature names
feature_names = data.drop(columns=["Label"]).columns

# User input fields
user_input = []
for feature in feature_names:
    value = st.number_input(f"{feature}", min_value=0.0, step=0.1)
    user_input.append(value)

# Convert input to numpy array and scale
user_data = np.array(user_input).reshape(1, -1)
user_data = scaler.transform(user_data)  # Standardize
user_data = np.c_[np.ones((user_data.shape[0], 1)), user_data]  # Add bias term

# Predict user input
if st.button("Predict Water Quality"):
    prediction = predict(user_data, theta)
    class_label = "Safe" if prediction[0] == 1 else "Unsafe"
    st.subheader(f"Prediction: {class_label}")

# Display model accuracy and confusion matrix
st.write(f"### Model Accuracy: {accuracy * 100:.2f}%")
st.write("### Confusion Matrix")
fig, ax = plt.subplots()
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
st.pyplot(fig)
