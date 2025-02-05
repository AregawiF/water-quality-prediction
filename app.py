import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv('src/cleaned_data/cleaned_data.csv')


# Separate features (X) and target (y)
X = df.drop('Label', axis=1)
y = df['Label']

# Split into training and test sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42, 
    stratify=y  # Preserve class balance
)

# Initialize and train the model
def train_model(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=100,   # Number of trees
        max_depth=10,       # Control tree depth to prevent overfitting
        min_samples_split=5,  # Minimum samples to split a node
        class_weight='balanced',  # Adjust for class imbalance
        random_state=42     # Reproducibility
    )
    model.fit(X_train, y_train)
    return model

model = train_model(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)



# Streamlit UI
st.title("🚰 Random Forest Water Quality Classification")
st.write("Enter the water quality parameters below to predict water quality.")


# Get feature names
feature_names = X.columns

# User input fields
user_input = []
for feature in feature_names:
    value = st.number_input(f"{feature}", min_value=0.0, step=0.1)
    user_input.append(value)

# Convert input to numpy array
user_data = np.array(user_input).reshape(1, -1)

# Predict user input
if st.button("Predict Water Quality"):
    prediction = model.predict(user_data)
    class_label = "Safe" if prediction[0] == 1 else "Unsafe"
    st.subheader(f"Prediction: {class_label}")

