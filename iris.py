import streamlit as st
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Train a RandomForest Classifier
clf = RandomForestClassifier()
clf.fit(X, y)

# Define a function to take user input for the Iris flower features
def user_input_features():
    sepal_length = st.sidebar.slider('Sepal length (cm)', float(X[:,0].min()), float(X[:,0].max()), float(X[:,0].mean()))
    sepal_width = st.sidebar.slider('Sepal width (cm)', float(X[:,1].min()), float(X[:,1].max()), float(X[:,1].mean()))
    petal_length = st.sidebar.slider('Petal length (cm)', float(X[:,2].min()), float(X[:,2].max()), float(X[:,2].mean()))
    petal_width = st.sidebar.slider('Petal width (cm)', float(X[:,3].min()), float(X[:,3].max()), float(X[:,3].mean()))
    data = {'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width}
    features = pd.DataFrame(data, index=[0])
    return features

# Add a title and description
st.title('Iris Flower Classifier')
st.write("""
    This app classifies Iris flowers based on user input features. 
    Adjust the sliders to input the sepal and petal dimensions, and see the classification result.
""")

# Get user input
df = user_input_features()

# Display the user input
st.subheader('User Input Features')
st.write(df)

# Predict the class of the input features
prediction = clf.predict(df)
prediction_proba = clf.predict_proba(df)

# Display the prediction and corresponding probability
st.subheader('Prediction')
st.write(f'The model predicts the flower is of type: **{iris.target_names[prediction][0]}**')

st.subheader('Prediction Probability')
st.write(pd.DataFrame(prediction_proba, columns=iris.target_names))

# Optional: Add feature importance visualization
import matplotlib.pyplot as plt

st.subheader('Feature Importance')
feature_names = iris.feature_names
importances = clf.feature_importances_
indices = importances.argsort()

plt.figure()
plt.title("Feature importances")
plt.barh(range(len(indices)), importances[indices], align="center")
plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
plt.xlabel("Feature importance")
st.pyplot(plt)
