import spacy
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load spaCy model
nlp = spacy.load("en_core_web_sm")


# ================================================== #
# create problems list from .jsonl file:
import json
import pandas as pd
problems = []
with open('problems.jsonl', 'r') as json_file:
    json_list = list(json_file)
for json_str in json_list:
    result = json.loads(json_str)
    problems.append(result)
# print(problems[0]["text"])
# print(problems[0]["code"])
# exit(0)
# ================================================== #


# ================================================== #
# Extract features (word embeddings) from the text using spaCy
def extract_features(text):
    doc = nlp(text)
    # print("Doc vector for text: " + text)
    # print(doc.vector)
    return doc.vector
# Prepare data for training
X = np.array([extract_features(problem["text"]) for problem in problems])
y = np.array([1 if "import" in problem["code"] else 0 for problem in problems])
# # print a couple of the X and y values:
# print(X)
# print(y)
# exit(0)
# ================================================== #


# ================================================== #
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Train a simple linear regression model
model = LinearRegression()
model.fit(X_train, y_train)
# ================================================== #


# Make predictions on the test set
y_pred = model.predict(X_test)
# Convert predictions to binary (1 if the predicted value is greater than 0.5, else 0)
y_pred_binary = (y_pred > 0.5).astype(int)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred_binary)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Now you can use the trained model to predict import statements based on new text portions
# new_text = "Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target."
new_text = "Write a Python program to select a random element from a list, set, dictionary-value, and file from a directory."
new_text_features = extract_features(new_text)
predicted_import = model.predict([new_text_features])[0]

# Convert the prediction to binary
predicted_import_binary = int(predicted_import > 0.5)

if predicted_import_binary == 1:
    print("The model predicts that an import statement can be used for the problem: ")
    print(new_text)
else:
    print("The model predicts that no import statement is needed for the problem: ")
    print(new_text)
