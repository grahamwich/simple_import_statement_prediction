import spacy
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import tensorflow as tf
# from tensorflow.keras import layers, models # issue with current version of tensorflow
# from tensorflow.python.keras import layers, models
from keras.api._v2.keras import layers, models

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
print(problems[0]["text"])
print(problems[0]["code"])
exit(0)
# ================================================== #

# Extract features (word embeddings) from the text using spaCy
def extract_features(text):
    doc = nlp(text)
    return doc.vector

# Prepare data for training
X = np.array([extract_features(problem["text"]) for problem in problems])
y = np.array([1 if "import" in problem["code"] else 0 for problem in problems])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a simple neural network model
model = models.Sequential()
model.add(layers.Dense(64, activation="relu", input_shape=(X_train.shape[1],)))
model.add(layers.Dense(1, activation="sigmoid"))

# Compile the model
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

print("\nEvaluation:")
# Evaluate the model on the test set
_, accuracy = model.evaluate(X_test, y_test)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Now you can use the trained model to predict import statements based on new text portions
# new_text = "Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target."
# new_text = "Write a Python program to select a random element from a list, set, dictionary-value, and file from a directory."

new_text = "Write a function that will calculate the square root of square numbers."

new_text_features = extract_features(new_text)
new_text_features = np.reshape(new_text_features, (1, -1))  # Reshape to match model input shape
print("\nNew prediction on input problem:")
predicted_import_prob = model.predict(new_text_features)[0][0]

# Convert the probability to binary
predicted_import_binary = int(predicted_import_prob > 0.5)

if predicted_import_binary == 1:
    print("The model predicts that an import statement can be used for the problem: ")
    print(new_text)
else:
    print("The model predicts that no import statement is needed for the problem: ")
    print(new_text)
