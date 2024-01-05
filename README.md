# simple_import_statement_prediction
 Module prediction for simple Python problems.

Introduction to word embeddings for NLP (via spaCy), uses the Mostly Basic Python Problems dataset. Also my probably novice level attempt at understanding the general methodology for using NN tools.
https://github.com/google-research/google-research/tree/master/mbpp

//// Example use case ////
"Write a python function to identify non-prime numbers."
The basic way to solve this is to take the unidentified number N and divide it by all the numbers up to N/2 and see if the result is a whole number. This takes O(n) time, but can be done in O(sqrt(n)) time by dividing numbers until sqrt(N) and flagging that a number is not prime (this works because a larger factor of N can be the result of a smaller factor of N that has already been checked). Common practice is to import the Python math module for calculating the square root.


//// Methodology ////
Module import prediction:

1. take in 974 problems
2. seperate into "text" and code "sections"

In order to predict what import statement is necessary, must associate the text to
import statements already within the code.

This will split into training data and testing data:

training = problems with an import statement associated with a text portion
    - may not have enough? must find problems where an import statement can be used

testing?? = problems without and import statement that could use one



take descriptions of modules to help with training?
problem example: text = "write a function to convert....
                 import: "math"

module descr ex: text = description = "the math module is used for..."
                 import: "math"
