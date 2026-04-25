"""
FACE DATA PERCEPTRON:

    Classifies whether an image contains a face or not.

    Output:
        1 - face detected
        0 - no face detected

    Perceptron learning rule is used with:
        train: updates weights when a mistake/misclassification is made
        predict: calculate weighted sum and classify it as 0 or 1

"""

import random

class Perceptron: 

    # sets up class weights for algorithm
    def __init__(self):
        self.weights = []
        self.bias = 0
        self.alpha = 0.1
    

    # predicts and updates weights if it does not equal ground truth (y)
    def train(self, X, y):

        # initializes weights, random floats
        if len(self.weights) == 0:
            for i in range(len(X[0])):
                self.weights.append(random.random())
            print(f"\nWeights: {self.weights}") # sanity check after intializing weights

        for i in range(len(X)):
            x = X[i]    # one image, a subset of X (feature vector)
            actual = y[i]
            prediction = self.predict(x)

            # based on how (more below) the prediction was off, weights are adjusted towards ground truth
            if prediction != actual:
                for j in range(len(X[0])):
                    if actual == 1: # prediction was NOT face, but it was a face
                        self.weights[j] += self.alpha * x[j]  # adjust weights towards positive classifier (since it was FN)
                    else:   # prediction was 0, but it was NOT 0
                        self.weights[j] -= self.alpha * x[j] # adjust weights towards negative classifier (since it was FP)

                # shifts boundary toward positive/negative class
                if actual == 1:
                    self.bias += self.alpha
                else:
                    self.bias -= self.alpha


    # computes a score and defines it as true or false
    def predict(self, x):
        score = 0
        for j in range(len(x)):
            score += self.weights[j] * x[j]
        score += self.bias #
        
        # prediction -> 1 = image is a face, 0 = image is NOT a face
        if score >= 0:
            score = 1   # face
        else:
            score = 0   # not face

        return score