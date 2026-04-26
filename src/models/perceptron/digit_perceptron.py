"""
ONE-VS-ALL DIGIT PERCEPTRONS:

Each perceptron is trained to detect a single digit (0–9).
It outputs:
    1 - if image matches TARGET_DIGIT
    0 - if image is any other digit

Each class follows:
    __init__: initialize weights, bias, learning rate (alpha), target digit
    train: update weights using perceptron learning rule
    predict: calculate weighted sum and classify as 0 or 1

"""

import random

class Perceptron_0:

    def __init__(self):
        self.weights = []
        self.bias = 0
        self.alpha = 0.1
        self.TARGET_DIGIT = 0

    def train(self, X, y): 

        # initializes weights, random floats
        if len(self.weights) == 0:
            for i in range(len(X[0])):
                self.weights.append(random.random())
            print(f"\nWeights: {self.weights}")

        for i in range(len(X)):

            x = X[i]    # one image, a subset of X (feature vector)
            actual = y[i]
            prediction = self.predict(x)

            # converts ground truth label into binary classifier (y_binary)
            if actual == self.TARGET_DIGIT:
                y_binary = 1
            else:
                y_binary = 0

            # based on how (more below) the prediction was off, weights are adjusted towards ground truth

            if prediction != y_binary:
                for j in range(len(X[0])):
                    if y_binary == 1:   # prediction was NOT 0, but it was a 0
                        self.weights[j] += self.alpha * x[j]    # adjust weights towards positive classifier (since it was FN)
                    else:   # prediction was 0, but it was NOT 0
                        self.weights[j] -= self.alpha * x[j]    # adjust weights towards negative classifier (since it was FP)

                # shifts boundary toward positive/negative class
                if y_binary == 1:
                    self.bias += self.alpha
                else:
                    self.bias -= self.alpha
            

    def predict(self, x):
        score = 0

        # finding weighted sum (score)
        for j in range(len(x)):
            score += self.weights[j] * x[j]
        score += self.bias

        # prediction -> 1 = image is a 0 (target_digit), 0 = image is NOT a 0
        if score >= 0:
            score = 1
        else:
            score = 0

        return score

    # gets only score/confidence for DigitPerceptron
    def get_score(self, x):
        score = 0
        for j in range(len(x)):
            score += self.weights[j] * x[j]
        score += self.bias
        return score


class Perceptron_1:

    def __init__(self):
        self.weights = []
        self.bias = 0
        self.alpha = 0.1
        self.TARGET_DIGIT = 1

    def train(self, X, y):
        if len(self.weights) == 0:
            for i in range(len(X[0])):
                self.weights.append(random.random())
            print(f"\nWeights: {self.weights}")

        for i in range(len(X)):
            x = X[i]
            actual = y[i]
            prediction = self.predict(x)

            if actual == self.TARGET_DIGIT:
                y_binary = 1
            else:
                y_binary = 0

            if prediction != y_binary:
                for j in range(len(X[0])):
                    if y_binary == 1:
                        self.weights[j] += self.alpha * x[j]
                    else:
                        self.weights[j] -= self.alpha * x[j]

                if y_binary == 1:
                    self.bias += self.alpha
                else:
                    self.bias -= self.alpha



    def predict(self, x):
        score = 0
        for j in range(len(x)):
            score += self.weights[j] * x[j]
        score += self.bias

        if score >= 0:
            score = 1
        else:
            score = 0

        return score

    # gets only score/confidence for DigitPerceptron
    
    def get_score(self, x):
        score = 0
        for j in range(len(x)):
            score += self.weights[j] * x[j]
        score += self.bias
        return score


class Perceptron_2:

    def __init__(self):
        self.weights = []
        self.bias = 0
        self.alpha = 0.1
        self.TARGET_DIGIT = 2

    def train(self, X, y):
        if len(self.weights) == 0:
            for i in range(len(X[0])):
                self.weights.append(random.random())
            print(f"\nWeights: {self.weights}")

        for i in range(len(X)):
            x = X[i]
            actual = y[i]
            prediction = self.predict(x)

            if actual == self.TARGET_DIGIT:
                y_binary = 1
            else:
                y_binary = 0

            if prediction != y_binary:
                for j in range(len(X[0])):
                    if y_binary == 1:
                        self.weights[j] += self.alpha * x[j]
                    else:
                        self.weights[j] -= self.alpha * x[j]

                if y_binary == 1:
                    self.bias += self.alpha
                else:
                    self.bias -= self.alpha


    def predict(self, x):
        score = 0
        for j in range(len(x)):
            score += self.weights[j] * x[j]
        score += self.bias

        if score >= 0:
            score = 1
        else:
            score = 0

        return score

    # gets only score/confidence for DigitPerceptron
    
    def get_score(self, x):
        score = 0
        for j in range(len(x)):
            score += self.weights[j] * x[j]
        score += self.bias
        return score


class Perceptron_3:

    def __init__(self):
        self.weights = []
        self.bias = 0
        self.alpha = 0.1
        self.TARGET_DIGIT = 3

    def train(self, X, y):
        if len(self.weights) == 0:
            for i in range(len(X[0])):
                self.weights.append(random.random())
            print(f"\nWeights: {self.weights}")

        for i in range(len(X)):
            x = X[i]
            actual = y[i]
            prediction = self.predict(x)

            if actual == self.TARGET_DIGIT:
                y_binary = 1
            else:
                y_binary = 0

            if prediction != y_binary:
                for j in range(len(X[0])):
                    if y_binary == 1:
                        self.weights[j] += self.alpha * x[j]
                    else:
                        self.weights[j] -= self.alpha * x[j]

                if y_binary == 1:
                    self.bias += self.alpha
                else:
                    self.bias -= self.alpha



    def predict(self, x):
        score = 0
        for j in range(len(x)):
            score += self.weights[j] * x[j]
        score += self.bias

        if score >= 0:
            score = 1
        else:
            score = 0

        return score

    # gets only score/confidence for DigitPerceptron
    
    def get_score(self, x):
        score = 0
        for j in range(len(x)):
            score += self.weights[j] * x[j]
        score += self.bias
        return score

class Perceptron_4:

    def __init__(self):
        self.weights = []
        self.bias = 0
        self.alpha = 0.1
        self.TARGET_DIGIT = 4

    def train(self, X, y):
        if len(self.weights) == 0:
            for i in range(len(X[0])):
                self.weights.append(random.random())
            print(f"\nWeights: {self.weights}")

        for i in range(len(X)):
            x = X[i]
            actual = y[i]
            prediction = self.predict(x)

            if actual == self.TARGET_DIGIT:
                y_binary = 1
            else:
                y_binary = 0

            if prediction != y_binary:
                for j in range(len(X[0])):
                    if y_binary == 1:
                        self.weights[j] += self.alpha * x[j]
                    else:
                        self.weights[j] -= self.alpha * x[j]

                if y_binary == 1:
                    self.bias += self.alpha
                else:
                    self.bias -= self.alpha



    def predict(self, x):
        score = 0
        for j in range(len(x)):
            score += self.weights[j] * x[j]
        score += self.bias

        if score >= 0:
            score = 1
        else:
            score = 0

        return score

    # gets only score/confidence for DigitPerceptron
    
    def get_score(self, x):
        score = 0
        for j in range(len(x)):
            score += self.weights[j] * x[j]
        score += self.bias
        return score


class Perceptron_5:

    def __init__(self):
        self.weights = []
        self.bias = 0
        self.alpha = 0.1
        self.TARGET_DIGIT = 5

    def train(self, X, y):
        if len(self.weights) == 0:
            for i in range(len(X[0])):
                self.weights.append(random.random())
            print(f"\nWeights: {self.weights}")

        for i in range(len(X)):
            x = X[i]
            actual = y[i]
            prediction = self.predict(x)

            if actual == self.TARGET_DIGIT:
                y_binary = 1
            else:
                y_binary = 0

            if prediction != y_binary:
                for j in range(len(X[0])):
                    if y_binary == 1:
                        self.weights[j] += self.alpha * x[j]
                    else:
                        self.weights[j] -= self.alpha * x[j]

                if y_binary == 1:
                    self.bias += self.alpha
                else:
                    self.bias -= self.alpha



    def predict(self, x):
        score = 0
        for j in range(len(x)):
            score += self.weights[j] * x[j]
        score += self.bias

        if score >= 0:
            score = 1
        else:
            score = 0

        return score
    
    # gets only score/confidence for DigitPerceptron
    
    def get_score(self, x):
        score = 0
        for j in range(len(x)):
            score += self.weights[j] * x[j]
        score += self.bias
        return score


class Perceptron_6:

    def __init__(self):
        self.weights = []
        self.bias = 0
        self.alpha = 0.1
        self.TARGET_DIGIT = 6

    def train(self, X, y):
        if len(self.weights) == 0:
            for i in range(len(X[0])):
                self.weights.append(random.random())
            print(f"\nWeights: {self.weights}")

        for i in range(len(X)):
            x = X[i]
            actual = y[i]
            prediction = self.predict(x)

            if actual == self.TARGET_DIGIT:
                y_binary = 1
            else:
                y_binary = 0

            if prediction != y_binary:
                for j in range(len(X[0])):
                    if y_binary == 1:
                        self.weights[j] += self.alpha * x[j]
                    else:
                        self.weights[j] -= self.alpha * x[j]

                if y_binary == 1:
                    self.bias += self.alpha
                else:
                    self.bias -= self.alpha



    def predict(self, x):
        score = 0
        for j in range(len(x)):
            score += self.weights[j] * x[j]
        score += self.bias

        if score >= 0:
            score = 1
        else:
            score = 0

        return score

    # gets only score/confidence for DigitPerceptron
    
    def get_score(self, x):
        score = 0
        for j in range(len(x)):
            score += self.weights[j] * x[j]
        score += self.bias
        return score


class Perceptron_7:

    def __init__(self):
        self.weights = []
        self.bias = 0
        self.alpha = 0.1
        self.TARGET_DIGIT = 7

    def train(self, X, y):
        if len(self.weights) == 0:
            for i in range(len(X[0])):
                self.weights.append(random.random())
            print(f"\nWeights: {self.weights}")

        for i in range(len(X)):
            x = X[i]
            actual = y[i]
            prediction = self.predict(x)

            if actual == self.TARGET_DIGIT:
                y_binary = 1
            else:
                y_binary = 0

            if prediction != y_binary:
                for j in range(len(X[0])):
                    if y_binary == 1:
                        self.weights[j] += self.alpha * x[j]
                    else:
                        self.weights[j] -= self.alpha * x[j]

                if y_binary == 1:
                    self.bias += self.alpha
                else:
                    self.bias -= self.alpha


    def predict(self, x):
        score = 0
        for j in range(len(x)):
            score += self.weights[j] * x[j]
        score += self.bias

        if score >= 0:
            score = 1
        else:
            score = 0

        return score

    # gets only score/confidence for DigitPerceptron
    
    def get_score(self, x):
        score = 0
        for j in range(len(x)):
            score += self.weights[j] * x[j]
        score += self.bias
        return score


class Perceptron_8:

    def __init__(self):
        self.weights = []
        self.bias = 0
        self.alpha = 0.1
        self.TARGET_DIGIT = 8

    def train(self, X, y):
        if len(self.weights) == 0:
            for i in range(len(X[0])):
                self.weights.append(random.random())
            print(f"\nWeights: {self.weights}")

        for i in range(len(X)):
            x = X[i]
            actual = y[i]
            prediction = self.predict(x)

            if actual == self.TARGET_DIGIT:
                y_binary = 1
            else:
                y_binary = 0

            if prediction != y_binary:
                for j in range(len(X[0])):
                    if y_binary == 1:
                        self.weights[j] += self.alpha * x[j]
                    else:
                        self.weights[j] -= self.alpha * x[j]

                if y_binary == 1:
                    self.bias += self.alpha
                else:
                    self.bias -= self.alpha


    def predict(self, x):
        score = 0
        for j in range(len(x)):
            score += self.weights[j] * x[j]
        score += self.bias

        if score >= 0:
            score = 1
        else:
            score = 0

        return score

    # gets only score/confidence for DigitPerceptron
    
    def get_score(self, x):
        score = 0
        for j in range(len(x)):
            score += self.weights[j] * x[j]
        score += self.bias
        return score


class Perceptron_9:

    def __init__(self):
        self.weights = []
        self.bias = 0
        self.alpha = 0.1
        self.TARGET_DIGIT = 9

    def train(self, X, y):
        if len(self.weights) == 0:
            for i in range(len(X[0])):
                self.weights.append(random.random())
            print(f"\nWeights: {self.weights}")

        for i in range(len(X)):
            x = X[i]
            actual = y[i]
            prediction = self.predict(x)

            if actual == self.TARGET_DIGIT:
                y_binary = 1
            else:
                y_binary = 0

            if prediction != y_binary:
                for j in range(len(X[0])):
                    if y_binary == 1:
                        self.weights[j] += self.alpha * x[j]
                    else:
                        self.weights[j] -= self.alpha * x[j]

                if y_binary == 1:
                    self.bias += self.alpha
                else:
                    self.bias -= self.alpha

    def predict(self, x):
        score = 0
        for j in range(len(x)):
            score += self.weights[j] * x[j]
        score += self.bias

        if score >= 0:
            score = 1
        else:
            score = 0

        return score

    # gets only score/confidence for DigitPerceptron
    
    def get_score(self, x):
        score = 0
        for j in range(len(x)):
            score += self.weights[j] * x[j]
        score += self.bias
        return score

# Wrap all ten Perceptrons into one class 
# so we can easily use in RUNNER (runner.py)
class DigitPerceptron:
    def __init__(self):
        self.models = [
            Perceptron_0(), Perceptron_1(), Perceptron_2(),
            Perceptron_3(), Perceptron_4(), Perceptron_5(),
            Perceptron_6(), Perceptron_7(), Perceptron_8(),
            Perceptron_9()
        ]

    def train(self, X, y):
        for model in self.models:
            model.train(X, y)

    def predict(self, X):
        predictions = []
        """
        For one image, we compute the score across all models and record which model had the best score.

        With this wrapper, we are turning TEN individual classifiers into ONE classifier that picks the "best option."

            Best option: Which digit model is the MOST confident that this image is its digit?
            
        This returns predictions, a list that notes the predicted digit for each image (0-9)

        """
        for x in X:
            best_digit = 0
            best_score = float('-inf')

            # check all ten perceptrons for score
            for model in self.models:
                score = model.get_score(x)

                if score > best_score:
                    best_score = score
                    best_digit = model.TARGET_DIGIT

            predictions.append(best_digit)

        return predictions