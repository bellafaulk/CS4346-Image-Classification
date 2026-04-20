# perceptron implementation: using simplified formula f(x) = wx + b
# trains a model using weight updates, used for digit and face classification

# model object with initialized weights and the Perceptron algorithm for reuse
class Perceptron: 

    # sets up class weights for algorithm
    def __init__(self):
        self.weights = [0, 0, 0, 0] # needs to be random intialization
        self.bias = 0
    

    # predicts and updates weights if it does not equal ground truth (y)
    def train(self, X, y):
        
        for i in range(len(X)):
            x = X[i]
            actual = y[i]
            prediction = self.predict(x)

            # adjust weights towards actual answer if the prediction is incorrect

            if prediction != actual:
                if actual == 1:
                    for j in range(len(x)):
                        self.weights[j] = self.weights[j] + x[j]
                    self.bias += 1
                else:
                    for j in range(len(x)):
                        self.weights[j] = self.weights[j] - x[j]
                    self.bias -= 1


    # computes a score and defines it as true or false
    def predict(self, x):
        score = 0
        for j in range(len(x)):
            score += self.weights[j] * x[j]
        score += self.bias
        
        if score >= 0:
            score = 1   # true
        else:
            score = 0   # false

        return score

    