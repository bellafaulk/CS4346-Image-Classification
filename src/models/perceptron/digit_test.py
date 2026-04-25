from digit_perceptron import (
    Perceptron_0, Perceptron_1, Perceptron_2, Perceptron_3, Perceptron_4, 
    Perceptron_5, Perceptron_6, Perceptron_7, Perceptron_8, Perceptron_9
)

import random

def main():
    # defining dummy pixel feature data (X) and labels (y)
    # one image: 8x8, 1 = "+ or #", 0 = whitespace
    X = [
        [0, 0, 1, 1, 1, 1, 0, 0,
         0, 1, 0, 0, 0, 1, 0, 0,
         0, 1, 0, 0, 0, 1, 0, 0,
         0, 1, 0, 0, 0, 1, 0, 0,
         0, 1, 0, 0, 0, 1, 0, 0,
         0, 1, 0, 0, 0, 1, 0, 0,
         0, 0, 1, 1, 1, 1, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0],

        [0, 0, 0, 1, 1, 0, 0, 0,
         0, 0, 1, 1, 0, 0, 0, 0,
         0, 1, 0, 1, 0, 0, 0, 0,
         0, 0, 0, 1, 0, 0, 0, 0,
         0, 0, 0, 1, 0, 0, 0, 0,
         0, 0, 0, 1, 0, 0, 0, 0,
         0, 1, 1, 1, 1, 1, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0],

        [0, 1, 1, 1, 1, 0, 0, 0,
         1, 0, 0, 0, 1, 0, 0, 0,
         0, 0, 0, 0, 1, 0, 0, 0,
         0, 0, 1, 1, 0, 0, 0, 0,
         0, 1, 0, 0, 0, 0, 0, 0,
         1, 0, 0, 0, 0, 0, 0, 0,
         1, 1, 1, 1, 1, 1, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0],
        
        [0, 1, 1, 1, 1, 0, 0, 0,
         1, 0, 0, 0, 0, 1, 0, 0,
         0, 0, 0, 0, 0, 1, 0, 0,
         0, 0, 1, 1, 1, 0, 0, 0,
         0, 0, 0, 0, 0, 1, 0, 0,
         1, 0, 0, 0, 0, 1, 0, 0,
         0, 1, 1, 1, 1, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0],
    ]

    # ground truth labels
    y = [0, 1, 2, 3]

    # each model here looks at the sample image independently
    models = [
        Perceptron_0(), # "is this image a 0?"
        Perceptron_1(),
        Perceptron_2(),
        Perceptron_3(),
        Perceptron_4(),
        Perceptron_5(),
        Perceptron_6(),
        Perceptron_7(),
        Perceptron_8(),
        Perceptron_9()
    ]

    epochs = 10
    
    # each epoch is a "pass through" of the entire dataset
    for epoch in range(epochs):
        print(f"\nEpoch: {epoch + 1} out of {epochs}")
        
        # each model sees and learns from entire dataset
        for model in models:
            model.train(X, y)
        
        print("\nTEST RESULTS:\n")

        for i in range(len(X)):
            x = X[i]

            predictions = []

            # creates a vector of yes/no decisions from each model
            # each model is basically asking "is this image the digit i care about? (0-9)"
            for digit, model in enumerate(models):
                pred = model.predict(x)
                predictions.append((model.TARGET_DIGIT, model.predict(x)))


            # sanity checks
            print(f"\nSample {i}")
            print(f"Actual: {y[i]}")

            print("\nFORMAT: Digit model → prediction (1 = match, 0 = not match)\n")
            for digit, pred in predictions:
                print(f"Digit {digit} classifier: {pred}")
    
if __name__ == "__main__":
    main()