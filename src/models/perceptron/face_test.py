from face_perceptron import Perceptron
import random

def main():
    # defining dummy feature data (X) and labels (y)
    # one image (8x8, 1 = "+ or #", 0 = whitespace)
    X = [ # raw pixel feature
        [0, 0, 0, 0, 0, 0, 0, 0,
         0, 1, 1, 1, 1, 1, 1, 0,
         0, 1, 1, 1, 1, 1, 1, 0,
         0, 0, 0, 1, 1, 1, 1, 0,
         0, 1, 1, 1, 1, 1, 1, 0,
         0, 1, 1, 0, 0, 1, 1, 1, 
         1, 1, 1, 0, 0, 1, 1, 0, 
         0, 0, 0, 0, 0, 0, 0, 0],

        [0, 0, 0, 0, 0, 0, 0, 0,
         0, 1, 1, 0, 0, 1, 1, 0, 
         0, 1, 1, 0, 0, 1, 1, 0, 
         0, 0, 0, 1, 1, 0, 0, 0, 
         0, 1, 1, 1, 1, 1, 1, 0, 
         0, 1, 0, 0, 0, 0, 1, 0, 
         0, 0, 1, 1, 1, 1, 0, 0, 
         0, 0, 0, 0, 0, 0, 0, 0],

        [1, 0, 0, 0, 0, 0, 0, 1, 
         0, 1, 0, 0, 0, 0, 1, 0,
         0, 0, 1, 0, 0, 1, 0, 0,
         0, 0, 0, 1, 1, 0, 0, 0,
         0, 0, 0, 1, 1, 0, 0, 0,
         0, 0, 1, 0, 0, 1, 0, 0,
         0, 1, 0, 0, 0, 0, 1, 0,
         1, 0, 0, 0, 0, 0, 0, 1],
        
        [0, 1, 0, 1, 0, 1, 0, 1,
         1, 0, 1, 0, 1, 0, 1, 0,
         0, 1, 0, 1, 0, 1, 0, 1,
         1, 0, 1, 0, 1, 0, 1, 0,
         0, 1, 0, 1, 0, 1, 0, 1,
         1, 0, 1, 0, 1, 0, 1, 0,
         0, 1, 0, 1, 0, 1, 0, 1,
         1, 0, 1, 0, 1, 0, 1, 0],
    ]

    # ground truth labels - 1 = face, 0 = not face
    y = [1, 1, 0, 0]

    model = Perceptron()
    epochs = 10
    
    # each epoch is a "pass through" of the entire dataset
    for epoch in range(epochs):
        print("training model...")
        print(f"\nEpoch: {epoch + 1} out of {epochs}")
        model.train(X, y)
        predictions = []

        for i in range(len(X)):
            x = X[i]
            predictions.append(model.predict(x))

            # sanity checks
            print(f"\npredictions: {predictions}")
            print(f"\nInput: {X[i]}, Predicted: {predictions[i]}, Actual: {y[i]}")
            print(f"\nWeights: {model.weights.copy()}")

        
    
if __name__ == "__main__":
    main()