from face_perceptron import Perceptron
from src.data.parser import load_digit_data, load_face_data
from src.feature.feature_extractor import extract_all

import random

def main():
    model = Perceptron()

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
    
    model.train(X, y)
    predictions = []

        for i in range(len(X)):
            x = X[i]
            predictions.append(model.predict(x))

            # sanity check
            print(model.predict(X[0]))

        
if __name__ == "__main__":
    main()