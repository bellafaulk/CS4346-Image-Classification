from src.models.perceptron.digit_perceptron import DigitPerceptron

def main():
    model = DigitPerceptron()

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
    
    model.train(X, y)
    predictions = model.predict(X)

    print("\nPredictions:\n")

    for i in range(len(X)):
        print(f"Sample {i}: predicted {predictions[i]} | actual {y[i]}")
    
if __name__ == "__main__":
    main()