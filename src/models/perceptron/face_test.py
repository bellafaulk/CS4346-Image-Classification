from src.models.perceptron.face_perceptron import Perceptron

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
    
    predictions = model.predict(X)

    print("PREDICTIONS:")
    for i in range(len(X)):
        print(f"Sample {i}: predicted {predictions[i]} | actual {y[i]}")

        
if __name__ == "__main__":
    main()