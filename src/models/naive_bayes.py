#math
import math

# Defined a class called naiveBayes

class naiveBayes:

  # constructor: This sets initial values
  def __init__(self):

    # P(c) class prior
    self.priors = {}

    #P(x|c) likelihood
    self.conditionals = {}

    # categorical classes
    self.classes = []


  # train the model
  def train(self, X_train, y_train):

    # this tells you the length of X_train data
    # example in Digitdata : it will get the length of trainingimages
    num_samples = len(X_train)

    # this tells you the features in each data point
    # example in Digitdata: this is the features comeing from trainingimages (pixels)
    num_features = len(X_train[0])

    # this gets a list of all the unqiue categorical classes from your
    # training lables (classes)
    # example in Digitdata: this gives your all the possible numbers in trainiglabels
    self.classes = list(set(y_train))

    # Step 1. Calculate Priors: P(c) = (count of class c) / (total samples)
    # for each label in labels (class) this was saved in an array
    # example in Digitdata: This cacluclates by counting(c) and dividing it by
    # the trainingimage length.
    for c in self.classes:
        self.priors[c] = y_train.count(c) / num_samples

        # Initialize conditional storage for this class
        # We track how many times feature i is "1" for class c
        # eample in Digit: comes from train images
        self.conditionals[c] = [0] * num_features

    # Step 2. Calculate Conditionals: P(x_i | c)
    # Count occurrences of feature = 1 for each class
    for i in range(num_samples):
        current_class = y_train[i]
        for j in range(num_features):
            if X_train[i][j] == 1:
                self.conditionals[current_class][j] += 1

    # Convert counts to probabilities with Laplace Smoothing (+1)
    # This prevents multiplying by zero if a feature never appears
    # for each in my unique labels
    # get how many times that label appears 
    # store it in class_count
    for c in self.classes:
        class_count = y_train.count(c)

        # gets the propability and does a Laplace soomthing
        for j in range(num_features):
            # P(x_i=1 | c) = (count + 1) / (total_class_samples + 2)
            count = self.conditionals[c][j]
            self.conditionals[c][j] = (count + 1) / (class_count + 2)
    



  # predicts the model
  # images that we will test
  def predict(self, X_test):
  
    # empty arrary 
    predictions = []

    # assgined best_class = None 
    # for each image
    for image in X_test:
        best_class = None

        # getting the most negative infinity
        # becuase zeros matter so we set the most negative probiblity 
        max_log_prob = -float('inf')

        for c in self.classes:

            # Use Log probabilities to avoid floating point underflow
            # Log(P(c) * P(x1|c) * P(x2|c)...) = Log P(c) + Log P(x1|c) + ...
            # this is for the computer to handel to calculate small numbers
            current_log_prob = math.log(self.priors[c])
            
            # for each pixel
            for j in range(len(image)):
                prob_feat_is_one = self.conditionals[c][j]
                if image[j] == 1:
                    current_log_prob += math.log(prob_feat_is_one)
                else:
                    current_log_prob += math.log(1 - prob_feat_is_one)
            
            if current_log_prob > max_log_prob:
                max_log_prob = current_log_prob
                best_class = c
        
        predictions.append(best_class)
    return predictions
  
def main():

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

    # calling my function 
  model = naiveBayes()

  # training the model
  
'''
    print("Training the model: ")
    model.train(X, y)

    print("\nPredictions: ")
    predictions = model.predict(X)

    for i in range(len(X)):
        print(f"Input: {X[i]} , Predicted: {predictions[i]}, Actual: {y[i]}")
'''
      
main()