# loads data, extracts features, splits (10% -> 100%)
# runs multiple trials, compares models, and prints results
import time
import numpy as np

from src.data.parser import load_digit_data, load_face_data
from src.features.feature_extractor import extract_all

from src.models.perceptron.digit_perceptron import DigitPerceptron
from src.models.perceptron.face_perceptron import Perceptron as FacePerceptron
from src.models.naive_bayes import naiveBayes

from src.evaluation.metrics import accuracy as accuracy_func

# --------------------------------------------------
# STEP 1: Load dataset + convert to feature vectors
# --------------------------------------------------
def load_dataset(task="digit", mode="pixel"):

    # load raw images and labels
    if task == "digit":
        train_images, train_labels = load_digit_data("train")
        val_images, val_labels = load_digit_data("vali")
        test_images, test_labels = load_digit_data("test")
    else:
        train_images, train_labels = load_face_data("train")
        val_images, val_labels = load_face_data("vali")
        test_images, test_labels = load_face_data("test")

    # convert images to feature vectors (0/1)
    X_train = extract_all(train_images, mode=mode)
    X_val = extract_all(val_images, mode=mode)
    X_test = extract_all(test_images, mode=mode)

    # convert to numpy arrays to index easier

    return (
        X_train,
        train_labels,
        X_test,
        test_labels,
        X_val,
        val_labels
    )


# --------------------------------------------------
# STEP 2: Run ONE training/testing cycle
# --------------------------------------------------

def run_single_experiment(model_class, X_train, y_train, X_test, y_test, ratio):
    """
    model_class: Perceptron or Naive Bayes
    ratio: % training data to use (0.10 --> 1.0)

    """

    n = len(X_train)
    size = int(n * ratio)

    # pick a small subset using ratio (10%, 20%, ..., 100%)
    # indices is a random selection of training examples
    indices = np.random.choice(n, size=size, replace=False)

    # builds a random subset of images w matching labels
    # np extracts all rows and returns a new array
    X_sub = [X_train[i] for i in indices]
    y_sub = [y_train[i] for i in indices]

    # initialize model
    model = model_class()

    # train and time
    start = time.time()
    model.train(X_sub, y_sub)
    runtime = time.time() - start

    # predict on validation set for internal checking/debugging
    # val_predictions = model.predict(X_val) DEBUG, NOT FINAL EVAL
    # val_acc = accuracy_func(y_val, val_predictions) DEBUG, NOT FINAL EVAL

    # predict on full test set
    predictions = model.predict(X_test)

    # compute accuracy
    acc = accuracy_func(y_test, predictions)

    # return acc, runtime, val_acc DEBUG, NOT FINAL EVAL
    return acc, runtime


# --------------------------------------------------
# STEP 3: Run multiple trials per % (average results)
# --------------------------------------------------

def run_model_experiments(model_class, X_train, y_train, X_val, y_val, X_test, y_test, runs=5):
    """
    runs experiments for 10%, 20%, ..., 100% of the data
    repeats each percentage multiple times and averages the results
    
    """

    percentages = [i / 10 for i in range(1, 11)]

    print(f"\n==============================")
    print(f"MODEL: {model_class.__name__}")
    print(f"==============================")

    for p in percentages:
        accuracies = []
        # val_accuracies = [] DEBUG, NOT FINAL EVAL
        times = []

        # run multiple trials (reduces randomness)
        for _ in range(runs):
            acc, t = run_single_experiment( # add val_acc for DEBUG, NOT FINAL EVAL
                model_class,
                X_train, y_train,
                # X_val, y_val,
                X_test, y_test,
                p
            )
            accuracies.append(acc)
            # val_accuracies.append(val_acc) DEBUG, NOT FINAL EVAL
            times.append(t)

        print(
            f"{int(p*100)}% - "
            f"Accuracy: {np.mean(accuracies):.4f} +/- {np.std(accuracies):.4f} | "
            # f"Val Acc (debug): {np.mean(val_accuracies):.4f} +/- {np.std(val_accuracies):.4f} | " DEBUG, NOT FINAL EVAL
            f"Time: {np.mean(times):.4f}s"
        )


# --------------------------------------------------
# STEP 4: Run EVERYTHING (digits + faces)
# --------------------------------------------------
def run_all():
    """
    this is our main area for the project!
    runs both of the datasets (digit/face) and both models (Perceptron and Naive Bayes)

    """

    for task in ["digit", "face"]:
        for mode in ["pixel", "grid"]:
            print(f"\nLOADING {task.upper()} with {mode.upper()} features...")

            X_train, y_train, X_val, y_val, X_test, y_test = load_dataset(task, mode=mode)

            print(f"\n=== {task.upper()} ({mode}) ===")

            if task == "digit":
                perceptron_model = DigitPerceptron
            else:
                perceptron_model = FacePerceptron

            run_model_experiments(perceptron_model, X_train, y_train, X_val, y_val, X_test, y_test)
            run_model_experiments(naiveBayes, X_train, y_train, X_val, y_val, X_test, y_test)