# CS4346 Image Classification Project

## How to Run the Project

Clone the project, navigate to the root directory (the folder that contains the `src/` directory), then run:
```
python -m src.main
```

### Example

```
cd CS4346-Image-Classification

python -m src.main
```
## What the Program Does

When you run the command, the program will:

- Load digit and face datasets
- Extract features using pixel and grid representations
- Train two models:
  - Perceptron
  - Naive Bayes
- Evaluate performance across different training sizes (10% → 100%)
- Print accuracy results and runtime for each experiment

## Notes

- Run the command from the project root directory (not inside `src/`)
- Make sure Python is installed and available in your terminal
- Execution may take some time depending on dataset size and number of runs
