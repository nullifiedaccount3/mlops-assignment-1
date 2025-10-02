# Boston Housing Price Prediction

This project implements a complete machine learning workflow to predict house prices using classical machine learning models on the Boston Housing dataset.

## Project Structure

```
.
├── README.md           # This file
├── requirements.txt    # Python dependencies
├── misc.py            # Utility functions for ML workflow
├── train.py           # DecisionTreeRegressor model
└── train2.py          # KernelRidge model
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Setting up the Environment

1. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   ```

2. **Activate the virtual environment:**
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Running the Code

### DecisionTreeRegressor Model

To train and evaluate the DecisionTreeRegressor model:

```bash
python train.py
```

This will:
- Load the Boston Housing dataset
- Preprocess the data (split into train/test sets)
- Train a DecisionTreeRegressor model
- Evaluate and display the Mean Squared Error (MSE) on the test set

### KernelRidge Model

To train and evaluate the KernelRidge model:

```bash
python train2.py
```

This will:
- Load the Boston Housing dataset
- Preprocess and scale the data
- Train a KernelRidge model
- Evaluate and display the Mean Squared Error (MSE) on the test set

## Dataset Information

The Boston Housing dataset contains information collected by the U.S Census Service concerning housing in the area of Boston, Massachusetts. It includes 506 samples with 13 features:

- **CRIM**: Per capita crime rate by town
- **ZN**: Proportion of residential land zoned for lots over 25,000 sq.ft.
- **INDUS**: Proportion of non-retail business acres per town
- **CHAS**: Charles River dummy variable (1 if tract bounds river; 0 otherwise)
- **NOX**: Nitric oxides concentration (parts per 10 million)
- **RM**: Average number of rooms per dwelling
- **AGE**: Proportion of owner-occupied units built prior to 1940
- **DIS**: Weighted distances to five Boston employment centres
- **RAD**: Index of accessibility to radial highways
- **TAX**: Full-value property-tax rate per $10,000
- **PTRATIO**: Pupil-teacher ratio by town
- **B**: 1000(Bk - 0.63)^2 where Bk is the proportion of Black residents by town
- **LSTAT**: % lower status of the population

**Target Variable:**
- **MEDV**: Median value of owner-occupied homes in $1000's

## Models

### 1. DecisionTreeRegressor
A decision tree regression model that doesn't require feature scaling.

### 2. KernelRidge
A kernel ridge regression model that benefits from feature scaling.

## Project Workflow

The project follows a modular approach with utility functions in `misc.py`:

- `load_data()`: Loads the Boston Housing dataset
- `preprocess_data()`: Splits data into train/test sets and optionally scales features
- `train_model()`: Trains a given model
- `evaluate_model()`: Evaluates the model and calculates MSE
- `display_results()`: Displays evaluation results

## Results

The models are evaluated using Mean Squared Error (MSE) on the test set. Lower MSE values indicate better model performance.

## Notes

- The dataset is loaded directly from the original source at http://lib.stat.cmu.edu/datasets/boston
- Random state is set to 42 for reproducibility
- Test set size is 20% of the total dataset

## Author

Bhargav Nanekalva (G24AI2066@iitj.ac.in)

MLOps Assignment 1 - House Price Prediction
