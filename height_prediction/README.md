# Height Prediction with Linear Regression in JAX

## Overview

This project demonstrates a simple linear regression model built using JAX to predict a child's height based on their age. It serves as a beginner-friendly introduction to core machine learning concepts and JAX's capabilities.

## Project Structure

```
jaxml/
└── height_prediction/
    ├── main.py        # Python script containing the model, training logic, and data loading.
    └── data.json      # JSON file containing the age and height dataset.
```

- **`main.py`:** Contains the Python code that:
  - Loads the dataset from `data.json`.
  - Defines the linear regression model using JAX.
  - Implements the training loop using gradient descent.
  - Visualizes the results (actual vs. predicted heights).
- **`data.json`:** Stores the dataset in JSON format. Each data point is a dictionary with "age" and "height" keys:

## Running the Code

- Navigate to the `jaxml/height_prediction` directory in your terminal.
- Execute the `main.py` script: `python main.py`
- The script will print the loss during training at intervals of 100 epochs.
- After training, it will display a plot showing the actual vs. predicted heights.
- The trained weight (`w`) and bias (`b`) of the linear model will be printed.

## Concepts Illustrated

- **Linear Regression:** Building a simple linear model to predict a continuous target variable.
- **JAX:** Using JAX for numerical computation, automatic differentiation, and potential acceleration on GPUs/TPUs.
- **Gradient Descent:** Implementing a basic gradient descent algorithm to train the model.
- **Mean Squared Error (MSE):** Using MSE as the loss function to measure the model's performance.
- **Data Loading from JSON:** Reading data from a JSON file to use in the model.
- **Visualization:** Using Matplotlib to create a plot of the results.
