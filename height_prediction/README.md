# **Project 1:** Predicting Children’s Height with Linear Regression

## Linear Regression

In this project, we'll build our first machine-learning model using a technique called linear regression. We'll use it to predict a child's height based on their age. This is a simple yet powerful example that will introduce you to many fundamental concepts in machine learning.

Linear regression is a way to model the relationship between two variables (in our case, age and height) using a straight line. The idea is to find the line that best fits the data we have.

### The Equation of a Line

You might remember the equation of a line from school: y = mx + b. In our project, we'll use slightly different names for the variables, but the concept is the same:

```python
height = weight * age + bias
```

`height` = The value we want to predict (the dependent varialbe).
`age` = The input value we're using for the prediction (the independent variable).
`weight` = The slope of the line. It tells us how much heights is expected to increase for each one-unit increase in age (e.g., how many centimeters a child grows per year).
`bias` = This is the y-intercept. It represents the predicted height when age is zero.

### Finding the Best Line

Our goal is to find the values of weight and bias that create a line that best fits our data points (the actual ages and heights of children). This line will then allow us to predict the height of a child of any given age.

## Loading and Understanding the Data

Let's look at how we load and prepare our synthetic data for the model.

### `data.json`

Our data is stored in a file named data.json. This file is formatted as a JSON (JavaScript Object Notation) object, which is a common way to store data in a human-readable and easily parsable format. Here's what it looks like:

```json
{
  "data": [
    {
      "age": 2,
      "height": 85
    },
    {
      "age": 3,
      "height": 95
    },
    {
      "age": 4,
      "height": 103
    },
    {
      "age": 5,
      "height": 110
    },
    {
      "age": 6,
      "height": 118
    },
    {
      "age": 7,
      "height": 128
    },
    {
      "age": 8,
      "height": 135
    },
    {
      "age": 9,
      "height": 142
    },
    {
      "age": 10,
      "height": 150
    }
  ]
}
```

It's a dictionary with a single key, "data", which contains a list of dictionaries. Each dictionary in the list represents a child, with their "age" and "height" recorded.

### `load_data_from_json` Function

This function reads the data from data.json and converts it into JAX arrays that we can use in our model.

```python
import jax.numpy as jnp
import json

def load_data_from_json(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    ages = jnp.array([item["age"] for item in data["data"]], dtype=jnp.float32)
    heights = jnp.array([item["height"] for item in data["data"]], dtype=jnp.float32)
    return ages, heights
```

Let's break down what this code does:

1. `with open(filepath, 'r') as f` This opens the `data.json` file in read mode ('r'). The with statement ensures the file is properly closed even if errors occur.
2. `data = json.load(f)` This reads the JSON data from the file and loads it into a Python dictionary called data.
3. `ages = jnp.array([item["age"] for item in data["data"]], dtype=jnp.float32)` This line does the following:
   - `[item["age"] for item in data["data"]]` This is a list comprehension that extracts the "age" value from each dictionary in the data["data"] list.
   - `jnp.array(...)` This converts the list of ages into a JAX array.
   - `dtype=jnp.float32` This specifies that the numbers in the array should be stored as 32-bit floating-point numbers. This is a common practice in machine learning for efficiency.
4. `heights = jnp.array([item["height"] for item in data["data"]], dtype=jnp.float32)`:` This does the same as above but extracts the "height" values and creates a JAX array of heights.
5. `return ages, heights` The function returns the two JAX arrays, ages and heights.

## Initializing Parameters


<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />

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
