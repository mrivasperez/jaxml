# Spam Detection with Logistic Regression in JAX

## Overview

This project implements a simple spam detection model using logistic regression in JAX. It demonstrates how to classify emails as spam (1) or not spam (0) based on the presence of "spammy" keywords. This project serves as an introduction to binary classification and some core concepts in JAX.

## Project Structure

```
jaxml/
└── spam_detection/
    ├── main.py                # Python script with the model, training, and classification logic.
    ├── emails.json            # JSON file containing the training email dataset.
    ├── new_emails.json        # JSON file containing new emails to be classified.
    └── spammy_keywords.json   # JSON file containing a list of spammy keywords.
```

- **`main.py`:**
  - Loads training data from `emails.json`.
  - Loads spammy keywords from `spammy_keywords.json`.
  - Defines the logistic regression model, including the sigmoid function, prediction, loss function, and gradient update.
  - Trains the model using gradient descent.
  - Evaluates the model's accuracy on the training data.
  - Loads new emails from `new_emails.json` and classifies them as spam or not spam.
  - Optionally visualizes the results.
- **`emails.json`:**
  - Contains the training dataset of emails.
  - Each email is a dictionary with "text" (the email content) and "is_spam" (0 or 1) keys.

```json
{
  "emails": [
    { "text": "Normal email example.", "is_spam": 0 },
    { "text": "You've won a free vacation! Click here!", "is_spam": 1 }
  ]
}
```

- **`new_emails.json`:**
  - Contains a list of new emails to be classified.
  - Each email is a dictionary with a "text" key.

```json
{
  "emails": [
    { "text": "Another test email for the spam filter." },
    { "text": "Claim your lottery prize money now!" }
  ]
}
```

- **`spammy_keywords.json`:**
  - Contains a list of keywords considered indicative of spam.

```json
{
  "keywords": [
    "free",
    "money",
    "urgent",
    "win",
    "guaranteed",
    "click here",
    "discount",
    "offer"
  ]
}
```

## Running the Code

1. **Navigate:** Go to the `jaxml/spam_detection` directory in your terminal.
2. **Execute:** Run the script using: `python main.py`

## Output

The script will:

1. **Train:** Print the loss during training every 100 epochs.
2. **Evaluate:** Print the accuracy of the trained model on the training data.
3. **Classify:** Load emails from `new_emails.json`, classify them, and print the predictions:

```
Email: This is a test email to check the spam filter.
Prediction: Not Spam
--------------------
Email: You have won a lottery! Click here to claim your prize money now!
Prediction: Spam
--------------------
```

4. **Visualize (Optional):** Display a plot showing the decision boundary and how the model classifies the training data points.

## Concepts Illustrated

- **Logistic Regression:** Building a logistic regression model for binary classification.
- **JAX:** Using JAX for numerical operations, automatic differentiation (`jax.grad`), and array manipulation (`jax.numpy`).
- **Gradient Descent:** Implementing gradient descent to train the model.
- **Binary Cross-Entropy:** Using binary cross-entropy as the loss function.
- **Sigmoid Function:** Applying the sigmoid function to map model outputs to probabilities.
- **Data Loading from JSON:** Reading data and keywords from JSON files.
- **Text Preprocessing:** Converting text to lowercase and using regular expressions to count keyword occurrences.

# Code Break Down

**1. Loading Libraries and Data**

```python
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import json
import re
```

- **`import jax`:** Imports the JAX library, which is the foundation for numerical computation and automatic differentiation in this code.
- **`import jax.numpy as jnp`:** Imports JAX's version of NumPy. `jnp` will be used to create and manipulate arrays (which represent vectors and tensors).
- **`import matplotlib.pyplot as plt`:** Imports the Matplotlib library for plotting (used for visualization at the end).
- **`import json`:** Imports the `json` module for working with JSON files (loading data and keywords).
- **`import re`:** Imports the regular expression module for pattern matching in text (used to find spammy keywords in emails).

**2. Loading Spammy Keywords**

```python
def load_spammy_keywords(filepath):
    with open(filepath, 'r') as f:
        keywords = json.load(f)
    return keywords["keywords"]

spammy_keywords = load_spammy_keywords("spammy_keywords.json")
```

- **`load_spammy_keywords(filepath)`:** This function loads a list of spammy keywords from a JSON file.
- **`spammy_keywords = ...`:** The loaded keywords are stored in the `spammy_keywords` variable as a regular Python list. This list isn't a JAX array yet, but it will be used to process the email data.

**3. Loading and Preprocessing Training Data**

```python
def load_and_preprocess_data(filepath, spammy_keywords):
    with open(filepath, 'r') as f:
        data = json.load(f)

    features = []
    is_spam = []
    for item in data["emails"]:
        email_text = item["text"].lower()
        keyword_count = sum(1 for keyword in spammy_keywords if re.search(r'\b' + keyword + r'\b', email_text))
        features.append(keyword_count)
        is_spam.append(item["is_spam"])

    features = jnp.array(features, dtype=jnp.float32)
    is_spam = jnp.array(is_spam, dtype=jnp.float32)
    return features, is_spam

features, is_spam = load_and_preprocess_data("emails.json", spammy_keywords)
```

- **`load_and_preprocess_data(...)`:** This function loads the training data (emails) from a JSON file, preprocesses it, and converts it into JAX arrays.
  - **`features = []`, `is_spam = []`:** Empty lists are initialized to store the extracted features (number of spammy keywords) and the corresponding labels (spam or not spam).
  - **`for item in data["emails"]:`:** The code iterates through each email in the loaded JSON data.
    - **`email_text = item["text"].lower()`:** The email text is converted to lowercase.
    - **`keyword_count = ...`:** The number of spammy keywords in the email is counted using a list comprehension and regular expressions.
    - **`features.append(keyword_count)`:** The keyword count (a single number) is appended to the `features` list.
    - **`is_spam.append(item["is_spam"])`:** The spam label (0 or 1) is appended to the `is_spam` list.
  - **`features = jnp.array(features, dtype=jnp.float32)`:** The `features` list (which contains numbers) is converted into a JAX array (specifically, a 1-dimensional array, which is a **vector**). The `dtype=jnp.float32` ensures it's a float32 array, which is common for numerical data in machine learning.
  - **`is_spam = jnp.array(is_spam, dtype=jnp.float32)`:** The `is_spam` list (containing 0s and 1s) is also converted into a JAX array (another **vector**).
  - **`return features, is_spam`:** The function returns these two JAX arrays.

**4. Initializing Parameters**

```python
key = jax.random.PRNGKey(0)
weight = jax.random.normal(key)
bias = jax.random.normal(key)
```

- **`key = jax.random.PRNGKey(0)`:** A random number generator key is created. JAX uses these keys to manage randomness in a reproducible way.
- **`weight = jax.random.normal(key)`:** The `weight` parameter is initialized with a random number drawn from a normal distribution. This is a scalar value, but it can also be thought of as a JAX array with a single element.
- **`bias = jax.random.normal(key)`:** The `bias` parameter is also initialized randomly, similar to `weight`.

**5. Defining the Sigmoid Function**

```python
def sigmoid(z):
    return 1 / (1 + jnp.exp(-z))
```

- **`def sigmoid(z):`:** This function defines the sigmoid function, which is a core component of logistic regression.
- **`jnp.exp(-z)`:** JAX's `jnp.exp()` function is used to calculate the element-wise exponential of the input `z`. `z` can be a single number, a **vector**, or even a higher-dimensional **tensor** - `jnp.exp()` will handle it correctly.
- The sigmoid function itself returns a value between 0 and 1, which can be interpreted as a probability.

**6. Defining the Model (Prediction Function)**

```python
def predict(weight, bias, features):
    z = weight * features + bias
    return sigmoid(z)
```

- **`def predict(weight, bias, features):`:** This is the main prediction function of the logistic regression model.
  - **`z = weight * features + bias`:** This is the linear part of the model.
    - If `features` is a vector (like in our training data), then `weight * features` performs element-wise multiplication between the `weight` scalar and the `features` vector.
    - The `bias` (a scalar) is then added to each element of the resulting vector. Broadcasting rules are applied here, since we are adding a scalar to a vector.
    - The result `z` will be a vector of the same shape as `features`.
  - **`return sigmoid(z)`:** The `sigmoid` function is applied to the vector `z`, element by element, producing another vector of the same shape where each element is a probability (between 0 and 1).

**7. Defining the Loss Function (Binary Cross-Entropy)**

```python
def binary_cross_entropy_loss(weight, bias, features, is_spam):
    probabilities = predict(weight, bias, features)
    epsilon = 1e-7
    loss = -jnp.mean(is_spam * jnp.log(probabilities + epsilon) + (1 - is_spam) * jnp.log(1 - probabilities + epsilon))
    return loss
```

- **`def binary_cross_entropy_loss(...)`:** This function calculates the binary cross-entropy loss, which measures how well the model's predictions match the true labels.
  - **`probabilities = predict(weight, bias, features)`:** The model's predictions (probabilities) are calculated using the `predict` function.
  - **`epsilon = 1e-7`:** A small value `epsilon` is added to avoid taking the logarithm of zero (which would lead to numerical issues).
  - **`loss = -jnp.mean(...)`:** The binary cross-entropy formula is implemented.
    - `jnp.log(probabilities + epsilon)` and `jnp.log(1 - probabilities + epsilon)` calculate the element-wise logarithms of the probabilities (and 1 minus the probabilities).
    - `is_spam * ...` and `(1 - is_spam) * ...` perform element-wise multiplication between the true labels and the log probabilities. This selects the appropriate log probability term based on whether the true label is 0 or 1.
    - `jnp.mean(...)` calculates the average of the resulting vector, giving you a single loss value (a scalar).

**8. Defining the Update Function**

```python
loss_grad = jax.grad(binary_cross_entropy_loss, argnums=(0, 1))

def update(weight, bias, features, is_spam, learning_rate):
    dw, db = loss_grad(weight, bias, features, is_spam)
    weight_new = weight - learning_rate * dw
    bias_new = bias - learning_rate * db
    return weight_new, bias_new
```

- **`loss_grad = jax.grad(binary_cross_entropy_loss, argnums=(0, 1))`:** This is where JAX's automatic differentiation comes into play.
  - `jax.grad` calculates the gradient (partial derivatives) of the `binary_cross_entropy_loss` function.
  - `argnums=(0, 1)` specifies that we want the gradients with respect to the first two arguments of the loss function, which are `weight` and `bias`.
  - `loss_grad` is now a function that, when called, will return the gradients of the loss with respect to `weight` and `bias`.
- **`def update(...)`:** This function performs a gradient descent step to update the `weight` and `bias` parameters.
  - **`dw, db = loss_grad(weight, bias, features, is_spam)`:** The `loss_grad` function is called to get the gradients (`dw` for the weight and `db` for the bias).
  - **`weight_new = weight - learning_rate * dw`:** The weight is updated by subtracting the learning rate times the gradient of the loss with respect to the weight.
  - **`bias_new = bias - learning_rate * db`:** The bias is updated similarly.
  - **`return weight_new, bias_new`:** The updated `weight` and `bias` are returned.

**9. Training the Model**

```python
learning_rate = 0.1
num_epochs = 1000

for epoch in range(num_epochs):
    weight, bias = update(weight, bias, features, is_spam, learning_rate)
    if epoch % 100 == 0:
        loss = binary_cross_entropy_loss(weight, bias, features, is_spam)
        print(f"Epoch {epoch}, Loss: {loss}")
```

- **`learning_rate = 0.1`:** The learning rate is set (a hyperparameter that controls the step size in gradient descent).
- **`num_epochs = 1000`:** The number of training iterations (epochs) is set.
- **`for epoch in range(num_epochs):`:** The training loop iterates for the specified number of epochs.
  - **`weight, bias = update(weight, bias, features, is_spam, learning_rate)`:** The `update` function is called to update the model's parameters in each epoch.
  - **`if epoch % 100 == 0:`:** Every 100 epochs, the current loss is calculated and printed.

**10. Making Predictions and Evaluating**

```python
probabilities = predict(weight, bias, features)
predictions = (probabilities >= 0.5).astype(jnp.int32)
accuracy = jnp.mean(predictions == is_spam)
print(f"Accuracy: {accuracy * 100:.2f}%")
```

- **`probabilities = predict(weight, bias, features)`:** Predictions (probabilities) are made on the training data using the trained `weight` and `bias`.
- **`predictions = (probabilities >= 0.5).astype(jnp.int32)`:** Predictions are converted to binary labels (0 or 1) based on whether the probability is greater than or equal to 0.5. This creates a new vector `predictions`.
- **`accuracy = jnp.mean(predictions == is_spam)`:** The accuracy of the model is calculated by comparing the predicted labels (`predictions` vector) to the true labels (`is_spam` vector). `jnp.mean()` computes the proportion of correctly classified examples.

**11. Classifying New Emails**

```python
def classify_new_emails(emails_to_classify, weight, bias, spammy_keywords):
    predictions = []
    for email in emails_to_classify:
        text = email["text"].lower()
        keyword_count = sum(1 for keyword in spammy_keywords if re.search(r'\b' + keyword + r'\b', text))
        features = jnp.array([keyword_count], dtype=jnp.float32)
        probability = predict(weight, bias, features)
        prediction = (probability >= 0.5).astype(jnp.int32)[0]
        predictions.append(prediction)
    return predictions

def load_and_classify_new_emails(filepath, weight, bias, spammy_keywords):
    with open(filepath, 'r') as f:
        new_emails = json.load(f)

    predictions = classify_new_emails(new_emails["emails"], weight, bias, spammy_keywords)

    for i, email in enumerate(new_emails["emails"]):
        print(f"Email: {email['text']}")
        print(f"Prediction: {'Spam' if predictions[i] == 1 else 'Not Spam'}")
        print("-" * 20)

load_and_classify_new_emails("new_emails.json", weight, bias, spammy_keywords)
```

- **`classify_new_emails(...)`:**
  - Takes a list of new emails (`emails_to_classify`), the trained `weight` and `bias`, and the `spammy_keywords` list.
  - Iterates through each email:
    - Preprocesses the email text (lowercase, keyword counting).
    - Creates a `features` JAX array (a vector with a single element in this case) representing the keyword count for the email.
    - Calls the `predict` function to get the probability of the email being spam.
    - Classifies the email as spam (1) or not spam (0) based on the probability.
    - Appends the prediction to the `predictions` list.
  - Returns the list of predictions.
- **`load_and_classify_new_emails(...)`:**
  - Loads new emails from a JSON file (`new_emails.json`).
  - Calls `classify_new_emails` to get predictions for these emails.
  - Prints the email text and the corresponding prediction for each email.

**12. Visualization (Optional)**

```python
# --- Visualize (Optional) ---
plt.figure(figsize=(8, 6))
plt.scatter(features, is_spam, c=is_spam, cmap='bwr', label='Actual')
plt.scatter(features, probabilities, c=probabilities, cmap='bwr', marker='x', label='Predicted Probability')
plt.xlabel("Number of Spammy Keywords")
plt.ylabel("Probability of being Spam")
plt.title("Logistic Regression - Spam Prediction")

boundary_x = jnp.linspace(0, jnp.max(features), 100)
boundary_y = sigmoid(weight * boundary_x + bias)
plt.plot(boundary_x, boundary_y, color='black', linestyle='--', label='Decision Boundary (p=0.5)')

plt.legend()
plt.colorbar(label='Probability')
plt.grid(True)
plt.show()

print(f"Trained weight: {weight}, Trained bias: {bias}")
```

- The code uses `matplotlib.pyplot` to create a scatter plot:
  - The x-axis represents the number of spammy keywords.
  - The y-axis represents the probability of being spam.
  - Blue dots represent actual non-spam emails.
  - Red dots represent actual spam emails.
  - "x" markers represent predicted probabilities.
  - A dashed line shows the decision boundary (where the probability is 0.5).

**In Summary**

- **Vectors:** In this code, `features`, `is_spam`, `probabilities`, and `predictions` are all examples of 1-dimensional JAX arrays, which are considered **vectors**.
- **Tensors:** While we don't have higher-dimensional tensors in this specific example, if you were to process images or other more complex data, you might use 2D arrays (matrices) or 3D/4D tensors.
- **Arrays:** JAX arrays (`jnp.array`) are the fundamental data structure used to represent vectors, matrices, and tensors.
- **Scalars:** `weight`, `bias`, and `learning_rate` are examples of scalars (single numerical values). They can also be considered as JAX arrays with a single element.
- **Broadcasting:** Broadcasting is implicitly used in operations like `weight * features + bias` where a scalar (`weight` or `bias`) is applied to a vector (`features`).
- **JAX's Power:** JAX provides automatic differentiation (`jax.grad`) and optimized numerical computation (using `jax.numpy` functions) that are essential for training machine learning models like this logistic regression example.

This detailed explanation connects the code to the concepts of vectors, tensors, and arrays and highlights how JAX is used to implement a machine-learning model. Please let me know if you have any more questions!
