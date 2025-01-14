> [!NOTE]
> I wrote this guide for anyone curious about machine learning, even if you're just starting with programming with Python or have an elementary understanding of math. We'll use the powerful [JAX](https://jax.readthedocs.io/en/latest/) library to explore the fundamentals of machine learning, focusing on practical projects that will help you build a solid intuition for how these systems work.

- [What is Machine Learning?](#what-is-machine-learning)
- [What You'll Learn](#what-youll-learn)
- [Getting Started](#getting-started)
  - [Option 1: Clone the Repository (Recommended)](#option-1-clone-the-repository-recommended)
  - [Option 2: Starting from Scratch](#option-2-starting-from-scratch)
- [Why JAX?](#why-jax)
- [JAX Concepts](#jax-concepts)
  - [`jax.numpy`](#jaxnumpy)
  - [Immutability](#immutability)
  - [Device Agnostic](#device-agnostic)
  - [`jax.Array`](#jaxarray)
- [Vectors, Matrices, and Tensors](#vectors-matrices-and-tensors)
  - [Vectors](#vectors)
    - [Role in Machine Learning](#role-in-machine-learning)
  - [Matrices](#matrices)
    - [Role in Machine Learning](#role-in-machine-learning-1)
  - [Tensors](#tensors)
    - [Role in Machine Learning](#role-in-machine-learning-2)
- [Projects](#projects)
  - [1. Predicting Children's Height with Linear Regression](#1-predicting-childrens-height-with-linear-regression)
  - [2. Spam Detection with Logistic Regression](#2-spam-detection-with-logistic-regression)

# What is Machine Learning?

Imagine you want to teach a child to identify different fruits. You wouldn't give them a long list of rules like, "If it's red and round, it's an apple, unless it's small and has a stem, then it's a cherry." Instead, you'd show them many examples of apples, oranges, bananas, and so on, letting them learn the differences on their own. Eventually, they'd be able to identify new fruits they've never seen before.

Machine learning is similar. It's about enabling computers to learn from data without being explicitly programmed with rigid rules. Instead of writing specific instructions for every task, we feed the computer a large amount of data and let it discover patterns, relationships, and insights on its own. This ability to learn from data allows computers to perform tasks that would be incredibly complex or even impossible to program traditionally.

# What You'll Learn

We'll focus on two fundamental machine learning techniques.

1. Linear Regression
   - We'll start by exploring linear regression, a method for modeling the relationship between variables using a straight line.
   - You'll learn how to use linear regression to predict a continuous value, such as a child's height based on their age in our first project.
   - We'll cover key concepts like:
     - Data loading and preprocessing with JAX
     - Model parameters (weight and bias)
     - The equation of a line
     - Loss functions (Mean Squared Error)
     - Gradient descent optimization
     - Making predictions
     - Visualizing results with Matplotlib
2. Logistic Regression
   - In our second project, you'll learn how to use logistic regression to build a spam filter that can classify emails as "spam" or "not spam."
   - We'll build upon the concepts from linear regression and introduce new ones, including:
     - The sigmoid function
     - Probability in classification
     - Binary cross-entropy loss
     - Evaluating model performance (accuracy)
     - Creating a decision boundary to separate classes
     - Feature engineering (counting spammy keywords)

# Getting Started

This section will guide you through setting up the project on your local machine. We provide two options: cloning the existing repository (recommended for following along with the guide) or starting from scratch and copying the necessary files.

## Option 1: Clone the Repository (Recommended)

1. **Clone the Repository**

   Open your terminal or command prompt and navigate to the directory where you want to store the project. Then, run the following command:

   ```bash
   git clone https://github.com/mrivasperez/jaxml.git/
   ```

   This will create a new folder with the project's name containing all the files.

2. **Create a Virtual Environemnt**

   It's highly recommended to use a virtual environment to keep the project's dependencies isolated from your global Python installation. Navigate into the newly created project directory:

   ```bash
   cd jaxml
   ```

   Now, create a virtual environment.

   ```bash
   python3 -m venv .venv
   ```

   This will create a .venv directory inside your project folder.

3. **Activate the Virtual Enviornment**

   Before you can use the virtual environment, you need to activate it.

   On macOS/Linux

   ```bash
   source .venv/bin/activate
   ```

   On Windows:

   ```powershell
   .venv\Scripts\activate
   ```

   You'll know the virtual environment is active when you see (.venv) at the beginning of your terminal prompt.

> [!TIP]
> **Deactivating the Virtual Environment.** When you're finished working on the project, you can deactivate the virtual environment by running `deactivate` in your terminal.

4. **Install Dependencies**

   The project's dependencies are listed in the `requirements.txt` file. Install them using pip, the package installer for Python:

   ```
   pip install -r requirements.txt
   ```

   This will install JAX, Matplotlib, and other necessary libraries within your virtual environment.

## Option 2: Starting from Scratch

If you prefer to build the project from the ground up, follow these steps.

1. **Create a Project Directory**

   Create a new folder for your project and navigate into it using your terminal or command prompt:

   ```
   mkdir jaxml
   cd jaxml
   ```

2. **Create a Virtual Environment**

   As in Option 1, create and activate a virtual environment.

   ```bash
   python3 -m venv .venv
   ```

   On macOS/Linux

   ```bash
   source .venv/bin/activate
   ```

   On Windows:

   ```powershell
   .venv\Scripts\activate
   ```

   You'll know the virtual environment is active when you see (.venv) at the beginning of your terminal prompt.

> [!TIP] > **Deactivating the Virtual Environment.** When you're finished working on the project, you can deactivate the virtual environment by running `deactivate` in your terminal.

3. **Create `requirements.txt`**

   In your project's root directory, create a new file named requirements.txt and paste the following content into it:

   ```
   contourpy==1.3.1
   cycler==0.12.1
   fonttools==4.55.3
   jax==0.4.38
   jaxlib==0.4.38
   kiwisolver==1.4.8
   matplotlib==3.10.0
   ml_dtypes==0.5.1
   numpy==2.2.1
   opt_einsum==3.4.0
   packaging==24.2
   pillow==11.1.0
   pyparsing==3.2.1
   python-dateutil==2.9.0.post0
   scipy==1.15.0
   six==1.17.0
   ```

4. **Install Dependencies**

   Install the dependencies from the requirements.txt file you just created:

   ```
   pip install -r requirements.txt
   ```

# Why JAX?

[JAX](https://jax.readthedocs.io/en/latest/) is a powerful library that combines the ease of use of NumPy with the ability to automatically calculate gradients (which we'll use for optimization) and run computations on GPUs and TPUs for significant speedups. Throughout this guide, you will learn the basics of using JAX:

- Using jax.numpy for numerical computation.
- Working with immutable arrays.
- Understanding JAX's unified array type: jax.Array.
- Using jax.grad for automatic differentiation, a crucial tool for optimizing machine learning models.

By the end of this guide, you will have a solid understanding of these core machine learning concepts and how to implement them using JAX. You'll also have two working projects to showcase your new skills and a foundation for exploring more advanced machine learning topics.

# JAX Concepts

[JAX](https://jax.readthedocs.io/en/latest/) builds upon the foundation of [NumPy](https://numpy.org/), a popular library for working with arrays in Python, but adds some unique features that make it ideal for high-performance computing and machine learning.

## `jax.numpy`

If you've used [NumPy](https://numpy.org/) before, jax.numpy will feel very familiar. It's a NumPy-like library that provides functions for creating and manipulating arrays, but it's designed to work seamlessly with JAX. You can often replace `import numpy as np` with `import jax.numpy as jnp` and your code will work with JAX arrays.

## Immutability

One of the key differences between JAX arrays and NumPy arrays is that JAX arrays are immutable. This means that once you create a JAX array, you cannot change its values in place. Instead of modifying an array, you create a new array with the desired changes.

```python
import jax.numpy as jnp

x = jnp.array([1, 2, 3])
# x[0] = 10  # This would raise an error because you can't modify x in place

y = x.at[0].set(10)  # Correct way: Create a new array y with the updated value

print(x)
print(y)
```

This immutability might seem like a limitation at first, but it's a crucial aspect of JAX that enables its powerful transformations like automatic differentiation and just-in-time compilation.

## Device Agnostic

Code written using JAX can run seamlessly on CPUs, GPUs, and TPUs without requiring significant modifications. JAX abstracts away the details of the underlying hardware, allowing you to write code once and run it anywhere. This is a huge advantage for machine learning, where training on specialized hardware like GPUs and TPUs can lead to substantial speedups.

## `jax.Array`

In older versions of JAX, there were different types of arrays (like `DeviceArray`, `ndarray`). However, in recent versions, JAX has introduced a unified array type called `jax.Array`. This simplifies things, as you now have a single array type that works across all devices.

# Vectors, Matrices, and Tensors

Now, let's explore the fundamental data structures used in machine learning: vectors, matrices, and tensors. We'll use jax.numpy to create and manipulate them.

## Vectors

A vector is simply a list of numbers. You can think of it like a single column in a spreadsheet.

```python
import jax.numpy as jnp

# Creating a vector (1D tensor)
vector = jnp.array([1, 2, 3])
print(vector, type(vector))

# Output
## [1 2 3] <class 'jax.Array'>
```

This creates a 1-dimensional JAX array (a vector) containing the numbers 1, 2, and 3. Notice that the output indicates the type is jax.Array.

### Role in Machine Learning

In machine learning, vectors are often used to represent individual data points or features. For instance, in our height prediction project, a single data point could be represented as a vector: [age, height]. Each element in the vector corresponds to a specific feature of the data point (in this case, age and height).

## Matrices

A matrix is a grid of numbers arranged in rows and columns, like a table.

```python
import jax.numpy as jnp

# Creating a matrix (2D tensor)
matrix = jnp.array([[1, 2], [3, 4]])
print(matrix, type(matrix))

# Output
[[1 2]
 [3 4]] <class 'jax.Array'>
```

### Role in Machine Learning

Matrices are commonly used to represent collections of data points or to store model parameters. For example, in our height prediction project, we could store the ages and heights of multiple children in a matrix where each row represents a child (a data point) and each column represents a feature (age or height). In a more complex model, the weights of a neural network are often stored in matrices.

## Tensors

A tensor is a generalization of vectors and matrices to higher dimensions. A vector is a 1-dimensional tensor, a matrix is a 2-dimensional tensor, and you can have tensors with 3, 4, or more dimensions.

A helpful way to visualize a 3D tensor is as a stack of matrices (like a stack of tables).

```
import jax.numpy as jnp

# Creating a 3D tensor
tensor_3d = jnp.ones((2, 3, 4))  # A 2x3x4 tensor filled with ones
print(tensor_3d, type(tensor_3d))
```

**Output.** This creates a 3-dimensional JAX array with shape (2, 3, 4). You can think of it as 2 matrices stacked on top of each other, where each matrix has 3 rows and 4 columns.

```python
[[[1. 1. 1. 1.]
  [1. 1. 1. 1.]
  [1. 1. 1. 1.]]

 [[1. 1. 1. 1.]
  [1. 1. 1. 1.]
  [1. 1. 1. 1.]]] <class 'jax.Array'>
```

### Role in Machine Learning

Tensors are essential for representing complex, multi-dimensional data. For example, a color image can be represented as a 3D tensor with dimensions (height, width, color_channels), where each color channel (red, green, blue) is a matrix representing the intensity of that color at each pixel. In deep learning, tensors are used extensively to store and process data as it flows through the layers of a neural network.

# Projects

## 1. Predicting Children's Height with Linear Regression

In this project, we'll build our first machine-learning model using a technique called linear regression. We'll use it to predict a child's height based on their age. This is a simple yet powerful example that will introduce you to many fundamental concepts in machine learning.

[➡️ Get Started](./height_prediction/README.md)

## 2. Spam Detection with Logistic Regression

In this project, we'll tackle a different type of machine learning problem: classification. Specifically, we'll build a model that can classify emails as either "spam" or "not spam". We'll use a technique called logistic regression for this task.

[➡️ Get Started](./spam/README.md)
