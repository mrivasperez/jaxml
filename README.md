> [!NOTE]
> I wrote this guide for anyone curious about machine learning, even if you're just starting with programming or have an elementary understanding of math. We'll use the powerful JAX library to explore the fundamentals of machine learning, focusing on practical projects that will help you build a solid intuition for how these systems work.

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

# Why JAX?

AX is a powerful library that combines the ease of use of NumPy with the ability to automatically calculate gradients (which we'll use for optimization) and run computations on GPUs and TPUs for significant speedups. Throughout this guide, you will learn the basics of using JAX:

- Using jax.numpy for numerical computation.
- Working with immutable arrays.
- Understanding JAX's unified array type: jax.Array.
- Using jax.grad for automatic differentiation, a crucial tool for optimizing machine learning models.

By the end of this guide, you will have a solid understanding of these core machine learning concepts and how to implement them using JAX. You'll also have two working projects to showcase your new skills and a foundation for exploring more advanced machine learning topics.

# Getting Started

This section will guide you through setting up the project on your local machine. We provide two options: cloning the existing repository (recommended for following along with the guide) or starting from scratch and copying the necessary files.

## Option 1: Clone the Repository (Recommended)

1. **Clone the Repository**

   Open your terminal or command prompt and navigate to the directory where you want to store the project. Then, run the following command, replacing [repository URL] with the actual URL of the GitHub repository:

   ```bash
   git clone [repository URL]
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
