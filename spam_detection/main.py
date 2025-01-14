import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import json
import re

# --- Load Spammy Keywords ---


def load_spammy_keywords(filepath):
    with open(filepath, 'r') as f:
        keywords = json.load(f)
    return keywords["keywords"]

# --- Load and Preprocess Training Data ---


def load_and_preprocess_data(filepath, spammy_keywords):
    with open(filepath, 'r') as f:
        data = json.load(f)

    features = []
    is_spam = []
    for item in data["emails"]:
        email_text = item["text"].lower()
        keyword_count = sum(1 for keyword in spammy_keywords if re.search(
            r'\b' + keyword + r'\b', email_text))
        features.append(keyword_count)
        is_spam.append(item["is_spam"])

    features = jnp.array(features, dtype=jnp.float32)
    is_spam = jnp.array(is_spam, dtype=jnp.float32)
    return features, is_spam

# --- Classify New Emails ---


def classify_new_emails(emails_to_classify, weight, bias, spammy_keywords):
    """Classifies a list of new emails as spam (1) or not spam (0).

    Args:
        emails_to_classify: A list of dictionaries, each with a "text" key for the email content.
        weight: The trained weight parameter.
        bias: The trained bias parameter.
        spammy_keywords: A list of spammy keywords.

    Returns:
        A list of predictions (0 or 1) for each email.
    """
    predictions = []
    for email in emails_to_classify:
        text = email["text"].lower()
        keyword_count = sum(1 for keyword in spammy_keywords if re.search(
            r'\b' + keyword + r'\b', text))
        features = jnp.array([keyword_count], dtype=jnp.float32)
        probability = predict(weight, bias, features)
        prediction = (probability >= 0.5).astype(jnp.int32)[0]
        predictions.append(prediction)
    return predictions

# --- Load and Classify New Emails from JSON ---


def load_and_classify_new_emails(filepath, weight, bias, spammy_keywords):
    """Loads new emails from a JSON file, classifies them, and prints results.

    Args:
        filepath: Path to the JSON file containing new emails.
        weight: The trained weight parameter.
        bias: The trained bias parameter.
        spammy_keywords: A list of spammy keywords.
    """
    with open(filepath, 'r') as f:
        new_emails = json.load(f)

    predictions = classify_new_emails(
        new_emails["emails"], weight, bias, spammy_keywords)

    for i, email in enumerate(new_emails["emails"]):
        print(f"Email: {email['text']}")
        print(f"Prediction: {'Spam' if predictions[i] == 1 else 'Not Spam'}")
        print("-" * 20)

# --- Initialization, Model, Loss, Update (Same as before) ---


# 2. Initialize Parameters
key = jax.random.PRNGKey(0)
weight = jax.random.normal(key)
bias = jax.random.normal(key)

# 3. Define the Sigmoid Function


def sigmoid(z):
    return 1 / (1 + jnp.exp(-z))

# 4. Define the Model (Prediction Function)


def predict(weight, bias, features):
    z = weight * features + bias
    return sigmoid(z)

# 5. Define the Loss Function (Binary Cross-Entropy)


def binary_cross_entropy_loss(weight, bias, features, is_spam):
    probabilities = predict(weight, bias, features)
    epsilon = 1e-7
    loss = -jnp.mean(is_spam * jnp.log(probabilities + epsilon) +
                     (1 - is_spam) * jnp.log(1 - probabilities + epsilon))
    return loss


# 6. Define the Update Function
loss_grad = jax.grad(binary_cross_entropy_loss, argnums=(0, 1))


def update(weight, bias, features, is_spam, learning_rate):
    dw, db = loss_grad(weight, bias, features, is_spam)
    weight_new = weight - learning_rate * dw
    bias_new = bias - learning_rate * db
    return weight_new, bias_new


# --- Train the Model ---
learning_rate = 0.1
num_epochs = 1000

spammy_keywords = load_spammy_keywords("spam_keywords.json")
features, is_spam = load_and_preprocess_data("data.json", spammy_keywords)

for epoch in range(num_epochs):
    weight, bias = update(weight, bias, features, is_spam, learning_rate)
    if epoch % 100 == 0:
        loss = binary_cross_entropy_loss(weight, bias, features, is_spam)
        print(f"Epoch {epoch}, Loss: {loss}")

# --- Evaluate on Training Data ---
probabilities = predict(weight, bias, features)
predictions = (probabilities >= 0.5).astype(jnp.int32)
accuracy = jnp.mean(predictions == is_spam)
print(f"Accuracy: {accuracy * 100:.2f}%")

# --- Classify New Emails from JSON ---
load_and_classify_new_emails("validate.json", weight, bias, spammy_keywords)

# --- Visualize (Optional) ---
plt.figure(figsize=(8, 6))
plt.scatter(features, is_spam, c=is_spam, cmap='bwr', label='Actual')
plt.scatter(features, probabilities, c=probabilities, cmap='bwr',
            marker='x', label='Predicted Probability')
plt.xlabel("Number of Spammy Keywords")
plt.ylabel("Probability of being Spam")
plt.title("Logistic Regression - Spam Prediction")

boundary_x = jnp.linspace(0, jnp.max(features), 100)
boundary_y = sigmoid(weight * boundary_x + bias)
plt.plot(boundary_x, boundary_y, color='black',
         linestyle='--', label='Decision Boundary (p=0.5)')

plt.legend()
plt.colorbar(label='Probability')
plt.grid(True)
plt.show()

print(f"Trained weight: {weight}, Trained bias: {bias}")
