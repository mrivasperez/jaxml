import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import json


# 1. Load the Dataset from JSON
def load_data_from_json(filepath):
    with open(filepath, 'r') as f:
        data = json.load(f)
    ages = jnp.array([item["age"] for item in data["data"]], dtype=jnp.float32)
    heights = jnp.array([item["height"]
                        for item in data["data"]], dtype=jnp.float32)
    return ages, heights


# 2. Initialize Parameters
key = jax.random.PRNGKey(0)
weight = jax.random.normal(key)  # More descriptive name
bias = jax.random.normal(key)    # More descriptive name


# 3. Define the Model
def predict(weight, bias, ages):  # Updated parameter names
    return weight * ages + bias


# 4. Define the Loss Function
def mse_loss(weight, bias, ages, heights):  # Updated parameter names
    predictions = predict(weight, bias, ages)
    return jnp.mean((predictions - heights)**2)


# 5. Define the Update Function
loss_grad = jax.grad(mse_loss, argnums=(0, 1))


def update(weight, bias, ages, heights, learning_rate):  # Updated parameter names
    dw, db = loss_grad(weight, bias, ages, heights)
    weight_new = weight - learning_rate * dw
    bias_new = bias - learning_rate * db
    return weight_new, bias_new


# 6. Train the Model
learning_rate = 0.01
num_epochs = 1000

ages, heights = load_data_from_json("data.json")

for epoch in range(num_epochs):
    weight, bias = update(weight, bias, ages, heights,
                          learning_rate)  # Updated variable names
    if epoch % 100 == 0:
        loss = mse_loss(weight, bias, ages, heights)
        print(f"Epoch {epoch}, Loss: {loss}")

# 7. Make Predictions
predicted_heights = predict(weight, bias, ages)  # Updated variable names


# 8. Visualize the Results
plt.scatter(ages, heights, label="Actual")
plt.plot(ages, predicted_heights, label="Predicted", color="red")
plt.xlabel("Age (years)")
plt.ylabel("Height (cm)")
plt.legend()
plt.show()

print(f"Trained weight: {weight}, Trained bias: {
      bias}")  # Updated variable names
