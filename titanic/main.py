import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import re
import csv
import pandas as pd
from sklearn.model_selection import KFold, ParameterGrid

# --- Data Loading ---


def load_titanic_data(filepath):
    """Loads the Titanic training data from a CSV file.

    Args:
        filepath: Path to the 'train.csv' file.

    Returns:
        A dictionary containing JAX arrays for each column of the data,
        except for columns with strings, which are returned as lists.
    """
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)  # Read the header row
        data = list(reader)    # Read the remaining rows

    # Convert data to a dictionary of lists
    data_dict = {}
    for i, col_name in enumerate(header):
        data_dict[col_name] = [row[i] for row in data]

    # Convert lists to JAX arrays with appropriate data types
    jax_data = {}
    for col_name, values in data_dict.items():
        if col_name in ['PassengerId', 'Survived', 'Pclass', 'SibSp', 'Parch']:
            jax_data[col_name] = jnp.array(values, dtype=jnp.int32)
        elif col_name in ['Age', 'Fare']:
            # Handle missing values by converting them to NaN
            values = [float(val) if val != '' else float('nan')
                      for val in values]
            jax_data[col_name] = jnp.array(values, dtype=jnp.float32)
        elif col_name in ['Name', 'Sex', 'Ticket', 'Cabin', 'Embarked']:
            jax_data[col_name] = values  # Keep string columns as lists
        else:
            jax_data[col_name] = jnp.array(values)

    return jax_data

# --- Data Preprocessing ---


def preprocess_data(data):
    """Preprocesses the Titanic data.

    Args:
        data: A dictionary of JAX arrays or lists, as returned by load_titanic_data.

    Returns:
        A new dictionary with preprocessed data.
    """
    processed_data = data.copy()

    # 1. Handle Missing Values:
    # Age: Fill missing values with the median age
    if 'Age' in processed_data and isinstance(processed_data['Age'], jnp.ndarray):
        age = processed_data['Age']
        median_age = jnp.nanmedian(age)
        processed_data['Age'] = jnp.nan_to_num(age, nan=median_age)

    # Fare: Fill missing values with the median fare
    if 'Fare' in processed_data and isinstance(processed_data['Fare'], jnp.ndarray):
        fare = processed_data['Fare']
        median_fare = jnp.nanmedian(fare)
        processed_data['Fare'] = jnp.nan_to_num(fare, nan=median_fare)

    # Embarked: Drop rows with missing values
    if 'Embarked' in processed_data and isinstance(processed_data['Embarked'], list):
        mask = [embarked != '' for embarked in processed_data['Embarked']]
        for key in processed_data:
            if isinstance(processed_data[key], list):
                processed_data[key] = [item for item, keep in zip(
                    processed_data[key], mask) if keep]
            elif isinstance(processed_data[key], jnp.ndarray):
                processed_data[key] = processed_data[key][jnp.array(mask)]

    # 2. Convert Categorical Features to Numerical:
    # Sex: Convert 'male' to 0 and 'female' to 1
    if 'Sex' in processed_data and isinstance(processed_data['Sex'], list):
        processed_data['Sex'] = jnp.array(
            [0 if s == 'male' else 1 for s in processed_data['Sex']])

    # Embarked: One-hot encode
    if 'Embarked' in processed_data and isinstance(processed_data['Embarked'], list):
        embarked_one_hot = jax.nn.one_hot(
            jnp.array([0 if x == 'S' else 1 if x ==
                      'C' else 2 for x in processed_data['Embarked']]),
            num_classes=3
        )
        processed_data['Embarked_S'] = embarked_one_hot[:, 0]
        processed_data['Embarked_C'] = embarked_one_hot[:, 1]
        processed_data['Embarked_Q'] = embarked_one_hot[:, 2]
        del processed_data['Embarked']

    return processed_data

# --- Feature Engineering ---


def extract_titles(names):
    """Extracts titles from passenger names."""
    titles = []
    for name in names:
        match = re.search(r',\s(.*?)\.', name)
        if match:
            titles.append(match.group(1))
        else:
            titles.append("")
    return titles


def engineer_features(data):
    """
    Engineers new features from the Titanic dataset to improve model accuracy.
    """
    engineered_data = data.copy()

    # --- 1. Cabin Deck ---
    if 'Cabin' in engineered_data and isinstance(engineered_data['Cabin'], list):
        engineered_data['CabinDeck'] = [c[0] if isinstance(
            c, str) and c != '' else 'U' for c in engineered_data['Cabin']]

        # One-hot encode CabinDeck
        unique_decks = sorted(list(set(engineered_data['CabinDeck'])))
        deck_to_int = {deck: i for i, deck in enumerate(unique_decks)}
        deck_indices = jnp.array([deck_to_int[deck]
                                 for deck in engineered_data['CabinDeck']])
        decks_one_hot = jax.nn.one_hot(
            deck_indices, num_classes=len(unique_decks))

        for i, deck in enumerate(unique_decks):
            engineered_data[f'CabinDeck_{deck}'] = decks_one_hot[:, i]
        del engineered_data['CabinDeck']

    # --- 2. Ticket Prefix ---
    if 'Ticket' in engineered_data and isinstance(engineered_data['Ticket'], list):
        engineered_data['TicketPrefix'] = [re.split(r'\s|\.', t)[0] if len(
            re.split(r'\s|\.', t)) > 1 else 'UNK' for t in engineered_data['Ticket']]

        # One-hot encode TicketPrefix (consider only the most frequent prefixes to avoid too many features)
        unique_prefixes = sorted(list(set(engineered_data['TicketPrefix'])))
        prefix_to_int = {prefix: i for i, prefix in enumerate(unique_prefixes)}
        prefix_indices = jnp.array([prefix_to_int[prefix]
                                   for prefix in engineered_data['TicketPrefix']])
        prefixes_one_hot = jax.nn.one_hot(
            prefix_indices, num_classes=len(unique_prefixes))

        for i, prefix in enumerate(unique_prefixes):
            engineered_data[f'TicketPrefix_{prefix}'] = prefixes_one_hot[:, i]
        del engineered_data['TicketPrefix']

    # --- 3. Fare per Person ---
    if 'Fare' in engineered_data and 'FamilySize' in engineered_data:
        engineered_data['FarePerPerson'] = engineered_data['Fare'] / \
            engineered_data['FamilySize']

    # --- 4. Is Child/Mother ---
    if 'Age' in engineered_data and 'Sex' in engineered_data and 'Parch' in engineered_data:
        engineered_data['IsChildMother'] = jnp.where(
            (engineered_data['Age'] < 18) | (
                (engineered_data['Sex'] == 1) & (engineered_data['Parch'] > 0)), 1, 0
        )

    # --- 5. Title (Existing Feature - Improved Grouping) ---
    if 'Name' in engineered_data and isinstance(engineered_data['Name'], list):
        titles = extract_titles(engineered_data['Name'])
        engineered_data['Title'] = titles

        # More refined grouping of uncommon titles
        title_mapping = {
            "Capt": "Officer",
            "Col": "Officer",
            "Major": "Officer",
            "Dr": "Rare",
            "Rev": "Rare",
            "Jonkheer": "Royalty",
            "Don": "Royalty",
            "Sir": "Royalty",
            "the Countess": "Royalty",
            "Dona": "Royalty",
            "Lady": "Royalty",
            "Mme": "Mrs",
            "Ms": "Miss",
            "Mlle": "Miss"
        }
        engineered_data['Title'] = [title_mapping.get(
            t, t) for t in engineered_data['Title']]

        # One-hot encode Titles
        unique_titles = sorted(list(set(engineered_data['Title'])))
        title_to_int = {title: i for i, title in enumerate(unique_titles)}
        title_indices = jnp.array([title_to_int[title]
                                  for title in engineered_data['Title']])
        titles_one_hot = jax.nn.one_hot(
            title_indices, num_classes=len(unique_titles))

        for i, title in enumerate(unique_titles):
            engineered_data[f'Title_{title}'] = titles_one_hot[:, i]
        del engineered_data['Title']

    # --- 6. AgeGroup (Existing Feature - Fine-tuning) ---
    if 'Age' in engineered_data:
        # Adjust age group bins slightly
        age_groups = jnp.digitize(
            engineered_data['Age'],
            bins=jnp.array([0, 4, 13, 18, 30, 45, 60, 120]
                           ),  # Changed bin edges
            right=True
        )
        engineered_data['AgeGroup'] = age_groups

        # One-hot encode Age Groups
        age_groups_one_hot = jax.nn.one_hot(
            age_groups, num_classes=8)  # Changed num_classes to 8
        # Added a specific label for babies
        engineered_data['AgeGroup_Baby'] = age_groups_one_hot[:, 0]
        engineered_data['AgeGroup_Child'] = age_groups_one_hot[:, 1]
        engineered_data['AgeGroup_Teenager'] = age_groups_one_hot[:, 2]
        engineered_data['AgeGroup_YoungAdult'] = age_groups_one_hot[:, 3]
        engineered_data['AgeGroup_Adult'] = age_groups_one_hot[:, 4]
        # Added a label for middle-aged
        engineered_data['AgeGroup_MiddleAged'] = age_groups_one_hot[:, 5]
        engineered_data['AgeGroup_Senior'] = age_groups_one_hot[:, 6]
        # Assuming 7 represents an unknown group
        engineered_data['AgeGroup_Unknown'] = age_groups_one_hot[:, 7]
        del engineered_data['AgeGroup']

    # --- 7. Family Size and Is Alone (Existing Features) ---
    if 'SibSp' in engineered_data and 'Parch' in engineered_data:
        engineered_data['FamilySize'] = engineered_data['SibSp'] + \
            engineered_data['Parch'] + 1
        engineered_data['IsAlone'] = jnp.where(
            engineered_data['FamilySize'] == 1, 1, 0)

    return engineered_data

# --- Model (Logistic Regression) ---


def sigmoid(z):
    return 1 / (1 + jnp.exp(-z))


def predict(weights, biases, features):
    z = jnp.dot(features, weights) + biases
    return sigmoid(z)


def binary_cross_entropy_loss(weights, biases, features, labels, l2_lambda=0.0):
    probabilities = predict(weights, biases, features)
    epsilon = 1e-7
    loss = -jnp.mean(labels * jnp.log(probabilities + epsilon) +
                     (1 - labels) * jnp.log(1 - probabilities + epsilon))

    # Add L2 regularization term
    l2_penalty = l2_lambda * jnp.sum(weights**2)
    loss += l2_penalty

    return loss


def update(weights, biases, features, labels, learning_rate, l2_lambda=0.0):
    probabilities = predict(weights, biases, features)
    dw = jnp.dot(features.T, (probabilities - labels)) / labels.shape[0]
    db = jnp.mean(probabilities - labels)

    # Add L2 regularization to gradient
    dw += 2 * l2_lambda * weights

    weights_new = weights - learning_rate * dw
    biases_new = biases - learning_rate * db
    return weights_new, biases_new

# --- Training ---


def train_model(data, learning_rate, num_epochs, l2_lambda, feature_cols):
    """Trains the logistic regression model.

    Args:
        data: A dictionary of JAX arrays containing the training data.
        learning_rate: The learning rate for gradient descent.
        num_epochs: The number of training epochs.
        l2_lambda: L2 regularization strength.
        feature_cols: List of feature columns to use.

    Returns:
        The trained weights and biases.
    """
    # Split data into training and validation sets (80/20 split)
    split_index = int(0.8 * len(data['PassengerId']))

    train_data = {}
    val_data = {}
    for key in data:
        if isinstance(data[key], jnp.ndarray):
            train_data[key] = data[key][:split_index]
            val_data[key] = data[key][split_index:]
        else:
            train_data[key] = data[key][:split_index]
            val_data[key] = data[key][split_index:]

    # Prepare features and labels
    train_features = jnp.column_stack([
        jnp.array(train_data[col]) for col in feature_cols if col in train_data
    ])
    train_labels = jnp.array(train_data['Survived'])
    val_features = jnp.column_stack([
        jnp.array(val_data[col]) for col in feature_cols if col in val_data
    ])
    val_labels = jnp.array(val_data['Survived'])

    # Initialize weights and biases
    key = jax.random.PRNGKey(0)
    weights = jax.random.normal(key, shape=(train_features.shape[1],))
    biases = jax.random.normal(key)

    # Training loop
    for epoch in range(num_epochs):
        weights, biases = update(
            weights, biases, train_features, train_labels, learning_rate, l2_lambda
        )
        if epoch % 100 == 0:
            loss = binary_cross_entropy_loss(
                weights, biases, train_features, train_labels, l2_lambda
            )
            print(f"Epoch {epoch}, Loss: {loss}")

    return weights, biases, val_features, val_labels

# --- Evaluation ---


def evaluate_model(weights, biases, features, labels):
    """Evaluates the model's accuracy.

    Args:
        weights: The trained weights.
        biases: The trained biases.
        features: The validation features.
        labels: The validation labels.

    Returns:
        The accuracy of the model on the validation set.
    """
    probabilities = predict(weights, biases, features)
    predictions = (probabilities >= 0.5).astype(jnp.int32)
    accuracy = jnp.mean(predictions == labels)
    print(f"Validation Accuracy: {accuracy * 100:.2f}%")
    return accuracy

# --- Cross-Validation ---


def cross_validate(data, learning_rate, num_epochs, l2_lambda, feature_cols, n_splits=5):
    """Performs k-fold cross-validation.

    Args:
        data: A dictionary of JAX arrays containing the data.
        learning_rate: The learning rate for gradient descent.
        num_epochs: The number of training epochs.
        l2_lambda: L2 regularization strength.
        feature_cols: List of feature columns to use.
        n_splits: Number of folds for cross-validation.

    Returns:
        The average cross-validation accuracy.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    accuracies = []

    for train_index, val_index in kf.split(data['PassengerId']):
        train_data = {}
        val_data = {}
        for key in data:
            if isinstance(data[key], jnp.ndarray):
                train_data[key] = data[key][train_index]
                val_data[key] = data[key][val_index]
            else:
                train_data[key] = [data[key][i] for i in train_index]
                val_data[key] = [data[key][i] for i in val_index]

        # Prepare features and labels
        train_features = jnp.column_stack([
            jnp.array(train_data[col]) for col in feature_cols if col in train_data
        ])
        train_labels = jnp.array(train_data['Survived'])
        val_features = jnp.column_stack([
            jnp.array(val_data[col]) for col in feature_cols if col in val_data
        ])
        val_labels = jnp.array(val_data['Survived'])

        # Initialize weights and biases
        key = jax.random.PRNGKey(0)
        weights = jax.random.normal(key, shape=(train_features.shape[1],))
        biases = jax.random.normal(key)

        # Train the model on the training fold
        weights, biases = train_model_kfold(
            train_data, learning_rate, num_epochs, l2_lambda, weights, biases, train_features, train_labels, feature_cols
        )

        # Evaluate on the validation fold
        accuracy = evaluate_model(weights, biases, val_features, val_labels)
        accuracies.append(accuracy)

    avg_accuracy = jnp.mean(jnp.array(accuracies))
    print(f"Average Cross-Validation Accuracy: {avg_accuracy:.4f}")
    return avg_accuracy


def train_model_kfold(data, learning_rate, num_epochs, l2_lambda, weights, biases, train_features, train_labels, feature_cols):
    # Training loop
    for epoch in range(num_epochs):
        weights, biases = update(
            weights, biases, train_features, train_labels, learning_rate, l2_lambda)

    return weights, biases

# --- Visualization ---


def visualize_data(data):
    """Creates some basic visualizations of the data."""
    df = pd.DataFrame(data)

    # Example: Survival rate by class
    plt.figure()
    survival_by_class = df.groupby('Pclass')['Survived'].mean()
    plt.bar(survival_by_class.index, survival_by_class.values)
    plt.xlabel("Passenger Class")
    plt.ylabel("Survival Rate")
    plt.title("Survival Rate by Passenger Class")
    plt.xticks(survival_by_class.index)
    plt.show()

    # Example: Age distribution of survivors vs. non-survivors
    plt.figure()
    plt.hist(df['Age'][df['Survived'] == 0], bins=20,
             alpha=0.5, label="Did not survive")
    plt.hist(df['Age'][df['Survived'] == 1],
             bins=20, alpha=0.5, label="Survived")
    plt.xlabel("Age")
    plt.ylabel("Number of Passengers")
    plt.title("Age Distribution of Survivors vs. Non-Survivors")
    plt.legend()
    plt.show()

# --- Main ---


if __name__ == "__main__":
    data = load_titanic_data("./data/train.csv")
    processed_data = preprocess_data(data)
    engineered_data = engineer_features(processed_data)

    # Convert engineered data to pandas DataFrame for visualization
    engineered_df = pd.DataFrame({
        col: (engineered_data[col] if isinstance(
            engineered_data[col], list) else engineered_data[col].tolist())
        for col in engineered_data
        if col in ['Pclass', 'Survived', 'Age', 'SibSp', 'Parch', 'Fare', 'FamilySize', 'IsAlone', 'FarePerPerson', 'IsChildMother'] or 'Title_' in col or 'Embarked_' in col or 'AgeGroup_' in col or 'CabinDeck_' in col or 'TicketPrefix_' in col
    })

    # Now call visualize_data
    visualize_data(engineered_df)

    # --- Hyperparameter Tuning with Cross-Validation ---
    param_grid = {
        'learning_rate': [0.01, 0.001],
        'num_epochs': [1000, 2000, 3000],
        'l2_lambda': [0.01, 0.1, 0.5]
    }

    feature_cols = [
        'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'FamilySize', 'IsAlone',
        'Embarked_S', 'Embarked_C', 'Embarked_Q',
        'FarePerPerson', 'IsChildMother',
        'Title_Mr', 'Title_Miss', 'Title_Mrs', 'Title_Master', 'Title_Royalty', 'Title_Officer', 'Title_Rare',
        'AgeGroup_Baby', 'AgeGroup_Child', 'AgeGroup_Teenager', 'AgeGroup_YoungAdult', 'AgeGroup_Adult', 'AgeGroup_MiddleAged', 'AgeGroup_Senior', 'AgeGroup_Unknown',
    ]

    # Dynamically add CabinDeck and TicketPrefix features if they exist
    cabin_deck_features = [
        col for col in engineered_data.keys() if col.startswith('CabinDeck_')]
    ticket_prefix_features = [
        col for col in engineered_data.keys() if col.startswith('TicketPrefix_')]
    feature_cols.extend(cabin_deck_features)
    feature_cols.extend(ticket_prefix_features)

    best_accuracy = 0
    best_params = {}
    for params in ParameterGrid(param_grid):
        print(f"Training with params: {params}")
        avg_accuracy = cross_validate(
            engineered_data, params['learning_rate'], params['num_epochs'], params['l2_lambda'], feature_cols, n_splits=5
        )

        if avg_accuracy > best_accuracy:
            best_accuracy = avg_accuracy
            best_params = params

    print(f"Best Cross-Validation Accuracy: {best_accuracy:.4f}")
    print(f"Best Parameters: {best_params}")

    # --- Train Final Model with Best Hyperparameters ---
    # You can now train the final model on the entire dataset using the best hyperparameters
    # and evaluate it on a held-out test set (if you have one).
    # Or, if you're submitting to Kaggle, you would use the best model to make predictions on the Kaggle test set.
    # For example:

    final_weights, final_biases, _, _ = train_model(
        engineered_data, best_params['learning_rate'], best_params['num_epochs'], best_params['l2_lambda'], feature_cols
    )

    # Then, use final_weights and final_biases to make predictions on new data (e.g., the Kaggle test set).
