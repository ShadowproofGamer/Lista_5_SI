import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load the Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# Convert to DataFrame
X_df = pd.DataFrame(X, columns=iris.feature_names)
y_df = pd.Series(y, name='target')

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_df, y_df, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Train a basic MLP model
mlp = MLPClassifier(max_iter=200, random_state=42)
mlp.fit(X_train, y_train)

# Predict and calculate accuracy
y_pred_train = mlp.predict(X_train)
y_pred_val = mlp.predict(X_val)
accuracy_train = accuracy_score(y_train, y_pred_train)
accuracy_val = accuracy_score(y_val, y_pred_val)

print(f'Accuracy on Training Set: {accuracy_train}')
print(f'Accuracy on Validation Set: {accuracy_val}')

# Plot loss curve
plt.plot(mlp.loss_curve_)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Curve for MLP Model')
plt.show()

# Tune Learning Rate
learning_rates = [0.001, 0.01, 0.1]
results = []

for lr in learning_rates:
    mlp = MLPClassifier(learning_rate_init=lr, max_iter=200, random_state=42)
    mlp.fit(X_train, y_train)
    y_pred_val = mlp.predict(X_val)
    accuracy_val = accuracy_score(y_val, y_pred_val)
    results.append((lr, accuracy_val))

for lr, accuracy in results:
    print(f'Learning Rate: {lr}, Validation Accuracy: {accuracy}')

# Tune Model Size
layer_sizes = [(50,), (100,), (100, 50)]
results = []

for layers in layer_sizes:
    mlp = MLPClassifier(hidden_layer_sizes=layers, max_iter=200, random_state=42)
    mlp.fit(X_train, y_train)
    y_pred_val = mlp.predict(X_val)
    accuracy_val = accuracy_score(y_val, y_pred_val)
    results.append((layers, accuracy_val))

for layers, accuracy in results:
    print(f'Hidden Layers: {layers}, Validation Accuracy: {accuracy}')
