import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

# Load the joke texts
jokes_df = pd.read_excel('Dataset4JokeSet.xlsx', sheet_name='Sheet1', names=['joke'])

# Remove unrated jokes
unrated_jokes = {1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 14, 20, 27, 31, 43, 51, 52, 61, 73, 80, 100, 116}
filtered_jokes_df = jokes_df.drop(unrated_jokes)

# Initialize BERT model
model = SentenceTransformer('rithwik-db/bert-base-cased-10')

# Encode the joke texts
embeddings = model.encode(filtered_jokes_df['joke'].tolist())

# Convert embeddings to DataFrame
embeddings_df = pd.DataFrame(embeddings)

# Load the ratings
ratings_df = pd.read_excel('[final] April 2015 to Nov 30 2019 - Transformed Jester Data - .xlsx', sheet_name='Sheet1')

# Filter the ratings DataFrame to include only the columns for the rated jokes
rated_columns = [i for i in range(1, 159) if i not in unrated_jokes]
ratings_df = ratings_df.iloc[:, [0] + rated_columns]

# Adjust column names to match the joke indices in filtered_jokes_df
ratings_df.columns = ['rated_jokes'] + [i-1 for i in rated_columns]

# Melt the ratings DataFrame to long format
ratings_melted = ratings_df.melt(id_vars=['rated_jokes'], var_name='joke_id', value_name='rating')
ratings_melted['joke_id'] = ratings_melted['joke_id'].astype(int)

# Remove null ratings (99)
ratings_melted = ratings_melted[ratings_melted['rating'] != 99]

# Merge BERT embeddings with ratings
merged_df = pd.merge(ratings_melted, embeddings_df, left_on='joke_id', right_index=True)

# Separate features and target
X = merged_df.drop(['rated_jokes', 'joke_id', 'rating'], axis=1)
y = merged_df['rating']

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# # Standardize the data
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_val = scaler.transform(X_val)

# Build MLP model for regression
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1)  # Single output for regression
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])

# Train the model and save the history
history = model.fit(X_train, y_train, epochs=50, validation_data=(X_val, y_val))

# Plot loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Experiment with learning rates
learning_rates = [0.001, 0.01, 0.1]

plt.figure(figsize=(10, 6))

for lr in learning_rates:
    print(f"Training with learning rate: {lr}")

    # Compile model with different learning rate
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
                  loss='mean_squared_error',
                  metrics=['mean_absolute_error'])

    # Train model and save history
    history = model.fit(X_train, y_train, epochs=50, validation_data=(X_val, y_val))

    # Plot loss
    plt.plot(history.history['loss'], label=f'Training Loss (learning rate={lr})')
    plt.plot(history.history['val_loss'], label=f'Validation Loss (learning rate={lr})')

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Function to build and train model with different neurons
def build_and_train_model(neurons):
    model = Sequential([
        Dense(neurons, activation='relu', input_shape=(X_train.shape[1],)),
        Dense(neurons, activation='relu'),
        Dense(1)  # Single output for regression
    ])
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])
    history = model.fit(X_train, y_train, epochs=50, validation_data=(X_val, y_val))
    return history

# List of neurons to test
neurons_list = [5, 10, 20, 50]

plt.figure(figsize=(10, 6))

for neurons in neurons_list:
    print(f"Training with {neurons} neurons per layer")
    history = build_and_train_model(neurons)

    # Plot loss
    plt.plot(history.history['loss'], label=f'Training Loss ({neurons} neurons)')
    plt.plot(history.history['val_loss'], label=f'Validation Loss ({neurons} neurons)')

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
