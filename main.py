import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns

# Load the joke texts
# jokes_df = pd.read_csv('Dataset4JokeSet.csv', names=['joke'], encoding='1250', delimiter='\n')
jokes_df = pd.read_excel('Dataset4JokeSet.xlsx', sheet_name='Sheet1', names=['joke'])

# Remove unrated jokes
unrated_jokes = {1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 14, 20, 27, 31, 43, 51, 52, 61, 73, 80, 100, 116}
filtered_jokes_df = jokes_df.drop(unrated_jokes)

# Initialize BERT model
# model = SentenceTransformer('bert-base-cased')
model = SentenceTransformer('Davlan/bert-base-multilingual-cased-finetuned-yoruba')

# model.save('/bert-model')

# Encode the joke texts
embeddings = model.encode(filtered_jokes_df['joke'].tolist())
print(embeddings.shape)

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

# Train basic MLP model
mlp = MLPRegressor(max_iter=200, random_state=42, solver='sgd', alpha=0.0, learning_rate='constant')
mlp.fit(X_train, y_train)

# Predict and calculate RMSE
y_pred_train = mlp.predict(X_train)
y_pred_val = mlp.predict(X_val)
rmse_train = mean_squared_error(y_train, y_pred_train, squared=False)
rmse_val = mean_squared_error(y_val, y_pred_val, squared=False)

print(f'RMSE on Training Set: {rmse_train}')
print(f'RMSE on Validation Set: {rmse_val}')

# Plot loss curve
plt.plot(mlp.loss_curve_)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Curve for MLP Model')
plt.show()

















### Exercise 3: Investigate the Impact of Learning Rate
learning_rates = [0.001, 0.01, 0.1]
results_lr = []

for lr in learning_rates:
    mlp = MLPRegressor(max_iter=200, random_state=42, solver='sgd', learning_rate_init=lr)
    mlp.fit(X_train, y_train)

    y_pred_val = mlp.predict(X_val)
    rmse_val = mean_squared_error(y_val, y_pred_val, squared=False)

    results_lr.append({'learning_rate': lr, 'rmse_val': rmse_val})
    print(f'Learning Rate: {lr}, Validation RMSE: {rmse_val}')

    # Plot loss curve
    plt.plot(mlp.loss_curve_, label=f'LR: {lr}')

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Curve for Different Learning Rates')
plt.legend()
plt.show()

# Convert results to DataFrame for plotting
results_lr_df = pd.DataFrame(results_lr)
sns.barplot(x='learning_rate', y='rmse_val', data=results_lr_df)
plt.xlabel('Learning Rate')
plt.ylabel('Validation RMSE')
plt.title('Validation RMSE for Different Learning Rates')
plt.show()

### Exercise 4: Investigate the Impact of Model Size
layer_sizes = [(50,), (100,), (100, 50)]
results_size = []

for layers in layer_sizes:
    mlp = MLPRegressor(hidden_layer_sizes=layers, max_iter=200, random_state=42, solver='sgd', learning_rate_init=0.01)
    mlp.fit(X_train, y_train)

    y_pred_val = mlp.predict(X_val)
    rmse_val = mean_squared_error(y_val, y_pred_val, squared=False)

    results_size.append({'hidden_layers': str(layers), 'rmse_val': rmse_val})
    print(f'Hidden Layers: {layers}, Validation RMSE: {rmse_val}')

    # Plot loss curve
    plt.plot(mlp.loss_curve_, label=f'Layers: {layers}')

plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Curve for Different Model Sizes')
plt.legend()
plt.show()

# Convert results to DataFrame for plotting
results_size_df = pd.DataFrame(results_size)
sns.barplot(x='hidden_layers', y='rmse_val', data=results_size_df)
plt.xlabel('Hidden Layers')
plt.ylabel('Validation RMSE')
plt.title('Validation RMSE for Different Model Sizes')
plt.show()

### Exercise 5: Test the Best Model with a New Joke
# Select the best model based on validation RMSE
best_model = MLPRegressor(hidden_layer_sizes=(100,), max_iter=200, random_state=42, solver='sgd', learning_rate_init=0.01)
best_model.fit(X_train, y_train)

# Test with a new joke
new_joke = "Why don't scientists trust atoms? Because they make up everything!"
new_joke_embedding = model.encode([new_joke])
new_joke_rating = best_model.predict(new_joke_embedding)

print(f'Predicted rating for the new joke: {new_joke_rating[0]}')