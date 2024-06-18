import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import LearningCurveDisplay, ShuffleSplit
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

# Load the joke texts
# jokes_df = pd.read_csv('Dataset4JokeSet.csv', names=['joke'], encoding='1250', delimiter='\n')
jokes_df = pd.read_excel('Dataset4JokeSet.xlsx', sheet_name='Sheet1', names=['joke'])

# Remove unrated jokes
unrated_jokes = {1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 14, 20, 27, 31, 43, 51, 52, 61, 73, 80, 100, 116}
filtered_jokes_df = jokes_df.drop(unrated_jokes)

# Initialize BERT model
# model = SentenceTransformer('bert-base-cased')
# model = SentenceTransformer('google-bert/bert-base-cased')
# model = SentenceTransformer('Davlan/bert-base-multilingual-cased-finetuned-yoruba')
# model = SentenceTransformer('all-MiniLM-L6-v2')
model = SentenceTransformer('rithwik-db/bert-base-cased-10')


# model.save('/bert-model')

# Encode the joke texts
embeddings = model.encode(filtered_jokes_df['joke'].tolist())
print(embeddings.shape)

# Convert embeddings to DataFrame
embeddings_df = pd.DataFrame(embeddings)

# Load the ratings
ratings_df = pd.read_excel('[final] April 2015 to Nov 30 2019 - Transformed Jester Data - .xlsx', sheet_name='Sheet1')

print(ratings_df)

# Filter the ratings DataFrame to include only the columns for the rated jokes
rated_columns = [i for i in range(1, 159) if i not in unrated_jokes]
ratings_df = ratings_df.iloc[:, [0] + rated_columns]
#


# Adjust column names to match the joke indices in filtered_jokes_df
ratings_df.columns = ['rated_jokes'] + [i-1 for i in rated_columns]
print(ratings_df)

# Melt the ratings DataFrame to long format
ratings_melted = ratings_df.melt(id_vars=['rated_jokes'], var_name='joke_id', value_name='rating')
ratings_melted['joke_id'] = ratings_melted['joke_id'].astype(int)

# Remove null ratings (99)
ratings_melted = ratings_melted[ratings_melted['rating'] != 99]
print(ratings_melted)

# Merge BERT embeddings with ratings
merged_df = pd.merge(ratings_melted, embeddings_df, left_on='joke_id', right_index=True)

# print(merged_df)

# Separate features and target
X = merged_df.drop(['rated_jokes', 'joke_id', 'rating'], axis=1)
y = merged_df['rating']

# Split into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)








# Train basic MLP model
mlp = MLPRegressor(max_iter=200, random_state=42, solver='sgd', alpha=0.0, learning_rate='constant')
common_params = {
    "X": X,
    "y": y,
    "train_sizes": np.linspace(0.1, 1.0, 5),
    "cv": ShuffleSplit(n_splits=50, test_size=0.2, random_state=0)
}


fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10, 6), sharey=True)
LearningCurveDisplay.from_estimator(mlp, **common_params, ax=ax)
handles, label = ax.get_legend_handles_labels()
ax.legend(handles[:2], ["Training Score", "Test Score"])
ax.set_title(f"Learning Curve for MLPRegressor")