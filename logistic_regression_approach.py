import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
import os
from kaggle.api.kaggle_api_extended import KaggleApi

# Specify the directory where you want to store the dataset
dataset_directory = os.path.join(os.getcwd(),
                                 "data")  # This will create a 'data' folder in the current working directory

# Kaggle API authentication
api = KaggleApi()
api.authenticate()

# Download the dataset
dataset_name = 'aunanya875/suicidal-tweet-detection-dataset'
dataset_path = os.path.join(dataset_directory, api.dataset_list_files(dataset_name).files[0].name)

if not os.path.exists(dataset_path):
    api.dataset_download_files(dataset_name, path=dataset_directory, unzip=True)
    print("Dataset downloaded and extracted to:", dataset_directory)
# Load your dataset (replace with your dataset loading code)
data = pd.read_csv(dataset_path)  # Assuming CSV format

import pandas as pd

data = pd.read_csv(os.path.join(dataset_directory, api.dataset_list_files(dataset_name).files[0].name))
data.rename(columns={'Suicide': 'label', 'Tweet': 'text'}, inplace=True)

data['label'] = data['label'].astype('category').cat.codes.astype(int)

import nltk

nltk.download('punkt')
nltk.download('stopwords')

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
from sklearn.linear_model import LogisticRegression


def preprocess_text(text):
    # Lowercase the text
    text = str(text).lower()

    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))

    # Tokenize the text
    tokens = word_tokenize(text)

    # Remove stopwords
    stop_words = set(stopwords.words("english"))
    tokens = [token for token in tokens if token not in stop_words]

    # Stem the tokens
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens]

    return " ".join(tokens)


data['text'] = data['text'].apply(preprocess_text)

# Assuming your dataset has "text" column and "label" column (0 or 1)
X = data["text"]
y = data["label"]

# Convert text to TF-IDF features
vectorizer = TfidfVectorizer()
X_tfidf = vectorizer.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Define parameter grid to search over
param_grid = {
    "C": [0.1, 1.0, 10.0],
    "penalty": ["l1", "l2"],
    "solver": ["liblinear"],
    "max_iter": [100, 200, 300]  # Increase max_iter value
}

# Initialize logistic regression model
logreg = LogisticRegression()

# Initialize GridSearchCV
grid_search = GridSearchCV(logreg, param_grid, cv=5, scoring="accuracy", n_jobs=-1)

# Fit the GridSearchCV to the data
grid_search.fit(X_train, y_train)

# Print best parameters and best cross-validation score
print("Best Parameters:", grid_search.best_params_)
print("Best Cross-Validation Score:", grid_search.best_score_)

# Evaluate the best model on the test set
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Test Set Accuracy with Best Model:", accuracy)
