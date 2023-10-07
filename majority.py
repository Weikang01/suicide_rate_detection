import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
import nltk

nltk.download('punkt')
nltk.download('stopwords')

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
from sklearn.linear_model import LogisticRegression

import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import pandas as pd

# Load your dataset
df = pd.read_excel(r"D:\LakeheadUCourse\3rd_year_winter\BigData_COMP4311\suicideRate\data\RedditData.xlsx")
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df = df[:10000]

# Rename columns
df.rename(columns={'Suicidal': 'label', 'title': 'text'}, inplace=True)

# Split your data into features (X) and labels (y)
X = df['text'].astype(str)
y = df['label'].astype('category').cat.codes.astype(int)


def get_ft_result():
    _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_test = X_test.tolist()
    y_test = y_test.tolist()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_path = r"D:\LakeheadUCourse\3rd_year_winter\BigData_COMP4311\suicideRate\distilbert-base-uncased-RedditData-fine-tuned"  # Replace with your model's path

    model = DistilBertForSequenceClassification.from_pretrained(model_path).to(device)
    tokenizer = DistilBertTokenizer.from_pretrained(model_path)
    inputs = tokenizer(X_test, padding=True, truncation=True, return_tensors="pt", max_length=256).to(device)

    length = 100
    start = 0
    end = length

    decoded_predictions = []
    while start < len(y_test):
        # Perform inference
        with torch.no_grad():
            if end > len(y_test):
                logits = model(**inputs[start:]).logits
            else:
                logits = model(**inputs[start:end]).logits
            predictions = torch.argmax(logits, dim=1)

        predictions = predictions.cpu().tolist()

        # Decode the labels back to their original values
        unique_labels = {0: 'Non Suicide', 1: 'Potential Suicide Post'}
        decoded_predictions.extend([unique_labels[pred] for pred in predictions])

        print(f'{start}-{end} is finished')
        start += length
        end += length

    return [label for label in y_test[:len(decoded_predictions)]]


# Vectorize the text data using TF-IDF (you can use other text vectorization methods as well)
tfidf_vectorizer = TfidfVectorizer(max_features=1000)  # You can adjust max_features as needed
X_tfidf = tfidf_vectorizer.fit_transform(X)


def get_rf_result():
    # Split the dataset into a training set and a testing set
    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

    # Create a Random Forest classifier
    # You can customize the hyperparameters as needed
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

    # Train the classifier on the training data
    rf_classifier.fit(X_train, y_train)

    # Make predictions on the test data
    return rf_classifier.predict(X_test)


def get_lr_result():
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

    logreg = LogisticRegression(
        C=10.0,
        penalty="l1",
        solver="liblinear",
        max_iter=100
    )

    # Fit the logistic regression model to the training data
    logreg.fit(X_train, y_train)

    # Evaluate the model on the test set
    return logreg.predict(X_test)


import gevent
from gevent import monkey

monkey.patch_all()
import numpy as np


# Define a function to calculate the majority vote score
def majority_vote(predictions_list):
    # Calculate the majority vote for each data point
    majority_votes = []
    for i in range(len(predictions_list[0])):
        votes = [pred[i] for pred in predictions_list]
        majority_vote = int(np.bincount(votes).argmax())
        majority_votes.append(majority_vote)
    return majority_votes


# Define a function to calculate and return predictions from each algorithm
def get_predictions():
    ft_future = gevent.spawn(get_ft_result)
    rf_future = gevent.spawn(get_rf_result)
    lr_future = gevent.spawn(get_lr_result)

    gevent.joinall([ft_future, rf_future])

    ft_predictions = ft_future.get()
    rf_predictions = rf_future.get()
    lr_predictions = lr_future.get()

    return ft_predictions, rf_predictions, lr_predictions


# Get predictions from each algorithm
ft_predictions, rf_predictions, lr_predictions = get_predictions()

from sklearn.metrics import classification_report

# Calculate the majority vote score
majority_votes = majority_vote([ft_predictions, rf_predictions, lr_predictions])

X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Calculate precision, recall, and f1-score based on majority vote
classification_rep = classification_report(y_test, majority_votes,
                                           target_names=['Non Suicide', 'Potential Suicide Post'])
print("Majority Vote Classification Report:\n", classification_rep)
