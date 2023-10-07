import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load your dataset
df = pd.read_excel(r"D:\LakeheadUCourse\3rd_year_winter\BigData_COMP4311\suicideRate\data\RedditData.xlsx")
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Rename columns
df.rename(columns={'Suicidal': 'label', 'title': 'text'}, inplace=True)

# Split your data into features (X) and labels (y)
X = df['text'].astype(str)
y = df['label']

# Vectorize the text data using TF-IDF (you can use other text vectorization methods as well)
tfidf_vectorizer = TfidfVectorizer(max_features=1000)  # You can adjust max_features as needed
X_tfidf = tfidf_vectorizer.fit_transform(X)

# Split the dataset into a training set and a testing set
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# Create a Random Forest classifier
# You can customize the hyperparameters as needed
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the classifier on the training data
rf_classifier.fit(X_train, y_train)

# Make predictions on the test data
y_pred = rf_classifier.predict(X_test)

# Evaluate the classifier's performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# You can also print a classification report for more detailed metrics
classification_rep = classification_report(y_test, y_pred)
print("Classification Report:\n", classification_rep)
