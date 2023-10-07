import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from sklearn.metrics import classification_report
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the fine-tuned DistilBERT model and tokenizer
model_path = r"D:\LakeheadUCourse\3rd_year_winter\BigData_COMP4311\suicideRate\distilbert-base-uncased-RedditData-fine-tuned"  # Replace with your model's path
model = DistilBertForSequenceClassification.from_pretrained(model_path).to(device)
tokenizer = DistilBertTokenizer.from_pretrained(model_path)

# Load and preprocess the test dataset
df = pd.read_excel(
    r"D:\LakeheadUCourse\3rd_year_winter\BigData_COMP4311\suicideRate\data\RedditData.xlsx")  # Replace with your test dataset's path
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df = df[:1200]

df.rename(columns={'Suicidal': 'label', 'title': 'text'}, inplace=True)

unique_suicide_values = df['label'].unique()
df['label'] = df['label'].astype('category').cat.codes.astype(float)

# Tokenize and convert the texts to tensors
test_texts = df['text'].astype(str).tolist()
test_labels = df['label'].tolist()

inputs = tokenizer(test_texts, padding=True, truncation=True, return_tensors="pt", max_length=256).to(device)
print('tokenization finished!')

length = 100
start = 0
end = length

decoded_predictions = []
while start < len(test_labels):
    # Perform inference
    with torch.no_grad():
        if end > len(test_labels):
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

test_labels = [int(label) for label in test_labels[:len(decoded_predictions)]]
decoded_test_labels = [unique_labels[label] for label in test_labels]
# Calculate evaluation metrics
report = classification_report(decoded_test_labels, decoded_predictions, target_names=["Negative", "Positive"])
print(report)
