import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

# Define the model path and validation dataset path
model_path = './Creating_models/Supervised/models/distilbert_article_classifier_train_syn_V2'  # Path to your model
validation_data_path = './Data/validation_data.csv'  # Path to your validation dataset

# Load the model and tokenizer
model = DistilBertForSequenceClassification.from_pretrained(model_path)
tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)  # Load the tokenizer from the same path

# Set the model to evaluation mode
model.eval()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Load the validation dataset
validation_data = pd.read_csv(validation_data_path)

# Tokenize the validation data
tokenized_validation = tokenizer(
    validation_data['OmsNederlands'].tolist(),
    padding=True,
    truncation=True,
    max_length=512,
    return_tensors="pt"
)

# Create a DataLoader for the validation dataset
val_loader = DataLoader(
    TensorDataset(
        tokenized_validation['input_ids'],
        tokenized_validation['attention_mask'],
        torch.tensor(validation_data['L5_ARTIKELTYPE'].values)  # True labels
    ),
    batch_size=16,  # Adjust batch size as needed
    shuffle=False
)

# Function to evaluate the model on the validation set, including top-3 and top-5 accuracy
def evaluate_model(model, val_loader):
    all_preds = []
    all_labels = []
    all_top_k_preds = {3: [], 5: []}

    with torch.no_grad():
        for batch in val_loader:
            input_ids, attention_mask, labels = batch
            input_ids, attention_mask, labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            # Get the top-1 predictions
            top_1_preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(top_1_preds)
            all_labels.extend(labels.cpu().numpy())
            
            # Get the top-k predictions for top-3 and top-5 accuracy
            for k in all_top_k_preds:
                top_k_preds = torch.topk(logits, k).indices.cpu().numpy()  # Get top-k predictions
                all_top_k_preds[k].extend(top_k_preds)

    # Calculate basic evaluation metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')

    # Calculate top-3 and top-5 accuracy
    def top_k_accuracy(k, all_labels, all_top_k_preds):
        correct = 0
        for label, top_preds in zip(all_labels, all_top_k_preds[k]):
            if label in top_preds:
                correct += 1
        return correct / len(all_labels)

    top_3_accuracy = top_k_accuracy(3, all_labels, all_top_k_preds)
    top_5_accuracy = top_k_accuracy(5, all_labels, all_top_k_preds)

    return accuracy, precision, recall, f1, top_3_accuracy, top_5_accuracy

# Run the evaluation function and print the metrics
accuracy, precision, recall, f1, top_3_accuracy, top_5_accuracy = evaluate_model(model, val_loader)

print(f"Validation Results:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Top-3 Accuracy: {top_3_accuracy:.4f}")
print(f"Top-5 Accuracy: {top_5_accuracy:.4f}")
