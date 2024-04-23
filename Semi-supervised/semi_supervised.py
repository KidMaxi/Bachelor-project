import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast, Trainer, TrainingArguments
from transformers import EarlyStoppingCallback
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Early stopping configuration
patience = 2  # Number of epochs without improvement before early stopping
batch_size = 16  # Batch size for DataLoaders
learning_rate = 2e-5  # Learning rate for the optimizer
num_epochs = 10  # Number of training epochs
threshold = 0.95  # Confidence threshold for pseudo-labeling

# Model and tokenizer paths
model_name = 'distilbert_article_classifier_train_syn_V2'
model_path = f'./Creating_models/Supervised/models/{model_name}'
labeled_data_path = './Data/labeled_data.csv'
unlabeled_data_path = './Data/unlabeled_data.csv'

# Load the pre-trained model
model = DistilBertForSequenceClassification.from_pretrained(model_path)
tokenizer = DistilBertTokenizerFast.from_pretrained(model_path)

# Set device and load datasets
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

labeled_data = pd.read_csv(labeled_data_path)
unlabeled_data = pd.read_csv(unlabeled_data_path)

# Encode labels
label_encoder = LabelEncoder()
labeled_data['L5_ARTIKELTYPE'] = label_encoder.fit_transform(labeled_data['L5_ARTIKELTYPE'])

# Tokenize both labeled and unlabeled datasets
def tokenize_data(df, tokenizer):
    return tokenizer(
        df['OmsNederlands'].tolist(),
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    )

tokenized_labeled = tokenize_data(labeled_data, tokenizer)
tokenized_unlabeled = tokenize_data(unlabeled_data, tokenizer)

# Create PyTorch datasets
labeled_dataset = TensorDataset(
    tokenized_labeled['input_ids'],
    tokenized_labeled['attention_mask'],
    torch.tensor(labeled_data['L5_ARTIKELTYPE'], dtype=torch.long),
)

unlabeled_dataset = TensorDataset(
    tokenized_unlabeled['input_ids'],
    tokenized_unlabeled['attention_mask'],
)

# DataLoader for unlabeled data
unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=batch_size, shuffle=True)

# Function to perform pseudo-labeling
def pseudo_labeling(data_loader, model, threshold):
    pseudo_labels = []
    model.to(device)
    with torch.no_grad():
        for batch in data_loader:
            input_ids, attention_mask = batch
            input_ids, attention_mask = input_ids.to(device), attention_mask.to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
            confidences, predictions = torch.max(probabilities, dim=-1)
            pseudo_labels.extend([(pred.item(), conf.item()) for pred, conf in zip(predictions, confidences) if conf > threshold])
    return pseudo_labels

# Perform pseudo-labeling with the defined threshold
pseudo_label_data = pseudo_labeling(unlabeled_loader, model, threshold=threshold)

# Create pseudo-labeled DataFrame
high_confidence_indices = [
    idx for idx, (_, conf) in enumerate(pseudo_label_data) if conf > threshold
]
pseudo_labeled_df = unlabeled_data.iloc[high_confidence_indices].copy()
pseudo_labeled_df['L5_ARTIKELTYPE'] = [label_encoder.inverse_transform([label])[0] for label, _ in pseudo_label_data]

# Save the pseudo-labeled data to a CSV file
pseudo_labeled_df.to_csv(f'./Creating_models/Semi-supervised/pseudo_data/{model_name}', index=False)

# Combine with labeled data and split into training and validation sets
combined_data = pd.concat([labeled_data, pseudo_labeled_df])

label_encoder = LabelEncoder()  # Use the same label encoder as before
combined_data['L5_ARTIKELTYPE'] = label_encoder.fit_transform(combined_data['L5_ARTIKELTYPE'])

train_data, val_data = train_test_split(combined_data, test_size=0.1)

# Create training and validation DataLoaders
tokenized_train = tokenize_data(train_data, tokenizer)
tokenized_val = tokenize_data(val_data, tokenizer)

train_loader = DataLoader(
    TensorDataset(
        tokenized_train['input_ids'],
        tokenized_train['attention_mask'],
        torch.tensor(train_data['L5_ARTIKELTYPE'], dtype=torch.long),
    ),
    batch_size=batch_size,
    shuffle=True,
)

val_loader = DataLoader(
    TensorDataset(
        tokenize_data(val_data, tokenizer)['input_ids'],
        tokenize_data(val_data, tokenizer)['attention_mask'],
        torch.tensor(val_data['L5_ARTIKELTYPE'], dtype=torch.long),
    ),
    batch_size=batch_size,
    shuffle=False,
)

# Define training arguments with early stopping
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=num_epochs,
    per_device_train_batch_size=batch_size,
    evaluation_strategy='epoch',  # Evaluate at the end of each epoch
    learning_rate=learning_rate,
    report_to='none',  # Disable reporting if needed
)

# Create a Trainer with early stopping callback
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_loader.dataset,
    eval_dataset=val_loader.dataset,
    compute_metrics=lambda pred: {
        'accuracy': accuracy_score(pred.label_ids, pred.predictions.argmax(axis=1)),
        'precision': precision_score(pred.label_ids, pred.predictions.argmax(axis=1), average='weighted', zero_division=0),
        'recall': recall_score(pred.label_ids, pred.predictions.argmax(axis=1), average='weighted', zero_division=0),
        'f1': f1_score(pred.label_ids, pred.predictions.argmax(axis=1), average='weighted', zero_division=0),
    },
    callbacks=[EarlyStoppingCallback(early_stopping_patience=patience)],
)

# Start training with early stopping
trainer.train()

print("Training complete with early stopping.")
