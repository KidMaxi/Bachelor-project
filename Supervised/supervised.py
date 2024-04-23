import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import matplotlib.pyplot as plt
from scipy.special import softmax
import numpy as np

patience = 3
epochs = 300
batch_size = 4
weight_decay = 0.015
warmup_ratio = 0.10

# Load the datasets
file_path_labelled = './Data/labeled_data.csv'
file_path_syn = './Data/syn_labeled_data.csv'
df_labelled = pd.read_csv(file_path_labelled, encoding='ISO-8859-1')
df_syn = pd.read_csv(file_path_syn, encoding='ISO-8859-1')

print("Datasets loaded successfully.")

# Encode the article types if they're categorical
encoder = LabelEncoder()
df_labelled['L5_ARTIKELTYPE'] = encoder.fit_transform(df_labelled['L5_ARTIKELTYPE'])
df_syn['L5_ARTIKELTYPE'] = encoder.transform(df_syn['L5_ARTIKELTYPE'])  # Ensure same encoding as df_labelled
print("Article types encoded.")

# Split the labelled dataset
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df_labelled['OmsNederlands'], df_labelled['L5_ARTIKELTYPE'], test_size=0.2)
print("Dataset split into training and validation sets.")

# Append synthetic data to the training set
train_texts = pd.concat([train_texts, df_syn['OmsNederlands']])
train_labels = pd.concat([train_labels, df_syn['L5_ARTIKELTYPE']])
print("Synthetic data added to the training set.")

# Tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-multilingual-cased')
print("\nTokenizer initialized.")

# Tokenize data
train_encodings = tokenizer(train_texts.tolist(), truncation=True, padding='max_length', max_length=212)
val_encodings = tokenizer(val_texts.tolist(), truncation=True, padding='max_length', max_length=212)
print("\nTokenization complete.")

class ArticleDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)

# Convert to dataset
train_dataset = ArticleDataset(train_encodings, train_labels.tolist())
val_dataset = ArticleDataset(val_encodings, val_labels.tolist())
print("\nTraining and validation datasets prepared.")

# Model initialization
print("\nInitializing the model...\n")
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-multilingual-cased', num_labels=len(encoder.classes_))
print("\nModel initialized.")

# Training arguments
training_args = TrainingArguments(
    output_dir='./Creating_models/Supervised/results',
    num_train_epochs=epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    warmup_steps=500,
    weight_decay=weight_decay,
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    warmup_ratio=warmup_ratio,
    save_total_limit=1,
    load_best_model_at_end=True,  # Load the best model after training
    metric_for_best_model="eval_loss",  # Choose the metric for determining the best model
    greater_is_better=False,  # False because lower eval_loss is better
)

print("\nTraining arguments set up complete.")

def top_n_accuracy(preds, labels):
    # Apply softmax to convert logits to probabilities
    probs = softmax(preds, axis=1)
    # Get the top 3 predictions
    top_3_preds = np.argsort(probs, axis=1)[:, -3:]
    top_5_preds = np.argsort(probs, axis=1)[:, -5:]
    # Check if the true label is in the top 3 predictions
    match_top_3 = np.any(top_3_preds == labels.reshape(-1, 1), axis=1)
    match_top_5 = np.any(top_5_preds == labels.reshape(-1, 1), axis=1)
    # Calculate top-3 accuracy
    top_3_accuracy = np.mean(match_top_3)
    top_5_accuracy = np.mean(match_top_5)
    return top_3_accuracy, top_5_accuracy

def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted', zero_division=1)
    acc = accuracy_score(labels, preds)
    # Calculate top-3 accuracy
    top_3_acc = top_n_accuracy(pred.predictions, labels)[0]
    top_5_acc = top_n_accuracy(pred.predictions, labels)[1]

    return {
        'accuracy': acc, 
        'f1': f1, 
        'precision': precision, 
        'recall': recall,
        'top_3_accuracy': top_3_acc,
        'top_5_accuracy': top_5_acc
    }

print("\nTraining the model...\n")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=patience)]
)

trainer.train()
print("\nTraining complete.")

# Save the model and tokenizer with error handling
try:
    model_name = 'distilbert_article_classifier_train_syn_V4'
    model_path = f'./Creating_models/Supervised/models/{model_name}'
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    print("\nModel and tokenizer saved.")
except Exception as e:
    print(f"Error saving model or tokenizer: {e}")

print("\nEvaluating the model...")
eval_results = trainer.evaluate()
print("\nEvaluation results:", eval_results)

images_dir = "./Creating_models/Supervised/plots"
os.makedirs(images_dir, exist_ok=True)

# Collect the learning rates and loss values from the log history
learning_rates = [log['learning_rate'] for log in trainer.state.log_history if 'learning_rate' in log]
loss_values = [log['loss'] for log in trainer.state.log_history if 'loss' in log]

# Hyperparameters and metrics for text display
hyperparams_text = f"Batch Size: {batch_size}\nEpochs: {epochs}\nPatience: {patience}\nWeight Decay: {weight_decay}\nWarmup Ratio: {warmup_ratio}"
metrics_text = (
    f"Accuracy: {eval_results['eval_accuracy']:.4f}\n"
    f"Precision: {eval_results['eval_precision']:.4f}\n"
    f"Recall: {eval_results['eval_recall']:.4f}\n"
    f"F1: {eval_results['eval_f1']:.4f}\n"
    f"Top-3 Accuracy: {eval_results['eval_top_3_accuracy']:.4f}\n"
    f"\nTop-5 Accuracy: {eval_results['eval_top_5_accuracy']:.4f}"
)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))

# Plot learning rate with hyperparams text
ax1.plot(learning_rates, label='Learning Rate')
ax1.set_title('Learning Rate Over Training Steps')
ax1.set_xlabel('Training Steps')
ax1.set_ylabel('Learning Rate')
ax1.legend(loc='upper left')
ax1.text(0.5, 0.5, hyperparams_text, transform=ax1.transAxes, fontsize=9,
         horizontalalignment='center', verticalalignment='center', 
         bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.5))

# Plot training loss with metrics text
ax2.plot(loss_values, '-', label='Training Loss')
ax2.set_title('')
ax2.set_xlabel('Steps')
ax2.set_ylabel('Loss')
ax2.legend(loc='upper left')
ax2.text(0.5, 0.5, metrics_text, transform=ax2.transAxes, fontsize=9,
         horizontalalignment='center', verticalalignment='center', 
         bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.5))

plt.tight_layout()
plt.savefig(f"{images_dir}/{model_name}_plot.png")
plt.show()