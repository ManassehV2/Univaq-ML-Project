import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertForSequenceClassification, BertConfig, get_scheduler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_curve, auc,
    precision_recall_curve
)
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Ensure output directory exists
output_dir = "results_exp_2"
os.makedirs(output_dir, exist_ok=True)

# Load Augmented Data
data = torch.load(
    "preprocessed_data_aug_fr_it_and_de.pt", weights_only=True)

# Extract data
train_inputs = data["train_inputs"]
val_inputs = data["val_inputs"]
train_masks = data["train_masks"]
val_masks = data["val_masks"]
train_labels = data["train_labels"]
val_labels = data["val_labels"]

# Token length distribution
# Calculate token lengths
token_lengths = [input_ids.size(0) for input_ids in train_inputs]
plt.figure(figsize=(8, 6))
sns.histplot(token_lengths, bins=20, kde=True)
plt.title("Token Length Distribution")
plt.xlabel("Number of Tokens")
plt.ylabel("Frequency")
plt.savefig(os.path.join(output_dir, "token_length_distribution.png"))

# Create datasets and dataloaders
batch_size = 16
train_dataset = TensorDataset(train_inputs, train_masks, train_labels)
val_dataset = TensorDataset(val_inputs, val_masks, val_labels)
train_dataloader = DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

# Configure BERT model
dropout_rate = 0.11
config = BertConfig.from_pretrained(
    "bert-base-uncased",
    hidden_dropout_prob=dropout_rate,
    attention_probs_dropout_prob=dropout_rate,
    num_labels=5
)
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", config=config)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Optimizer and scheduler
learning_rate = 2e-05
optimizer = torch.optim.AdamW(
    model.parameters(), lr=learning_rate, weight_decay=0.01)
total_steps = len(train_dataloader) * 3  # Assuming 3 epochs
scheduler = get_scheduler(
    "linear", optimizer=optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
)

# Compute class weights
unique_classes = torch.unique(train_labels)
class_weights = compute_class_weight(
    class_weight="balanced",
    classes=unique_classes.cpu().numpy(),
    y=train_labels.cpu().numpy()
)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

# Loss function
loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

# Training loop
epochs = 4
best_val_loss = float("inf")
train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []
f1_scores_per_epoch = []

for epoch in range(epochs):
    model.train()
    total_loss = 0
    correct_preds = 0
    total_samples = 0

    for batch in train_dataloader:
        optimizer.zero_grad()
        input_ids, attention_mask, labels = batch[0].to(
            device), batch[1].to(device), batch[2].to(device)

        # Forward pass
        outputs = model(input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        loss = loss_fn(logits, labels)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        correct_preds += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += labels.size(0)

    train_loss = total_loss / len(train_dataloader)
    train_accuracy = correct_preds / total_samples
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)

    # Validation step
    model.eval()
    val_loss = 0
    val_correct_preds = 0
    val_samples = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in val_dataloader:
            input_ids, attention_mask, labels = batch[0].to(
                device), batch[1].to(device), batch[2].to(device)
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            loss = loss_fn(logits, labels)

            val_loss += loss.item()
            val_correct_preds += (logits.argmax(dim=1) == labels).sum().item()
            val_samples += labels.size(0)
            all_preds.extend(logits.argmax(dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    val_loss /= len(val_dataloader)
    val_accuracy = val_correct_preds / val_samples
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)

    # Calculate F1-score for each class
    report = classification_report(all_labels, all_preds, output_dict=True)
    f1_scores = [report[str(i)]["f1-score"] for i in range(5)]
    f1_scores_per_epoch.append(f1_scores)

    print(f"Epoch {epoch + 1}/{epochs}")
    print(
        f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")
    print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

    # Save best model
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), os.path.join(
            output_dir, "best_model.pth"))

# Save plots
# 1. Loss and Accuracy Plot
plt.figure(figsize=(8, 6))
plt.plot(range(1, epochs + 1), train_losses, label="Train Loss")
plt.plot(range(1, epochs + 1), val_losses, label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.savefig(os.path.join(output_dir, "loss_plot.png"))

plt.figure(figsize=(8, 6))
plt.plot(range(1, epochs + 1), train_accuracies, label="Train Accuracy")
plt.plot(range(1, epochs + 1), val_accuracies, label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")
plt.legend()
plt.savefig(os.path.join(output_dir, "accuracy_plot.png"))

# 2. Confusion Matrix
conf_matrix = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))

# 3. ROC-AUC Curves
plt.figure(figsize=(10, 8))
for i in range(5):
    y_true = np.array(all_labels) == i
    y_score = np.array(all_preds) == i
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"Class {i} (AUC = {roc_auc:.2f})")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC-AUC Curves")
plt.legend()
plt.savefig(os.path.join(output_dir, "roc_auc_curve.png"))

# 4. Precision-Recall Curves (Per Class)
plt.figure(figsize=(10, 8))
for i in range(5):
    y_true = np.array(all_labels) == i
    y_score = np.array(all_preds) == i
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    plt.plot(recall, precision, label=f"Class {i}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.savefig(os.path.join(output_dir, "precision_recall_curve.png"))

# 5. F1-Score Trend
plt.figure(figsize=(10, 8))
for i in range(5):
    f1_scores = [epoch_f1[i] for epoch_f1 in f1_scores_per_epoch]
    plt.plot(range(1, epochs + 1), f1_scores, label=f"Class {i}")
plt.xlabel("Epochs")
plt.ylabel("F1-Score")
plt.title("F1-Score Trend per Class")
plt.legend()
plt.savefig(os.path.join(output_dir, "f1_score_trend.png"))

# Print and Save Classification Report
print(pd.DataFrame(report).transpose())
pd.DataFrame(report).transpose().to_csv(
    os.path.join(output_dir, "classification_report.csv"))
