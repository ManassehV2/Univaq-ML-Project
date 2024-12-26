import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertForSequenceClassification, AdamW, get_scheduler
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, roc_curve, auc, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from sklearn.preprocessing import label_binarize

# Ensure output directory exists
output_dir = "results_exp_3_final_optuna_three"
os.makedirs(output_dir, exist_ok=True)

# Load Augmented Data
data = torch.load(
    "preprocessed_data_aug_fr_it_and_de_rating_4.pt", weights_only=True)
train_inputs = data["train_inputs"]
val_inputs = data["val_inputs"]
train_masks = data["train_masks"]
val_masks = data["val_masks"]
train_labels = data["train_labels"]
val_labels = data["val_labels"]

# Create datasets
train_dataset = TensorDataset(train_inputs, train_masks, train_labels)
val_dataset = TensorDataset(val_inputs, val_masks, val_labels)

# Optimal Hyperparameters
best_learning_rate = 4.7758653044470604e-05
best_batch_size = 8
best_dropout_rate = 0.23586043586075478
num_epochs = 2

# Prepare dataloaders
train_dataloader = DataLoader(
    train_dataset, batch_size=best_batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=best_batch_size)

# Configure BERT model with optimal dropout rate
model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=5)
model.config.hidden_dropout_prob = best_dropout_rate
model.config.attention_probs_dropout_prob = best_dropout_rate
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Optimizer and Scheduler
optimizer = AdamW(model.parameters(), lr=best_learning_rate)
num_training_steps = len(train_dataloader) * num_epochs
scheduler = get_scheduler(
    "linear", optimizer=optimizer, num_warmup_steps=0.1 * num_training_steps, num_training_steps=num_training_steps
)

# Training loop
train_losses, val_losses, train_accuracies, val_accuracies = [], [], [], []
model.train()

for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}/{num_epochs}")
    total_loss, correct_preds, total_samples = 0, 0, 0

    for batch in train_dataloader:
        optimizer.zero_grad()
        input_ids, attention_mask, labels = [b.to(device) for b in batch]
        outputs = model(input_ids=input_ids,
                        attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        logits = outputs.logits

        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        correct_preds += (logits.argmax(dim=-1) == labels).sum().item()
        total_samples += labels.size(0)

    train_loss = total_loss / len(train_dataloader)
    train_accuracy = correct_preds / total_samples
    train_losses.append(train_loss)
    train_accuracies.append(train_accuracy)
    print(
        f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")

    # Validation loop
    model.eval()
    val_loss, val_correct_preds, val_total_samples = 0, 0, 0
    all_preds, all_labels = [], []
    all_logits = []

    with torch.no_grad():
        for batch in val_dataloader:
            input_ids, attention_mask, labels = [b.to(device) for b in batch]
            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            val_loss += loss.item()
            val_correct_preds += (logits.argmax(dim=-1) == labels).sum().item()
            val_total_samples += labels.size(0)

            all_preds.extend(logits.argmax(dim=-1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_logits.extend(logits.cpu().numpy())

    val_loss /= len(val_dataloader)
    val_accuracy = val_correct_preds / val_total_samples
    val_losses.append(val_loss)
    val_accuracies.append(val_accuracy)
    print(
        f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")

# Save loss and accuracy plots
epochs = range(1, len(train_losses) + 1)

plt.figure(figsize=(8, 6))
plt.plot(epochs, train_losses, label="Training Loss")
plt.plot(epochs, val_losses, label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Validation Loss")
plt.legend()
plt.savefig(os.path.join(output_dir, "train_val_loss.png"))
plt.close()

plt.figure(figsize=(8, 6))
plt.plot(epochs, train_accuracies, label="Training Accuracy")
plt.plot(epochs, val_accuracies, label="Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.title("Training and Validation Accuracy")
plt.legend()
plt.savefig(os.path.join(output_dir, "train_val_accuracy.png"))
plt.close()

# Save confusion matrix
conf_matrix = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues",
            xticklabels=range(5), yticklabels=range(5))
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
plt.close()

# Generate precision-recall and ROC-AUC curves
y_true = label_binarize(all_labels, classes=range(5))  # One-hot encode labels
y_scores = np.array(all_logits)

# Precision-Recall Curve
plt.figure(figsize=(10, 8))
for i in range(5):
    precision, recall, _ = precision_recall_curve(y_true[:, i], y_scores[:, i])
    plt.plot(recall, precision, label=f"Class {i}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.savefig(os.path.join(output_dir, "precision_recall_curve.png"))
plt.close()

# ROC-AUC Curve
plt.figure(figsize=(10, 8))
for i in range(5):
    fpr, tpr, _ = roc_curve(y_true[:, i], y_scores[:, i])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f"Class {i} (AUC = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], 'k--')  # Diagonal line
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.savefig(os.path.join(output_dir, "roc_curve.png"))
plt.close()

# Save classification report
report = classification_report(all_labels, all_preds, target_names=[
                               str(i) for i in range(5)])
print(report)

with open(os.path.join(output_dir, "classification_report.txt"), "w") as f:
    f.write(report)

print("Training completed. All results saved in the output directory.")
