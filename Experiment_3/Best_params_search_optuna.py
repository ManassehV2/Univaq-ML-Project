import joblib
import optuna
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertForSequenceClassification, AdamW, get_scheduler
from sklearn.metrics import accuracy_score
import os

# Ensure output directory exists
output_dir = "results_exp_3_optuna"
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

# Define the Optuna objective function


def objective(trial):
    # Suggest hyperparameters
    learning_rate = trial.suggest_float("learning_rate", 1e-6, 5e-5, log=True)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    dropout_rate = trial.suggest_float("dropout_rate", 0.1, 0.5)

    # Prepare dataloaders
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    # Configure BERT model
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=5)
    model.config.hidden_dropout_prob = dropout_rate
    model.config.attention_probs_dropout_prob = dropout_rate
    model.to("cuda")

    # Optimizer and Scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    num_training_steps = len(train_dataloader) * 3  # Assuming 3 epochs
    scheduler = get_scheduler(
        "linear", optimizer=optimizer, num_warmup_steps=0.1 * num_training_steps, num_training_steps=num_training_steps
    )

    # Training loop
    model.train()
    for epoch in range(3):  # Fixed number of epochs
        for batch in train_dataloader:
            # Unpack the batch and move tensors to GPU
            input_ids, attention_mask, labels = [b.to("cuda") for b in batch]
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()

    # Validation loop
    model.eval()
    val_preds, val_labels_all = [], []
    with torch.no_grad():
        for batch in val_dataloader:
            input_ids, attention_mask, labels = [b.to("cuda") for b in batch]
            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask, labels=labels)
            logits = outputs.logits
            val_preds.extend(torch.argmax(logits, dim=-1).cpu().numpy())
            val_labels_all.extend(labels.cpu().numpy())

    # Compute validation accuracy
    accuracy = accuracy_score(val_labels_all, val_preds)
    return accuracy


# Run Optuna optimization
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)

# Print best parameters and accuracy
print("Best Parameters:", study.best_params)
print("Best Validation Accuracy:", study.best_value)

# Save the study for future use
joblib.dump(study, os.path.join(output_dir, "optuna_study.pkl"))

# Save visualizations as images
try:
    import optuna.visualization as viz
    from optuna.visualization.matplotlib import plot_optimization_history, plot_param_importances

    # Save optimization history
    fig1 = plot_optimization_history(study)
    fig1.savefig(os.path.join(output_dir, "optimization_history.png"))

    # Save parameter importances
    fig2 = plot_param_importances(study)
    fig2.savefig(os.path.join(output_dir, "param_importances.png"))

    print("Visualizations saved in", output_dir)
except ImportError:
    print("Optuna visualization libraries not installed.")
