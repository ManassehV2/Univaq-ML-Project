import pandas as pd
import re
from transformers import MarianMTModel, MarianTokenizer
import torch
from tqdm import tqdm
import seaborn as sns
from matplotlib import pyplot as plt
import os

# Ensure Output Directory Exists
output_dir = "output_download_models"
os.makedirs(output_dir, exist_ok=True)

# Load Dataset
print("Loading dataset...")
df = pd.read_csv(
    'https://raw.githubusercontent.com/ManassehV2/Univaq-ML-Project/refs/heads/master/data/tripadvisor_hotel_reviews.csv')

# Preprocessing Function


def preprocess_text(text):
    """
    Preprocesses the text by cleaning and normalizing:
    - Converts text to lowercase
    - Removes special characters and extra whitespaces
    """
    text = text.lower()  # Convert to lowercase
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)  # Remove special characters
    text = re.sub(r"\s+", " ", text).strip()  # Remove excessive whitespaces
    return text


# Apply Preprocessing
print("Preprocessing text...")
df['Review'] = df['Review'].apply(preprocess_text)

# Save Preprocessed Data
preprocessed_path = os.path.join(output_dir, "preprocessed_data.csv")
df.to_csv(preprocessed_path, index=False)
print(f"Preprocessed data saved to {preprocessed_path}")

# Plot Class Distribution


def plot_class_distribution(df, title, save_path):
    """
    Plots the distribution of classes in the dataset.
    """
    plt.figure(figsize=(8, 6))
    ax = sns.countplot(x=df['Rating'])
    for container in ax.containers:
        ax.bar_label(container)  # Annotate bars with counts
    plt.title(title)
    plt.xlabel('Rating')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# Save Old Distribution
old_distribution_path = os.path.join(output_dir, "old_class_distribution.png")
plot_class_distribution(
    df, "Original Class Distribution", old_distribution_path)
print(f"Old class distribution plot saved to {old_distribution_path}")

# Back-Translation Setup


def load_translation_models():
    """
    Load MarianMT models directly from Hugging Face.
    """
    tokenizer_en_to_fr = MarianTokenizer.from_pretrained(
        "Helsinki-NLP/opus-mt-en-fr")
    model_en_to_fr = MarianMTModel.from_pretrained(
        "Helsinki-NLP/opus-mt-en-fr")

    tokenizer_fr_to_en = MarianTokenizer.from_pretrained(
        "Helsinki-NLP/opus-mt-fr-en")
    model_fr_to_en = MarianMTModel.from_pretrained(
        "Helsinki-NLP/opus-mt-fr-en")

    return tokenizer_en_to_fr, model_en_to_fr, tokenizer_fr_to_en, model_fr_to_en


# Load Translation Models
print("Downloading translation models...")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
tokenizer_en_to_fr, model_en_to_fr, tokenizer_fr_to_en, model_fr_to_en = load_translation_models()
model_en_to_fr.to(device)
model_fr_to_en.to(device)

# Back-Translation Function


def back_translate(text, tokenizer_en_to_fr, model_en_to_fr, tokenizer_fr_to_en, model_fr_to_en):
    """
    Perform back-translation: English -> French -> English.
    """
    tokens_en_to_fr = tokenizer_en_to_fr(
        text, return_tensors="pt", padding=True, truncation=True).to(device)
    french_text = model_en_to_fr.generate(**tokens_en_to_fr)
    french_text = tokenizer_en_to_fr.batch_decode(
        french_text, skip_special_tokens=True)[0]

    tokens_fr_to_en = tokenizer_fr_to_en(
        french_text, return_tensors="pt", padding=True, truncation=True).to(device)
    back_translated_text = model_fr_to_en.generate(**tokens_fr_to_en)
    return tokenizer_fr_to_en.batch_decode(back_translated_text, skip_special_tokens=True)[0]


# Apply Back-Translation to Minority Classes
print("Applying back-translation...")
minority_classes = [1, 2, 3]  # Underrepresented classes
minority_data = df[df['Rating'].isin(minority_classes)].copy()

tqdm.pandas()  # Enable progress bar
minority_data['Review'] = minority_data['Review'].progress_apply(
    lambda x: back_translate(
        x, tokenizer_en_to_fr, model_en_to_fr, tokenizer_fr_to_en, model_fr_to_en)
)

# Combine Original and Augmented Data
augmented_df = pd.concat([df, minority_data])
augmented_df = augmented_df.sample(frac=1).reset_index(drop=True)

# Save New Distribution
new_distribution_path = os.path.join(output_dir, "new_class_distribution.png")
plot_class_distribution(
    augmented_df, "Augmented Class Distribution", new_distribution_path)
print(f"New class distribution plot saved to {new_distribution_path}")

# Save Augmented Data
augmented_path = os.path.join(output_dir, "augmented_with_backtranslation.csv")
augmented_df.to_csv(augmented_path, index=False)
print(f"Augmented dataset saved to {augmented_path}")
