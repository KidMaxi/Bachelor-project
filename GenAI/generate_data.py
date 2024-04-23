import pandas as pd
from transformers import pipeline
from tqdm import tqdm
import nlpaug.augmenter.word as naw

# Load your dataset
file_path = './Data/AI_Dataset__ArtikelBuilding_maart_2024_V1_relevant.csv'
df = pd.read_csv(file_path)

# Initialize the augmenter
# Synonym augmenter, which replaces words with their synonyms based on WordNet
aug = naw.SynonymAug(aug_src='wordnet')

augmented_texts = []
labels = []

for index, row in df.iterrows():
    original_text = row['OmsNederlands']
    label = row['L5_ARTIKELTYPE']

    # Augment the text
    augmented_text = aug.augment(original_text)

    # Save the augmented text and the original label
    augmented_texts.append(augmented_text)
    labels.append(label)

# Create a DataFrame with the augmented data
df_augmented = pd.DataFrame({
    'OmsNederlands': augmented_texts,
    'L5_ARTIKELTYPE': labels
})

# Combine the original and augmented data
df_combined = pd.concat([df, df_augmented], ignore_index=True)

# Save the augmented dataset to a new CSV file
df_combined.to_csv('augmented_dataset.csv', index=False)

print("Augmented dataset saved.")
