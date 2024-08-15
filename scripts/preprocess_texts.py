import os
import ufal.morphodita as morphodita

# Setting to specify how many reports to process
report_limit = 20000  # You can change this value to process more or fewer reports

# Paths to directories
text_files_dir = '../data/csi_reports/'
processed_dir = '../data/processed/'
os.makedirs(processed_dir, exist_ok=True)

# Step 1: Initialize Morphodita for Lemmatization and POS Tagging
model_path = '../models/czech-morfflex2.0-pdtc1.0-220710/czech-morfflex2.0-pdtc1.0-220710-no_dia.tagger'
tagger = morphodita.Tagger.load(model_path)
if tagger is None:
    print("Failed to load Morphodita model")
    exit()

tokenizer = tagger.newTokenizer()

# Step 2: Preprocessing loop for each document (Tokenization, Lemmatization, POS Filtering)
documents = []
for idx, filename in enumerate(os.listdir(text_files_dir)):
    if filename.endswith(".txt"):
        # Stop processing if the report limit is reached
        if idx >= report_limit:
            break

        # Set the path for the processed file with the same name as the original
        processed_file_path = os.path.join(processed_dir, filename)

        # Check if the file has already been processed and skip if it has
        if os.path.exists(processed_file_path):
            print(f"File {filename} already processed. Skipping...")
            continue

        file_path = os.path.join(text_files_dir, filename)
        with open(file_path, 'r', encoding='utf-8') as text_file:
            text = text_file.read().split('\n', 2)[2]

            # Lemmatize and filter by POS (nouns, adjectives, verbs)
            processed_text = []
            tokenizer.setText(text)
            forms = morphodita.Forms()
            token_ranges = morphodita.TokenRanges()
            while tokenizer.nextSentence(forms, token_ranges):
                tagged_lemmas = morphodita.TaggedLemmas()
                tagger.tag(forms, tagged_lemmas)
                for i in range(len(forms)):
                    lemma = tagged_lemmas[i].lemma
                    pos = tagged_lemmas[i].tag[:1]  # Get POS (first character)
                    if pos in ['N', 'A', 'V']:  # Nouns, Adjectives, Verbs
                        processed_text.append(lemma)

            print(f"Processing report {idx + 1}/{report_limit}")
            documents.append(' '.join(processed_text))

            # Save the processed text with the same file name as the original
            with open(processed_file_path, 'w', encoding='utf-8') as processed_file:
                processed_file.write(' '.join(processed_text))

print("Text preprocessing complete.")
