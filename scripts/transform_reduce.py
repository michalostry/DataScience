import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy import sparse

# Setting to specify how many processed files to load
process_limit = None  # Set to None to process all files, or set a specific number
# Setting to specify how many processed files to load per chunk
chunk_size = 2000  # Adjust this based on your memory capacity
processing_number = 0

# Paths to directories
processed_dir = '../data/processed/'
processed_finished = '../data/transform'
os.makedirs(processed_finished, exist_ok=True)

# List of files to process
file_list = os.listdir(processed_dir)

# Limit the number of files to process if specified
if process_limit is not None:
    file_list = file_list[:process_limit]

# First, create a combined vocabulary by fitting on a subset of data or the whole dataset
print("Fitting TF-IDF Vectorizer to establish a common vocabulary...")
combined_documents = []
for filename in file_list:
    processing_number += 1
    print(f'[{processing_number}/{len(file_list)}] Reading {filename}...')
    if filename.endswith(".txt"):
        file_path = os.path.join(processed_dir, filename)
        with open(file_path, 'r', encoding='utf-8') as processed_file:
            combined_documents.append(processed_file.read())
    if processing_number >= chunk_size:  # Optionally stop early for performance
        break

# Fit the vectorizer on the combined documents to get the vocabulary
vectorizer = TfidfVectorizer(max_df=0.85, min_df=2)
vectorizer.fit(combined_documents)
print("Vocabulary established.")


tfidf_chunks = []
metadata_chunks = []

# processing in chunks, because of memory issues
for i in range(0, len(file_list), chunk_size):
    chunk_files = file_list[i:i + chunk_size]
    chunk_documents = []
    chunk_metadata = []

    for filename in chunk_files:
        processing_number += 1
        print(f'[{processing_number}/{len(file_list)}] Processing {filename}')
        if filename.endswith(".txt"):
            file_parts = filename.split('_')
            redizo = file_parts[0]
            year = file_parts[1]

            chunk_metadata.append({'redizo': redizo, 'year': year})

            file_path = os.path.join(processed_dir, filename)
            with open(file_path, 'r', encoding='utf-8') as processed_file:
                chunk_documents.append(processed_file.read())

    metadata_df = pd.DataFrame(chunk_metadata)

    print("Transforming chunk", i // chunk_size + 1, "with established vocabulary.")
    X_chunk = vectorizer.transform(chunk_documents)

    print("Saving TF-IDF chunk", i // chunk_size + 1)
    sparse.save_npz(f"{processed_finished}/tfidf_chunk_{i}.npz", X_chunk)
    metadata_df.to_csv(f"{processed_finished}/metadata_chunk_{i}.csv", index=False)

    tfidf_chunks.append(f"{processed_finished}/tfidf_chunk_{i}.npz")
    metadata_chunks.append(f"{processed_finished}/metadata_chunk_{i}.csv")

# After processing all chunks, load and concatenate
print("Concatenating all chunks...")
tfidf_matrices = [sparse.load_npz(file) for file in tfidf_chunks]
metadata_frames = [pd.read_csv(file) for file in metadata_chunks]

X_full = sparse.vstack(tfidf_matrices)
metadata_full = pd.concat(metadata_frames, axis=0).reset_index(drop=True)

# Save full TF-IDF matrix before PCA
sparse.save_npz(f"{processed_finished}/tfidf_full.npz", X_full)

# Step 3: Dimensionality Reduction using PCA
# Calculate the maximum number of components that can be used for PCA
max_components = min(X_full.shape[0], X_full.shape[1])
max_components = 250

# Adjust PCA to use this value
pca = PCA(n_components=min(200, max_components))
X_reduced = pca.fit_transform(X_full.toarray())

# Adjust the column names to match the number of components
n_components = X_reduced.shape[1]
feature_df = pd.DataFrame(X_reduced, columns=[f'PC{i}' for i in range(1, n_components + 1)])

# Step 4: Save the resulting features with redizo and year
feature_df = pd.concat([metadata_full, feature_df], axis=1)
feature_df.to_csv(os.path.join(processed_finished, 'processed_features.csv'), index=False)

print("Feature extraction and dimensionality reduction complete.")

# # Step 5: Visualize the explained variance by each component
# explained_variance = pca.explained_variance_ratio_
# print(f"Explained variance by each component: {explained_variance}")
# print(f"Cumulative explained variance: {explained_variance.cumsum()}")

# # Step 5a: Scree Plot (Visualizing the explained variance by each component)
# plt.figure(figsize=(8, 6))
# plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o', linestyle='--')
# plt.title('Scree Plot')
# plt.xlabel('Principal Component')
# plt.ylabel('Explained Variance')
# plt.show()

# Step 5b: Cumulative Explained Variance Plot
cumulative_variance = explained_variance.cumsum()

plt.figure(figsize=(8, 6))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='--', color='orange')
plt.title('Cumulative Explained Variance')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.savefig('plots/Cumulative Explained Variance.png')
plt.show()

# # Step 5c: 2D PCA Scatter Plot (Visualizing the data in the space of the first two principal components)
# plt.figure(figsize=(8, 6))
# plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c='blue', edgecolor='k', s=50)
# plt.title('2D PCA Scatter Plot')
# plt.xlabel('Principal Component 1')
# plt.ylabel('Principal Component 2')
# plt.grid(True)
# plt.show()

# # Step 5d: Pairplot of the first few principal components
# pca_df = pd.DataFrame(X_reduced[:, :5], columns=[f'PC{i}' for i in range(1, 6)])
# sns.pairplot(pca_df)
# plt.suptitle('Pairplot of First 5 Principal Components')
# plt.show()
#
# print("Data transformation and visualization complete.")