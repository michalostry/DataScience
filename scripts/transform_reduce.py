import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Setting to specify how many processed files to load
process_limit = None  # Set to None to process all files, or set a specific number
processing_number = 0

# Paths to directories
processed_dir = '../data/processed/'
processed_finished = '../data/'
os.makedirs(processed_finished, exist_ok=True)

# Step 1: Load the processed text files and metadata (redizo and year)
metadata = []
documents = []
file_list = os.listdir(processed_dir)

# Limit the number of files to process if specified
if process_limit is not None:
    file_list = file_list[:process_limit]

for idx, filename in enumerate(file_list):
    processing_number = processing_number + 1
    print(processing_number,"/",process_limit, f'| Processing {filename}')
    if filename.endswith(".txt"):
        # Assuming the filename format is: REDIZO_year_schoolname_inspection_report.txt
        file_parts = filename.split('_')
        redizo = file_parts[0]
        year = file_parts[1]

        metadata.append({'redizo': redizo, 'year': year})

        file_path = os.path.join(processed_dir, filename)
        with open(file_path, 'r', encoding='utf-8') as processed_file:
            documents.append(processed_file.read())

metadata_df = pd.DataFrame(metadata)

print("Step 2: TF-IDF Transformation")
# Step 2: TF-IDF Transformation (Converting text to numerical features)
vectorizer = TfidfVectorizer(max_df=0.85, min_df=2)
X = vectorizer.fit_transform(documents)

print("Save the TF-IDF matrix before PCA")
# Save the TF-IDF matrix before PCA
tfidf_df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
tfidf_df = pd.concat([metadata_df, tfidf_df], axis=1)
#tfidf_df.to_csv(os.path.join(processed_finished, 'tfidf_features.csv'), index=False)

# Step 3: Dimensionality Reduction using PCA
# Calculate the maximum number of components that can be used for PCA
max_components = min(X.shape[0], X.shape[1])
max_components = 250

#max also to 50
# Adjust PCA to use this value
pca = PCA(n_components=min(200, max_components))
X_reduced = pca.fit_transform(X.toarray())

# Adjust the column names to match the number of components
n_components = X_reduced.shape[1]
feature_df = pd.DataFrame(X_reduced, columns=[f'PC{i}' for i in range(1, n_components + 1)])

# Step 4: Save the resulting features with redizo and year
feature_df = pd.concat([metadata_df, feature_df], axis=1)
feature_df.to_csv(os.path.join(processed_finished, 'processed_features.csv'), index=False)

print("Feature extraction and dimensionality reduction complete.")

# Step 5: Visualize the explained variance by each component
explained_variance = pca.explained_variance_ratio_
print(f"Explained variance by each component: {explained_variance}")
print(f"Cumulative explained variance: {explained_variance.cumsum()}")

# Step 5a: Scree Plot (Visualizing the explained variance by each component)
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(explained_variance) + 1), explained_variance, marker='o', linestyle='--')
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance')
plt.show()

# Step 5b: Cumulative Explained Variance Plot
cumulative_variance = explained_variance.cumsum()

plt.figure(figsize=(8, 6))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='--', color='orange')
plt.title('Cumulative Explained Variance')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance')
plt.show()

# Step 5c: 2D PCA Scatter Plot (Visualizing the data in the space of the first two principal components)
plt.figure(figsize=(8, 6))
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c='blue', edgecolor='k', s=50)
plt.title('2D PCA Scatter Plot')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.grid(True)
plt.show()

# Step 5d: Pairplot of the first few principal components
pca_df = pd.DataFrame(X_reduced[:, :5], columns=[f'PC{i}' for i in range(1, 6)])
sns.pairplot(pca_df)
plt.suptitle('Pairplot of First 5 Principal Components')
plt.show()

print("Data transformation and visualization complete.")
