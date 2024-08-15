import pandas as pd

# Load the CSV data
data_path = 'C:/++4630/++ University, education/03 -- Tilburg_v02/Data Science/data/1_index_minimalist.csv'
data = pd.read_csv(data_path)

# Rename 'red_izo' to 'redizo' to match the processed features dataset
data = data.rename(columns={'red_izo': 'redizo'})

# Filter out special schools (spec != 1) and rows where nschool != 0
filtered_data = data[(data['spec'] != 1) & (data['nschool'] == 0)]


# Convert 'redizo' to string for consistency
filtered_data['redizo'] = filtered_data['redizo'].astype(str)

# Load the processed PCA features
processed_pca_features_path = 'C:/++4630/++ University, education/03 -- Tilburg_v02/Data Science/data/processed_features.csv'
processed_pca_features = pd.read_csv(processed_pca_features_path)
# Load the processed TF-IDF features (full dimensions)
#processed_tfidf_features_path = 'C:/++4630/++ University, education/03 -- Tilburg_v02/Data Science/data/tfidf_features.csv'
#processed_tfidf_features = pd.read_csv(processed_tfidf_features_path)


# Convert 'redizo' to string for consistency
processed_pca_features['redizo'] = processed_pca_features['redizo'].astype(str)
#processed_tfidf_features['redizo'] = processed_tfidf_features['redizo'].astype(str)
# Merge the index values onto the processed PCA features using 'redizo' as the key
merged_pca_data = pd.merge(processed_pca_features, filtered_data, on='redizo', how='right')
#merged_tfidf_data = pd.merge(processed_tfidf_features, filtered_data, on='redizo', how='right')

# Filter out rows with missing 'value' column to ensure data completeness
merged_pca_data = merged_pca_data.dropna(subset=['value'])
merged_pca_data = merged_pca_data.dropna(subset=['PC1'])
#merged_tfidf_data = merged_tfidf_data.dropna(subset=['value'])

# Check the resulting complete dataset
print("Complete PCA Data Sample:")
print(merged_pca_data.head())
#print("Complete TFIDF Data Sample:")
#print(merged_tfidf_data.head())

# Optionally, save the data for further analysis
merged_pca_data.to_csv('C:/++4630/++ University, education/03 -- Tilburg_v02/Data Science/data/complete_merged_data.csv', index=False)
#merged_tfidf_data.to_csv('C:/++4630/++ University, education/03 -- Tilburg_v02/Data Science/data/complete_merged_data.csv', index=False)

print("Data merging and filtering complete.")


# #######################################
# ######################################
# ### ANALYSIS ####
#
# import pandas as pd                # Data manipulation and analysis
# import numpy as np                 # Numerical operations
# import matplotlib.pyplot as plt    # Plotting and visualizations
# import seaborn as sns              # Enhanced data visualization
# from sklearn.model_selection import train_test_split  # Data splitting for training/testing
# from sklearn.linear_model import LinearRegression     # Linear regression model
# from sklearn.metrics import mean_squared_error, r2_score  # Model evaluation metrics
# from sklearn.decomposition import PCA                # Principal Component Analysis (PCA)
# from sklearn.linear_model import LassoCV
#
#
# # Assuming complete_pca_data is already loaded and prepared
#
# # Select only numeric columns
# numeric_columns = complete_pca_data.select_dtypes(include=[np.number])
#
# # Correlation with 'value'
# correlation_with_value = numeric_columns.corr()['value'].drop('value')
# print(correlation_with_value)
#
# # Define the ranges for different PC groups
# pc_ranges = [(1, 10), (11, 20), (21, 30), (31, 40), (41, 50)]
#
# # Plot the correlations in groups
# for start, end in pc_ranges:
#     plt.figure(figsize=(10, 6))
#     sns.barplot(x=correlation_with_value[f'PC{start}':f'PC{end}'].index,
#                 y=correlation_with_value[f'PC{start}':f'PC{end}'])
#     plt.title(f'Correlation between Value and PC{start}-{end}')
#     plt.xlabel('Principal Component')
#     plt.ylabel('Correlation with Value')
#     plt.xticks(rotation=45)
#     plt.show()
#
# ########### LASSO
#
# # Define features (PCs) and target variable (value)
# X = complete_pca_data.drop(columns=['redizo', 'year_x', 'spec', 'value', 'bound', 'nschool'])
# y = complete_pca_data['value']
#
# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# # Initialize Lasso model with cross-validation to find the best alpha
# lasso = LassoCV(cv=5, random_state=42).fit(X_train, y_train)
#
# # Predict on the test set
# y_pred = lasso.predict(X_test)
#
# # Evaluate the model
# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)
#
# print(f"Mean Squared Error (MSE): {mse}")
# print(f"R-squared (R2): {r2}")
# print(f"Optimal Alpha: {lasso.alpha_}")
#
# # Plot the true vs predicted values
# plt.figure(figsize=(8, 6))
# plt.scatter(y_test, y_pred, edgecolor='k', alpha=0.7)
# plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2, color='red')
# plt.xlabel('True Value')
# plt.ylabel('Predicted Value')
# plt.title('True vs Predicted Values (Lasso Regression)')
# plt.show()
#
# # Plot the coefficients to see which features are selected
# plt.figure(figsize=(10, 6))
# plt.plot(range(len(lasso.coef_)), lasso.coef_, marker='o')
# plt.title('Lasso Coefficients')
# plt.xlabel('Feature Index')
# plt.ylabel('Coefficient Value')
# plt.show()