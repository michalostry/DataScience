import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV
from sklearn.metrics import mean_squared_error, r2_score

# Load the CSV data
data_path = 'C:/++4630/++ University, education/03 -- Tilburg_v02/Data Science/data/raw/1_index_minimalist.csv'
data = pd.read_csv(data_path)

# Rename 'red_izo' to 'redizo' to match the processed features dataset
data = data.rename(columns={'red_izo': 'redizo'})

# Filter out special schools (spec != 1) and rows where nschool != 0 >> to get rid of values for neighboring schools
filtered_data = data[(data['spec'] != 1) & (data['nschool'] == 0)]

# Convert 'redizo' to string for consistency
filtered_data['redizo'] = filtered_data['redizo'].astype(str)

# Load the processed PCA features
processed_pca_features_path = 'C:/++4630/++ University, education/03 -- Tilburg_v02/Data Science/data/transform/processed_features.csv'
processed_pca_features = pd.read_csv(processed_pca_features_path)

# Convert 'redizo' to string for consistency
processed_pca_features['redizo'] = processed_pca_features['redizo'].astype(str)

# Calculate the distance in years and match the closest report
def closest_report(row, reports):
    school_reports = reports[reports['redizo'] == row['redizo']]
    if not school_reports.empty:
        school_reports['year_difference'] = abs(school_reports['year'] - row['year'])
        closest_report = school_reports.loc[school_reports['year_difference'].idxmin()]
        return pd.Series([closest_report['year'], closest_report['year_difference']] + closest_report.iloc[2:].tolist())
    return pd.Series([np.nan, np.nan] + [np.nan] * (len(reports.columns) - 2))

# Apply the closest_report function to the dataset
print("Applying closest report matching...")
report_values = filtered_data.apply(closest_report, axis=1, reports=processed_pca_features)

# Add the new columns to the filtered_data DataFrame
filtered_data = pd.concat([filtered_data, report_values], axis=1)

# Rename the columns for consistency
filtered_data.rename(columns={
    filtered_data.columns[6]: 'inspection_year',
    filtered_data.columns[7]: 'inspection_value_distance',
    **{filtered_data.columns[i]: f'PC{i-7}' for i in range(8, 208)}
}, inplace=True)

# Filter out rows with missing 'value' or 'matched_year' to ensure data completeness
filtered_data = filtered_data.dropna(subset=['value', 'year', 'PC1'])



# Drop the redundant last column
filtered_data.drop(columns=[filtered_data.columns[208]], inplace=True)

# Save the complete dataset without applying any filter
print("Saving complete dataset without filter...")
filtered_data.to_csv('C:/++4630/++ University, education/03 -- Tilburg_v02/Data Science/data/complete_all_merged_data_with_distance.csv', index=False)

# Apply the filtering criteria for certain disadvantaged or not disadvantaged schools
filtered_certain_data = filtered_data[((filtered_data['bound'] == 'lower') & (filtered_data['value'] > 0.83)) |
                                      ((filtered_data['bound'] == 'upper') & (filtered_data['value'] < 0.5))]

# Save the filtered dataset
print("Saving filtered dataset...")
filtered_certain_data.to_csv('C:/++4630/++ University, education/03 -- Tilburg_v02/Data Science/data/complete_certain_merged_data_with_distance.csv', index=False)

print("Data merging, filtering, and distance calculation complete.")

# Print the number of rows in the datasets
num_rows = len(filtered_data)
print(f"The unfiltered DataFrame has {num_rows} rows.")
num_rows_filtered = len(filtered_certain_data)
print(f"The filtered DataFrame has {num_rows_filtered} rows.")



########################################
######################################
### ANALYSIS ####

# Select only numeric columns
numeric_columns = filtered_certain_data.select_dtypes(include=[np.number])

# Correlation with 'value'
correlation_with_value = numeric_columns.corr()['value'].drop('value')
print(correlation_with_value)

# Define the ranges for different PC groups
pc_ranges = [(1, 50), (51, 100), (101, 150), (151, 200)]

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

########### LASSO

# Define features (PCs) and target variable (value)
X = filtered_certain_data.drop(columns=['redizo', 'year', 'spec', 'value', 'bound', 'nschool'])
y = filtered_certain_data['value']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize Lasso model with cross-validation to find the best alpha
lasso = LassoCV(cv=5, random_state=42).fit(X_train, y_train)

# Predict on the test set
y_pred = lasso.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R2): {r2}")
print(f"Optimal Alpha: {lasso.alpha_}")

# Plot the true vs predicted values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, edgecolor='k', alpha=0.7)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2, color='red')
plt.xlabel('True Value')
plt.ylabel('Predicted Value')
plt.title('True vs Predicted Values (Lasso Regression)')
plt.savefig('plots/True vs Predicted Values (Lasso Regression).png')
plt.show()

# # Plot the coefficients to see which features are selected
# plt.figure(figsize=(10, 6))
# plt.plot(range(len(lasso.coef_)), lasso.coef_, marker='o')
# plt.title('Lasso Coefficients')
# plt.xlabel('Feature Index')
# plt.ylabel('Coefficient Value')
# plt.show()