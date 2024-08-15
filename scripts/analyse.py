import pandas as pd
import numpy as np

# Load the CSV data
data_path = 'C:/++4630/++ University, education/03 -- Tilburg_v02/Data Science/data/1_index_minimalist.csv'
data = pd.read_csv(data_path)

# Rename 'red_izo' to 'redizo' to match the processed features dataset
data = data.rename(columns={'red_izo': 'redizo'})

# Filter out special schools (spec != 1) and rows where nschool != 0 >> to get rid of values for neighboring schools
filtered_data = data[(data['spec'] != 1) & (data['nschool'] == 0)]

# Filter data based on certainty where upper - lower < 0.2
filtered_certain_data = filtered_data[(filtered_data['bound'] == 'upper') &
                                      (filtered_data['value'] - filtered_data['value'] < 0.2)]

# Convert 'redizo' to string for consistency
filtered_certain_data['redizo'] = filtered_certain_data['redizo'].astype(str)

# Load the processed PCA features
processed_pca_features_path = 'C:/++4630/++ University, education/03 -- Tilburg_v02/Data Science/data/processed_features.csv'
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

# Apply the closest_report function
report_values = filtered_certain_data.apply(closest_report, axis=1, reports=processed_pca_features)

# Add the new columns to the filtered_certain_data DataFrame
filtered_certain_data = pd.concat([filtered_certain_data, report_values], axis=1)

# Filter out rows with missing 'value' or 'matched_year' to ensure data completeness
filtered_certain_data = filtered_certain_data.dropna(subset=['value', 'year'])
# Rename the columns
filtered_certain_data.rename(columns={
    filtered_certain_data.columns[6]: 'inspection_year',
    filtered_certain_data.columns[7]: 'inspection_value_distance',
    **{filtered_certain_data.columns[i]: f'PC{i-7}' for i in range(8, 58)}
}, inplace=True)
filtered_certain_data.drop(columns=[filtered_certain_data.columns[59]], inplace=True)

filtered_certain_data = filtered_certain_data.dropna(subset=['PC1'])

# Check the resulting complete dataset
print("Complete PCA Data Sample:")
print(filtered_certain_data.head())

# Optionally, save the data for further analysis
filtered_certain_data.to_csv('C:/++4630/++ University, education/03 -- Tilburg_v02/Data Science/data/complete_merged_data_with_distance.csv', index=False)

print("Data merging, filtering, and distance calculation complete.")
#
#
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