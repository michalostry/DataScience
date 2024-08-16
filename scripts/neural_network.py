import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler

# Step 1: Load the data
data_path_significant = 'C:/++4630/++ University, education/03 -- Tilburg_v02/Data Science/data/complete_certain_merged_data_with_distance.csv'
data_path_all = 'C:/++4630/++ University, education/03 -- Tilburg_v02/Data Science/data/complete_all_merged_data_with_distance.csv'

data_all = pd.read_csv(data_path_all)
data_significant = pd.read_csv(data_path_significant)

data_for_filter = data_all #which data to use

# Remove duplicate years x inspection year combinations from the data
data_unique = data_for_filter.loc[data_for_filter.groupby(['redizo', 'inspection_year'])['inspection_value_distance'].idxmin()]

# Keep only the row with the lowest 'inspection_value_distance' for each 'redizo'
data_filtered = data_unique.loc[data_unique.groupby('redizo')['inspection_value_distance'].idxmin()]

#Display rows where the same 'redizo' appears more than once
multiple_reports = data_unique.groupby('redizo').filter(lambda x: len(x) > 1)
print("Rows with the same redizo (multiple inspection reports matched):")
print(multiple_reports.head(10))  # Adjust the number to see more rows if needed

data = data_filtered #data to use for the model

# Step 2: Prepare the data
# Define input features (PC1 to PC200) and target variable (value)
X = data.loc[:, 'PC1':'PC200']
y = data['value']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 3: Define the Neural Network model
model = Sequential()
model.add(Dense(128, input_dim=X_train.shape[1], activation='relu'))  # First hidden layer
model.add(Dense(64, activation='relu'))  # Second hidden layer
model.add(Dense(32, activation='relu'))  # Third hidden layer (new)
model.add(Dense(1, activation='sigmoid'))  # Output layer for regression


# Step 4: Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Step 5: Train the model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=32)

# Step 6: Evaluate the model
loss = model.evaluate(X_test, y_test)
print(f'Test Loss (MSE): {loss}')

# Step 7: Plot the learning curves
import matplotlib.pyplot as plt

plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Learning Curves')
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error')
plt.legend()
plt.show()

# Step 8: Make predictions and compare
y_pred = model.predict(X_test)

# Scatter plot: True vs Predicted values
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, edgecolor='k', alpha=0.7)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')
plt.xlabel('True Value')
plt.ylabel('Predicted Value')
plt.title('True vs Predicted Values (Neural Network: 64-32-16)')
plt.show()

# Step 9: Residual Analysis
residuals = y_test - y_pred.flatten()

# Histogram of residuals
plt.figure(figsize=(8, 6))
plt.hist(residuals, bins=30, edgecolor='k', alpha=0.7)
plt.title('Histogram of Residuals')
plt.xlabel('Residual')
plt.ylabel('Frequency')
plt.show()

# Save the model in a subdirectory of the current working directory
model.save('my_model.keras')



# Ok, what to do tomorrow
# then lets run this model on the dataset, which also immediatelly allowes for comparison with their prediction
#
# and then let's wrup up'