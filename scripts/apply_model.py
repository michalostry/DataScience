import pandas as pd
from keras.models import load_model
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Step 1: Load the data
data_path_significant = 'C:/++4630/++ University, education/03 -- Tilburg_v02/Data Science/data/complete_certain_merged_data_with_distance.csv'
data_path_all = 'C:/++4630/++ University, education/03 -- Tilburg_v02/Data Science/data/complete_all_merged_data_with_distance.csv'

data_all = pd.read_csv(data_path_all)
data_significant = pd.read_csv(data_path_significant)

data = data_significant #which data to use

# Step 2: Prepare the Data
# Extract features (PC1 to PC200) and standardize them
# Make sure to drop non-numeric columns
X_full = data.loc[:, 'PC1':'PC200']  # Only select PC1 to PC200 columns
y_full = data['value']  # This is optional if you want to compare later


# Standardize the features using the same scaler as during model training
scaler = StandardScaler()
X_full_scaled = scaler.fit_transform(X_full)

# Step 3: Load the Pre-trained Model
model = load_model('my_model.keras')

# Step 4: Make Predictions on the Entire Dataset
y_full_pred = model.predict(X_full_scaled)

# Add the predictions to the original DataFrame
data['predicted_value'] = y_full_pred

# Step 5: Save the Predictions
output_path = 'C:/++4630/++ University, education/03 -- Tilburg_v02/Data Science/data/predicted_features.csv'
data.to_csv(output_path, index=False)

print(f"Predictions saved to {output_path}")


# Step 6: Plot True vs Predicted Values
plt.figure(figsize=(8, 6))
plt.scatter(y_full, y_full_pred, edgecolor='k', alpha=0.7)
plt.plot([y_full.min(), y_full.max()], [y_full.min(), y_full.max()], 'r--', lw=2)
plt.xlabel('True Value')
plt.ylabel('Predicted Value')
plt.title('True vs Predicted Values (Neural Network)')
plt.savefig('plots/True vs Predicted Values applied')
plt.show()