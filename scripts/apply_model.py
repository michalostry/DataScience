import pandas as pd
from keras.models import load_model
from sklearn.preprocessing import StandardScaler

# Step 1: Load the data
data_path_all = 'C:/++4630/++ University, education/03 -- Tilburg_v02/Data Science/data/complete_certain_merged_data_with_distance.csv'
data_path_significant = 'C:/++4630/++ University, education/03 -- Tilburg_v02/Data Science/data/complete_all_merged_data_with_distance.csv'

data_all = pd.read_csv(data_path_all)
data_significant = pd.read_csv(data_path_significant)

# Step 1: Load the Dataset
data_path = 'C:/++4630/++ University, education/03 -- Tilburg_v02/Data Science/data/transform/processed_features.csv'
full_data = pd.read_csv(data_path)

# Step 2: Prepare the Data
# Extract features (PC1 to PC200) and standardize them
X_full = full_data.drop(columns=['redizo', 'year'])
y_full = full_data['value']  # This is optional if you want to compare later

# Standardize the features using the same scaler as during model training
scaler = StandardScaler()
X_full_scaled = scaler.fit_transform(X_full)

# Step 3: Load the Pre-trained Model
model = load_model('my_model.keras')

# Step 4: Make Predictions on the Entire Dataset
y_full_pred = model.predict(X_full_scaled)

# Add the predictions to the original DataFrame
full_data['predicted_value'] = y_full_pred

# Step 5: Save the Predictions
output_path = 'C:/++4630/++ University, education/03 -- Tilburg_v02/Data Science/data/predicted_features.csv'
full_data.to_csv(output_path, index=False)

print(f"Predictions saved to {output_path}")
