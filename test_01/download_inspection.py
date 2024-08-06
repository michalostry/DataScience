import requests
import os
import pandas as pd

# Get the current working directory
current_directory = os.getcwd()

# Load the CSV file
csv_file = r'C:\++4630\++ University, education\03 -- Tilburg_v02\Data Science\test_01\inspekcni_zpravy.csv'
data = pd.read_csv(csv_file)

# Display the first few rows of the dataset
print(data.head())

# Select a specific row (e.g., the first row)
selected_row = data.iloc[0]  # Select the first row

# Extract the URL from the selected row using iloc
pdf_url = selected_row.iloc[4]  # Access the fifth column by position

# Print the selected row and the extracted URL
print(f"Selected School: {selected_row.iloc[1]}")
print(f"PDF URL: {pdf_url}")

# Define the filename for the PDF
pdf_filename = 'inspection_report.pdf'

# Create the full path
file_path = os.path.join(current_directory, pdf_filename)

# Download the PDF file
response = requests.get(pdf_url)

# Save the PDF to the current working directory
with open(file_path, 'wb') as pdf_file:
    pdf_file.write(response.content)

print(f"Downloaded {pdf_filename} and saved it to {current_directory}")
