import os
import requests
import pandas as pd
import fitz  # PyMuPDF
import re
import time
import random

# Function to sanitize the file name
def sanitize_filename(filename):
    return re.sub(r'[<>:"/\\|?*]', '', filename)

# Retry parameters
max_retries = 1
retry_delay = 1  # seconds

# Set the limit for the number of reports to process
report_limit = 15000

# Get the current working directory
current_directory = os.getcwd()

# Load the CSV file containing the report URLs
csv_file = os.path.join(current_directory, '..', 'data', 'raw', 'inspekcni_zpravy.csv')
data = pd.read_csv(csv_file)

# Initialize a counter and a list to record problematic REDIZOs
processed_count = 0
problematic_redizos = []

# Loop through the dataset and process each report
for index, row in data.iterrows():
    if processed_count >= report_limit:
        break

    pdf_url = row.iloc[4]  # Adjust the column index based on your CSV structure
    redizo = row.iloc[0]  # Assuming the REDIZO is in the first column
    school_name = row.iloc[1]  # Use school name or another unique identifier for file naming
    inspection_year = str(row['DatumOd'])[:4]  # Extract the year from DatumOd

    # Sanitize the school name for use in file paths
    sanitized_school_name = sanitize_filename(school_name)

    # Define file path for the extracted text
    text_filename = f"{redizo}_{inspection_year}_{sanitized_school_name}_inspection_report.txt"
    text_output_path = os.path.join(current_directory, '..', 'data', 'csi_reports', text_filename)

    # Check if the text file already exists
    if os.path.exists(text_output_path):
        print(f"Text file already exists for REDIZO {redizo}. Skipping download and extraction.")
    else:
        # Attempt to download the PDF file

        # small chance to delay the download to prevent the server from blocking frequent downloads
        chance_sleep = random.uniform(0, 1)
        print(chance_sleep)
        if chance_sleep >= 0.9:
            print("sleeping")
            time.sleep(random.uniform(3, 10))


        for attempt in range(max_retries):
            try:
                print(f"Downloading PDF for REDIZO {redizo} (Attempt {attempt + 1}/{max_retries})...")
                response = requests.get(pdf_url)

                # Check if the response is successful
                if response.status_code == 200 and len(response.content) > 0:
                    pdf_data = response.content
                    doc = fitz.open("pdf", pdf_data)

                    # Extract text from the PDF
                    print(f"Extracting text from the PDF for REDIZO {redizo}...")
                    text = ""
                    for page_num in range(doc.page_count):
                        page = doc.load_page(page_num)
                        text += page.get_text()

                    # Close the document
                    doc.close()

                    # Save the REDIZO, year, and extracted text to a file
                    with open(text_output_path, 'w', encoding='utf-8') as text_file:
                        text_file.write(f"REDIZO: {redizo}\n")
                        text_file.write(f"Year: {inspection_year}\n\n")
                        text_file.write(text)
                    print(f"Text extracted and saved to {text_output_path}")

                    # Exit the retry loop if successful
                    break

                else:
                    print(f"Failed to download PDF for REDIZO {redizo}, status code {response.status_code}.")
                    raise Exception("Empty or unsuccessful response.")

            except Exception as e:
                print(f"Error: {e}")
                if attempt < max_retries - 1:
                    print(f"Retrying in {retry_delay} seconds...")
                    time.sleep(retry_delay)
                else:
                    print(f"Failed to download and process PDF for REDIZO {redizo} after {max_retries} attempts.")
                    problematic_redizos.append(redizo)
                    break

    # Increment the processed count
    processed_count += 1

# Save the list of problematic REDIZOs to a file
if problematic_redizos:
    problem_log_path = os.path.join(current_directory, '..', 'data', 'raw', 'problematic_redizos.txt')
    with open(problem_log_path, 'w', encoding='utf-8') as log_file:
        log_file.write("\n".join(map(str, problematic_redizos)))
    print(f"Problematic REDIZOs logged to {problem_log_path}")

print(f"Processing complete. {processed_count} reports processed.")
