import fitz  # PyMuPDF

# Path to the PDF file
pdf_path = '../data/csi_reports/inspection_report.pdf'

# Open the PDF file
doc = fitz.open(pdf_path)

# Initialize an empty string to hold the text
text = ""

# Loop through each page
for page_num in range(doc.page_count):
    page = doc.load_page(page_num)
    text += page.get_text()

# Close the document
doc.close()

# Save the extracted text to a file
text_output_path = '../data/csi_reports/inspection_report.txt'
with open(text_output_path, 'w', encoding='utf-8') as text_file:
    text_file.write(text)

print("Text extracted and saved.")
