import os
import requests
from pathlib import Path
# url = "http://0.0.0.0:8000"
# resp = requests.get(url)
# print(resp.status_code)
# print(resp.json())


# url = "http://0.0.0.0:8000/documents"
# query = "What is ETF performance?"
# payload = {"question": query,}

# # Send the POST request with the JSON payload
# response = requests.post(url, json=payload)

# # Check the response
# if response.status_code == 200:
#     print("Request successful!")
#     print(response.json())


# ##upload documents
# url = "http://0.0.0.0:8000/feed"

# # List of PDF file paths
# pdf_file_paths = [
#     "data/20130208-etf-performance-and-perspectives.pdf",
#    # "data/E_BOCHK_AR.pdf"
# ]

# Prepare files and data for multiple PDFs
files = []
data = []

# Open each PDF file and add to files list
for i, pdf_file_path in enumerate(pdf_file_paths):
    if os.path.exists(pdf_file_path):
        # Read the file content and store it
        with open(pdf_file_path, 'rb') as pdf_file:
            file_content = pdf_file.read()
        
        # Create tuple for requests: (filename, file_content, content_type)
        files.append((f'files', (os.path.basename(pdf_file_path), file_content, 'application/pdf')))
        # Add filepath as a separate form field
        data.append(('filepaths', Path(pdf_file_path).stem))
    else:
        print(f"Warning: File {pdf_file_path} not found, skipping...")

# Send the POST request with multiple files
if files:  # Only send if we have files to upload
    response = requests.post(url, files=files, data=data)
    
    # Check the response
    if response.status_code == 200:
        print("Request successful!")
        print(response.json())
    else:
        print(f"Request failed with status code: {response.status_code}")
        print(response.text)
else:
    print("No valid PDF files found to upload.")


# ##ask question
# url = "http://0.0.0.0:8000/ask"
# query = "What is ETF performance?"
# payload = {"question": query,}
# response = requests.post(url, json=payload)
# if response.status_code == 200:
#     print("Request successful!")
#     print(response.json())
# else:
#     print(f"Request failed with status code: {response.status_code}")
#     print(response.text)
#print(response.json())