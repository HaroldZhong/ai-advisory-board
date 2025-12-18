import requests
import os
from docx import Document

BASE_URL = "http://localhost:8001/api/upload"

def create_sample_files():
    # 1. Text file
    with open("test.txt", "w") as f:
        f.write("This is a test text file content.")
        
    # 2. DOCX file
    doc = Document()
    doc.add_paragraph("This is a test DOCX file content.")
    doc.save("test.docx")
    
    # 3. Large file (for truncation test)
    with open("large.txt", "w") as f:
        f.write("A" * 25000)

def test_upload(filename):
    print(f"Testing {filename}...")
    with open(filename, "rb") as f:
        files = {"file": (filename, f, "application/octet-stream")} # Let backend sniff or use extension
        try:
            response = requests.post(BASE_URL, files=files)
            if response.status_code == 200:
                data = response.json()
                print(f"✅ Success: {filename}")
                print(f"   Text length: {len(data['text'])}")
                print(f"   Truncated: {data['truncated']}")
                print(f"   Snippet: {data['text'][:50]}...")
            else:
                print(f"❌ Failed: {filename} - {response.status_code} - {response.text}")
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    create_sample_files()
    
    test_upload("test.txt")
    test_upload("test.docx")
    test_upload("large.txt")
    
    # Clean up
    if os.path.exists("test.txt"): os.remove("test.txt")
    if os.path.exists("test.docx"): os.remove("test.docx")
    if os.path.exists("large.txt"): os.remove("large.txt")
