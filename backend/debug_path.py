from pathlib import Path
from app.services.document_processor import document_processor

print("Document processor docs path:", document_processor.docs_path)
print("Does the path exist?", document_processor.docs_path.exists())

# List markdown files found
md_files = list(document_processor.docs_path.rglob("*.md"))
print("Number of markdown files found:", len(md_files))
for i, md_file in enumerate(md_files[:5]):  # Print first 5 files
    print(f"  {i+1}: {md_file}")