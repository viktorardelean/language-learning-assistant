import chromadb
import os
from pathlib import Path

# setup Chroma in-memory, for easy prototyping. Can add persistence easily!
client = chromadb.Client()

# Create collection. get_collection, get_or_create_collection, delete_collection also available!
collection = client.create_collection("spanish-listening-comprehension")

# Function to read text files from transcripts directory
def load_transcripts(transcripts_dir):
    documents = []
    metadatas = []
    ids = []
    
    # Convert string path to Path object
    transcript_path = Path(transcripts_dir)
    
    # Iterate through all .txt files in the transcripts directory
    for file_path in transcript_path.glob('*.txt'):
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                documents.append(content)
                metadatas.append({"source": str(file_path)})
                ids.append(f"doc_{file_path.stem}")  # Using filename without extension as ID
        except Exception as e:
            print(f"Error reading file {file_path}: {e}")
    
    return documents, metadatas, ids

# Load documents from transcripts directory
transcripts_dir = "transcripts"  # Adjust this path as needed
documents, metadatas, ids = load_transcripts(transcripts_dir)

# Add documents to the collection. Can also update and delete. Row-based API coming soon!
if documents:
    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids,
    )
    print(f"Added {len(documents)} documents to the collection")
else:
    print("No documents found in the transcripts directory")

# Query/search 2 most similar results. You can also .get by id
results = collection.query(
    query_texts=["This is a query document"],
    n_results=2,
    # where={"metadata_field": "is_equal_to_this"}, # optional filter
    # where_document={"$contains":"search_string"}  # optional filter
)

print(results)