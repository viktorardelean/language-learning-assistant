import chromadb
import os
import json
import boto3
import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TranscriptVectorStore:
    def __init__(self, collection_name="spanish-listening-comprehension"):
        """Initialize ChromaDB client and collection for transcript storage"""
        # Use persistent storage
        self.client = chromadb.PersistentClient(path="./chroma_db")
        
        # Create or get collection
        try:
            self.collection = self.client.get_or_create_collection(collection_name)
            logger.info(f"Using collection: {collection_name}")
        except Exception as e:
            logger.error(f"Error getting/creating collection: {str(e)}")
            raise
        
        # Initialize Bedrock client for embeddings
        self.bedrock_client = boto3.client('bedrock-runtime', region_name="us-east-1")
        self.model_id = "amazon.titan-embed-text-v2:0"
        
        # Get the script's directory
        script_dir = Path(__file__).parent
        
        # Directory paths - updated to use absolute paths
        self.data_dir = script_dir / "data"
        self.transcripts_dir = self.data_dir / "transcripts"
        self.structured_dir = self.data_dir / "structured_transcripts"
        
        # Create directories if they don't exist
        self._ensure_directories()

    def _ensure_directories(self):
        """Create necessary directories if they don't exist"""
        for directory in [self.data_dir, self.transcripts_dir, self.structured_dir]:
            if not directory.exists():
                directory.mkdir(parents=True)
                logger.info(f"Created directory: {directory}")

    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embeddings using Amazon Titan Text Embeddings V2
        
        Args:
            text (str): Text to embed
            
        Returns:
            List[float]: Embedding vector
        """
        try:
            response = self.bedrock_client.invoke_model(
                modelId=self.model_id,
                contentType="application/json",
                accept="application/json",
                body=json.dumps({
                    "inputText": text
                })
            )
            
            response_body = json.loads(response["body"].read())
            return response_body["embedding"]
            
        except Exception as e:
            logger.error(f"Error generating embedding: {str(e)}")
            return []

    def load_structured_transcripts(self) -> Tuple[List[str], List[Dict], List[str], List[List[float]]]:
        """
        Load structured transcript JSON files from structured_transcripts directory
        
        Returns:
            Tuple containing:
                - documents: List of text content
                - metadatas: List of metadata dictionaries
                - ids: List of document IDs
                - embeddings: List of embedding vectors
        """
        documents = []
        metadatas = []
        ids = []
        embeddings = []
        
        # Add this to the load_structured_transcripts method
        logger.info(f"Looking for files in: {self.structured_dir}")
        structured_files = list(self.structured_dir.glob('*.json'))
        logger.info(f"Files found: {[str(f) for f in structured_files]}")
        
        if not structured_files:
            logger.warning("No structured transcript files found")
            return documents, metadatas, ids, embeddings
        
        logger.info(f"Found {len(structured_files)} structured transcript files")
        
        # Process each structured file
        for file_path in structured_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    # Load the JSON data directly
                    content = json.load(file)
                    
                    # Extract video ID from filename
                    video_id = file_path.stem
                    
                    # Process introduction
                    if "introduction" in content and content["introduction"]:
                        intro_text = content["introduction"]
                        doc_id = f"{video_id}_introduction"
                        documents.append(intro_text)
                        metadatas.append({
                            "video_id": video_id,
                            "type": "introduction",
                            "source": f"{video_id}.json"
                        })
                        ids.append(doc_id)
                        embeddings.append(self.generate_embedding(intro_text))
                    
                    # Process conversation
                    if "conversation" in content and content["conversation"]:
                        conv_text = content["conversation"]
                        doc_id = f"{video_id}_conversation"
                        documents.append(conv_text)
                        metadatas.append({
                            "video_id": video_id,
                            "type": "conversation",
                            "source": f"{video_id}.json"
                        })
                        ids.append(doc_id)
                        embeddings.append(self.generate_embedding(conv_text))
                    
                    # Process Q&A pairs
                    if "qa_pairs" in content and content["qa_pairs"]:
                        for i, qa_pair in enumerate(content["qa_pairs"], 1):
                            # Process question
                            if "question" in qa_pair and qa_pair["question"]:
                                q_text = qa_pair["question"]
                                q_id = f"{video_id}_question_{i}"
                                documents.append(q_text)
                                metadatas.append({
                                    "video_id": video_id,
                                    "type": "question",
                                    "question_number": i,
                                    "source": f"{video_id}.json"
                                })
                                ids.append(q_id)
                                embeddings.append(self.generate_embedding(q_text))
                            
                            # Process answer
                            if "answer" in qa_pair and qa_pair["answer"]:
                                a_text = qa_pair["answer"]
                                a_id = f"{video_id}_answer_{i}"
                                documents.append(a_text)
                                metadatas.append({
                                    "video_id": video_id,
                                    "type": "answer",
                                    "question_number": i,
                                    "source": f"{video_id}.json"
                                })
                                ids.append(a_id)
                                embeddings.append(self.generate_embedding(a_text))
                
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {str(e)}")
        
        return documents, metadatas, ids, embeddings

    def add_to_vector_store(self, documents: List[str], metadatas: List[Dict], 
                           ids: List[str], embeddings: List[List[float]]) -> bool:
        """
        Add documents to the vector store
        
        Args:
            documents (List[str]): List of document texts
            metadatas (List[Dict]): List of metadata dictionaries
            ids (List[str]): List of document IDs
            embeddings (List[List[float]]): List of embedding vectors
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Filter out documents with empty embeddings
        valid_docs = []
        valid_metadatas = []
        valid_ids = []
        valid_embeddings = []
        
        for doc, metadata, doc_id, embedding in zip(documents, metadatas, ids, embeddings):
            if embedding and len(embedding) > 0:
                valid_docs.append(doc)
                valid_metadatas.append(metadata)
                valid_ids.append(doc_id)
                valid_embeddings.append(embedding)
        
        if not valid_docs:
            logger.warning("No valid documents with embeddings to add to vector store")
            return False
        
        try:
            self.collection.add(
                documents=valid_docs,
                metadatas=valid_metadatas,
                ids=valid_ids,
                embeddings=valid_embeddings
            )
            logger.info(f"Added {len(valid_docs)} documents to the vector store")
            return True
        except Exception as e:
            logger.error(f"Error adding documents to vector store: {str(e)}")
            return False

    def query_similar(self, query_text: str, n_results: int = 3, 
                     filter_criteria: Optional[Dict] = None) -> Dict:
        """
        Query the vector store for similar documents
        
        Args:
            query_text (str): Query text
            n_results (int): Number of results to return
            filter_criteria (Optional[Dict]): Filter criteria for the query
            
        Returns:
            Dict: Query results
        """
        try:
            # Generate embedding for the query
            query_embedding = self.generate_embedding(query_text)
            
            # Log detailed embedding information
            logger.info(f"Query: {query_text}")
            logger.info(f"Query Embedding Length: {len(query_embedding)}")
            logger.info(f"Query Embedding (first 5 values): {query_embedding[:5]}")
            
            # Check if the collection is empty
            collection_count = self.collection.count()
            logger.info(f"Total documents in collection: {collection_count}")
            
            # Query the collection
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=filter_criteria
            )
            
            return results
        except Exception as e:
            logger.error(f"Error querying vector store: {str(e)}")
            return {"ids": [], "documents": [], "metadatas": [], "distances": []}

    def process_all_transcripts(self) -> bool:
        """
        Process all structured transcripts and add to vector store
        
        Returns:
            bool: True if successful, False otherwise
        """
        # Load structured transcripts
        documents, metadatas, ids, embeddings = self.load_structured_transcripts()
        
        # Add to vector store
        if documents:
            return self.add_to_vector_store(documents, metadatas, ids, embeddings)
        else:
            return False

    def migrate_existing_files(self) -> None:
        """
        Migrate existing files to the new folder structure
        """
        # Check for old transcript files in ./transcripts
        old_transcripts_dir = Path("./transcripts")
        if old_transcripts_dir.exists():
            # Move transcript files
            for file_path in old_transcripts_dir.glob('*.txt'):
                target_path = self.transcripts_dir / file_path.name
                if not target_path.exists():
                    try:
                        import shutil
                        shutil.copy2(file_path, target_path)
                        logger.info(f"Migrated transcript file: {file_path.name}")
                    except Exception as e:
                        logger.error(f"Error migrating transcript file {file_path.name}: {str(e)}")
            
            # Move structured files
            for file_path in old_transcripts_dir.glob('*_structured.json'):
                target_path = self.structured_dir / file_path.name
                if not target_path.exists():
                    try:
                        import shutil
                        shutil.copy2(file_path, target_path)
                        logger.info(f"Migrated structured file: {file_path.name}")
                    except Exception as e:
                        logger.error(f"Error migrating structured file {file_path.name}: {str(e)}")

def main():
    """Main function to process all structured transcripts"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Process structured transcripts into vector store')
    parser.add_argument('--query', '-q', help='Query the vector store')
    parser.add_argument('--filter', '-f', help='Filter criteria (JSON string)')
    parser.add_argument('--results', '-r', type=int, default=3, help='Number of results to return')
    parser.add_argument('--migrate', '-m', action='store_true', help='Migrate existing files to new folder structure')
    args = parser.parse_args()
    
    vector_store = TranscriptVectorStore()
    
    # Migrate existing files if requested
    if args.migrate:
        vector_store.migrate_existing_files()
        print("Migration of existing files completed")
        return
    
    if args.query:
        # Query the vector store
        filter_criteria = json.loads(args.filter) if args.filter else None
        results = vector_store.query_similar(args.query, args.results, filter_criteria)
        
        print("\nQuery Results:")
        for i, (doc_id, doc, metadata, distance) in enumerate(zip(
            results["ids"][0] if results["ids"] else [], 
            results["documents"][0] if results["documents"] else [],
            results["metadatas"][0] if results["metadatas"] else [],
            results["distances"][0] if results["distances"] else []
        )):
            print(f"\nResult {i+1}:")
            print(f"ID: {doc_id}")
            print(f"Distance: {distance}")
            print(f"Metadata: {metadata}")
            print(f"Document: {doc[:150]}..." if len(doc) > 150 else f"Document: {doc}")
    else:
        # Process all transcripts
        success = vector_store.process_all_transcripts()
        if success:
            print("Successfully processed all structured transcripts into vector store")
        else:
            print("Failed to process structured transcripts")

if __name__ == "__main__":
    main()