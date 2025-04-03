import json
import logging
import boto3
from typing import Dict, List, Optional
from .rag import TranscriptVectorStore

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LanguageLearningAssistant:
    def __init__(self, model_id="amazon.nova-micro-v1:0"):
        """
        Initialize the Language Learning Assistant
        
        Args:
            model_id (str): Amazon Bedrock model ID for question generation
        """
        # Initialize vector store
        self.vector_store = TranscriptVectorStore()
        
        # Debug: Print all documents in the vector store
        try:
            results = self.vector_store.collection.get()
            logger.info("Documents in vector store:")
            for doc, metadata in zip(results['documents'], results['metadatas']):
                logger.info(f"Document: {doc[:100]}...")
                logger.info(f"Metadata: {metadata}")
        except Exception as e:
            logger.error(f"Error getting documents: {e}")
        
        # Process transcripts if vector store is empty
        if self.vector_store.collection.count() == 0:
            logger.info("Vector store is empty. Loading structured transcripts...")
            success = self.vector_store.process_all_transcripts()
            if success:
                logger.info("Successfully loaded transcripts into vector store")
            else:
                logger.warning("Failed to load transcripts into vector store")
        else:
            logger.info(f"Vector store contains {self.vector_store.collection.count()} documents")
        
        # Initialize Bedrock client
        self.bedrock_client = boto3.client('bedrock-runtime', region_name="us-east-1")
        self.model_id = model_id

    def retrieve_similar_context(self, query: str, n_results: int = 3) -> List[str]:
        """
        Retrieve similar context from the vector store
        
        Args:
            query (str): Query to find similar context
            n_results (int): Number of similar contexts to retrieve
            
        Returns:
            List[str]: List of similar context documents
        """
        try:
            # Get similar documents without filtering
            results = self.vector_store.query_similar(query, n_results)
            
            # Debug: Log full results
            logger.info(f"Query results: {results}")
            
            # Extract documents from results
            similar_contexts = results.get('documents', [[]])[0]
            logger.info(f"Found {len(similar_contexts)} similar contexts")
            
            # Process each context
            processed_contexts = []
            for context in similar_contexts:
                logger.info(f"Processing context: {context[:100]}...")  # Log first 100 chars
                
                # If the context is too short, it might not be a complete conversation
                if len(context.split('\n')) < 3:
                    logger.info("Context too short, skipping")
                    continue
                
                # Clean up the context
                cleaned_context = context.strip()
                if cleaned_context:
                    processed_contexts.append(cleaned_context)
                    logger.info("Added context to processed list")
            
            # If we have contexts, combine them
            if processed_contexts:
                combined_context = "Conversation:\n" + "\n".join(processed_contexts)
                logger.info(f"Combined {len(processed_contexts)} contexts")
                return [combined_context]
            
            logger.warning("No contexts processed")
            return []
            
        except Exception as e:
            logger.error(f"Error retrieving similar context: {str(e)}")
            return []

    def generate_question(self, context: str, question_type: str) -> Optional[Dict]:
        """
        Generate a language learning question using Bedrock Nova Micro
        
        Args:
            context (str): Context to base the question on
            question_type (str): Type of question to generate
            
        Returns:
            Optional[Dict]: Generated question data if successful, None otherwise
        """
        prompt = f"""You are an expert Spanish language teacher. Here's a sample conversation for reference:

{context}

Create a NEW conversation following a similar pattern but with different:
- Names
- Numbers
- Places
- Professions
- Details

The new conversation should:
- Be at A1 level Spanish
- Use similar grammar structures
- Cover similar topics
- Be about 3-4 lines long
- Include personal information (name, age, job, etc.)
- Include some numbers and locations

After creating the conversation, generate a {question_type} multiple choice question about it.

Return your response in this exact JSON format:
{{
    "conversation": "your new conversation in Spanish",
    "question_spanish": "your question about the NEW conversation in Spanish",
    "question_english": "English translation of your question",
    "answers": [
        {{
            "text_spanish": "correct answer in Spanish",
            "text_english": "correct answer in English",
            "is_correct": true
        }},
        {{
            "text_spanish": "first incorrect answer in Spanish",
            "text_english": "first incorrect answer in English",
            "is_correct": false
        }},
        {{
            "text_spanish": "second incorrect answer in Spanish",
            "text_english": "second incorrect answer in English",
            "is_correct": false
        }},
        {{
            "text_spanish": "third incorrect answer in Spanish",
            "text_english": "third incorrect answer in English",
            "is_correct": false
        }}
    ]
}}"""

        try:
            request_body = {
                "messages": [
                    {
                        "role": "user",
                        "content": [{"text": prompt}]
                    }
                ]
            }

            response = self.bedrock_client.invoke_model(
                modelId=self.model_id,
                contentType="application/json",
                accept="application/json",
                body=json.dumps(request_body)
            )
            
            # Extract and parse the response - updated to match Nova's response format
            response_body = json.loads(response["body"].read())
            response_text = response_body.get("output", {}).get("message", {}).get("content", [{}])[0].get("text", "").strip()
            
            logger.info(f"Raw response: {response_text}")  # Debug logging
            
            try:
                # First try direct JSON parsing
                try:
                    return json.loads(response_text)
                except json.JSONDecodeError:
                    # If direct parsing fails, try to extract JSON
                    start = response_text.find("{")
                    end = response_text.rfind("}") + 1
                    
                    if start >= 0 and end > start:
                        json_str = response_text[start:end]
                        try:
                            question_data = json.loads(json_str)
                            # Validate required fields
                            required_fields = ["conversation", "question_spanish", "question_english", "answers"]
                            if all(field in question_data for field in required_fields):
                                return question_data
                        except json.JSONDecodeError:
                            pass
                    
                    logger.error("Could not extract valid JSON from response")
                    logger.error(f"Response text: {response_text}")
                    return None
                
            except Exception as e:
                logger.error(f"Error processing response: {str(e)}")
                return None
            
        except Exception as e:
            logger.error(f"Error generating question: {str(e)}")
            return None

    def generate_learning_exercise(self, question_type: str = "comprehension") -> Optional[Dict]:
        """Generate a complete learning exercise"""
        try:
            # Simple query to get any conversation
            results = self.vector_store.collection.get(
                where={"type": "conversation"}
            )
            
            if not results or not results['documents']:
                # Try getting any document
                results = self.vector_store.collection.get(
                    limit=5
                )
                
            logger.info(f"Got {len(results.get('documents', []))} documents from vector store")
            
            if results and results['documents']:
                # Use the first document as a template
                template = results['documents'][0]
                response = self.generate_question(template, question_type)
                
                if response and "conversation" in response:
                    return {
                        "context": response["conversation"],  # Use the newly generated conversation
                        "question": {
                            "question_spanish": response["question_spanish"],
                            "question_english": response["question_english"],
                            "answers": response["answers"]
                        }
                    }
            
            return None
            
        except Exception as e:
            logger.error(f"Error in generate_learning_exercise: {e}")
            return None 