import os
import json
import boto3
import logging
from typing import Dict, List, Optional, Tuple
import re

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model ID
MODEL_ID = "amazon.nova-micro-v1:0"

class TranscriptStructurer:
    def __init__(self, model_id: str = MODEL_ID):
        """Initialize Bedrock client for transcript structuring"""
        self.bedrock_client = boto3.client('bedrock-runtime', region_name="us-east-1")
        self.model_id = model_id
        
        # Update paths to use the new folder structure
        self.data_dir = "./data"
        self.transcripts_dir = os.path.join(self.data_dir, "transcripts")
        self.structured_dir = os.path.join(self.data_dir, "structured_transcripts")
        
        # Create directories if they don't exist
        for directory in [self.data_dir, self.transcripts_dir, self.structured_dir]:
            if not os.path.exists(directory):
                os.makedirs(directory)
                logger.info(f"Created directory: {directory}")

    def load_transcript(self, filename: str) -> Optional[str]:
        """
        Load transcript from file
        
        Args:
            filename (str): Transcript filename
            
        Returns:
            Optional[str]: Transcript text if successful, None otherwise
        """
        filepath = os.path.join(self.transcripts_dir, filename)
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Error loading transcript: {str(e)}")
            return None

    def list_transcripts(self) -> List[str]:
        """
        List all transcript files in the transcripts directory
        
        Returns:
            List[str]: List of transcript filenames
        """
        try:
            return [f for f in os.listdir(self.transcripts_dir) if f.endswith('.txt')]
        except Exception as e:
            logger.error(f"Error listing transcripts: {str(e)}")
            return []

    def structure_transcript(self, transcript_text: str) -> Optional[Dict]:
        """
        Structure transcript into Introduction, Conversation, and Questions
        
        Args:
            transcript_text (str): Raw transcript text
            
        Returns:
            Optional[Dict]: Structured transcript if successful, None otherwise
        """
        prompt = f"""
        You are an expert Spanish language teacher. I have a transcript from a Spanish A1 level listening comprehension video.
        Please analyze this transcript and structure it into a JSON format with the following structure:

        {{
            "introduction": "Introduction text here",
            "conversation": "Conversation text here",
            "qa_pairs": [
                {{
                    "question": "Question 1 text here",
                    "answer": "Answer 1 text here"
                }},
                {{
                    "question": "Question 2 text here",
                    "answer": "Answer 2 text here"
                }},
                ...
            ]
        }}
        
        Here is the transcript:
        
        {transcript_text}
        
        Return only the JSON object with no additional text or explanation.
        """

        try:
            response = self.bedrock_client.converse(
                modelId=self.model_id,
                messages=[{
                    "role": "user",
                    "content": [{"text": prompt}]
                }],
                inferenceConfig={"temperature": 0.2}
            )
            
            response_text = response['output']['message']['content'][0]['text']
            
            # Try to parse the response as JSON
            try:
                # Remove any markdown code block formatting if present
                if "```json" in response_text:
                    response_text = response_text.split("```json")[1].split("```")[0].strip()
                elif "```" in response_text:
                    response_text = response_text.split("```")[1].split("```")[0].strip()
                    
                structured_data = json.loads(response_text)
                return structured_data
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing JSON response: {str(e)}")
                logger.error(f"Response text: {response_text}")
                return None
                
        except Exception as e:
            logger.error(f"Error structuring transcript: {str(e)}")
            return None

    def save_structured_data(self, filename: str, structured_data: Dict) -> bool:
        """
        Save structured data to file
        
        Args:
            filename (str): Output filename
            structured_data (Dict): Structured transcript data
            
        Returns:
            bool: True if successful, False otherwise
        """
        # Extract video ID from filename (remove .txt extension)
        video_id = os.path.splitext(os.path.basename(filename))[0]
        
        # Create output filename without the _structured suffix
        output_filename = f"{video_id}.json"
        output_path = os.path.join(self.structured_dir, output_filename)
        
        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(structured_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Structured data saved to {output_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving structured data: {str(e)}")
            return False

    def process_transcript(self, filename: str) -> Optional[Dict]:
        """
        Process a transcript file: load, structure, and save
        
        Args:
            filename (str): Transcript filename
            
        Returns:
            Optional[Dict]: Structured transcript if successful, None otherwise
        """
        # Load transcript
        transcript_text = self.load_transcript(filename)
        if not transcript_text:
            return None
            
        # Structure transcript
        structured_data = self.structure_transcript(transcript_text)
        if not structured_data:
            return None
            
        # Save structured data
        self.save_structured_data(filename, structured_data)
        
        return structured_data

def main():
    """Main function to process all transcripts or a specific one"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Structure Spanish transcripts')
    parser.add_argument('--file', '-f', help='Specific transcript file to process')
    parser.add_argument('--all', '-a', action='store_true', help='Process all transcript files')
    args = parser.parse_args()
    
    structurer = TranscriptStructurer()
    
    if args.file:
        # Process specific file
        result = structurer.process_transcript(args.file)
        if result:
            print(f"Successfully processed {args.file}")
            print("\nStructured Data:")
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            print(f"Failed to process {args.file}")
    
    elif args.all:
        # Process all files
        transcript_files = structurer.list_transcripts()
        if not transcript_files:
            print("No transcript files found")
            return
            
        print(f"Found {len(transcript_files)} transcript files")
        for filename in transcript_files:
            print(f"Processing {filename}...")
            result = structurer.process_transcript(filename)
            if result:
                print(f"Successfully processed {filename}")
            else:
                print(f"Failed to process {filename}")
    
    else:
        # No arguments provided, show help
        parser.print_help()

if __name__ == "__main__":
    main()
