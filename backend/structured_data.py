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
        self.transcripts_dir = "./transcripts"

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
        Please analyze this transcript and structure it into three sections:
        
        1. Introduction - The part that introduces the lesson or topic
        2. Conversation - The main dialogue or conversation
        3. Questions - Any questions or exercises at the end
        
        For each section, provide the text and a brief analysis in English about what's happening.
        If a section doesn't exist in the transcript, indicate that it's not present.
        
        Here is the transcript:
        
        {transcript_text}
        
        Format your response as a JSON object with the following structure:
        {{
            "introduction": {{
                "text": "Spanish text of introduction",
                "analysis": "Brief analysis in English"
            }},
            "conversation": {{
                "text": "Spanish text of conversation",
                "analysis": "Brief analysis in English"
            }},
            "questions": {{
                "text": "Spanish text of questions",
                "analysis": "Brief analysis in English"
            }}
        }}
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
            
            # Extract JSON from response
            json_match = re.search(r'```json\n(.*?)\n```', response_text, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
            else:
                # Try to find JSON without markdown formatting
                json_match = re.search(r'({.*})', response_text, re.DOTALL)
                if json_match:
                    json_str = json_match.group(1)
                else:
                    json_str = response_text
            
            # Clean up and parse JSON
            try:
                structured_data = json.loads(json_str)
                return structured_data
            except json.JSONDecodeError:
                logger.error(f"Failed to parse JSON from response: {response_text}")
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
        output_filename = filename.replace('.txt', '_structured.json')
        output_path = os.path.join(self.transcripts_dir, output_filename)
        
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
