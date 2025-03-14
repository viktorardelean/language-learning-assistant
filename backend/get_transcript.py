from youtube_transcript_api import YouTubeTranscriptApi
from typing import Optional, List, Dict
import argparse
import sys


class YouTubeTranscriptDownloader:
    def __init__(self, languages: List[str] = ["es", "en"]):
        self.languages = languages

    def extract_video_id(self, url: str) -> Optional[str]:
        """
        Extract video ID from YouTube URL
        
        Args:
            url (str): YouTube URL
            
        Returns:
            Optional[str]: Video ID if found, None otherwise
        """
        if "v=" in url:
            return url.split("v=")[1][:11]
        elif "youtu.be/" in url:
            return url.split("youtu.be/")[1][:11]
        return None

    def get_transcript(self, video_id: str) -> Optional[List[Dict]]:
        """
        Download YouTube Transcript
        
        Args:
            video_id (str): YouTube video ID or URL
            
        Returns:
            Optional[List[Dict]]: Transcript if successful, None otherwise
        """
        # Extract video ID if full URL is provided
        if "youtube.com" in video_id or "youtu.be" in video_id:
            video_id = self.extract_video_id(video_id)
            
        if not video_id:
            print("Invalid video ID or URL")
            return None

        print(f"Downloading transcript for video ID: {video_id}")
        
        try:
            return YouTubeTranscriptApi.get_transcript(video_id, languages=self.languages)
        except Exception as e:
            print(f"An error occurred: {str(e)}")
            return None

    def save_transcript(self, transcript: List[Dict], filename: str) -> bool:
        """
        Save transcript to file
        
        Args:
            transcript (List[Dict]): Transcript data
            filename (str): Output filename
            
        Returns:
            bool: True if successful, False otherwise
        """
        filename = f"./transcripts/{filename}.txt"
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                for entry in transcript:
                    f.write(f"{entry['text']}\n")
            return True
        except Exception as e:
            print(f"Error saving transcript: {str(e)}")
            return False

def main():
    # Check if help is requested directly
    if '--help' in sys.argv or '-h' in sys.argv:
        print("Usage: python get_transcript.py [OPTIONS] URL")
        print("\nOptions:")
        print("  --print, -p    Print transcript to console")
        print("  --help, -h     Show this help message")
        print("\nExamples:")
        print("  python get_transcript.py https://www.youtube.com/watch?v=VIDEO_ID")
        print("  python get_transcript.py -p https://www.youtube.com/watch?v=VIDEO_ID")
        return

    # Set up argument parser for other cases
    parser = argparse.ArgumentParser(description='Download YouTube video transcripts')
    
    # Add arguments
    parser.add_argument('url', help='YouTube video URL')
    parser.add_argument('--print', '-p', action='store_true', 
                       help='Print transcript to console')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Initialize downloader
    downloader = YouTubeTranscriptDownloader()
    
    # Get transcript
    transcript = downloader.get_transcript(args.url)
    if transcript:
        # Save transcript
        video_id = downloader.extract_video_id(args.url)
        if downloader.save_transcript(transcript, video_id):
            print(f"Transcript saved successfully to {video_id}.txt")
            # Print transcript if --print flag is used
            if args.print:
                for entry in transcript:
                    print(f"{entry['text']}")
        else:
            print("Failed to save transcript")
    else:
        print("Failed to get transcript")

if __name__ == "__main__":
    main()
        