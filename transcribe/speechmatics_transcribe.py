#!/usr/bin/env python3
"""
Speechmatics Batch API transcriber.
Uses the Enhanced operating point for best accuracy.
"""

import time
import os
import json
import requests
from pathlib import Path

from base_transcriber import BaseTranscriber, TranscriptionResult


class SpeechmaticsTranscriber(BaseTranscriber):
    """Speechmatics Batch API transcriber."""
    
    def __init__(self, 
                 operating_point: str = "enhanced",
                 api_key: str = None, 
                 results_dir: str = None):
        # Model name for results directory
        display_name = f"speechmatics-{operating_point}"
        super().__init__(display_name, results_dir)
        
        self.api_key = api_key or os.getenv('SPEECHMATICS_API_KEY')
        if not self.api_key:
            raise ValueError("Speechmatics API key required. Set SPEECHMATICS_API_KEY environment variable or pass api_key parameter.")
        
        self.operating_point = operating_point
        self.base_url = "https://asr.api.speechmatics.com/v2/jobs"
        self.headers = {"Authorization": f"Bearer {self.api_key}"}
    
    def transcribe_file(self, audio_file: str) -> TranscriptionResult:
        """
        Transcribe a single audio file using Speechmatics Batch API.
        
        Args:
            audio_file: Path to audio file
            
        Returns:
            TranscriptionResult object
        """
        if not Path(audio_file).exists():
            raise FileNotFoundError(f"Audio file not found: {audio_file}")
        
        start_time = time.time()
        
        try:
            # 1. Submit transcription job
            config = {
                "type": "transcription",
                "transcription_config": {
                    "language": "en",
                    "operating_point": self.operating_point,
                    "enable_entities": True
                }
            }
            
            with open(audio_file, 'rb') as f:
                files = {
                    'data_file': (os.path.basename(audio_file), f, 'audio/wav')
                }
                data = {
                    'config': json.dumps(config)
                }
                
                response = requests.post(
                    self.base_url, 
                    files=files, 
                    data=data, 
                    headers=self.headers
                )
                response.raise_for_status()
                job_id = response.json()["id"]
            
            # 2. Poll for completion
            while True:
                time.sleep(3)
                response = requests.get(
                    f"{self.base_url}/{job_id}", 
                    headers=self.headers
                )
                response.raise_for_status()
                status = response.json()["job"]["status"]
                
                if status == "done":
                    break
                elif status == "rejected":
                    raise Exception(f"Job rejected: {response.json()}")
            
            # 3. Get transcript
            response = requests.get(
                f"{self.base_url}/{job_id}/transcript?format=json-v2",
                headers=self.headers
            )
            response.raise_for_status()
            
            result = response.json()
            
            # Extract text from results
            words = []
            for item in result.get("results", []):
                alternatives = item.get("alternatives", [])
                if alternatives:
                    content = alternatives[0].get("content", "")
                    if content:
                        words.append(content)
            
            text = " ".join(words)
            duration = time.time() - start_time
            
            if not text:
                raise ValueError("Empty transcription result")
            
            return TranscriptionResult(
                text=text,
                duration=duration,
                model_name=self.model_name,
                audio_file=audio_file
            )
            
        except requests.exceptions.RequestException as e:
            raise Exception(f"Speechmatics API request failed: {e}")
        except Exception as e:
            raise Exception(f"Speechmatics transcription failed: {e}")


def main():
    """Main function for standalone usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Speechmatics Batch API Transcription")
    parser.add_argument("--audio_dir", required=True, help="Directory containing audio files")
    parser.add_argument("--operating_point", default="enhanced", 
                       choices=["standard", "enhanced"], 
                       help="Operating point (standard or enhanced)")
    parser.add_argument("--api_key", help="Speechmatics API key (or set SPEECHMATICS_API_KEY env var)")
    parser.add_argument("--results_dir", default="results", help="Results directory")
    parser.add_argument("--pattern", default="*_conversation.wav", help="Audio file pattern")
    
    args = parser.parse_args()
    
    # Initialize transcriber
    try:
        transcriber = SpeechmaticsTranscriber(
            operating_point=args.operating_point,
            api_key=args.api_key,
            results_dir=args.results_dir
        )
    except ValueError as e:
        print(f"❌ {e}")
        return
    
    # Get audio files
    try:
        audio_files = transcriber.get_audio_files(args.audio_dir, args.pattern)
        print(f"Found {len(audio_files)} audio files")
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return
    
    # Process files
    results = transcriber.transcribe_batch(audio_files)
    
    if results:
        print(f"\n✅ Successfully processed {len(results)} files")
        print(f"📁 Transcripts saved to: {transcriber.transcripts_dir}")
        print(f"📊 Metrics saved to: {transcriber.metrics_dir}")
    else:
        print("\n❌ No files were processed successfully")


if __name__ == "__main__":
    main()
