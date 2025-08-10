#!/usr/bin/env python3
"""
Create test audio samples for entity-dense benchmarking

This script creates text files with entity-dense content that can be used
to generate synthetic speech for testing name/URL recognition accuracy.

For actual testing, these would need to be converted to audio using TTS
or recorded manually.
"""

from pathlib import Path

def create_entity_test_samples():
    """Create text samples with names, URLs, and technical terms"""
    
    test_dir = Path(__file__).parent / "test_audio" / "entities"
    test_dir.mkdir(exist_ok=True)
    
    # Sample texts with various entities
    samples = [
        {
            "filename": "news_urls.txt",
            "text": "Please visit bbc.com/news for the latest updates. You can also check cnn.com and reuters.com for international coverage."
        },
        {
            "filename": "tech_companies.txt", 
            "text": "The meeting includes representatives from Microsoft, Google, Apple, Amazon, and Meta. We'll discuss partnerships with OpenAI and Anthropic."
        },
        {
            "filename": "email_addresses.txt",
            "text": "Send the report to john.doe@company.com and mary.smith@example.org. CC sarah.wilson@university.edu on important updates."
        },
        {
            "filename": "mixed_entities.txt",
            "text": "John Smith from Microsoft will present at github.com/microsoft/project. Contact him at j.smith@microsoft.com or visit linkedin.com/in/johnsmith."
        },
        {
            "filename": "technical_terms.txt",
            "text": "Configure the API endpoint at api.example.com/v1/users. Use OAuth2 authentication with client ID and secret from developer.example.com."
        },
        {
            "filename": "phone_numbers.txt",
            "text": "Call the office at plus one four one five five five five one two three four or mobile at plus one six five zero two three four five six seven eight."
        },
        {
            "filename": "addresses.txt", 
            "text": "The office is located at One Hundred Twenty Three Main Street, San Francisco, California nine four one zero two."
        },
        {
            "filename": "complex_entities.txt",
            "text": "Dr. Sarah Johnson-Williams from Stanford University will discuss machine learning at ai-conference.org. The event is sponsored by OpenAI and hosted at conference-center.com."
        }
    ]
    
    # Write text files
    for sample in samples:
        text_file = test_dir / sample["filename"]
        with open(text_file, 'w') as f:
            f.write(sample["text"])
        print(f"Created: {text_file}")
    
    # Create a master reference file
    reference_file = test_dir / "references.txt"
    with open(reference_file, 'w') as f:
        for sample in samples:
            f.write(f"{sample['filename'].replace('.txt', '.wav')}\t{sample['text']}\n")
    
    print(f"Created reference file: {reference_file}")
    print("\nNOTE: These are text files. To create actual test audio:")
    print("1. Use TTS to generate WAV files from these texts")
    print("2. Or record them manually with clear pronunciation")
    print("3. Save as 16kHz mono WAV files in the same directory")

def create_dictation_test_samples():
    """Create samples for dictation/meeting scenarios"""
    
    test_dir = Path(__file__).parent / "test_audio" / "realworld"
    test_dir.mkdir(exist_ok=True)
    
    samples = [
        {
            "filename": "meeting_minutes.txt",
            "text": "The board meeting was attended by CEO Michael Rodriguez, CFO Lisa Chen, and CTO David Kumar. We discussed Q4 projections and the partnership with TechCorp Inc."
        },
        {
            "filename": "dictation_email.txt",
            "text": "Dear Mr. Thompson, Please send the quarterly report to accounting@company.com by Friday. The file should be uploaded to sharepoint.company.com/reports. Best regards, Jennifer Martinez."
        },
        {
            "filename": "technical_discussion.txt", 
            "text": "The API integration with stripe.com requires updating the webhook at webhook.oursite.com/stripe. Contact the DevOps team at devops@company.com for deployment."
        }
    ]
    
    # Write text files
    for sample in samples:
        text_file = test_dir / sample["filename"]
        with open(text_file, 'w') as f:
            f.write(sample["text"])
        print(f"Created: {text_file}")
    
    # Create a master reference file
    reference_file = test_dir / "references.txt"
    with open(reference_file, 'w') as f:
        for sample in samples:
            f.write(f"{sample['filename'].replace('.txt', '.wav')}\t{sample['text']}\n")
    
    print(f"Created reference file: {reference_file}")

if __name__ == "__main__":
    print("Creating entity-dense test samples...")
    create_entity_test_samples()
    
    print("\nCreating real-world test samples...")
    create_dictation_test_samples()
    
    print("\nDone! Text samples created.")
    print("To generate actual audio files:")
    print("1. Use your TTS system: python -c \"from kokoro_tts import KokoroTTS; tts=KokoroTTS(); tts.synthesize('text', 'output.wav')\"")
    print("2. Or use online TTS services")
    print("3. Or record manually with good microphone")
    print("4. Ensure all files are 16kHz mono WAV format")