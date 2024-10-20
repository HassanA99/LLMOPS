from typing import Dict, List, Optional
import unicodedata
from utils.adlam import AdlamProcessor
from utils.phonemes import PhonemeProcessor

class TextProcessor:
    """Text processing pipeline"""
    def __init__(self):
        self.adlam_processor = AdlamProcessor()
        self.phoneme_processor = PhonemeProcessor()

    def process_text(self, text: str, script: str = 'adlam') -> Dict[str, str]:
        """Process text in specified script"""
        if script == 'adlam':
            normalized = self.adlam_processor.normalize_text(text)
            phonemes = self.phoneme_processor.text_to_phonemes(normalized)
        else:
            phonemes = self.phoneme_processor.text_to_phonemes(text)
        
        return {
            'original_text': text,
            'normalized_text': normalized if script == 'adlam' else text,
            'phonemes': phonemes,
            'script': script
        }