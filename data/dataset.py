from torch.utils.data import Dataset
from typing import Dict, Any
import pandas as pd
import os

class FulaDataset(Dataset):
    """Fula TTS dataset"""
    def __init__(
        self,
        root_dir: str,
        metadata_file: str,
        config: BaseConfig,
        script: str = 'adlam'
    ):
        self.root_dir = root_dir
        self.metadata = pd.read_csv(os.path.join(root_dir, metadata_file))
        self.config = config
        self.script = script
        
        # Initialize processors
        self.text_processor = TextProcessor()
        self.audio_processor = AudioProcessor(config)
        
    def __len__(self) -> int:
        return len(self.metadata)
        
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.metadata.iloc[idx]
        
        # Process text
        text_data = self.text_processor.process_text(row['text'], self.script)
        
        # Process audio
        audio_path = os.path.join(self.root_dir, 'wavs', row['wav_file'])
        audio = self.audio_processor.load_audio(audio_path)
        
        return {
            'text': text_data['original_text'],
            'normalized_text': text_data['normalized_text'],
            'phonemes': text_data['phonemes'],
            'audio': audio,
            'speaker_id': row['speaker_id'],
            'emotion': row['emotion'],
            'dialect': row['dialect']
        }