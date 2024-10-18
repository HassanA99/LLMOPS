from typing import Dict, List, Tuple
from datasets import Dataset
from transformers import Pipeline
import logging

logger = logging.getLogger(__name__)

def load_fula_tts_samples(dataset: Dataset, asr_pipeline: Pipeline, config: Dict) -> Tuple[List, List]:
    train_samples = []
    eval_samples = []
    
    for split in ['train', 'test']:
        for item in dataset[split]:
            try:
                audio_path = item['audio']['path']
                
                # Transcribe audio using the ASR pipeline
                transcription = asr_pipeline(audio_path)['text']
                
                sample = [transcription, audio_path, item.get('speaker_id')]
                
                if split == 'train':
                    train_samples.append(sample)
                else:
                    eval_samples.append(sample)
            except Exception as e:
                logger.error(f"Error processing sample {item.get('id', 'unknown')}: {str(e)}")
    
    # Implement split logic if needed
    if config.get('eval_split_size'):
        split_index = int(len(train_samples) * (1 - config['eval_split_size']))
        eval_samples.extend(train_samples[split_index:])
        train_samples = train_samples[:split_index]
    
    logger.info(f"Loaded {len(train_samples)} training samples and {len(eval_samples)} evaluation samples")
    return train_samples, eval_samples

def augment_data(samples: List[List[str]]) -> List[List[str]]:
    """Apply data augmentation techniques to the samples."""
    augmented_samples = []
    for sample in samples:
        augmented_samples.append(sample)
        # Add augmentation techniques here, e.g.:
        # augmented_samples.append(pitch_shift(sample))
        # augmented_samples.append(time_stretch(sample))
    return augmented_samples
