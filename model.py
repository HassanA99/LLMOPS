from TTS.tts.models.glow_tts import GlowTTS
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor
from typing import Dict, Any

def initialize_model(config: Dict[str, Any]):
    """Initialize the TTS model with error handling."""
    try:
        ap = AudioProcessor.init_from_config(config)
        tokenizer, config = TTSTokenizer.init_from_config(config)
        model = GlowTTS(config, ap, tokenizer, speaker_manager=None)
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to initialize model: {str(e)}")

def save_model_checkpoint(model: GlowTTS, path: str):
    """Save a model checkpoint."""
    try:
        model.save_checkpoint(config=model.config, path=path)
    except Exception as e:
        raise RuntimeError(f"Failed to save model checkpoint: {str(e)}")

def load_model_checkpoint(config: Dict[str, Any], checkpoint_path: str):
    """Load a model from a checkpoint."""
    try:
        model = initialize_model(config)
        model.load_checkpoint(config, checkpoint_path)
        return model
    except Exception as e:
        raise RuntimeError(f"Failed to load model checkpoint: {str(e)}")
