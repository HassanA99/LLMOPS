# config/model_config.py
@dataclass
class VITSConfig(BaseConfig):
    """VITS model configuration"""
    model_name: str = "vits"
    hidden_channels: int = 192
    num_speakers: int = 10
    use_speaker_embedding: bool = True
    use_emotion_embedding: bool = True
    phoneme_embedding_dim: int = 256
    num_phonemes: int = 128

@dataclass
class TrainingConfig(BaseConfig):
    """Training configuration"""
    batch_size: int = 32
    num_epochs: int = 1000
    learning_rate: float = 0.0001
    checkpoint_dir: str = "checkpoints"
    log_dir: str = "logs"