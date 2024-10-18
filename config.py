from TTS.tts.configs.glow_tts_config import GlowTTSConfig
from TTS.tts.configs.shared_configs import BaseDatasetConfig
import os
from typing import Dict, Any

def get_dataset_config(output_path: str) -> BaseDatasetConfig:
    return BaseDatasetConfig(
        formatter="custom", 
        meta_file_train="metadata.csv", 
        path=output_path,
        language="ful"
    )

def get_tts_config(output_path: str, dataset_config: BaseDatasetConfig) -> GlowTTSConfig:
    return GlowTTSConfig(
        batch_size=32,
        eval_batch_size=16,
        num_loader_workers=4,
        num_eval_loader_workers=4,
        run_eval=True,
        test_delay_epochs=-1,
        epochs=1000,
        text_cleaner="phoneme_cleaners",
        use_phonemes=True,
        phoneme_language="ful",
        phoneme_cache_path=os.path.join(output_path, "phoneme_cache"),
        print_step=25,
        print_eval=False,
        mixed_precision=True,
        output_path=output_path,
        datasets=[dataset_config],
        use_speaker_embedding=True,
        use_d_vector_file=True,
        d_vector_dim=256,
        optimizer="AdamW",
        optimizer_params={"betas": [0.8, 0.99], "weight_decay": 0.01},
        lr_scheduler="NoamLR",
        lr_scheduler_params={"warmup_steps": 4000, "min_lr": 1e-5},
    )

def update_config(config: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """Update configuration with new values."""
    for key, value in updates.items():
        if isinstance(value, dict) and key in config:
            config[key] = update_config(config[key], value)
        else:
            config[key] = value
    return config
