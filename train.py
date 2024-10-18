import os
import argparse
from datasets import load_dataset
from transformers import pipeline
from TTS.trainer import Trainer, TrainerArgs

from config import get_tts_config, get_dataset_config, update_config
from data_loader import load_fula_tts_samples, augment_data
from model import initialize_model, save_model_checkpoint, load_model_checkpoint
from utils import setup_output_path, setup_logger, save_metadata

def main(args):
    # Set up paths and configs
    output_path = setup_output_path(args.output_path)
    logger = setup_logger("train_log", os.path.join(output_path, "train.log"))
    
    dataset_config = get_dataset_config(output_path)
    config = get_tts_config(output_path, dataset_config)
    
    if args.config_updates:
        config = update_config(config, args.config_updates)

    # Load dataset and ASR pipeline
    logger.info("Loading dataset and ASR pipeline...")
    ds = load_dataset("cawoylel/FulaPretrainingSpeechCorpora")
    asr_pipe = pipeline("automatic-speech-recognition", model="cawoylel/mawdo-windanam-3000")

    # Load data samples
    logger.info("Loading and processing data samples...")
    train_samples, eval_samples = load_fula_tts_samples(ds, asr_pipe, config)
    
    if args.augment_data:
        logger.info("Augmenting training data...")
        train_samples = augment_data(train_samples)

    # Save metadata
    save_metadata(train_samples, os.path.join(output_path, "train_metadata.csv"))
    save_metadata(eval_samples, os.path.join(output_path, "eval_metadata.csv"))

    # Initialize model and trainer
    logger.info("Initializing model and trainer...")
    if args.checkpoint:
        model = load_model_checkpoint(config, args.checkpoint)
    else:
        model = initialize_model(config)
    
    trainer = Trainer(
        TrainerArgs(), config, output_path, model=model, train_samples=train_samples, eval_samples=eval_samples
    )

    # Start training
    logger.info("Starting training...")
    trainer.fit()

    # Save final model
    logger.info("Saving final model...")
    save_model_checkpoint(model, os.path.join(output_path, "final_model.pth"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Fula TTS model")
    parser.add_argument("--output_path", type=str, help="Custom output path")
    parser.add_argument("--checkpoint", type=str, help="Path to a model checkpoint to resume training")
    parser.add_argument("--augment_data", action="store_true", help="Apply data augmentation")
    parser.add_argument("--config_updates", type=dict, help="Updates to the default configuration")
    args = parser.parse_args()
    
    main(args)
