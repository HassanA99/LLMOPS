from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from config.model_config import VITSConfig, TrainingConfig
from models.vits_model import VITSModel
from data.dataset import FulaDataset

def train(config: TrainingConfig):
    """Training pipeline"""
    # Setup
    model_config = VITSConfig()
    model = VITSModel(model_config)
    model.to(config.device)
    
    # Data
    dataset = FulaDataset(
        root_dir=config.data_dir,
        metadata_file="metadata.csv",
        config=config
    )
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory
    )
    
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    
    # Training loop
    for epoch in range(config.num_epochs):
        for batch in dataloader:
            optimizer.zero_grad()
            outputs = model(batch)
            loss = model.compute_loss(outputs, batch)
            loss.backward()
            optimizer.step()
            
        # Save checkpoint
        if (epoch + 1) % config.save_interval == 0:
            save_checkpoint(model, optimizer, epoch, config)