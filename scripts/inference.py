def synthesize_speech(
    text: str,
    model: VITSModel,
    config: VITSConfig,
    script: str = 'adlam',
    speaker_id: int = 0,
    emotion: str = 'neutral'
) -> np.ndarray:
    """Generate speech from text"""
    # Process text
    text_processor = TextProcessor()
    text_data = text_processor.process_text(text, script)
    
    # Prepare input
    with torch.no_grad():
        inputs = {
            'text': text_data['normalized_text'],
            'phonemes': text_data['phonemes'],
            'speaker_id': speaker_id,
            'emotion': emotion
        }
        
        # Generate audio
        outputs = model(inputs)
        audio = outputs['audio'].cpu().numpy()
        
    return audio



# 6. Main Entry Point
if __name__ == "__main__":
    # Training
    config = TrainingConfig()
    train(config)
    
    # Inference example
    model = VITSModel(VITSConfig())
    text = "𞤖𞤢𞤤𞤢"  # "Hello" in Adlam
    audio = synthesize_speech(text, model, VITSConfig())